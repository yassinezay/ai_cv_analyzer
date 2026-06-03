"""
streamlit_app.py – AI CV & LinkedIn Analyzer – Interactive Demo
Run: streamlit run app/streamlit_app.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import tempfile
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.dataset import (
    clean_text, extract_skills, compute_ats_score,
    load_dataset, load_job_descriptions, match_cv_to_jobs,
    prepare_data, SKILLS_DB, download_resume_dataset, download_jobs_dataset,
)
from src.train import train_baseline, save_baseline, load_baseline

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI CV Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        border: 1px solid #d0d3d9;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        color: #0e1117 !important;
    }
    .metric-card b, .metric-card small { color: #0e1117 !important; }
    .score-high   { color: #28a745 !important; font-weight: 700; }
    .score-medium { color: #e6a817 !important; font-weight: 700; }
    .score-low    { color: #dc3545 !important; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ─── PDF extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_bytes)
        path = f.name
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception:
        pass
    if not text.strip():
        try:
            import fitz
            doc = fitz.open(path)
            text = "\n".join(page.get_text() for page in doc)
        except Exception:
            pass
    os.unlink(path)
    return re.sub(r"\s+", " ", text).strip()


# ─── Cached resources (Kaggle downloads + model training) ────────────────────

@st.cache_resource(show_spinner="Downloading & training on Kaggle Resume Dataset … (first run only)")
def get_classifier():
    """
    Load saved model, or download the Kaggle dataset and train on it.
    Uses: snehaanbhawal/resume-dataset
    """
    clf_path = "data/models/baseline_clf.pkl"
    vec_path  = "data/models/baseline_vec.pkl"
    le_path   = "data/models/label_encoder.pkl"

    if os.path.exists(clf_path) and os.path.exists(vec_path):
        clf, vec = load_baseline("data/models")
        le = joblib.load(le_path) if os.path.exists(le_path) else None
        return clf, vec, le

    # Download real dataset and train
    df = load_dataset()          # calls kagglehub internally
    X_tr, X_te, y_tr, y_te, le = prepare_data(df, max_samples=2000)
    clf, vec, _ = train_baseline(X_tr, y_tr, X_te, y_te)

    os.makedirs("data/models", exist_ok=True)
    save_baseline(clf, vec, "data/models")
    joblib.dump(le, le_path)
    return clf, vec, le


@st.cache_data(show_spinner="Loading Kaggle Job Descriptions Dataset …", ttl=3600)
def get_jobs_df() -> pd.DataFrame:
    """
    Load the job descriptions dataset.
    Uses: ravindrasinghrana/job-description-dataset
    Returns a DataFrame ready for cosine-similarity matching.
    """
    return load_job_descriptions(max_rows=5000)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def predict_job(text: str, clf, vec, le):
    clean = clean_text(text)
    x = vec.transform([clean])
    proba = clf.predict_proba(x)[0]
    classes = le.classes_ if le is not None else clf.classes_
    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [{"job": classes[i], "score": float(proba[i])} for i in top3_idx]
    return top3[0]["job"], top3[0]["score"], top3


def ats_grade(score: float) -> str:
    return "A" if score >= 85 else ("B" if score >= 70 else
           ("C" if score >= 55 else ("D" if score >= 40 else "F")))


# ─── Charts ──────────────────────────────────────────────────────────────────

def gauge_chart(value: float, title: str, max_val: float = 100):
    color = "#28a745" if value >= 70 else ("#ffc107" if value >= 50 else "#dc3545")
    step = max_val / 4
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 36, "color": color}},
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {
                "range": [0, max_val],
                "tickvals": [0, step, step*2, step*3, max_val],
                "ticktext": [f"{int(v)}" for v in [0, step, step*2, step*3, max_val]],
                "tickfont": {"size": 11},
            },
            "bar": {"color": color},
            "steps": [
                {"range": [0,          max_val*0.5], "color": "#ffeaea"},
                {"range": [max_val*0.5, max_val*0.7], "color": "#fff8e1"},
                {"range": [max_val*0.7, max_val],    "color": "#e8f5e9"},
            ],
        },
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=40, r=40, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#0e1117",
    )
    return fig


def skills_bar(skills_dict: dict):
    cats   = [c for c, s in skills_dict.items() if s]
    counts = [len(skills_dict[c]) for c in cats]
    if not cats:
        return None
    fig = px.bar(x=counts, y=cats, orientation="h",
                 labels={"x": "Skills found", "y": ""},
                 color=counts, color_continuous_scale="Blues",
                 title="Skills Detected by Category")
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=10),
                      coloraxis_showscale=False)
    return fig


def job_bar(top3: list):
    jobs   = [d["job"] for d in top3]
    scores = [d["score"] * 100 for d in top3]
    fig = go.Figure(go.Bar(
        x=scores, y=jobs, orientation="h",
        marker_color=["#0066CC", "#3399FF", "#99CCFF"],
        text=[f"{s:.1f}%" for s in scores], textposition="auto",
    ))
    fig.update_layout(title="Job Category Prediction", xaxis_title="Confidence (%)",
                      height=200, margin=dict(l=0, r=0, t=40, b=10))
    return fig


def ats_breakdown_bar(details: dict):
    cats   = [d.replace("_", " ").title() for d in details]
    scores = [details[d]["score"] for d in details]
    maxes  = [details[d]["max"]   for d in details]
    colors = ["#28a745" if s/m >= 0.7 else ("#ffc107" if s/m >= 0.4 else "#dc3545")
              for s, m in zip(scores, maxes)]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Score", x=cats, y=scores, marker_color=colors))
    fig.add_trace(go.Bar(name="Max",   x=cats, y=maxes,  marker_color="lightgrey", opacity=0.4))
    fig.update_layout(title="ATS Score Breakdown", barmode="overlay",
                      yaxis_title="Points", height=300,
                      margin=dict(l=0, r=0, t=40, b=10))
    return fig


# ─── Recommendations ─────────────────────────────────────────────────────────

def generate_recommendations(text, skills, ats_details):
    recs = []
    t  = text.lower()
    wc = len(text.split())

    if not re.search(r"\S+@\S+", text):
        recs.append(("HIGH",   "Add email address",         "No email detected – essential for ATS and recruiters.",           "name@email.com"))
    if not re.search(r"\+?\d[\d\s\-().]{7,}\d", text):
        recs.append(("HIGH",   "Add phone number",          "No phone number found.",                                          "+33 6 12 34 56 78"))
    if "skills" not in t and "compétences" not in t:
        recs.append(("HIGH",   "Add a Skills section",      "ATS parsers look for an explicit Skills section.",               "Skills: Python, SQL, Docker, Git"))
    if "experience" not in t and "emploi" not in t:
        recs.append(("HIGH",   "Add Experience section",    "Experience is the #1 section recruiters look for.",               "Work Experience: Role | Company | Dates"))
    if "summary" not in t and "profile" not in t and "objective" not in t:
        recs.append(("MEDIUM", "Add professional summary",  "A 3-line summary boosts ATS score and recruiter attention.",      "Data Scientist with 2 years experience in NLP…"))
    if "projects" not in t and "projet" not in t:
        recs.append(("MEDIUM", "Add Projects section",       "Projects prove skills – critical for early-career profiles.",    "Project: AI CV Analyzer (Python, NLP, Streamlit)"))
    if wc < 300:
        recs.append(("MEDIUM", "Expand your CV",             f"Only {wc} words found. Target 300–700 words.",                  "Add bullet points for each role/project."))
    elif wc > 800:
        recs.append(("LOW",    "Shorten your CV",            f"{wc} words is too long for a 1-page CV.",                       "Remove old/irrelevant experience."))
    n_tech = sum(len(v) for v in skills.values())
    if n_tech < 5:
        recs.append(("HIGH",   "List more technical skills", f"Only {n_tech} skills detected. List all tools explicitly.",     "Python • PyTorch • Docker • AWS • PostgreSQL"))
    if "github" not in t:
        recs.append(("LOW",    "Add GitHub profile",         "A GitHub link with projects is a strong signal for tech roles.", "github.com/your-username"))
    if "linkedin" not in t:
        recs.append(("LOW",    "Add LinkedIn profile",       "Recruiters verify LinkedIn. Include the URL.",                   "linkedin.com/in/firstname-lastname"))
    return recs


# ─── Main app ─────────────────────────────────────────────────────────────────

def main():
    st.title("📄 AI CV & LinkedIn Analyzer")
    st.caption("Powered by Kaggle Resume Dataset · Bidirectional LSTM · TF-IDF + LogReg")
    st.markdown("---")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📎 Upload your CV")
        uploaded = st.file_uploader("Drop a PDF here", type=["pdf"])
        st.markdown("---")
        st.subheader("✏️ Or paste text directly")
        raw_text_input = st.text_area("Resume text", height=200,
                                       placeholder="Paste your CV text here…")
        st.markdown("---")
        st.caption("**Datasets used:**\n"
                   "- snehaanbhawal/resume-dataset\n"
                   "- ravindrasinghrana/job-description-dataset\n\n"
                   "**Models:** TF-IDF + LogReg · BiLSTM (PyTorch)")

    # ── Resolve input ─────────────────────────────────────────────────────────
    raw_text = ""
    if uploaded:
        with st.spinner("Extracting text from PDF …"):
            raw_text = extract_text_from_pdf(uploaded.read())
        if not raw_text:
            st.error("Could not extract text. Try pasting the text directly.")
            return
        st.success(f"PDF processed – {len(raw_text.split())} words extracted.")
    elif raw_text_input.strip():
        raw_text = raw_text_input.strip()
    else:
        st.info("Upload a CV PDF or paste text in the sidebar to begin.")
        with st.expander("How it works"):
            st.markdown("""
**Step 1 – Upload** your CV as PDF (or paste the text)

**Step 2 – The AI analyses your CV:**
- **ATS Score** – structured scoring on 5 criteria (0–100)
- **Skills Detection** – 150+ skills across 6 categories
- **Job Prediction** – ML model trained on 2400+ real Kaggle resumes
- **Recommendations** – prioritised improvement tips

**Step 3 – Job Matching** – match your CV against 5000 real Kaggle job postings

**Models:**
- Baseline: TF-IDF + Logistic Regression (scikit-learn)
- Deep Learning: Bidirectional LSTM (PyTorch) – see notebook
""")
        return

    # ── Load classifier (trains from Kaggle dataset if not cached) ────────────
    with st.spinner("Loading model …"):
        clf, vec, le = get_classifier()

    # ── Run analysis ──────────────────────────────────────────────────────────
    with st.spinner("Analysing your CV …"):
        ats_score, ats_details = compute_ats_score(raw_text)
        skills_dict = extract_skills(raw_text)
        top_job, confidence, top3 = predict_job(raw_text, clf, vec, le)
        recs = generate_recommendations(raw_text, skills_dict, ats_details)
        grade = ats_grade(ats_score)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Overview", "🛠 Skills", "💡 Recommendations", "🔍 Job Matching"]
    )

    # ── Tab 1: Overview ───────────────────────────────────────────────────────
    with tab1:
        c1, c2, c3 = st.columns(3)
        grade_css = "score-high" if grade in ("A","B") else ("score-medium" if grade=="C" else "score-low")
        CARD = "background:#f0f2f6; border:1px solid #d0d3d9; border-radius:10px; padding:14px 18px; margin:8px 0; color:#111111;"
        grade_color = "#28a745" if grade in ("A","B") else ("#e6a817" if grade=="C" else "#dc3545")
        with c1:
            st.plotly_chart(gauge_chart(ats_score, "ATS Score"), use_container_width=True)
            st.markdown(f"<div style='{CARD}'>"
                        f"<b style='color:#111111;'>Grade:</b> "
                        f"<span style='color:{grade_color}; font-weight:700;'>{grade}</span><br>"
                        f"<small style='color:#444444;'>{ats_score:.0f}/100 pts</small></div>",
                        unsafe_allow_html=True)
        with c2:
            st.plotly_chart(gauge_chart(confidence*100, "Prediction Confidence"), use_container_width=True)
            st.markdown(f"<div style='{CARD}'>"
                        f"<b style='color:#111111;'>Predicted Role:</b><br>"
                        f"<span style='color:#28a745; font-weight:700;'>{top_job}</span></div>",
                        unsafe_allow_html=True)
        with c3:
            n_sk = sum(len(v) for v in skills_dict.values())
            st.plotly_chart(gauge_chart(min(100, n_sk*5), "Skills Coverage"), use_container_width=True)
            st.markdown(f"<div style='{CARD}'>"
                        f"<b style='color:#111111;'>Skills found:</b> <span style='color:#111111;'>{n_sk}</span><br>"
                        f"<b style='color:#111111;'>Word count:</b> <span style='color:#111111;'>{len(raw_text.split())}</span></div>",
                        unsafe_allow_html=True)

        st.markdown("---")
        cl, cr = st.columns(2)
        with cl: st.plotly_chart(job_bar(top3), use_container_width=True)
        with cr: st.plotly_chart(ats_breakdown_bar(ats_details), use_container_width=True)

        st.subheader("Sections Detected")
        found_secs = ats_details["sections"]["found"]
        all_s = ["contact","summary","experience","education","skills","projects","certifications"]
        cols = st.columns(len(all_s))
        for col, sec in zip(cols, all_s):
            col.markdown(f"{'✅' if sec in found_secs else '❌'} **{sec.title()}**")

    # ── Tab 2: Skills ─────────────────────────────────────────────────────────
    with tab2:
        fig = skills_bar(skills_dict)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No skills detected. List your technologies explicitly in your CV.")

        st.markdown("---")
        st.subheader("Detected skills")
        for cat, found in skills_dict.items():
            if found:
                st.markdown(f"**{cat}:** " + " · ".join(f"`{s}`" for s in found))

        st.markdown("---")
        st.subheader("Missing skills (from taxonomy)")
        for cat, all_sk in SKILLS_DB.items():
            missing = [s for s in all_sk if s not in (skills_dict.get(cat) or [])]
            if missing:
                st.markdown(f"**{cat}:** " + " · ".join(f"`{s}`" for s in missing[:6]))

    # ── Tab 3: Recommendations ────────────────────────────────────────────────
    with tab3:
        for priority, label, color in [("HIGH","🔴 High Priority","#dc3545"),
                                        ("MEDIUM","🟡 Medium Priority","#ffc107"),
                                        ("LOW","🟢 Nice to Have","#28a745")]:
            group = [r for r in recs if r[0] == priority]
            if group:
                st.subheader(label)
                for _, title, desc, example in group:
                    with st.expander(title):
                        st.write(desc)
                        st.code(example)
        if not recs:
            st.success("Your CV looks great! No major issues detected.")

    # ── Tab 4: Job Matching (Kaggle dataset) ──────────────────────────────────
    with tab4:
        st.subheader("Match your CV against real Kaggle job postings")
        st.caption("Dataset: ravindrasinghrana/job-description-dataset · 5,000 job postings")

        match_mode = st.radio(
            "Match against:",
            ["🗄️ Kaggle Job Descriptions Database", "✏️ Paste a specific job description"],
            horizontal=True,
        )

        if match_mode.startswith("🗄️"):
            with st.spinner("Loading job descriptions from Kaggle …"):
                jobs_df = get_jobs_df()

            if jobs_df.empty:
                st.error("Job descriptions dataset unavailable. "
                         "Check your Kaggle credentials (see data/README.md).")
            else:
                role_filter = ""
                if "Role" in jobs_df.columns:
                    roles = ["All roles"] + sorted(jobs_df["Role"].dropna().unique().tolist())
                    selected_role = st.selectbox("Filter by role (optional)", roles)
                    if selected_role != "All roles":
                        jobs_df = jobs_df[jobs_df["Role"] == selected_role]

                if st.button("🔍 Find matching jobs", type="primary"):
                    with st.spinner("Computing TF-IDF similarity …"):
                        matches = match_cv_to_jobs(raw_text, jobs_df, top_k=5)

                    if matches.empty:
                        st.warning("No matches found.")
                    else:
                        st.markdown(f"**Top {len(matches)} matches from {len(jobs_df)} job postings:**")
                        for _, row in matches.iterrows():
                            score_pct = row["match_score"] * 100
                            score_color = "#28a745" if score_pct >= 30 else ("#e6a817" if score_pct >= 15 else "#dc3545")
                            title   = row.get("Job Title", "N/A")
                            role    = row.get("Role", "")
                            company = row.get("Company Name", "")
                            loc     = row.get("Location", "")
                            desc    = str(row.get("Job Description", ""))[:300] + "…"
                            skills  = str(row.get("Skills", ""))

                            role_html    = f" <span style='color:#555555;'>· {role}</span>" if role else ""
                            company_html = f"<br><small style='color:#555555;'>🏢 {company}</small>" if company else ""
                            loc_html     = f"<small style='color:#555555;'> · 📍 {loc}</small>" if loc else ""
                            skills_html  = f"<p style='margin:4px 0 0; color:#333333;'><b style='color:#111111;'>Skills required:</b> {skills[:200]}</p>" if skills else ""

                            st.markdown(f"""
<div style='background:#ffffff; border:2px solid #e0e4ea; border-radius:12px;
            padding:16px 20px; margin:10px 0; color:#111111;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);'>
  <span style='font-size:1.05em; font-weight:700; color:#111111;'>{title}</span>{role_html}
  {company_html}{loc_html}
  <br>
  <span style='color:{score_color}; font-weight:700; font-size:1.1em;'>Match: {score_pct:.1f}%</span>
  <details style='margin-top:8px;'>
    <summary style='cursor:pointer; color:#444444; font-size:0.9em; user-select:none;'>
      Description preview
    </summary>
    <p style='font-size:0.85em; color:#333333; margin:8px 0 0;'>{desc}</p>
    {skills_html}
  </details>
</div>""", unsafe_allow_html=True)

                        # Missing keywords
                        st.markdown("---")
                        st.subheader("Keywords from top job missing in your CV")
                        top_jd = matches.iloc[0]["Job Description"]
                        jd_words = set(clean_text(top_jd).split())
                        cv_words = set(clean_text(raw_text).split())
                        missing_kw = sorted(jd_words - cv_words)[:25]
                        if missing_kw:
                            st.markdown(" · ".join(f"`{w}`" for w in missing_kw))
                        else:
                            st.success("Your CV already covers most keywords from the top match!")

        else:
            # Manual paste mode
            job_desc = st.text_area("Paste a job description", height=200,
                                     placeholder="Senior Data Scientist – 3+ years Python, ML…")
            if job_desc.strip():
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity

                tfidf = TfidfVectorizer(stop_words="english")
                tfidf.fit([raw_text, job_desc])
                match = float(cosine_similarity(
                    tfidf.transform([raw_text]),
                    tfidf.transform([job_desc])
                )[0][0]) * 100

                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(gauge_chart(match, "CV ↔ Job Match Score"), use_container_width=True)
                with c2:
                    jd_words = set(clean_text(job_desc).split())
                    cv_words = set(clean_text(raw_text).split())
                    missing  = sorted(jd_words - cv_words)[:20]
                    st.subheader("Missing keywords")
                    if missing:
                        st.markdown(" · ".join(f"`{w}`" for w in missing))
                    else:
                        st.success("Great coverage of the job description!")


if __name__ == "__main__":
    main()
