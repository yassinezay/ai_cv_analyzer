"""
dataset.py – Data loading (Kaggle via kagglehub), preprocessing, vocabulary, PyTorch Dataset
"""
import os
import re
import glob
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import torch
from torch.utils.data import Dataset

nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)

# ── Real Kaggle Resume Dataset categories (snehaanbhawal/resume-dataset) ────
JOB_CATEGORIES = [
    "ACCOUNTANT", "ADVOCATE", "AGRICULTURE", "APPAREL", "ARTS",
    "AUTOMOBILE", "AVIATION", "BANKING", "BPO", "BUSINESS-DEVELOPMENT",
    "CHEF", "CONSTRUCTION", "CONSULTANT", "DESIGNER", "DIGITAL-MEDIA",
    "ENGINEERING", "FINANCE", "FITNESS", "HEALTHCARE", "HR",
    "INFORMATION-TECHNOLOGY", "PUBLIC-RELATIONS", "SALES", "TEACHER",
]

# ── Weak classes merged into AUTRE (F1 < 0.40 on the full 24-class model) ───
WEAK_CLASSES = [
    "AGRICULTURE", "APPAREL", "ARTS", "AUTOMOBILE", "BPO", "FITNESS",
]


def remap_categories(df: pd.DataFrame, weak_classes: list = None) -> pd.DataFrame:
    """Replace low-F1 categories with AUTRE so the model focuses on learnable classes."""
    if weak_classes is None:
        weak_classes = WEAK_CLASSES
    df = df.copy()
    df["Category"] = df["Category"].apply(
        lambda c: "AUTRE" if c in weak_classes else c
    )
    return df


# ── Skills taxonomy ─────────────────────────────────────────────────────────
SKILLS_DB = {
    "Programming":    ["python", "java", "javascript", "typescript", "c++", "c#",
                       "ruby", "php", "go", "rust", "r", "scala", "sql", "bash"],
    "Data / ML":      ["machine learning", "deep learning", "neural network", "nlp",
                       "tensorflow", "pytorch", "keras", "scikit-learn", "pandas",
                       "numpy", "xgboost", "bert", "transformers", "computer vision"],
    "Cloud / DevOps": ["docker", "kubernetes", "aws", "azure", "gcp", "ci/cd",
                       "terraform", "jenkins", "linux", "git", "github", "gitlab"],
    "Web Dev":        ["react", "vue", "angular", "node.js", "django", "flask",
                       "fastapi", "html", "css", "rest api", "graphql"],
    "Databases":      ["mysql", "postgresql", "mongodb", "redis", "elasticsearch",
                       "oracle", "sqlite", "dynamodb", "cassandra"],
    "Soft Skills":    ["leadership", "communication", "teamwork", "agile", "scrum",
                       "project management", "problem solving", "analytical"],
}

ACTION_VERBS = [
    "achieved", "built", "created", "designed", "developed", "implemented",
    "improved", "increased", "led", "managed", "optimized", "reduced", "solved",
    "delivered", "launched", "analyzed", "coordinated", "generated", "maintained",
    "produced", "supervised", "transformed", "automated", "deployed", "architected",
]

CV_SECTIONS = [
    "education", "experience", "skills", "summary", "objective",
    "projects", "certifications", "contact", "profile", "formation",
]


# ── Kaggle download helpers ───────────────────────────────────────────────────

def _find_csv(directory: str, keywords: list = None) -> str | None:
    """Return first CSV found in directory, optionally filtered by filename keywords."""
    csvs = glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True)
    if not csvs:
        return None
    if keywords:
        for csv in csvs:
            if any(kw in os.path.basename(csv).lower() for kw in keywords):
                return csv
    return sorted(csvs)[0]


def download_resume_dataset() -> str | None:
    """
    Download snehaanbhawal/resume-dataset via kagglehub.
    Returns path to the CSV file, or None on failure.
    kagglehub caches the download – subsequent calls are instant.
    """
    try:
        import kagglehub
        print("[Kaggle] Downloading resume dataset (snehaanbhawal/resume-dataset)...")
        path = kagglehub.dataset_download("snehaanbhawal/resume-dataset")
        csv = _find_csv(path, keywords=["resume"])
        print(f"[Kaggle] Resume CSV: {csv}")
        return csv
    except Exception as e:
        print(f"[Kaggle] Resume download failed: {e}")
        print("         Make sure your Kaggle credentials are configured.")
        print("         See data/README.md for setup instructions.")
        return None


def download_jobs_dataset() -> str | None:
    """
    Download ravindrasinghrana/job-description-dataset via kagglehub.
    Returns path to the CSV file, or None on failure.
    """
    try:
        import kagglehub
        print("[Kaggle] Downloading job descriptions dataset...")
        path = kagglehub.dataset_download("ravindrasinghrana/job-description-dataset")
        csv = _find_csv(path)
        print(f"[Kaggle] Jobs CSV: {csv}")
        return csv
    except Exception as e:
        print(f"[Kaggle] Jobs download failed: {e}")
        return None


# ── Text cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str, remove_stops: bool = True) -> str:
    """Lowercase, remove URLs/emails, replace numbers, remove punctuation, optionally drop stopwords."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\d+", " NUM ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if remove_stops:
        stops = set(stopwords.words("english"))
        tokens = [t for t in text.split() if t not in stops and len(t) > 2]
        text = " ".join(tokens)
    return text


# ── Resume dataset loading ────────────────────────────────────────────────────

def load_dataset(csv_path: str = None) -> pd.DataFrame:
    """
    Load the Kaggle Resume Dataset.
    Priority:
      1. csv_path if provided and exists
      2. Auto-download via kagglehub
      3. Synthetic fallback (warns user)
    """
    # Path provided explicitly
    if csv_path and os.path.exists(csv_path):
        return _load_resume_csv(csv_path)

    # Try kagglehub auto-download
    downloaded = download_resume_dataset()
    if downloaded:
        return _load_resume_csv(downloaded)

    # Last resort: synthetic data
    print("[Dataset] WARNING: Using synthetic data. "
          "Configure Kaggle credentials to use real data.")
    return _synthetic_dataset()


def _load_resume_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalise column names: Kaggle dataset uses 'Resume_str' or 'Resume'
    if "Resume" in df.columns and "Resume_str" not in df.columns:
        df = df.rename(columns={"Resume": "Resume_str"})
    df = df[["Resume_str", "Category"]].dropna()
    print(f"[Dataset] Loaded {len(df)} resumes, "
          f"{df['Category'].nunique()} categories")
    print(f"          Categories: {sorted(df['Category'].unique().tolist())}")
    return df


def _synthetic_dataset() -> pd.DataFrame:
    """Minimal fallback dataset – only used when Kaggle is unavailable."""
    templates = {
        "Information-Technology": (
            "software engineer java python javascript react aws docker kubernetes "
            "microservices rest api postgresql git agile scrum backend frontend cloud"
        ),
        "HR": (
            "human resources recruitment talent acquisition onboarding performance "
            "management payroll employee relations training hr information systems"
        ),
        "Sales": (
            "sales account management b2b crm salesforce lead generation revenue "
            "business development negotiation customer success pipeline quota"
        ),
        "Finance": (
            "finance accounting financial analysis excel sql reporting budgeting "
            "forecasting audit compliance risk management balance sheet"
        ),
        "Healthcare": (
            "healthcare nursing medical clinical patient care hospital physician "
            "pharmacist laboratory diagnosis treatment ehr electronic health"
        ),
        "Engineering": (
            "mechanical engineer autocad solidworks manufacturing design simulation "
            "project management cad cam materials thermodynamics quality control"
        ),
        "Designer": (
            "ui ux designer figma photoshop illustrator html css javascript react "
            "responsive design wireframe prototype user research visual identity"
        ),
        "Teacher": (
            "teacher educator curriculum lesson plan classroom management student "
            "assessment pedagogy instruction learning outcomes academic"
        ),
        "Banking": (
            "banking finance investment portfolio risk credit loan compliance "
            "regulatory financial products trading analyst relationship manager"
        ),
        "Consultant": (
            "management consultant strategy business analysis client stakeholder "
            "project delivery process improvement change management advisory"
        ),
    }
    np.random.seed(42)
    records = []
    for cat, base in templates.items():
        words = base.split()
        for _ in range(60):
            n = np.random.randint(50, 85)
            idx = np.random.choice(len(words), n, replace=True)
            text = " ".join(words[j] for j in sorted(idx))
            records.append({"Resume_str": text, "Category": cat})
    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[Dataset] Synthetic: {len(df)} samples, {df['Category'].nunique()} categories")
    return df


# ── Job descriptions dataset ──────────────────────────────────────────────────

def load_job_descriptions(csv_path: str = None, max_rows: int = 5000) -> pd.DataFrame:
    """
    Load the Kaggle Job Descriptions Dataset (ravindrasinghrana/job-description-dataset).
    Returns a clean DataFrame with columns: Job Title, Role, Job Description, Skills, Company Name, Location.
    """
    if not csv_path:
        csv_path = download_jobs_dataset()
    if not csv_path or not os.path.exists(csv_path):
        print("[Jobs] Dataset unavailable. Job matching will use text input only.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path, nrows=max_rows)

    # Select relevant columns (tolerant of column-name variations)
    col_map = {
        "Job Title":       ["Job Title", "job_title", "title"],
        "Role":            ["Role", "role", "job_role"],
        "Job Description": ["Job Description", "job_description", "description"],
        "Skills":          ["Skills", "skills", "required_skills"],
        "Company Name":    ["Company Name", "company_name", "company"],
        "Location":        ["Location", "location", "city"],
    }
    rename = {}
    for standard, variants in col_map.items():
        for v in variants:
            if v in df.columns:
                rename[v] = standard
                break

    df = df.rename(columns=rename)
    keep = [c for c in col_map if c in df.columns]
    df = df[keep].dropna(subset=["Job Description"])

    # Build combined text for similarity matching
    df["match_text"] = (
        df.get("Job Title", "").fillna("") + " " +
        df.get("Role", "").fillna("") + " " +
        df.get("Skills", "").fillna("") + " " +
        df.get("Job Description", "").fillna("")
    )
    df["match_text"] = df["match_text"].apply(clean_text)

    print(f"[Jobs] Loaded {len(df)} job descriptions")
    return df.reset_index(drop=True)


def match_cv_to_jobs(cv_text: str, jobs_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """
    Compute TF-IDF cosine similarity between a CV and all job descriptions.
    Returns top_k best-matching rows with an added 'match_score' column.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if jobs_df.empty:
        return pd.DataFrame()

    cv_clean = clean_text(cv_text)
    corpus = [cv_clean] + jobs_df["match_text"].tolist()

    tfidf = TfidfVectorizer(max_features=15000, stop_words="english", ngram_range=(1, 2))
    matrix = tfidf.fit_transform(corpus)

    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    jobs_df = jobs_df.copy()
    jobs_df["match_score"] = scores
    return jobs_df.nlargest(top_k, "match_score").reset_index(drop=True)


# ── Train / test split ───────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame, max_samples: int = 2000, test_size: float = 0.2):
    """
    Stratified train/test split with cleaned text and encoded labels.
    max_samples enforces the CPU < 10 min training constraint.
    """
    if len(df) > max_samples:
        n_per_class = max_samples // df["Category"].nunique()
        df = (
            df.groupby("Category", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), max(1, n_per_class)), random_state=42))
            .reset_index(drop=True)
        )

    le = LabelEncoder()
    df = df.copy()
    df["text_clean"] = df["Resume_str"].apply(clean_text)
    df["label"] = le.fit_transform(df["Category"])

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text_clean"].tolist(), df["label"].tolist(),
        test_size=test_size, random_state=42, stratify=df["label"],
    )
    print(f"[Data] Train={len(X_tr)}  Test={len(X_te)}  Classes={len(le.classes_)}")
    return X_tr, X_te, y_tr, y_te, le


# ── Vocabulary ────────────────────────────────────────────────────────────────

class Vocabulary:
    PAD_IDX, UNK_IDX = 0, 1

    def __init__(self, max_size: int = 8000, min_freq: int = 2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    @property
    def size(self):
        return len(self.word2idx)

    def build(self, texts: list):
        counter = Counter(w for t in texts for w in t.split())
        vocab = [w for w, c in counter.most_common(self.max_size - 2) if c >= self.min_freq]
        for w in vocab:
            idx = len(self.word2idx)
            self.word2idx[w] = idx
            self.idx2word[idx] = w
        print(f"[Vocab] Size: {self.size}")
        return self

    def encode(self, text: str, max_len: int = 150) -> list:
        tokens = text.split()[:max_len]
        ids = [self.word2idx.get(t, 1) for t in tokens]
        ids += [0] * (max_len - len(ids))
        return ids


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class ResumeDataset(Dataset):
    def __init__(self, texts: list, labels: list, vocab: Vocabulary, max_len: int = 150):
        self.samples = [
            (
                torch.tensor(vocab.encode(t, max_len), dtype=torch.long),
                torch.tensor(lbl, dtype=torch.long),
            )
            for t, lbl in zip(texts, labels)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── CV analysis helpers ───────────────────────────────────────────────────────

def extract_skills(text: str) -> dict:
    """Return {category: [matched_skills]} for all skill categories."""
    t = text.lower()
    return {
        cat: [s for s in skills if re.search(r"\b" + re.escape(s) + r"\b", t)]
        for cat, skills in SKILLS_DB.items()
    }


def compute_ats_score(text: str) -> tuple:
    """Compute ATS score 0–100 and return (score, details_dict)."""
    t = text.lower()
    score, details = 0, {}

    found_sec = [s for s in CV_SECTIONS if s in t]
    s = min(40, len(found_sec) * 8)
    score += s
    details["sections"] = {"score": s, "found": found_sec, "max": 40}

    n_verbs = sum(1 for v in ACTION_VERBS if v in t)
    s = min(20, n_verbs * 2)
    score += s
    details["action_verbs"] = {"score": s, "count": n_verbs, "max": 20}

    numbers = re.findall(r"\d+%|\d+\+|\b\d{3,}\b", text)
    s = min(20, len(numbers) * 4)
    score += s
    details["quantified"] = {"score": s, "count": len(numbers), "max": 20}

    wc = len(text.split())
    s = 10 if 300 <= wc <= 700 else (5 if wc < 300 else 7)
    score += s
    details["length"] = {"score": s, "word_count": wc, "max": 10}

    all_skills = [sk for skills in SKILLS_DB.values() for sk in skills]
    n_skills = sum(1 for sk in all_skills if sk in t)
    s = min(10, n_skills)
    score += s
    details["skills_keywords"] = {"score": s, "count": n_skills, "max": 10}

    return score, details
