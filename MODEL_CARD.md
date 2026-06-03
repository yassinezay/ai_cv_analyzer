# Model Card – AI CV & LinkedIn Analyzer

## Model Overview

| Field | Details |
|-------|---------|
| **Task** | Multi-class text classification (job category prediction) |
| **Model type** | Bidirectional LSTM (PyTorch) + TF-IDF baseline (scikit-learn) |
| **Input** | Resume plain text (extracted from PDF) |
| **Output** | Job category probabilities (10–25 classes) |
| **Language** | English |
| **Framework** | PyTorch 2.x + scikit-learn 1.x |

---

## Dataset

| Field | Value |
|-------|-------|
| **Name** | Kaggle Resume Dataset |
| **Source** | [Livecareer.com](https://www.livecareer.com/) via Kaggle |
| **Link** | https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset |
| **Samples** | ~2,400 labelled resumes |
| **Classes** | 25 job categories |
| **Features** | Raw resume text (cleaned and tokenised) |
| **Target variable** | Job category (string label) |
| **Train/Test split** | 80% / 20%, stratified by class |

---

## Model Architecture

### Baseline
- **TF-IDF Vectorizer** (10,000 features, unigrams + bigrams, sublinear TF)
- **Logistic Regression** (C=1.0, L2 regularisation, multi-class: multinomial)

### Deep Learning
```
Embedding(vocab_size=8000, embed_dim=128)
    ↓
Dropout(p=0.3)
    ↓
BiLSTM(hidden=128, num_layers=2, bidirectional=True, dropout=0.3)
    ↓
Concat [forward_h, backward_h] → (batch, 256)
    ↓
Dropout(p=0.3)
    ↓
Linear(256 → num_classes)
    ↓
CrossEntropyLoss
```

**Trainable parameters:** ~2.3 M  
**Optimizer:** Adam (lr=1e-3, weight_decay=1e-5)  
**LR scheduler:** StepLR (step=3, gamma=0.5)  
**Gradient clipping:** norm=1.0  
**Epochs:** 5 | **Batch size:** 32 | **Max sequence length:** 150 tokens

---

## Performance

| Metric | TF-IDF + LogReg | BiLSTM (best) |
|--------|-----------------|---------------|
| Accuracy | see notebook | see notebook |
| F1-score (weighted) | see notebook | see notebook |
| Precision (weighted) | see notebook | see notebook |
| Recall (weighted) | see notebook | see notebook |
| Training time | < 5s | < 5 min (CPU) |

> Run `notebooks/project_analysis.ipynb` to see actual metric values.

---

## Hyperparameter Experiments

| Run | Hidden | Dropout | LR | Val Acc | F1 |
|-----|--------|---------|-----|---------|-----|
| 1 | 64 | 0.3 | 1e-3 | see notebook | see notebook |
| 2 | 128 | 0.3 | 1e-3 | see notebook | see notebook |
| 3 | 128 | 0.5 | 1e-3 | see notebook | see notebook |

---

## Intended Use

### Primary use cases
- Help job seekers understand how ATS systems categorise their CV
- Provide actionable recommendations to improve CV–job alignment
- Educational tool for demonstrating NLP classification pipelines

### Out-of-scope use cases
- Automated hiring or candidate rejection without human review
- Legal or medical professional screening without domain experts
- Non-English CVs (model was not trained on multilingual data)

---

## Limitations

1. **Language:** English-only. Performance degrades on French, Arabic, or mixed-language CVs.
2. **Fixed categories:** The 25 job categories are from 2020-era Livecareer data. Emerging roles (MLOps Engineer, Prompt Engineer) may be misclassified.
3. **Text extraction quality:** Complex PDF layouts (tables, 2-column, images) may produce garbled text that degrades accuracy.
4. **Small dataset:** ~2000 training samples is insufficient for production use. Enterprise systems use millions of labelled examples.
5. **No seniority detection:** The model classifies *role type* but not experience level.

---

## Bias and Risks

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Demographic bias** | Training data from one platform may over-represent certain demographics | Audit model predictions by subgroup |
| **Geographic bias** | Western CV conventions rewarded by ATS scorer | Acknowledge in docs; plan multilingual support |
| **Feedback loop** | If CVs are optimised for this model, diversity may decrease | Use as advisory tool only |
| **False confidence** | High confidence score ≠ correct prediction for uncommon roles | Always show top-3 predictions |

---

## Failure Cases

- **Career changers** (mixed-domain text) → ambiguous predictions near decision boundary
- **Very short CVs** (< 100 words) → insufficient signal for classification
- **Overloaded academic CVs** → classified as Research Scientist regardless of target role
- **CVs with only acronyms** (e.g. "AWS, GCP, CI/CD") without context → lower accuracy

---

## ATS Scoring Module

The rule-based ATS scorer evaluates:

| Criterion | Weight | Details |
|-----------|--------|---------|
| Sections present | 40 pts | education, experience, skills, summary, contact, projects, certifications |
| Action verbs | 20 pts | achieved, built, improved, led, reduced, etc. |
| Quantified results | 20 pts | percentages, large numbers, "+N" patterns |
| CV length | 10 pts | Optimal: 300–700 words |
| Skill keywords | 10 pts | Matched against curated skills taxonomy |

---

## Citation

```bibtex
@misc{ai-cv-analyzer-2026,
  title  = {AI CV \& LinkedIn Analyzer -- Deep Learning NLP Pipeline},
  author = {Student Team, BTS Applied ML 2026},
  year   = {2026},
  note   = {Academic project -- BTS Applied ML course}
}
```
