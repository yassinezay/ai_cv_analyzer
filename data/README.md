# Data Directory

This project downloads both datasets **automatically** from Kaggle using `kagglehub`.  
Downloads are cached locally — only happens once per machine.

---

## Setup Kaggle Credentials (required once)

### Step 1 – Create a Kaggle account
Go to https://www.kaggle.com (free account).

### Step 2 – Generate API key
Account → Settings → API → **Create New API Token**  
This downloads `kaggle.json`.

### Step 3 – Place the file

**Windows:**
```
C:\Users\<YourUsername>\.kaggle\kaggle.json
```
**Linux / Mac:**
```
~/.kaggle/kaggle.json
```

Content of the file:
```json
{"username": "your_kaggle_username", "key": "your_api_key_here"}
```

**Alternative – environment variables:**
```bash
set KAGGLE_USERNAME=your_username   # Windows
set KAGGLE_KEY=your_api_key

export KAGGLE_USERNAME=your_username  # Linux/Mac
export KAGGLE_KEY=your_api_key
```

---

## Datasets Used

### 1. Resume Dataset — `snehaanbhawal/resume-dataset`
- 2400+ resumes from livecareer.com
- Columns: `Resume_str` (text), `Category` (label)
- 23 job categories: HR, Designer, Information-Technology, Teacher, Advocate, Business-Development, Healthcare, Fitness, Agriculture, BPO, Sales, Consultant, Digital-Media, Automobile, Chef, Finance, Apparel, Engineering, Accountant, Construction, Public-Relations, Banking, Arts
- **Used for:** Training baseline + BiLSTM classifier

### 2. Job Description Dataset — `ravindrasinghrana/job-description-dataset`
- 5000 synthetic job postings (Faker + ChatGPT)
- Columns: `Job Title`, `Role`, `Job Description`, `Skills`, `Company Name`, `Location`
- **Used for:** CV ↔ Job matching (TF-IDF cosine similarity)

---

## How automatic download works

```python
import kagglehub

# In src/dataset.py – called automatically on first use
path = kagglehub.dataset_download("snehaanbhawal/resume-dataset")
# Cached at: ~/.cache/kagglehub/datasets/...
```

---

## Generated model files (created by the notebook)

After running `notebooks/project_analysis.ipynb`:

```
data/models/
├── baseline_clf.pkl      ← TF-IDF + LogReg classifier
├── baseline_vec.pkl      ← TF-IDF vectorizer
├── label_encoder.pkl     ← Category label encoder
├── lstm_best.pt          ← Best BiLSTM weights (PyTorch)
├── vocab.pkl             ← Vocabulary object
└── lstm_config.json      ← Model hyperparameter config
```
