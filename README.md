# ICD-10 Medical Code Classification Project

Automatic classification of Spanish/Catalan medical diagnosis literals into ICD-10 categories, built for the **UAB-ASHO AI Codification** Kaggle competition as part of *Fundamentals of Natural Language Processing* course.

---

## Problem

Given a short clinical **literal** (e.g. `"HTA irc 6"`, `"miocardiopatía dilatada"`, `"cos estrany"`), predict its **ICD-10 category**: the first character of the code.

The dataset contains both standard ICD-10 letter codes (A–Z) and legacy numeric codes (0–9, likely CIE-9 procedure codes), which together form 36 possible output classes.

| Category | Description |
|----------|-------------|
| A–B | Certain infectious and parasitic diseases |
| C–D48 | Neoplasms |
| D50–D89 | Diseases of the blood and blood-forming organs |
| E | Endocrine, nutritional and metabolic diseases |
| F | Mental and behavioural disorders |
| G | Diseases of the nervous system |
| H00–H59 | Diseases of the eye and adnexa |
| H60–H95 | Diseases of the ear and mastoid process |
| I | Diseases of the circulatory system |
| J | Diseases of the respiratory system |
| K | Diseases of the digestive system |
| L | Diseases of the skin and subcutaneous tissue |
| M | Diseases of the musculoskeletal system |
| N | Diseases of the genitourinary system |
| O | Pregnancy, childbirth and the puerperium |
| P | Certain conditions originating in the perinatal period |
| Q | Congenital malformations and chromosomal abnormalities |
| R | Symptoms and signs not elsewhere classified |
| S–T | Injury, poisoning and other consequences of external causes |
| U | Codes for special purposes (e.g. COVID-19) |
| V–Y | External causes of morbidity and mortality |
| Z | Factors influencing health status and contact with health services |
| 0–9 | Legacy numeric codes (CIE-9 / ICD-9 procedure codes) |

---

## Dataset

| Split | Samples | Notes |
|-------|---------|-------|
| Training | 13,700 | Labeled `(Literal, Code)` pairs |
| Test (leaderboard) | 6,667 | Literals only |
| ICD-10 dictionary | 179,742 | Official `(Code, Description)` pairs |

The label space has **4,059 unique ICD-10 codes** with a heavy long tail: 47% of codes appear only once in training.

Optional external data: **CodiEsp** ([zenodo.org/records/3837305](https://zenodo.org/records/3837305)) — ~8k real clinical mention→ICD pairs that add ~+1.6pp on local CV when available.

---

## Pipeline

```
Literal
  │
  ├─ Regex shortcut ──────────► return code directly (if literal IS an ICD code)
  │
  ▼
Preprocessing
  ├─ Lowercase + remove accents
  ├─ Catalan → Spanish translation (28 term dictionary)
  └─ Medical abbreviation expansion (50 entries: HTA, IAM, EPOC, VIH, …)
  │
  ▼
TF-IDF Feature Extraction (FeatureUnion)
  ├─ Word n-grams  (1–2), max 20k features, sublinear TF
  └─ Char n-grams  (3–5, char_wb), max 40k features — captures medical morphology (-itis, -ectomía, …)
  │
  ▼
LinearSVC (C=0.7)   ← primary classifier
  │
  └─ Fallback: cosine similarity against ICD-10 official descriptions
```

**Data augmentation:** official ICD-10 descriptions are appended as synthetic training examples for each seen code (+2,618 examples on top of the 13,700 real ones).

---

## Repository Structure

```
FNLProject/
├── data/
│   ├── raw/
│   │   ├── training_codification_data.csv   # labeled training literals
│   │   ├── test_leaderboard_data.csv        # test literals
│   │   └── icd_d_p_pairs.csv               # ICD-10 dictionary
│   ├── preprocessed/
│   │   ├── preprocessed_data.csv
│   │   ├── test_preprocessed.csv
│   │   └── icd_preprocessed_descriptions.csv
│   └── utils/
│       ├── cross_referenced_data.csv
│       ├── preprocessed_cross.csv
│       └── icd10_training_set.csv
│
├── final/
│   ├── final_model.ipynb                        # Production model (LinearSVC + CodiEsp)
│   ├── exploratory_analysis_training_data.ipynb # EDA: label distribution, long tail analysis
│   ├── extract_abbreviations.ipynb              # Medical abbreviation dictionary extraction
│   ├── hierarchical_model_experiment.ipynb      # Hierarchical ICD prediction experiment
│   └── report.pdf                               # Project report
│
└── dev/
    ├── codification.ipynb         # Kaggle baseline (similarity + LogReg)
    ├── preliminary_model.ipynb    # First hybrid pipeline (TF-IDF + LogReg)
    ├── preliminary_model_v2.ipynb # Improved pipeline (LinearSVC + CodiEsp support)
    ├── project.ipynb              # Consolidated pipeline notebook
    ├── eda_train.ipynb            # Early exploratory data analysis
    ├── cross.ipynb                # Cross-referencing ICD codes with training data
    ├── inheritance.ipynb          # ICD-10 code hierarchy exploration
    └── just_icd10.ipynb           # ICD-only training experiments
```

---

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install pandas scikit-learn numpy scipy tqdm
```

---

## Running

Open and run **`final/final_model.ipynb`** for the best local result. The notebook auto-detects whether it is running on Kaggle or locally and adjusts data paths accordingly.

To use the CodiEsp external dataset, download it from [zenodo.org/records/3837305](https://zenodo.org/records/3837305) and place it at `data/codiesp/final_dataset_v4_to_publish/`. The notebook will pick it up automatically.

---

## Results

| Model | Approach | Score |
|-------|----------|-------|
| Cosine similarity baseline | char TF-IDF → nearest ICD description | ~35% |
| LogReg | word+char TF-IDF on training data | ~32% |
| LogReg + ICD augmentation | training + synthetic ICD descriptions | improved |
| **LinearSVC + ICD augmentation** | word+char TF-IDF, C=0.7 | **best** |
| LinearSVC + CodiEsp | + ~8k external clinical examples | +1.6pp on CV |

---

## Key Design Decisions

- **Char n-grams** (`char_wb`, 3–5) are essential for Spanish medical morphology — they handle suffixes like `-itis`, `-ectomía`, `-plasty` that word n-grams miss.
- **No `class_weight='balanced'`**: the ICD dictionary is procedure-heavy (S/T codes) while the leaderboard mirrors the training distribution (Z/O-heavy); balancing would re-weight in the wrong direction.
- **Abbreviation expansion before vectorization**: `"HTA"` → `"hipertension arterial"` vastly improves TF-IDF overlap with ICD descriptions.
- **ICD shortcut**: if a literal is already a valid ICD code string, skip the model entirely.

---

## Authors

- Andrei-Adrian Ragman
- Mihail Alexe
- Kacper Kotowski
- Sonia Serra Grivina
