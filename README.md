### 📘 `README.md`

````markdown
# 💊🏡 Drug & EPC Model Prediction System

This project implements two machine learning pipelines:
1. **Drug Sentiment Analysis** — Classifies user drug reviews into positive, neutral, or negative sentiments.
2. **EPC Prediction** — Predicts the **Potential Energy Rating** for properties in Birmingham, UK using Energy Performance Certificate (EPC) data.

Each pipeline includes:
- Data preprocessing
- Model training & evaluation
- Best model selection and saving
- API endpoints (FastAPI)
- Interactive dashboards (Streamlit)
- Secure access via API key

---

## 📁 Project Folder Structure

```bash
Drug_epc_model/
│
├── api/                                  # FastAPI apps for inference
│   ├── drug_review_fastapi.py            # Load the drug model and return sentiment prediction (JSON)
│   └── epc_rating_fastapi.py                    # Load the EPC model and return energy rating prediction (JSON)
│
├── dashboard/                            # Streamlit dashboards
│   ├── drug_review_dashboard.py          # UI for entering drug review and seeing sentiment
│   └── epc_dashboard.py                  # UI for EPC search by address/LMK_KEY and view predictions
│
├── data/
│   ├── drug_reviews/
│   │   ├── cleaned_dat/                  # Intermediate preprocessed files
│   │   ├── Test_clean.csv                # Cleaned test data
│   │   ├── train_clean.csv               # Cleaned train data
│   │   ├── EDA/                          # Visuals for EDA (e.g., sentiment dist)
│   │   │   └── sentiment_analysis_train.png
│   │   ├── drugsComTest_raw.csv          # Original test dataset
│   │   └── drugsComTrain_raw.csv         # Original training dataset
│   │
│   └── epc/
│       ├── Cleaned_data/
│       │   └── Clean_data.csv            # Final merged and cleaned EPC dataset
│       ├── EDA/
│       │   └── potential_energy_rating_dstribution.png  # EPC rating distribution
│       ├── Certificates.csv              # Raw EPC certificate file (93 features)
│       ├── Columns.csv                   # Description of all columns in EPC files
│       └── Recommendations.csv           # EPC improvement suggestions data
│
├── models/
│   ├── drug_reviews/
│   │   ├── Decision Tree_sentiment_model.pkl
│   │   ├── Gradient Boosting_sentiment_model.pkl
│   │   ├── Logistic Regression_sentiment_model.pkl
│   │   ├── Naive Bayes_sentiment_model.pkl
│   │   ├── Random Forest_sentiment_model.pkl
│   │   ├── best_model.pkl                # Final best sentiment model
│   │   └── Important_words_plot.png      # Word importance visual (EDA)
│   │
│   └── epc/
│       ├── gradient_boosting.pkl
│       ├── logistic_regression.pkl
│       ├── random_forest.pkl
│       └── best_model.pkl                # Final best EPC model
│
├── src/
│   ├── drug_reviews/
│   │   ├── preprocess.py                 # Clean, preprocess and EDA functions
│   │   ├── model.py                      # Train and evaluate models (TF-IDF etc.)
│   │   └── main.py                       # Executes preprocessing and model training pipeline
│   │
│   └── epc/
│       ├── preprocess.py
│       ├── model.py
│       └── main.py
│
├── read.me                               # Project overview and usage guide (you are here)
├── requirements.txt                      # List of dependencies
└── .gitignore                            # Git exclusions (models, pycache etc.)

````

---

## ⚙️ Installation & Setup

1. **Download the project files**

Download the complete project folder (including code, models, and data) from the shared Google Drive

2. **Extract the zip**

```bash
unzip drug_epc_model.zip
cd drug_epc_model

2. **Create a virtual environment and install dependencies**

```bash
python -m venv drug_epc_env
source drug_epc_env/bin/activate  
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### 🧠 Run Preprocessing & Training

Run both pipelines individually to preprocess data and train models:

#### Drug Reviews:

```bash
cd src/drug_reviews
python main.py
```

#### EPC:

```bash
cd src/epc
python main.py
```

---

## 🌐 Run APIs (FastAPI)

### 📌 EPC FastAPI

```bash
uvicorn api.epc_rating_fastapi:app --reload
```

### 📌 Drug Review Sentiment API

```bash
uvicorn api.drug_review_fastapi:app --reload
```

#### 🔍 API Test (FastAPI Swagger UI)

Open in browser:

* **Drug Review**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **EPC**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (or port 8001 if you run them on separate ports)

#### 📬 API Test via Postman

**Drug Review (POST)**:

```
POST http://127.0.0.1:8000/predict_sentiment
Body: { "review": "This drug helped me a lot and had no side effects." }
```

**EPC (GET)**:

```
GET http://localhost:8000/predict_epc_rating
Example: http://127.0.0.1:8000/predict/123456789123
```

---

## 📊 Run Dashboards (Streamlit)

### 📌 Drug Review Dashboard

```bash
cd Dashboard
streamlit run drug_review_dashboard.py
```

### 📌 EPC Dashboard

```bash
cd Dashboard
streamlit run epc_rating_dashboard.py
```

Dashboards allow:

* ✅ Drug Review: User inputs a review, gets sentiment prediction.
* ✅ EPC: Search property by **address** or **LMK\_KEY**, view predicted energy rating.

---

## 🔐 API Security
- API KEY : "bPMYxNcKo8Z82YoDtLT5lxAob7H_xDw3AgpTcFbRgls"
Security implementation includes API key .

---

## 📦 Requirements

All dependencies are listed in `requirements.txt`. Some major libraries:

* `fastapi`, `uvicorn`
* `streamlit`
* `scikit-learn`, `pandas`, `joblib`
* `imblearn` for SMOTE
* `nltk` or `spacy` for text preprocessing

---

### ✅ What to Do

1. Save this as `README.md` in your project root.
2. Run all `main.py` files to generate `best_model.pkl` in each model folder.
3. Run both APIs (`epc_rating_fastapi.py` and `drug_review_fastapi.py`).
4. Test via Swagger or Postman.
5. Launch Streamlit dashboards.


