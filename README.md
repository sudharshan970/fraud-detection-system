# 🛡️ Fraud Detection System

A machine learning web application to detect fraudulent credit card transactions in real time using XGBoost.

## 🚀 Live Demo
[Click here to view the live dashboard](https://fraud-detection-system-4p3utaputndmcbn29wdbyq.streamlit.app)

## 📁 Download Project Files
These files are too large for GitHub — download from Google Drive:

| File | Size | Link |
|------|------|------|
| creditcard.csv | 143 MB | [Download Dataset](https://drive.google.com/file/d/1gjWlzZfaA4T0U1W4ApL-HfIti7n0wAei/view?usp=sharing) |
| fraud_detection_model.pkl | ~3 MB | [Download Model](https://drive.google.com/file/d/1pYrC72kfvhrbbxdfxqPN_u2owlUy45km/view?usp=sharing) |

## 📊 Project Overview
- **Dataset**   : Credit Card Fraud Detection (Kaggle) — 284,807 transactions
- **Models**    : Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Best Model**: XGBoost — AUC-ROC: 0.9997
- **Recall**    : 86.7% — 85 out of 98 fraud cases caught

## 🏆 Model Results

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9741 | 0.0604 | 0.9184 | 0.1133 | 0.9741 |
| Decision Tree | 0.9991 | 0.7426 | 0.8061 | 0.7731 | 0.9026 |
| Random Forest | 0.9995 | 0.9286 | 0.8367 | 0.8803 | 0.9853 |
| **XGBoost** | **0.9996** | **0.8805** | **0.9082** | **0.8941** | **0.9997** |

## 🧰 Tech Stack
- Python 3
- Streamlit
- XGBoost
- Scikit-learn
- Plotly
- Pandas
- NumPy
- SMOTE (imbalanced-learn)
- Joblib

## 📋 Dashboard Pages
- **Summary** — test set metrics, bar chart, pie chart
- **Model Scores** — all 4 models comparison with dropdown
- **ROC Curve** — XGBoost AUC-ROC curve
- **Confusion Matrix** — TP, FP, FN, TN with heatmap
- **Feature Importance** — top N features with slider
- **Transactions** — all 98 fraud cases with filter and probability slider
- **Live Predictor** — enter transaction values and detect fraud instantly

## ⚙️ How to Run Locally

1. Clone the repo
```bash
git clone https://github.com/sudharshan970/fraud-detection-system.git
cd fraud-detection-system
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download the large files from Google Drive links above and place them in the project folder

4. Run the dashboard
```bash
streamlit run app.py
```

5. Open browser at
```
http://localhost:8501
```

## 📂 Project Structure
```
fraud-detection-system/
├── app.py                 Streamlit dark theme dashboard
├── requirements.txt       Python dependencies
├── README.md              Project documentation
└── .gitignore             Git ignore rules
```

## 📈 Key Findings
- XGBoost achieved near-perfect AUC-ROC of 0.9997
- 85 out of 98 fraud cases correctly detected
- Only 55 false alarms out of 56,864 legitimate transactions
- Class imbalance handled using SMOTE oversampling

## 👨‍💻 Author
**Sudharshan**
- GitHub: [@sudharshan970](https://github.com/sudharshan970)
