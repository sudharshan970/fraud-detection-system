import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import gdown
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# ── Auto download files from Google Drive ──────────
MODEL_ID = "https://drive.google.com/file/d/1pYrC72kfvhrbbxdfxqPN_u2owlUy45km/view?usp=sharing "
CSV_ID   = "https://drive.google.com/file/d/1gjWlzZfaA4T0U1W4ApL-HfIti7n0wAei/view?usp=sharing  "

if not os.path.exists("fraud_detection_model.pkl"):
    with st.spinner("Downloading model... please wait"):
        gdown.download(
            f"https://drive.google.com/uc?id={MODEL_ID}",
            "fraud_detection_model.pkl", quiet=False
        )

if not os.path.exists("creditcard.csv"):
    with st.spinner("Downloading dataset... please wait (143MB)"):
        gdown.download(
            f"https://drive.google.com/uc?id={CSV_ID}",
            "creditcard.csv", quiet=False
        )

MODEL_PATH = "fraud_detection_model.pkl"
DATA_PATH  = "creditcard.csv"

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
* { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0a0e1a; color: #e2e8f0; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b27 100%);
    border-right: 1px solid #1e2d3d;
}
[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1e2d3d;
    border-radius: 12px;
    padding: 16px !important;
}
[data-testid="metric-container"] label { color: #64748b !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-size: 28px !important;
    font-weight: 600 !important;
}
h1 { color: #f1f5f9 !important; font-weight: 700 !important; }
.stButton button {
    background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
.stProgress > div > div {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
}
hr { border-color: #1e2d3d !important; }
.stat-banner {
    background: #111827;
    border: 1px solid #1e2d3d;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.stat-banner-title {
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: .05em;
    margin-bottom: 5px;
}
.stat-banner-value {
    font-size: 28px;
    font-weight: 700;
    line-height: 1;
}
.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled']   = scaler.fit_transform(df[['Time']])
    df.drop(['Amount','Time'], axis=1, inplace=True)
    X = df.drop('Class', axis=1)
    y = df['Class']
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return X, X_test, y_test

best_model        = load_model()
X, X_test, y_test = load_data()
y_pred            = best_model.predict(X_test)
y_proba           = best_model.predict_proba(X_test)[:, 1]

total   = len(y_test)
fraud   = int(y_test.sum())
caught  = int(((y_pred==1)&(y_test==1)).sum())
missed  = int(((y_pred==0)&(y_test==1)).sum())
alarms  = int(((y_pred==1)&(y_test==0)).sum())
cleared = int(((y_pred==0)&(y_test==0)).sum())

# ── Sidebar ────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align:center;padding:20px 0 10px'>
  <div style='font-size:40px'>🛡️</div>
  <div style='font-size:16px;font-weight:700;color:#e2e8f0;margin-top:6px'>Fraud Detection</div>
  <div style='font-size:12px;color:#475569'>System Dashboard</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio("", [
    "📊  Summary",
    "🏆  Model Scores",
    "📈  ROC Curve",
    "🔢  Confusion Matrix",
    "⭐  Feature Importance",
    "📋  Transactions",
    "🔍  Live Predictor"
])
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style='padding:12px;background:#111827;border-radius:10px;border:1px solid #1e2d3d'>
  <div style='font-size:11px;color:#475569;margin-bottom:6px;text-transform:uppercase'>Project Info</div>
  <div style='font-size:12px;color:#64748b;line-height:1.8'>
    Dataset: creditcard.csv<br>
    Model: XGBoost<br>
    Test size: 20%<br>
    Total records: 284,807
  </div>
</div>
""", unsafe_allow_html=True)

# ── PAGE 1 SUMMARY ─────────────────────────────────
if page == "📊  Summary":
    st.markdown("<h1>📊 Fraud Detection System — Summary</h1>", unsafe_allow_html=True)
    st.caption("XGBoost model — 56,962 real credit card transactions tested")
    st.markdown("---")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Tested",  f"{total:,}")
    c2.metric("Total Fraud",   f"{fraud}")
    c3.metric("Fraud Caught",  f"{caught}",  f"{caught/fraud*100:.1f}%")
    c4.metric("Fraud Missed",  f"{missed}",  f"-{missed/fraud*100:.1f}%", delta_color="inverse")
    c5.metric("False Alarms",  f"{alarms}")
    c6.metric("Legit Cleared", f"{cleared:,}")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Bar(
            x=['Caught','Missed','False Alarms','Legit Cleared'],
            y=[caught, missed, alarms, cleared],
            marker=dict(color=['#10b981','#ef4444','#f59e0b','#3b82f6']),
            text=[caught, missed, alarms, f'{cleared:,}'],
            textposition='outside',
            textfont=dict(color='#94a3b8')
        ))
        fig.update_layout(
            title=dict(text='Detection Results', font=dict(color='#e2e8f0')),
            height=380, plot_bgcolor='#111827', paper_bgcolor='#111827',
            font=dict(color='#94a3b8'),
            xaxis=dict(gridcolor='#1e2d3d'),
            yaxis=dict(gridcolor='#1e2d3d')
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure(go.Pie(
            labels=['Caught','Missed'],
            values=[caught, missed],
            marker=dict(colors=['#10b981','#ef4444'],
                        line=dict(color='#0a0e1a', width=2)),
            hole=0.6, textinfo='label+percent',
            textfont=dict(color='#e2e8f0')
        ))
        fig2.update_layout(
            title=dict(text='Recall Breakdown', font=dict(color='#e2e8f0')),
            height=380, paper_bgcolor='#111827',
            font=dict(color='#94a3b8')
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── PAGE 2 MODEL SCORES ────────────────────────────
elif page == "🏆  Model Scores":
    st.markdown("<h1>🏆 Model Performance Comparison</h1>", unsafe_allow_html=True)
    st.caption("All 4 models trained on creditcard.csv with SMOTE balancing")
    st.markdown("---")

    scores = pd.DataFrame({
        'Model':     ['Logistic Regression','Decision Tree','Random Forest','XGBoost'],
        'Accuracy':  [0.9741, 0.9991, 0.9995, 0.9996],
        'Precision': [0.0604, 0.7426, 0.9286, 0.8805],
        'Recall':    [0.9184, 0.8061, 0.8367, 0.9082],
        'F1 Score':  [0.1133, 0.7731, 0.8803, 0.8941],
        'AUC-ROC':   [0.9741, 0.9026, 0.9853, 0.9997],
    })

    st.dataframe(
        scores.style
              .highlight_max(subset=scores.columns[1:], color='#0f2a1e')
              .highlight_min(subset=scores.columns[1:], color='#2a0f0f')
              .format({c: '{:.4f}' for c in scores.columns[1:]}),
        use_container_width=True, hide_index=True
    )
    st.markdown("---")

    metric = st.selectbox("Select metric", scores.columns[1:].tolist())
    colors = ['#3b82f6','#8b5cf6','#f59e0b','#10b981']
    fig = go.Figure()
    for i, (m, v) in enumerate(zip(scores['Model'], scores[metric])):
        fig.add_trace(go.Bar(
            x=[m], y=[v], name=m,
            marker=dict(color=colors[i]),
            text=[f'{v:.4f}'], textposition='outside',
            textfont=dict(color='#94a3b8')
        ))
    fig.update_layout(
        title=dict(text=f'{metric} — All Models', font=dict(color='#e2e8f0')),
        height=420, showlegend=False,
        plot_bgcolor='#111827', paper_bgcolor='#111827',
        font=dict(color='#94a3b8'),
        xaxis=dict(gridcolor='#1e2d3d'),
        yaxis=dict(range=[scores[metric].min()-0.05, 1.05], gridcolor='#1e2d3d')
    )
    st.plotly_chart(fig, use_container_width=True)

# ── PAGE 3 ROC CURVE ───────────────────────────────
elif page == "📈  ROC Curve":
    st.markdown("<h1>📈 ROC Curve — XGBoost</h1>", unsafe_allow_html=True)
    st.caption("Higher AUC = better fraud separation")
    st.markdown("---")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc         = roc_auc_score(y_test, y_proba)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines',
        name=f'XGBoost (AUC={auc:.4f})',
        line=dict(color='#10b981', width=3),
        fill='tozeroy', fillcolor='rgba(16,185,129,0.06)'
    ))
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode='lines', name='Random',
        line=dict(color='#475569', dash='dash', width=1.5)
    ))
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=480,
        plot_bgcolor='#111827', paper_bgcolor='#111827',
        font=dict(color='#94a3b8'),
        xaxis=dict(gridcolor='#1e2d3d'),
        yaxis=dict(gridcolor='#1e2d3d'),
        legend=dict(x=0.6, y=0.15, bgcolor='#111827', bordercolor='#1e2d3d')
    )
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"AUC-ROC: **{auc:.4f}** — Near perfect fraud detection")

# ── PAGE 4 CONFUSION MATRIX ────────────────────────
elif page == "🔢  Confusion Matrix":
    st.markdown("<h1>🔢 Confusion Matrix — XGBoost</h1>", unsafe_allow_html=True)
    st.caption("Your real test set results")
    st.markdown("---")

    cm             = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f"<div class='stat-banner'><div class='stat-banner-title'>True Positive</div><div class='stat-banner-value' style='color:#10b981'>{tp}</div><div style='font-size:11px;color:#475569;margin-top:4px'>Fraud caught</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='stat-banner'><div class='stat-banner-title'>False Negative</div><div class='stat-banner-value' style='color:#ef4444'>{fn}</div><div style='font-size:11px;color:#475569;margin-top:4px'>Fraud missed</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='stat-banner'><div class='stat-banner-title'>False Positive</div><div class='stat-banner-value' style='color:#f59e0b'>{fp}</div><div style='font-size:11px;color:#475569;margin-top:4px'>False alarm</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='stat-banner'><div class='stat-banner-title'>True Negative</div><div class='stat-banner-value' style='color:#3b82f6'>{tn:,}</div><div style='font-size:11px;color:#475569;margin-top:4px'>Legit cleared</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    fig = px.imshow(
        cm, text_auto=True,
        color_continuous_scale=[[0,'#0a0e1a'],[0.5,'#1e3a5f'],[1,'#3b82f6']],
        labels=dict(x='Predicted', y='Actual'),
        x=['Not Fraud','Fraud'], y=['Not Fraud','Fraud'],
        title=f'Confusion Matrix  |  TP={tp}  FP={fp}  FN={fn}  TN={tn:,}'
    )
    fig.update_layout(
        height=450,
        plot_bgcolor='#111827', paper_bgcolor='#111827',
        font=dict(color='#94a3b8'),
        title=dict(font=dict(color='#e2e8f0'))
    )
    st.plotly_chart(fig, use_container_width=True)

# ── PAGE 5 FEATURE IMPORTANCE ──────────────────────
elif page == "⭐  Feature Importance":
    st.markdown("<h1>⭐ Feature Importance — XGBoost</h1>", unsafe_allow_html=True)
    st.caption("Top features the model uses to detect fraud")
    st.markdown("---")

    if hasattr(best_model, 'feature_importances_'):
        fi    = pd.Series(best_model.feature_importances_, index=X.columns)
        top_n = st.slider("Show top N features", 5, 28, 15)
        top   = fi.nlargest(top_n).sort_values()

        bar_colors = ['#10b981' if v > top.quantile(0.7)
                      else '#3b82f6' if v > top.quantile(0.4)
                      else '#8b5cf6'
                      for v in top.values]

        fig = go.Figure(go.Bar(
            x=top.values, y=top.index, orientation='h',
            marker=dict(color=bar_colors),
            text=[f'{v:.4f}' for v in top.values],
            textposition='outside',
            textfont=dict(color='#94a3b8')
        ))
        fig.update_layout(
            title=dict(text=f'Top {top_n} Features', font=dict(color='#e2e8f0')),
            xaxis_title='Importance Score',
            height=520,
            plot_bgcolor='#111827', paper_bgcolor='#111827',
            font=dict(color='#94a3b8'),
            xaxis=dict(gridcolor='#1e2d3d'),
            yaxis=dict(gridcolor='#1e2d3d')
        )
        st.plotly_chart(fig, use_container_width=True)

# ── PAGE 6 TRANSACTIONS ────────────────────────────
elif page == "📋  Transactions":
    st.markdown("<h1>📋 All Fraud Transactions</h1>", unsafe_allow_html=True)
    st.caption("98 actual fraud cases from your test set")
    st.markdown("---")

    fraud_df = pd.DataFrame({
        'Index':      y_test.index,
        'Actual':     y_test.values,
        'Predicted':  y_pred,
        'Fraud_Prob': (y_proba * 100).round(4)
    })
    fraud_df = fraud_df[fraud_df['Actual'] == 1].copy()
    fraud_df['Status'] = fraud_df.apply(
        lambda r: 'CAUGHT' if r['Predicted']==1 else 'MISSED', axis=1
    )
    fraud_df = fraud_df.sort_values('Fraud_Prob', ascending=False).reset_index(drop=True)

    col1, col2 = st.columns(2)
    filter_choice = col1.radio("Filter", ['All','Caught','Missed'], horizontal=True)
    min_prob      = col2.slider("Min probability (%)", 0.0, 100.0, 0.0, step=0.1)

    df_show = fraud_df.copy()
    if filter_choice == 'Caught':
        df_show = df_show[df_show['Status']=='CAUGHT']
    elif filter_choice == 'Missed':
        df_show = df_show[df_show['Status']=='MISSED']
    df_show = df_show[df_show['Fraud_Prob'] >= min_prob]

    c1,c2,c3 = st.columns(3)
    c1.markdown(f"<div class='stat-banner'><div class='stat-banner-title'>Showing</div><div class='stat-banner-value' style='color:#e2e8f0;font-size:22px'>{len(df_show)}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='stat-banner'><div class='stat-banner-title'>Caught</div><div class='stat-banner-value' style='color:#10b981;font-size:22px'>{len(df_show[df_show['Status']=='CAUGHT'])}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='stat-banner'><div class='stat-banner-title'>Missed</div><div class='stat-banner-value' style='color:#ef4444;font-size:22px'>{len(df_show[df_show['Status']=='MISSED'])}</div></div>", unsafe_allow_html=True)

    def color_row(row):
        c = 'background-color:#0f2a1e;color:#10b981' if row['Status']=='CAUGHT' \
            else 'background-color:#2a0f0f;color:#ef4444'
        return [c]*len(row)

    st.dataframe(
        df_show[['Index','Fraud_Prob','Status']]
               .style.apply(color_row, axis=1)
               .format({'Fraud_Prob':'{:.4f}%'}),
        use_container_width=True, hide_index=True
    )

# ── PAGE 7 LIVE PREDICTOR ──────────────────────────
elif page == "🔍  Live Predictor":
    st.markdown("<h1>🔍 Live Fraud Predictor</h1>", unsafe_allow_html=True)
    st.caption("Enter transaction values — model predicts instantly")
    st.markdown("---")

    sample_choice = st.selectbox("Load sample", [
        'Manual entry',
        'Fraud Sample 1 (~99%)',
        'Fraud Sample 2 (~99%)',
        'Legit Sample 1 (~0%)',
        'Legit Sample 2 (~0%)'
    ])
    samples = {
        'Fraud Sample 1 (~99%)': dict(amount=219.18, v1=-3.0435, v2=-3.1572, v3=1.0893,  v4=2.2358),
        'Fraud Sample 2 (~99%)': dict(amount=0.00,   v1=-2.3122, v2=1.9519,  v3=-1.6097, v4=3.9979),
        'Legit Sample 1 (~0%)':  dict(amount=149.62, v1=-1.3598, v2=-0.0728, v3=2.5363,  v4=1.3781),
        'Legit Sample 2 (~0%)':  dict(amount=2.69,   v1=1.1919,  v2=0.2661,  v3=0.1664,  v4=0.4482),
    }
    d = samples.get(sample_choice, samples['Legit Sample 1 (~0%)'])

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Amount (€)", value=float(d['amount']), step=0.01, format="%.2f")
        v1     = st.number_input("V1", value=float(d['v1']), step=0.0001, format="%.4f")
        v2     = st.number_input("V2", value=float(d['v2']), step=0.0001, format="%.4f")
    with col2:
        v3     = st.number_input("V3", value=float(d['v3']), step=0.0001, format="%.4f")
        v4     = st.number_input("V4", value=float(d['v4']), step=0.0001, format="%.4f")

    st.markdown("")
    if st.button("🔍 Run Fraud Detection", type="primary", use_container_width=True):
        sample = X_test.iloc[[0]].copy()
        sample.iloc[0, sample.columns.get_loc('V1')]            = v1
        sample.iloc[0, sample.columns.get_loc('V2')]            = v2
        sample.iloc[0, sample.columns.get_loc('V3')]            = v3
        sample.iloc[0, sample.columns.get_loc('V4')]            = v4
        sample.iloc[0, sample.columns.get_loc('Amount_scaled')] = amount / 100

        pred = best_model.predict(sample)[0]
        prob = best_model.predict_proba(sample)[0][1] * 100

        st.markdown("---")
        if pred == 1:
            st.markdown(f"""
            <div style='background:rgba(239,68,68,0.1);border:1px solid #ef4444;
                        border-radius:12px;padding:20px 24px'>
              <div style='font-size:22px;font-weight:700;color:#ef4444'>
                🚨 FRAUD DETECTED
              </div>
              <div style='font-size:14px;color:#94a3b8;margin-top:6px'>
                Fraud Probability:
                <span style='color:#ef4444;font-weight:600'>{prob:.4f}%</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:rgba(16,185,129,0.1);border:1px solid #10b981;
                        border-radius:12px;padding:20px 24px'>
              <div style='font-size:22px;font-weight:700;color:#10b981'>
                ✅ LEGITIMATE TRANSACTION
              </div>
              <div style='font-size:14px;color:#94a3b8;margin-top:6px'>
                Fraud Probability:
                <span style='color:#10b981;font-weight:600'>{prob:.4f}%</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(int(min(prob, 100)))

        r1,r2,r3 = st.columns(3)
        r1.markdown(f"<div class='stat-banner'><div class='stat-banner-title'>Prediction</div><div class='stat-banner-value' style='font-size:18px;color:{'#ef4444' if pred==1 else '#10b981'}'>{'FRAUD' if pred==1 else 'LEGIT'}</div></div>", unsafe_allow_html=True)
        r2.markdown(f"<div class='stat-banner'><div class='stat-banner-title'>Fraud Probability</div><div class='stat-banner-value' style='font-size:18px;color:{'#ef4444' if pred==1 else '#10b981'}'>{prob:.4f}%</div></div>", unsafe_allow_html=True)
        r3.markdown(f"<div class='stat-banner'><div class='stat-banner-title'>Amount</div><div class='stat-banner-value' style='font-size:18px;color:#3b82f6'>€{amount:.2f}</div></div>", unsafe_allow_html=True)
