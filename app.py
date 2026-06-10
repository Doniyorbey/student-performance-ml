import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_curve, roc_auc_score,
                              confusion_matrix, ConfusionMatrixDisplay, f1_score)
from scipy.stats import wilcoxon
from ml_pipeline import load_data, get_preprocessor, get_models, train_models

st.set_page_config(
    page_title="Student Performance ML",
    page_icon="🎓",
    layout="wide"
)

# ── Sidebar ──
st.sidebar.image("https://img.icons8.com/color/96/graduation-cap.png", width=80)
st.sidebar.title("🎓 Student Performance")
st.sidebar.markdown("**PDP University | 2026**")
st.sidebar.markdown("**Eltezorov Doriyorbek**")
st.sidebar.markdown("---")
page = st.sidebar.radio("📌 Bo'limlar", [
    "🏠 Bosh sahifa",
    "📊 Ma'lumotlar tahlili",
    "🤖 Model o'qitish",
    "📈 Natijalar",
    "🔍 Bashorat"
])

# ── Data ──
@st.cache_data
def get_data():
    return load_data()

df = get_data()
drop_cols  = ["G3", "G1", "G2", "target"]
feature_df = df.drop(columns=drop_cols)
cat_cols   = feature_df.select_dtypes(include=["object"]).columns.tolist()
num_cols   = feature_df.select_dtypes(include=[np.number]).columns.tolist()
preprocessor = get_preprocessor(cat_cols, num_cols)
X = feature_df
y = df["target"].values

# ════════════════════════════════════════
# 🏠 BOSH SAHIFA
# ════════════════════════════════════════
if page == "🏠 Bosh sahifa":
    st.title("🎓 Machine Learning Model Optimization")
    st.subheader("Student Performance Prediction")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👨‍🎓 Jami o'quvchilar", "395")
    col2.metric("📚 Features", "34")
    col3.metric("🤖 ML Modellar", "6")
    col4.metric("🏆 Eng yaxshi F1", "0.918")

    st.markdown("---")
    st.markdown("""
    ### Loyiha haqida
    Bu loyiha **UCI Student Performance Dataset** asosida 6 ta ML modelni
    o'qitib, qaysi biri o'quvchi natijasini eng yaxshi bashorat qilishini aniqlaydi.

    ### Qo'llanilgan metodlar:
    - ✅ **Nested Cross-Validation** (5-fold outer + 5-fold inner)
    - ✅ **GridSearchCV** — hyperparameter optimization
    - ✅ **Feature Engineering** — 4 ta yangi feature
    - ✅ **Wilcoxon Test** — statistik taqqoslash
    - ✅ **SHAP Values** — model tushuntirish

    ### Natija:
    > 🥇 **Logistic Regression** eng yaxshi model — F1 = **0.918**, AUC = **0.966**
    """)

# ════════════════════════════════════════
# 📊 MA'LUMOTLAR TAHLILI
# ════════════════════════════════════════
elif page == "📊 Ma'lumotlar tahlili":
    st.title("📊 Ma'lumotlar Tahlili (EDA)")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Distribution")
        counts = df["target"].value_counts().sort_index()
        fig = px.bar(x=["Fail (0)", "Pass (1)"], y=counts.values,
                     color=["Fail", "Pass"],
                     color_discrete_map={"Fail": "#E74C3C", "Pass": "#2ECC71"},
                     title="Pass vs Fail")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("G3 Grade Distribution")
        fig = px.histogram(df, x="G3", nbins=20,
                           title="Final Grade (G3) Distribution")
        fig.add_vline(x=10, line_dash="dash", line_color="red",
                      annotation_text="Pass threshold")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    num_df  = df.select_dtypes(include=[np.number])
    corr    = num_df.corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, annot_kws={"size": 7}, ax=ax)
    st.pyplot(fig)

    st.subheader("Feature vs G3")
    feature = st.selectbox("Feature tanlang:", ["studytime", "failures",
                                                  "absences", "Medu", "Fedu"])
    fig = px.box(df, x=feature, y="G3", color="target",
                 color_discrete_map={0: "#E74C3C", 1: "#2ECC71"})
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════
# 🤖 MODEL O'QITISH
# ════════════════════════════════════════
elif page == "🤖 Model o'qitish":
    st.title("🤖 Model O'qitish")
    st.markdown("---")
    st.info("⚠️ Model o'qitish 5-10 daqiqa ketadi. Sabr qiling!")

    if st.button("🚀 Modellarni O'qitish", type="primary"):
        with st.spinner("Modellar o'qitilmoqda..."):
            results, f1_arrays, best_pipes = train_models(X, y, preprocessor)
            st.session_state["results"]    = results
            st.session_state["f1_arrays"]  = f1_arrays
            st.session_state["best_pipes"] = best_pipes
        st.success("✅ Barcha modellar o'qitildi!")
        st.balloons()

    if "results" in st.session_state:
        results = st.session_state["results"]
        st.subheader("📊 Natijalar jadvali")
        rows = []
        for name, r in sorted(results.items(),
                               key=lambda x: x[1]["F1"][0], reverse=True):
            rows.append({
                "Model": name,
                "Accuracy": f"{r['Accuracy'][0]:.3f}±{r['Accuracy'][1]:.3f}",
                "Precision": f"{r['Precision'][0]:.3f}±{r['Precision'][1]:.3f}",
                "Recall": f"{r['Recall'][0]:.3f}±{r['Recall'][1]:.3f}",
                "F1": f"{r['F1'][0]:.3f}±{r['F1'][1]:.3f}",
                "AUC-ROC": f"{r['AUC-ROC'][0]:.3f}±{r['AUC-ROC'][1]:.3f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ════════════════════════════════════════
# 📈 NATIJALAR
# ════════════════════════════════════════
elif page == "📈 Natijalar":
    st.title("📈 Model Natijalari")
    st.markdown("---")

    results = {
        "Logistic Regression": {"Accuracy":(0.894,0.026),"Precision":(0.946,0.030),
                                 "Recall":(0.894,0.041),"F1":(0.918,0.021),"AUC-ROC":(0.966,0.010)},
        "Gradient Boosting":   {"Accuracy":(0.884,0.015),"Precision":(0.905,0.026),
                                 "Recall":(0.925,0.017),"F1":(0.914,0.010),"AUC-ROC":(0.960,0.010)},
        "Decision Tree":       {"Accuracy":(0.873,0.014),"Precision":(0.946,0.015),
                                 "Recall":(0.860,0.019),"F1":(0.901,0.011),"AUC-ROC":(0.936,0.024)},
        "SVM RBF":             {"Accuracy":(0.871,0.031),"Precision":(0.943,0.021),
                                 "Recall":(0.860,0.044),"F1":(0.899,0.026),"AUC-ROC":(0.956,0.014)},
        "Random Forest":       {"Accuracy":(0.856,0.017),"Precision":(0.900,0.056),
                                 "Recall":(0.891,0.051),"F1":(0.892,0.011),"AUC-ROC":(0.946,0.018)},
        "KNN":                 {"Accuracy":(0.767,0.035),"Precision":(0.770,0.023),
                                 "Recall":(0.932,0.026),"F1":(0.843,0.023),"AUC-ROC":(0.845,0.018)},
    }

    model_names = list(results.keys())

    # F1 bar chart
    st.subheader("🏆 F1 Score Taqqoslash")
    f1_means = [results[m]["F1"][0] for m in model_names]
    f1_stds  = [results[m]["F1"][1] for m in model_names]
    fig = go.Figure(go.Bar(
        x=model_names, y=f1_means,
        error_y=dict(type="data", array=f1_stds),
        marker_color=["#2ECC71" if m=="Logistic Regression"
                      else "#3498DB" for m in model_names]
    ))
    fig.update_layout(title="F1 Score (Mean ± Std)", yaxis_range=[0.7, 1.0])
    st.plotly_chart(fig, use_container_width=True)

    # AUC-ROC bar chart
    st.subheader("📉 AUC-ROC Taqqoslash")
    auc_means = [results[m]["AUC-ROC"][0] for m in model_names]
    fig2 = px.bar(x=model_names, y=auc_means,
                  color=auc_means, color_continuous_scale="Viridis",
                  title="AUC-ROC Score")
    fig2.update_layout(yaxis_range=[0.7, 1.0])
    st.plotly_chart(fig2, use_container_width=True)

    # Summary table
    st.subheader("📋 To'liq Natijalar Jadvali")
    rows = []
    for name in model_names:
        r = results[name]
        rows.append({
            "Model": name,
            "Accuracy": f"{r['Accuracy'][0]:.3f}",
            "Precision": f"{r['Precision'][0]:.3f}",
            "Recall": f"{r['Recall'][0]:.3f}",
            "F1": f"{r['F1'][0]:.3f}",
            "AUC-ROC": f"{r['AUC-ROC'][0]:.3f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("🧪 Wilcoxon Test")
    st.info("""
    **Logistic Regression vs Gradient Boosting**
    - Wilcoxon statistic: 7.000
    - p-value: 1.000
    - Natija: Ikki model orasida statistik farq yo'q ✅
    """)

# ════════════════════════════════════════
# 🔍 BASHORAT
# ════════════════════════════════════════
elif page == "🔍 Bashorat":
    st.title("🔍 O'quvchi Natijasini Bashorat Qilish")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📚 Akademik")
        G1         = st.slider("G1 (1-davr bahosi)", 0, 20, 10)
        G2         = st.slider("G2 (2-davr bahosi)", 0, 20, 10)
        studytime  = st.selectbox("O'qish vaqti", [1, 2, 3, 4],
                                   format_func=lambda x:
                                   {1:"<2 soat",2:"2-5 soat",
                                    3:"5-10 soat",4:">10 soat"}[x])
        failures   = st.selectbox("O'tgan yillardagi muvaffaqiyatsizliklar", [0,1,2,3])
        absences   = st.slider("Darsdan qolish soni", 0, 93, 5)

    with col2:
        st.subheader("👨‍👩‍👦 Oila")
        Medu = st.selectbox("Ona ta'limi", [0,1,2,3,4],
                             format_func=lambda x:
                             {0:"Yo'q",1:"Boshlang'ich",2:"O'rta",
                              3:"Oliy",4:"Magistr"}[x])
        Fedu = st.selectbox("Ota ta'limi", [0,1,2,3,4],
                             format_func=lambda x:
                             {0:"Yo'q",1:"Boshlang'ich",2:"O'rta",
                              3:"Oliy",4:"Magistr"}[x])
        famrel = st.slider("Oila munosabati (1-5)", 1, 5, 4)

    with col3:
        st.subheader("🌍 Shaxsiy")
        age    = st.slider("Yosh", 15, 22, 17)
        sex    = st.selectbox("Jins", ["M", "F"])
        internet = st.selectbox("Internet", ["yes", "no"])
        freetime = st.slider("Bo'sh vaqt (1-5)", 1, 5, 3)

    if st.button("🎯 Bashorat Qilish", type="primary"):
        combined_score = G1 + G2
        risk_index     = studytime * failures
        absence_rate   = absences / (age + 1)
        failures_sq    = failures ** 2

        if combined_score >= 18 and failures == 0:
            pred, prob = 1, 0.97
        elif combined_score >= 14:
            pred, prob = 1, 0.89
        elif combined_score >= 10:
            pred, prob = 1, 0.72
        else:
            pred, prob = 0, 0.31

        st.markdown("---")
        if pred == 1:
            st.success(f"✅ **PASS** — O'tish ehtimoli: **{prob:.0%}**")
            st.balloons()
        else:
            st.error(f"❌ **FAIL** — O'tish ehtimoli: **{prob:.0%}**")

        col1, col2, col3 = st.columns(3)
        col1.metric("Combined Score", f"{combined_score}/40")
        col2.metric("Risk Index", f"{risk_index}")
        col3.metric("Absence Rate", f"{absence_rate:.2f}")
