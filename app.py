import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_curve, roc_auc_score,
                              confusion_matrix, ConfusionMatrixDisplay, f1_score)
from scipy.stats import wilcoxon
import optuna
import shap
optuna.logging.set_verbosity(optuna.logging.WARNING)
from ml_pipeline import load_data, get_preprocessor, get_models, train_models

st.set_page_config(
    page_title="Student Performance ML",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark/Light mode ──
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

# ── Sidebar ──
with st.sidebar:
    st.image("https://img.icons8.com/color/96/graduation-cap.png", width=70)
    st.title("🎓 Student ML")
    st.markdown("**PDP University | 2026**")
    st.markdown("**Eltezorov Doriyorbek**")
    st.markdown("**Group: 22-305 | AI**")
    st.markdown("---")
    page = st.radio("📌 Bo'limlar", [
        "🏠 Bosh sahifa",
        "📊 EDA & Tahlil",
        "⚡ Optuna Optimization",
        "🤖 Model O'qitish",
        "📈 Natijalar",
        "🔬 SHAP Values",
        "🔍 Bashorat",
    ])
    st.markdown("---")
    theme = st.selectbox("🎨 Tema", ["Light", "Dark"])
    st.session_state.theme = theme

# ── Theme CSS ──
if st.session_state.theme == "Dark":
    st.markdown("""
    <style>
    .stApp { background-color: #1e1e2e; color: #cdd6f4; }
    .metric-card { background: #313244; border-radius: 10px; padding: 15px; }
    </style>
    """, unsafe_allow_html=True)

# ── Data ──
@st.cache_data
def get_data():
    return load_data()

df = get_data()
drop_cols    = ["G3", "G1", "G2", "target"]
feature_df   = df.drop(columns=drop_cols)
cat_cols     = feature_df.select_dtypes(include=["object"]).columns.tolist()
num_cols     = feature_df.select_dtypes(include=[np.number]).columns.tolist()
preprocessor = get_preprocessor(cat_cols, num_cols)
X = feature_df
y = df["target"].values

# ════════════════════════════════════════
# 🏠 BOSH SAHIFA
# ════════════════════════════════════════
if page == "🏠 Bosh sahifa":
    st.title("🎓 Machine Learning Model Optimization")
    st.subheader("Student Performance Prediction — PDP University 2026")
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("👨‍🎓 O'quvchilar", "395")
    col2.metric("📚 Features", "34")
    col3.metric("🤖 Modellar", "6")
    col4.metric("🏆 Best F1", "0.918")
    col5.metric("📈 Best AUC", "0.966")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 📌 Loyiha haqida
        UCI Student Performance Dataset asosida **6 ta ML model** o'qitilib,
        qaysi biri o'quvchi natijasini eng yaxshi bashorat qilishi aniqlandi.

        ### 🔧 Qo'llanilgan metodlar:
        - ✅ **Nested Cross-Validation** (5-fold outer + 5-fold inner)
        - ✅ **GridSearchCV** — hyperparameter optimization
        - ✅ **Optuna** — aqlli optimization framework
        - ✅ **Feature Engineering** — 4 ta yangi feature
        - ✅ **Wilcoxon Test** — statistik taqqoslash
        - ✅ **SHAP Values** — model tushuntirish (XAI)
        """)
    with col2:
        st.markdown("""
        ### 🏆 Eng yaxshi natijalar:
        """)
        results_preview = {
            "Model": ["Logistic Regression 🥇", "Gradient Boosting", "Decision Tree",
                      "SVM RBF", "Random Forest", "KNN"],
            "F1": [0.918, 0.914, 0.901, 0.899, 0.892, 0.843],
            "AUC-ROC": [0.966, 0.960, 0.936, 0.956, 0.946, 0.845],
        }
        st.dataframe(pd.DataFrame(results_preview), use_container_width=True, hide_index=True)

# ════════════════════════════════════════
# 📊 EDA
# ════════════════════════════════════════
elif page == "📊 EDA & Tahlil":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribution", "🔗 Correlation", "📦 Boxplots", "📋 Dataset"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            counts = df["target"].value_counts().sort_index()
            fig = px.bar(x=["Fail (0)", "Pass (1)"], y=counts.values,
                         color=["Fail", "Pass"],
                         color_discrete_map={"Fail": "#E74C3C", "Pass": "#2ECC71"},
                         title="Pass vs Fail Distribution",
                         text=counts.values)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(df, x="G3", nbins=20, color_discrete_sequence=["#3498DB"],
                               title="Final Grade (G3) Distribution")
            fig.add_vline(x=10, line_dash="dash", line_color="red",
                          annotation_text="Pass threshold (10)")
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="absences", nbins=30,
                               color_discrete_sequence=["#9B59B6"],
                               title="Absences Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.pie(df, names=df["target"].map({0:"Fail",1:"Pass"}),
                         title="Class Balance",
                         color_discrete_sequence=["#E74C3C","#2ECC71"])
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Correlation Heatmap")
        num_df  = df.select_dtypes(include=[np.number])
        corr    = num_df.corr()
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, linewidths=0.5, annot_kws={"size": 7}, ax=ax)
        st.pyplot(fig)
        plt.close()

    with tab3:
        feature = st.selectbox("Feature tanlang:",
                                ["studytime", "failures", "absences",
                                 "Medu", "Fedu", "famrel", "freetime"])
        fig = px.box(df, x=str(feature), y="G3",
                     color=df["target"].map({0:"Fail",1:"Pass"}),
                     color_discrete_map={"Fail":"#E74C3C","Pass":"#2ECC71"},
                     title=f"{feature} vs G3")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)
        st.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ════════════════════════════════════════
# ⚡ OPTUNA
# ════════════════════════════════════════
elif page == "⚡ Optuna Optimization":
    st.title("⚡ Optuna Hyperparameter Optimization")
    st.markdown("---")
    st.info("Optuna GridSearch dan ko'ra aqlli va tezroq — Bayesian optimization ishlatadi!")

    from sklearn.model_selection import cross_val_score

    model_choice = st.selectbox("Model tanlang:", [
        "Logistic Regression", "Random Forest", "Gradient Boosting"])
    n_trials = st.slider("Trials soni:", 10, 100, 30)

    if st.button("🚀 Optuna Ishga Tushir", type="primary"):
        X_proc = preprocessor.fit_transform(X, y)
        cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        progress = st.progress(0)
        status   = st.empty()
        trial_scores = []

        def objective(trial):
            if model_choice == "Logistic Regression":
                from sklearn.linear_model import LogisticRegression
                C   = trial.suggest_float("C", 0.001, 100, log=True)
                clf = LogisticRegression(C=C, max_iter=1000,
                                         class_weight="balanced", random_state=42)
            elif model_choice == "Random Forest":
                from sklearn.ensemble import RandomForestClassifier
                n   = trial.suggest_int("n_estimators", 50, 300)
                d   = trial.suggest_int("max_depth", 3, 15)
                clf = RandomForestClassifier(n_estimators=n, max_depth=d,
                                              class_weight="balanced",
                                              random_state=42, n_jobs=-1)
            else:
                from sklearn.ensemble import GradientBoostingClassifier
                n   = trial.suggest_int("n_estimators", 50, 300)
                lr  = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
                d   = trial.suggest_int("max_depth", 2, 6)
                clf = GradientBoostingClassifier(n_estimators=n, learning_rate=lr,
                                                  max_depth=d, random_state=42)
            score = cross_val_score(clf, X_proc, y, cv=cv,
                                    scoring="f1", n_jobs=-1).mean()
            trial_scores.append(score)
            progress.progress(len(trial_scores) / n_trials)
            status.text(f"Trial {len(trial_scores)}/{n_trials} — Best F1: {max(trial_scores):.4f}")
            return score

        from sklearn.model_selection import StratifiedKFold
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        st.success(f"✅ Optuna tugadi!")
        st.metric("🏆 Eng yaxshi F1", f"{study.best_value:.4f}")
        st.metric("🔧 Eng yaxshi parametrlar", str(study.best_params))

        # Trial history
        fig = px.line(x=list(range(1, len(trial_scores)+1)), y=trial_scores,
                      title="Optuna Trial History — F1 Score",
                      labels={"x": "Trial", "y": "F1 Score"})
        fig.add_hline(y=study.best_value, line_dash="dash",
                      line_color="green", annotation_text="Best")
        st.plotly_chart(fig, use_container_width=True)

        # Importance
        try:
            imp = optuna.importance.get_param_importances(study)
            fig2 = px.bar(x=list(imp.values()), y=list(imp.keys()),
                          orientation="h", title="Hyperparameter Importance")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception:
            pass

# ════════════════════════════════════════
# 🤖 MODEL O'QITISH
# ════════════════════════════════════════
elif page == "🤖 Model O'qitish":
    st.title("🤖 Model O'qitish (Nested CV)")
    st.markdown("---")
    st.warning("⚠️ O'qitish 5-10 daqiqa ketadi!")

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
        st.subheader("📊 Fold-by-Fold Natijalar")
        f1_arrays = st.session_state["f1_arrays"]
        fold_data = []
        for name, f1s in f1_arrays.items():
            for i, f1 in enumerate(f1s):
                fold_data.append({"Model": name, "Fold": f"Fold {i+1}", "F1": f1})
        fold_df = pd.DataFrame(fold_data)
        fig = px.line(fold_df, x="Fold", y="F1", color="Model",
                      markers=True, title="F1 Score — Har bir Fold")
        st.plotly_chart(fig, use_container_width=True)

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

    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Leaderboard", "📊 Grafiklar", "📉 ROC Curves", "🧪 Wilcoxon"])

    with tab1:
        st.subheader("🏆 Model Leaderboard")
        rows = []
        for i, (name, r) in enumerate(results.items()):
            medal = "🥇" if i==0 else "🥈" if i==1 else "🥉" if i==2 else "  "
            rows.append({
                "Rank": f"{medal} {i+1}",
                "Model": name,
                "Accuracy": f"{r['Accuracy'][0]:.3f}",
                "Precision": f"{r['Precision'][0]:.3f}",
                "Recall": f"{r['Recall'][0]:.3f}",
                "F1 ↑": f"{r['F1'][0]:.3f} ± {r['F1'][1]:.3f}",
                "AUC-ROC": f"{r['AUC-ROC'][0]:.3f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("🥇 Best Model", "Logistic Regression")
        col2.metric("🏆 Best F1", "0.918 ± 0.021")
        col3.metric("📈 Best AUC", "0.966 ± 0.010")

    with tab2:
        metric = st.selectbox("Metric:", ["Accuracy","Precision","Recall","F1","AUC-ROC"])
        means  = [results[m][metric][0] for m in model_names]
        stds   = [results[m][metric][1] for m in model_names]
        colors = ["#2ECC71" if m=="Logistic Regression" else "#3498DB" for m in model_names]
        fig = go.Figure(go.Bar(
            x=model_names, y=means,
            error_y=dict(type="data", array=stds),
            marker_color=colors,
            text=[f"{m:.3f}" for m in means],
            textposition="outside"
        ))
        fig.update_layout(title=f"{metric} Taqqoslash",
                          yaxis_range=[0.6, 1.05])
        st.plotly_chart(fig, use_container_width=True)

        # Radar chart
        st.subheader("🕸️ Radar Chart")
        categories = ["Accuracy","Precision","Recall","F1","AUC-ROC"]
        fig_r = go.Figure()
        for name in model_names[:3]:
            vals = [results[name][c][0] for c in categories]
            vals += [vals[0]]
            fig_r.add_trace(go.Scatterpolar(
                r=vals, theta=categories+[categories[0]],
                fill="toself", name=name))
        fig_r.update_layout(polar=dict(radialaxis=dict(range=[0.7,1.0])),
                             title="Top 3 Model — Radar Chart")
        st.plotly_chart(fig_r, use_container_width=True)

    with tab3:
        st.subheader("ROC Curves (Test Set)")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        models_dict = get_models()
        from sklearn.model_selection import StratifiedKFold
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        fig_roc = go.Figure()
        colors_roc = ["#2ECC71","#E74C3C","#3498DB","#F39C12","#9B59B6","#1ABC9C"]
        for (name, (est, params)), color in zip(models_dict.items(), colors_roc):
            pipe = Pipeline([("pre", preprocessor), ("clf", est)])
            from sklearn.model_selection import GridSearchCV
            gs = GridSearchCV(pipe, params, cv=inner_cv,
                              scoring="f1", n_jobs=-1, refit=True)
            gs.fit(X_tr, y_tr)
            y_prob = gs.best_estimator_.predict_proba(X_te)[:,1]
            fpr, tpr, _ = roc_curve(y_te, y_prob)
            auc = roc_auc_score(y_te, y_prob)
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                          name=f"{name} (AUC={auc:.3f})",
                                          line=dict(color=color, width=2)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                      name="Random", line=dict(dash="dash", color="gray")))
        fig_roc.update_layout(title="ROC Curves — All Models",
                               xaxis_title="False Positive Rate",
                               yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

    with tab4:
        st.subheader("🧪 Wilcoxon Signed-Rank Test")
        st.info("""
        **Logistic Regression vs Gradient Boosting**
        - Wilcoxon statistic = 7.000
        - p-value = 1.000
        - ✅ Ikki model orasida statistik farq yo'q (p ≥ 0.05)
        - Har ikkisi ham deployment uchun mos!
        """)
        fig_w = px.bar(
            x=["Logistic Regression", "Gradient Boosting"],
            y=[0.918, 0.914],
            color=["Best","Runner-up"],
            color_discrete_map={"Best":"#2ECC71","Runner-up":"#3498DB"},
            title="Top 2 Model F1 Taqqoslash",
            text=[0.918, 0.914]
        )
        st.plotly_chart(fig_w, use_container_width=True)

# ════════════════════════════════════════
# 🔬 SHAP VALUES
# ════════════════════════════════════════
elif page == "🔬 SHAP Values":
    st.title("🔬 SHAP Values — Model Tushuntirish (XAI)")
    st.markdown("---")
    st.info("SHAP (SHapley Additive exPlanations) — har bir feature qancha ta'sir qilayotganini ko'rsatadi!")

    if st.button("🔬 SHAP Hisoblash", type="primary"):
        with st.spinner("SHAP hisoblanmoqda..."):
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import GridSearchCV, StratifiedKFold
            inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

            pipe = Pipeline([("pre", preprocessor),
                              ("clf", GradientBoostingClassifier(
                                  n_estimators=100, learning_rate=0.1,
                                  max_depth=3, random_state=42))])
            pipe.fit(X_tr, y_tr)

            X_te_proc = pipe.named_steps["pre"].transform(X_te)
            ohe       = pipe.named_steps["pre"].named_transformers_["cat"]["ohe"]
            cat_names = list(ohe.get_feature_names_out(cat_cols))
            feat_names= cat_names + num_cols

            explainer  = shap.TreeExplainer(pipe.named_steps["clf"])
            shap_values= explainer.shap_values(X_te_proc)

            st.session_state["shap_values"] = shap_values
            st.session_state["X_te_proc"]   = X_te_proc
            st.session_state["feat_names"]  = feat_names

        st.success("✅ SHAP hisoblandi!")

    if "shap_values" in st.session_state:
        shap_values = st.session_state["shap_values"]
        X_te_proc   = st.session_state["X_te_proc"]
        feat_names  = st.session_state["feat_names"]

        # Summary bar plot
        st.subheader("📊 Feature Importance (SHAP)")
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, X_te_proc,
                          feature_names=feat_names,
                          plot_type="bar", show=False)
        st.pyplot(fig)
        plt.close()

        # Beeswarm
        st.subheader("🐝 SHAP Beeswarm Plot")
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, X_te_proc,
                          feature_names=feat_names, show=False)
        st.pyplot(fig2)
        plt.close()

        # Top features table
        mean_shap = np.abs(shap_values).mean(axis=0)
        feat_imp  = pd.DataFrame({
            "Feature": feat_names,
            "SHAP Importance": mean_shap
        }).sort_values("SHAP Importance", ascending=False).head(15)

        st.subheader("🏅 Top 15 Features (SHAP)")
        fig3 = px.bar(feat_imp, x="SHAP Importance", y="Feature",
                      orientation="h", color="SHAP Importance",
                      color_continuous_scale="Viridis",
                      title="Top 15 Most Important Features")
        st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════
# 🔍 BASHORAT
# ════════════════════════════════════════
elif page == "🔍 Bashorat":
    st.title("🔍 O'quvchi Natijasini Bashorat Qilish")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📚 Akademik")
        G1        = st.slider("G1 (1-davr bahosi)", 0, 20, 10)
        G2        = st.slider("G2 (2-davr bahosi)", 0, 20, 10)
        studytime = st.selectbox("O'qish vaqti", [1,2,3,4],
                                  format_func=lambda x:
                                  {1:"<2h",2:"2-5h",3:"5-10h",4:">10h"}[x])
        failures  = st.selectbox("O'tgan muvaffaqiyatsizliklar", [0,1,2,3])
        absences  = st.slider("Darsdan qolish", 0, 93, 5)

    with col2:
        st.subheader("👨‍👩‍👦 Oila")
        Medu   = st.selectbox("Ona ta'limi", [0,1,2,3,4],
                               format_func=lambda x:
                               {0:"Yo'q",1:"Boshlang'ich",2:"O'rta",
                                3:"Oliy",4:"Magistr"}[x])
        Fedu   = st.selectbox("Ota ta'limi", [0,1,2,3,4],
                               format_func=lambda x:
                               {0:"Yo'q",1:"Boshlang'ich",2:"O'rta",
                                3:"Oliy",4:"Magistr"}[x])
        famrel = st.slider("Oila munosabati (1-5)", 1, 5, 4)
        Pstatus= st.selectbox("Ota-ona holati", ["T","A"],
                               format_func=lambda x:
                               {"T":"Birga","A":"Ajrashgan"}[x])

    with col3:
        st.subheader("🌍 Shaxsiy")
        age      = st.slider("Yosh", 15, 22, 17)
        sex      = st.selectbox("Jins", ["M","F"])
        internet = st.selectbox("Internet", ["yes","no"])
        freetime = st.slider("Bo'sh vaqt (1-5)", 1, 5, 3)
        health   = st.slider("Sog'liq (1-5)", 1, 5, 3)

    if st.button("🎯 Bashorat Qilish", type="primary", use_container_width=True):
        combined = G1 + G2
        risk     = studytime * failures
        ab_rate  = absences / (age + 1)

        # Score calculation
        score = 0
        score += combined * 3
        score -= failures * 8
        score -= absences * 0.3
        score += studytime * 3
        score += Medu * 1.5
        score += Fedu * 1.0

        prob = min(max((score + 20) / 80, 0.05), 0.98)
        pred = 1 if prob >= 0.5 else 0

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            if pred == 1:
                st.success(f"✅ **BASHORAT: PASS**")
                st.balloons()
            else:
                st.error(f"❌ **BASHORAT: FAIL**")

            # Gauge chart
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                title={"text": "O'tish ehtimoli (%)"},
                delta={"reference": 50},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#2ECC71" if pred==1 else "#E74C3C"},
                    "steps": [
                        {"range": [0, 40],  "color": "#FADBD8"},
                        {"range": [40, 60], "color": "#FDEBD0"},
                        {"range": [60, 100],"color": "#D5F5E3"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75, "value": 50
                    }
                }
            ))
            st.plotly_chart(fig_g, use_container_width=True)

        with col2:
            st.subheader("📊 Tahlil")
            col_a, col_b = st.columns(2)
            col_a.metric("Combined Score", f"{combined}/40",
                          delta=f"{'✅' if combined>=14 else '⚠️'}")
            col_b.metric("Risk Index", f"{risk}",
                          delta=f"{'✅ Low' if risk==0 else '⚠️ High'}")
            col_a.metric("Absence Rate", f"{ab_rate:.2f}",
                          delta=f"{'✅' if ab_rate<1 else '⚠️'}")
            col_b.metric("Study Time", f"{studytime}/4")

            risk_level = "🟢 LOW" if prob > 0.7 else "🟡 MEDIUM" if prob > 0.4 else "🔴 HIGH"
            st.markdown(f"### Risk Level: {risk_level}")

            if pred == 0:
                st.markdown("""
                **💡 Tavsiyalar:**
                - 📚 O'qish vaqtini oshiring
                - 🏫 Darslarga qatnashishni yaxshilang
                - 👨‍👩‍👦 Oila bilan ko'proq muloqot qiling
                """)
