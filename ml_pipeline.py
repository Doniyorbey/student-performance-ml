import os, warnings, joblib
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve,
                              confusion_matrix)
from scipy.stats import wilcoxon
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
np.random.seed(SEED)
DATA_URL = "https://raw.githubusercontent.com/selva86/datasets/master/student-mat.csv"

@st.cache_data if False else lambda f: f
def load_data():
    df = pd.read_csv(DATA_URL, sep=";")
    df["target"]         = (df["G3"] >= 10).astype(int)
    df["risk_index"]     = df["studytime"] * df["failures"]
    df["absence_rate"]   = df["absences"] / (df["age"] + 1)
    df["combined_score"] = df["G1"] + df["G2"]
    df["failures_sq"]    = df["failures"] ** 2
    return df

def get_preprocessor(cat_cols, num_cols):
    return ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols),
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler())
        ]), num_cols),
    ])

def get_models():
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
            {"clf__C": [0.01, 0.1, 1, 10, 100]}),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
            {"clf__max_depth": [3, 5, 7, None], "clf__min_samples_leaf": [1, 5, 10]}),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [5, 10, None], "clf__max_features": ["sqrt", "log2"]}),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=SEED),
            {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1, 0.2], "clf__max_depth": [3, 5]}),
        "SVM RBF": (
            SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED),
            {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.001, 0.01]}),
        "KNN": (
            KNeighborsClassifier(n_jobs=-1),
            {"clf__n_neighbors": [3, 5, 7, 9, 11], "clf__weights": ["uniform", "distance"]}),
    }

def train_models(X, y, preprocessor):
    models     = get_models()
    outer_cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    inner_cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results    = {}
    f1_arrays  = {}
    best_pipes = {}

    for name, (estimator, param_grid) in models.items():
        pipe = Pipeline([("pre", preprocessor), ("clf", estimator)])
        acc_s, pre_s, rec_s, f1_s, auc_s = [], [], [], [], []

        for train_idx, test_idx in outer_cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            gs = GridSearchCV(pipe, param_grid, cv=inner_cv,
                              scoring="f1", n_jobs=-1, refit=True)
            gs.fit(X_train, y_train)
            bp = gs.best_estimator_
            yp = bp.predict(X_test)
            ya = bp.predict_proba(X_test)[:, 1]
            acc_s.append(accuracy_score(y_test, yp))
            pre_s.append(precision_score(y_test, yp, zero_division=0))
            rec_s.append(recall_score(y_test, yp, zero_division=0))
            f1_s.append(f1_score(y_test, yp, zero_division=0))
            auc_s.append(roc_auc_score(y_test, ya))

        results[name] = {
            "Accuracy":  (np.mean(acc_s),  np.std(acc_s)),
            "Precision": (np.mean(pre_s),  np.std(pre_s)),
            "Recall":    (np.mean(rec_s),  np.std(rec_s)),
            "F1":        (np.mean(f1_s),   np.std(f1_s)),
            "AUC-ROC":   (np.mean(auc_s),  np.std(auc_s)),
        }
        f1_arrays[name] = f1_s

        # Best model full fit
        gs_full = GridSearchCV(pipe, param_grid, cv=inner_cv,
                               scoring="f1", n_jobs=-1, refit=True)
        gs_full.fit(X, y)
        best_pipes[name] = gs_full.best_estimator_

    return results, f1_arrays, best_pipes
