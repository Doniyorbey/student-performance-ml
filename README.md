# 🎓 Student Performance ML

**Machine Learning Model Optimization for Student Performance Prediction**

---

## 👨‍🎓 Author
- **Name:** Eltezorov Doriyorbek Norqo'zi o'g'li
- **Group:** 22-305 | AI Track
- **Supervisor:** Muhammadjon
- **University:** PDP University, Tashkent | 2026

---

## 📌 About

This project applies 6 machine learning classification algorithms
to the UCI Student Performance Dataset (395 students, 33 features)
to predict whether a student will pass or fail their final exam.

**Key features:**
- Nested 5-fold Cross-Validation (unbiased evaluation)
- GridSearchCV hyperparameter optimization
- 4 engineered features (risk index, absence rate, etc.)
- Wilcoxon signed-rank statistical test
- Interactive Streamlit dashboard

---

## 🏆 Results

| Model | F1 | AUC-ROC |
|-------|-----|---------|
| **Logistic Regression** 🥇 | **0.918** | **0.966** |
| Gradient Boosting | 0.914 | 0.960 |
| Decision Tree | 0.901 | 0.936 |
| SVM RBF | 0.899 | 0.956 |
| Random Forest | 0.892 | 0.946 |
| KNN | 0.843 | 0.845 |

---

## 🔧 Tech Stack

- Python 3.12
- Scikit-learn, Pandas, NumPy
- Streamlit, Plotly, Seaborn
- Scipy, Optuna, SHAP

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
student-performance-ml/
├── app.py              # Streamlit dashboard
├── ml_pipeline.py      # ML training pipeline
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```
