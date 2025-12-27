# Student_Performance_XAI
Predicting student performance using Random Forest and explaining predictions using SHAP.# Student Performance Prediction & Explainability using Machine Learning

## Project Overview

This project focuses on predicting overall student academic performance using machine learning techniques and explaining the predictions with Explainable AI (XAI) methods. The goal is to provide insights into which factors impact student performance while ensuring the model is interpretable, transparent, and ethically sound.

## Problem Statement

Educational institutions aim not only to evaluate student performance but also to understand the underlying factors affecting academic outcomes. Traditional statistical methods often fail to capture complex relationships among demographic, behavioral, and lifestyle factors. Machine learning models can predict accurately but often act as black boxes. This project addresses this by combining predictive modeling with SHAP (SHapley Additive exPlanations) for interpretability.

## Dataset

Two real-world datasets were used:

1. `student-mat.csv` – Student performance in Mathematics.
2. `student-por.csv` – Student performance in Portuguese.

**Features include:**

* Demographics: age, sex, address
* Family background: parental education and jobs
* Academic behavior: study time, failures, absences
* Lifestyle: free time, alcohol consumption
* Grades: G1, G2, G3

## Data Merging

The datasets were merged on common demographic and family features such as age, parental education, study habits, and lifestyle attributes. The merged dataset enables a holistic analysis across both subjects and the creation of a combined target variable (`G3_avg`) representing overall academic performance.

```python
merged_df = pd.merge(math_df, por_df, on=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','nursery','internet'])
```

## Exploratory Data Analysis (EDA)

* Scatter plots and correlation analysis were performed to study the relationship between Math and Portuguese final grades.
* Positive correlation observed, justifying a combined performance metric.

```python
plt.scatter(merged_df['G3_math'], merged_df['G3_por'])
plt.xlabel('G3 Math')
plt.ylabel('G3 Portuguese')
plt.title('Math vs Portuguese Final Grades')
plt.show()
corr = merged_df['G3_math'].corr(merged_df['G3_por'])
print(f"Correlation: {corr}")
```

## Feature Engineering

* Created a new target variable `G3_avg` as the average of Math and Portuguese final grades:

```python
merged_df['G3_avg'] = (merged_df['G3_math'] + merged_df['G3_por']) / 2
```

* This reduces subject-specific bias and represents overall academic performance.

## Feature Selection & Data Leakage Prevention

* Intermediate grades (G1, G2, G3) were removed from features to prevent data leakage.
* Only behavioral, demographic, and lifestyle features were used for predictions.

```python
X = merged_df.drop(columns=['G1_math','G2_math','G3_math','G1_por','G2_por','G3_por','G3_avg'])
y = merged_df['G3_avg']
```

## Data Preprocessing

* One-hot encoding was applied to categorical variables to ensure compatibility with machine learning algorithms:

```python
X = pd.get_dummies(X, drop_first=True)
```

## Train-Test Split

* Dataset split: 80% training, 20% testing.
* Ensures unbiased evaluation of the model.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Model Selection: Random Forest Regressor

* Captures non-linear relationships.
* Robust to noise and outliers.
* Handles mixed data types.
* Provides feature importance for interpretability.

```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

## Model Evaluation

* Metric: Root Mean Squared Error (RMSE) to measure prediction accuracy.

```python
from sklearn.metrics import mean_squared_error
import numpy as np

y_pred = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")
```

* Lower RMSE indicates predictions are close to actual values.

## Feature Importance Analysis

* Identifies which features influence predictions the most.

```python
importances = rf.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)
```

* Key features: study time, failures, absences, parental education.

## Explainable AI Using SHAP

* SHAP provides global and local explanations for model predictions.

```python
import shap
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

* Local explanation for an individual student:

```python
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])
```

## Key Insights

* Study habits, parental support, and lifestyle significantly impact academic performance.
* The model is interpretable, transparent, and ethically sound.
* Provides actionable insights for educators.



## Technologies Used

* Python, Jupyter Notebook
* Pandas, NumPy, Matplotlib, Seaborn
* Scikit-learn (Random Forest Regressor)
* SHAP (Explainable AI)



