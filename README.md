CAR PRICE PREDICTION – MACHINE LEARNING PROJECT

1. Project Overview
This project predicts car prices using a complete Machine Learning pipeline.
 It includes data preprocessing, feature engineering, model selection, tuning, SHAP explainability, and a Streamlit web application.
This project demonstrates strong ML engineering skills, business understanding, and deployment experience.

2. Dataset
Source: Kaggle – Car Price Prediction Dataset
 Rows: 205
 Columns: 26
 Target Variable: price
Key features include:
 enginesize, curbweight, horsepower, carlength, carwidth, citympg, highwaympg.

3. Technologies Used
Python
Pandas, NumPy
Scikit-Learn
Random Forest, XGBoost
SHAP
Matplotlib, Seaborn
Streamlit
Joblib

4. Data Preprocessing Steps
Checked for missing values (none found).
Converted categorical columns using one-hot encoding.
Scaled numeric features using StandardScaler.
Split dataset into training and testing sets (80/20).
Selected top predictive features for the Streamlit app.
Top selected features:
enginesize, curbweight, horsepower, highwaympg, citympg, carwidth, carlength.

5. Model Building
Models trained and evaluated:
Linear Regression
R² ≈ 0.89, RMSE ≈ 2900, MAE ≈ 2100

Decision Tree
R² ≈ 0.88, RMSE ≈ 2930, MAE ≈ 2087

Random Forest (Final Model)
R² = 0.95, RMSE = 1810, MAE = 1231

XGBoost
R² ≈ 0.93, RMSE ≈ 2133, MAE ≈ 1474

The Random Forest model performed best and was selected for deployment.

6. Hyperparameter Tuning (XGBoost)
GridSearchCV was used to optimize:
learning_rate, n_estimators, max_depth, subsample, colsample_bytree.
Best parameters found:
Learning rate = 0.05
n_estimators = 200
max_depth = 4
subsample = 1.0
colsample_bytree = 0.8

7. Model Evaluation
Residual and error analysis showed:
No model bias
Errors centered around zero
Slight spread for higher-priced cars (expected)
Good generalization performance
RMSE = 1810 on Random Forest indicates strong prediction quality.

8. Feature Importance
Random Forest identified the following as the strongest predictors:
enginesize
curbweight
highwaympg
horsepower
carwidth
These directly influence car price (engine power, vehicle size, and efficiency).

9. Explainable AI (SHAP Analysis)
SHAP analysis revealed:
High enginesize increases predicted price
Low enginesize decreases predicted price
Heavy vehicles have higher predicted prices
High highwaympg (good mileage) decreases price
Stroke and compression ratio have minimal impact
SHAP helped transparently explain model decisions.

10. Streamlit Application
A clean and interactive UI was created using Streamlit.
Features included:
Slider inputs for numerical features
Real-time price prediction
Backend model loaded from joblib
Simple and user-friendly interface
App files:
 app.py, car_price_rf.pkl, scaler.pkl

11. Project Structure
CarPricePrediction
 │
 ├── app.py
 ├── car_price_rf.pkl
 ├── scaler.pkl
 ├── notebook.ipynb
 ├── README.md
 └── images folder (plots)

13. Conclusion
This project demonstrates:
End-to-end ML pipeline skills
Feature engineering
Model comparison
Explainable AI using SHAP
Real-time prediction app
Deployment readiness
It is portfolio-ready and strong for ML Engineer, Data Scientist, Data Engineer, and AI roles.


14. Future Enhancements
Add categorical feature inputs to the UI
Add SHAP visualization inside the Streamlit app
Deploy on HuggingFace Spaces
Include model comparison tab
Build complete API backend
