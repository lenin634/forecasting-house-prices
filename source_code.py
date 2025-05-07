# source_code.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('dataset/dataset.csv')

# Features and Target
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Support Vector Regressor
svm = SVR()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

# Evaluation Function
def evaluate(model_name, y_true, y_pred):
    print(f"{model_name} Evaluation:")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", mean_squared_error(y_true, y_pred, squared=False))
    print("R² Score:", r2_score(y_true, y_pred))
    print("-" * 30)

# Evaluate both models
evaluate("Random Forest", y_test, rf_pred)
evaluate("SVM", y_test, svm_pred)
