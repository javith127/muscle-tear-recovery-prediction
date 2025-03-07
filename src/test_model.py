import joblib
import numpy as np


scaler = joblib.load("models/scaler.pkl")
linear_model = joblib.load("models/LinearRegression.pkl")
random_forest = joblib.load("models/RandomForest.pkl")
svr_model = joblib.load("models/SVR.pkl")


test_input = np.array([[25, 1, 0, 2, 5, 0, 1]])  


test_input_scaled = scaler.transform(test_input) 


linear_pred = linear_model.predict(test_input_scaled)[0]
rf_pred = random_forest.predict(test_input_scaled)[0]
svr_pred = svr_model.predict(test_input_scaled)[0]

print(f"📊 Predictions for test input: {test_input}")
print(f"📈 Linear Regression Prediction: {linear_pred:.2f}")
print(f"🌲 Random Forest Prediction: {rf_pred:.2f}")
print(f"📉 SVR Prediction: {svr_pred:.2f}")

