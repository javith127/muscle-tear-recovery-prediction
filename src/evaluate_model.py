import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_models(X_test, y_test):
    models = ["LinearRegression", "DecisionTree", "XGBoost"]
    
    for name in models:
        model = joblib.load(f"models/{name}.pkl")
        predictions = model.predict(X_test)
        
        print(f"\n📊 {name} Performance:")
        print(f"  🔹 MAE: {mean_absolute_error(y_test, predictions)}")
        print(f"  🔹 MSE: {mean_squared_error(y_test, predictions)}")
        print(f"  🔹 R² Score: {r2_score(y_test, predictions)}")
