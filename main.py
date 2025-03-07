import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib


models = {
    "LinearRegression": joblib.load("models/LinearRegression.pkl"),
    "RandomForest": joblib.load("models/RandomForest.pkl"),
    "SVR": joblib.load("models/SVR.pkl"),
}

scaler = joblib.load("models/scaler.pkl")


def predict():
    try:
        
        age = float(entry_age.get())
        severity = int(entry_severity.get())
        inflammation = int(entry_inflammation.get())
        treatment = int(entry_treatment.get())
        pain_level = float(entry_pain.get())
        previous_injuries = int(entry_previous_injury.get())
        lifestyle = int(entry_lifestyle.get())

        
        input_features = np.array([[age, lifestyle, previous_injuries, pain_level, severity, inflammation, treatment]])
        input_features_scaled = scaler.transform(input_features)  

        
        predictions = {model: models[model].predict(input_features_scaled)[0] for model in models}

        
        result_msg = (f"Linear Regression Prediction: {predictions['LinearRegression']:.2f} days\n"
                      f"Random Forest Prediction: {predictions['RandomForest']:.2f} days\n"
                      f"SVR Prediction: {predictions['SVR']:.2f} days")
        messagebox.showinfo("Prediction", result_msg)
    
    except Exception as e:
        messagebox.showerror("Error", str(e))

#UI
root = tk.Tk()
root.title("Muscle Tear Recovery Prediction")
root.geometry("400x500")

tk.Label(root, text="Age:").pack()
entry_age = tk.Entry(root)
entry_age.pack()

tk.Label(root, text="Severity (1-Mild, 2-Moderate, 3-Severe):").pack()
entry_severity = tk.Entry(root)
entry_severity.pack()

tk.Label(root, text="Inflammation Level (1-Mild, 2-Moderate, 3-Severe):").pack()
entry_inflammation = tk.Entry(root)
entry_inflammation.pack()

tk.Label(root, text="Treatment Type (1-Physio, 2-Surgery, 3-Medication, 4-RICE):").pack()
entry_treatment = tk.Entry(root)
entry_treatment.pack()

tk.Label(root, text="Pain Level (1-10):").pack()
entry_pain = tk.Entry(root)
entry_pain.pack()

tk.Label(root, text="Previous Injuries (0-No, 1-Yes):").pack()
entry_previous_injury = tk.Entry(root)
entry_previous_injury.pack()

tk.Label(root, text="Lifestyle (1-Sedentary, 2-Active, 3-Athlete):").pack()
entry_lifestyle = tk.Entry(root)
entry_lifestyle.pack()


btn_predict = tk.Button(root, text="Predict", command=predict)
btn_predict.pack()

root.mainloop()
