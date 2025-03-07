import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


df = pd.read_csv("data/muscle_recovery_dataset_v2.csv")

df.rename(columns={
    'Age': 'age',
    'Lifestyle Factors': 'lifestyle',
    'Previous Injuries': 'previous_injuries',
    'Pain Level': 'pain_level',
    'Severity': 'severity',
    'Inflammation Level': 'inflammation',
    'Treatment Type': 'treatment',
    'Recovery Time': 'Recovery Time'
}, inplace=True)


categorical_cols = ['severity', 'inflammation', 'treatment', 'previous_injuries', 'lifestyle']
label_encoders = {}

for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

X = df.drop(columns=['Recovery Time'])
y = df['Recovery Time']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 


os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(),
    "SVR": SVR()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"models/{name}.pkl")
    print(f"✅ {name} model trained and saved successfully!")

print("🎉 All models trained and saved!")


