import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(file_path="data/muscle_recovery_dataset_v2.csv"):
    print("📂 Loading dataset...")
    df = pd.read_csv(file_path)

    print("✅ Dataset Loaded Successfully!")
    print("📝 Columns Found:", df.columns)  

   
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

  
    if 'Recovery Time' not in df.columns:
        raise ValueError("❌ Error: Column 'Recovery Time' is missing in the dataset!")


    X = df.drop(columns=['Recovery Time'])  
    y = df['Recovery Time'] 

    numerical_cols = ['age', 'pain_level']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("✅ Data Preprocessing Completed!")
    return X_train, X_test, y_train, y_test

load_and_preprocess_data()

