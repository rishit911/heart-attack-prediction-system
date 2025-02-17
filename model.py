import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

def predict_heart_attack_risk():
    # Load saved artifacts
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    columns = joblib.load('columns.joblib')
    
    user_input = {}
    for col in columns:
        user_input[col] = float(input(f"Enter {col}: "))

    # Convert input to DataFrame and scale it
    input_df = pd.DataFrame([user_input], columns=columns)
    input_scaled = scaler.transform(input_df)

    # Predict and return result
    prediction = model.predict(input_scaled)[0]
    result = "High Risk (1)" if prediction == 1 else "Low Risk (0)"
    print(f"Prediction: {result}")

if __name__ == "__main__":
    # MOVE ALL DATA LOADING AND MODEL TRAINING CODE HERE
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "Heart_Disease_Prediction.csv")
    df = pd.read_csv(file_path)

    # Convert target variable to binary (1 for Presence, 0 for Absence)
    df["Heart Disease"] = df["Heart Disease"].map({"Presence": 1, "Absence": 0})

    # Split dataset into features and target
    X = df.drop(columns=["Heart Disease"])
    y = df["Heart Disease"]

    # Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features for better model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression model with increased iterations and new solver
    model = LogisticRegression(solver='saga', max_iter=5000)
    model.fit(X_train_scaled, y_train)

    # Predict on test data
    y_pred = model.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Save the model and scaler
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(X.columns.tolist(), 'columns.joblib')
    
    print("Model trained and saved successfully!")
