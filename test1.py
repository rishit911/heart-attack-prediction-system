import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

# Load the dataset
data = pd.read_csv('ECS_Dataset.csv')

# Preprocessing
# Convert target to binary (1 for Presence, 0 for Absence)
data['Heart Disease'] = data['Heart Disease'].map({'Presence': 1, 'Absence': 0})

# Separate features and target
X = data.drop('Heart Disease', axis=1)
y = data['Heart Disease']

# Feature selection - select top 10 features
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create individual models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(kernel='rbf', probability=True, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

# Create ensemble model
ensemble = VotingClassifier(estimators=[
    ('rf', rf),
    ('svm', svm),
    ('lr', lr),
    ('knn', knn)
], voting='soft')

# Train the ensemble
ensemble.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = ensemble.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Ensemble Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Function to take user input and make prediction
def predict_heart_disease():
    print("\nEnter the following details to assess heart disease risk:")
    
    inputs = {}
    inputs['Age'] = int(input("Age: "))
    inputs['Sex'] = int(input("Sex (1 = male; 0 = female): "))
    inputs['Chest pain type'] = int(input("Chest pain type (1-4): "))
    inputs['BP'] = int(input("Resting blood pressure (mm Hg): "))
    inputs['Cholesterol'] = int(input("Serum cholesterol (mg/dl): "))
    inputs['FBS over 120'] = int(input("Fasting blood sugar > 120 mg/dl (1 = yes; 0 = no): "))
    inputs['EKG results'] = int(input("Resting electrocardiographic results (0-2): "))
    inputs['Max HR'] = int(input("Maximum heart rate achieved: "))
    inputs['Exercise angina'] = int(input("Exercise induced angina (1 = yes; 0 = no): "))
    inputs['ST depression'] = float(input("ST depression induced by exercise relative to rest: "))
    inputs['Slope of ST'] = int(input("Slope of the peak exercise ST segment (1-3): "))
    inputs['Number of vessels fluro'] = int(input("Number of major vessels (0-3) colored by flourosopy: "))
    inputs['Thallium'] = int(input("Thallium stress test result (3,6,7): "))
    inputs['SpO2 Level (%)'] = float(input("Blood oxygen saturation level (%): "))
    inputs['Temperature'] = float(input("Body temperature (Celsius): "))
    
    # Create dataframe from input
    input_df = pd.DataFrame([inputs])
    
    # Select the same features used in training
    input_selected = input_df[selected_features]
    
    # Scale the input
    input_scaled = scaler.transform(input_selected)
    
    # Make prediction
    prediction = ensemble.predict(input_scaled)
    probability = ensemble.predict_proba(input_scaled)
    
    print("\nPrediction Results:")
    if prediction[0] == 1:
        print("High risk of heart disease (Probability: {:.2f}%)".format(probability[0][1]*100))
    else:
        print("Low risk of heart disease (Probability: {:.2f}%)".format(probability[0][0]*100))


predict_heart_disease()