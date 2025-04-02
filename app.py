from flask import Flask, render_template, request, redirect, abort, session, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
import os
import datetime
import joblib
import pandas as pd
import pymongo
from twilio.rest import Client
import numpy as np
from fetch_thingSpeak_data import fetch_data
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# Load the dataset for feature selection
data = pd.read_csv('ECS_Dataset.csv')
data['Heart Disease'] = data['Heart Disease'].map({'Presence': 1, 'Absence': 0})
X = data.drop('Heart Disease', axis=1)
y = data['Heart Disease']

# Initialize feature selector and scaler
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

scaler = StandardScaler()
scaler.fit(X_selected)

# Create and train the ensemble model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(kernel='rbf', probability=True, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

ensemble = VotingClassifier(estimators=[
    ('rf', rf),
    ('svm', svm),
    ('lr', lr),
    ('knn', knn)
], voting='soft')

# Train the ensemble model
ensemble.fit(X_selected, y)

# Configure MongoDB
def get_db_connection():
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise ValueError("MONGO_URI environment variable is not set.")
    return MongoClient(uri)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Define chest pain type mapping
CHEST_PAIN_TYPES = {
    1: "Typical angina",
    2: "Atypical angina",
    3: "Non-anginal pain",
    4: "Asymptomatic"
}

class User(UserMixin):
    def __init__(self, user_id, role, username):
        self.id = str(user_id)
        self.role = role
        self.username = username
        
    @staticmethod
    def get(user_id):
        db = get_db_connection().heart_db
        # Check doctor collection first
        doctor = db.doctors.find_one({'_id': ObjectId(user_id)})
        if doctor:
            return User(doctor['_id'], 'doctor', doctor['username'])
            
        # Then check patient collection
        patient = db.patients.find_one({'_id': ObjectId(user_id)})
        if patient:
            return User(patient['_id'], 'patient', patient['username'])
            
        return None

@login_manager.user_loader
def load_user(user_id):
    db = get_db_connection().heart_db
    # Check both collections
    user_data = db.patients.find_one({'_id': ObjectId(user_id)}) or db.doctors.find_one({'_id': ObjectId(user_id)})
    return User.get(user_id) if user_data else None

# ======================
# PATIENT ROUTES
# ======================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            db = get_db_connection().heart_db
            user_data = db.patients.find_one({'username': request.form['username'].strip()})
            
            if user_data and check_password_hash(user_data['password'], request.form['password']):
                user = User.get(user_data['_id'])
                login_user(user)
                return redirect(url_for('patient_dashboard'))
            return render_template('patients/login.html', error='Invalid credentials')
        except Exception as e:
            return render_template('patients/login.html', error='Database error')
    return render_template('patients/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            db = get_db_connection().heart_db
            hashed_pw = generate_password_hash(request.form['password'])
            
            result = db.patients.insert_one({
                'username': request.form['username'],
                'password': hashed_pw,
                'email': request.form['email'],
                'phone': request.form['phone'],
                'gender': request.form['gender'],
                'age': int(request.form['age']),
                'role': 'patient',
                'created_at': datetime.datetime.utcnow()
            })
            
            return redirect('/login')
        except pymongo.errors.DuplicateKeyError:
            return render_template('patients/register.html', error='Username/Email already exists')
    return render_template('patients/register.html')

@app.route('/patient/dashboard')
@login_required
def patient_dashboard():
    if current_user.role != 'patient':
        abort(403)
    
    db = get_db_connection().heart_db
    patient = db.patients.find_one({'_id': ObjectId(current_user.id)})
    doctors = list(db.doctors.find())
    
    return render_template('patients/dashboard.html',
                         medical=patient.get('medical_profile', {}),
                         assessments=patient.get('assessments', [])[::-1],
                         doctors=doctors)

@app.route('/patient/assessment')
@login_required
def patient_assessment():
    db = get_db_connection().heart_db
    patient = db.patients.find_one({'_id': ObjectId(current_user.id)})
    assessments = patient.get('assessments', [])[::-1]
    
    # Get gender and age from patient profile
    gender = patient.get('gender', 'Male')  # Default to Male if not set
    age = patient.get('age', 30)  # Default to 30 if not set
    
    # Read the last record from health_data.csv
    try:
        health_data = pd.read_csv('health_data.csv')
        last_record = health_data.iloc[-1]
        # Check if heart rate is -999 (invalid reading)
        heart_rate = None if last_record.iloc[1] == -999 else last_record.iloc[1]
        # Check if SpO2 is -999 (invalid reading)
        spo2 = None if last_record.iloc[2] == -999 else last_record.iloc[2]
        # Get temperature from the health data
        temperature = last_record.iloc[3]
    except Exception as e:
        print(f"Health data read error: {str(e)}")
        heart_rate = None
        spo2 = None
        temperature = None

    return render_template('patients/assessment.html',
                         assessments=assessments,
                         features=selected_features,
                         heart_rate=heart_rate,
                         spo2=spo2,
                         temperature=temperature,
                         gender=gender,
                         age=age)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Check medical profile first
        db = get_db_connection().heart_db
        patient = db.patients.find_one({'_id': ObjectId(current_user.id)})
        medical = patient.get('medical_profile', {})
        
        if not medical:
            return render_template('patients/assessment.html',
                                error="Medical profile not complete - contact your doctor",
                                features=selected_features)
        
        # Get patient profile data
        gender = patient.get('gender', 'Male')
        age = patient.get('age', 30)
                                
        # Build input with patient data + medical profile
        user_input = {
            'Age': float(age),
            'Sex': 0 if gender == 'Male' else 1,
            'Chest pain type': medical['chest_pain_type'],
            'BP': medical['bp'],
            'Cholesterol': medical['cholesterol'],
            'FBS over 120': medical['fbs'],
            'EKG results': medical['ekg'],
            'Max HR': float(request.form['Max HR']),
            'Exercise angina': medical['exercise_angina'],
            'ST depression': medical['st_depression'],
            'Slope of ST': medical['slope_st'],
            'Number of vessels fluro': medical['vessels_fluro'],
            'Thallium': medical['thallium'],
            'SpO2 Level (%)': float(request.form['SpO2']),
            'Temperature': float(request.form['Temperature'])
        }
        
        # Convert to DataFrame and select features
        input_df = pd.DataFrame([user_input])
        input_selected = input_df[selected_features]
        
        # Scale the input
        input_scaled = scaler.transform(input_selected)
        
        # Make prediction
        prediction = ensemble.predict(input_scaled)[0]
        prediction_proba = ensemble.predict_proba(input_scaled)
        
        risk_percentage = round(prediction_proba[0][1] * 100, 2)
        
        # Store the assessment in the database with correct format
        assessment = {
            'date': datetime.datetime.utcnow(),
            'result': 'High Risk' if prediction == 1 else 'Low Risk',
            'risk_percentage': risk_percentage,
            'features': user_input
        }
        
        # Update the patient's assessments array
        db.patients.update_one(
            {'_id': ObjectId(current_user.id)},
            {'$push': {'assessments': assessment}}
        )
        
        # Send SMS alert if risk is high
        if risk_percentage >= 70:
            send_sms_alert(patient['phone'], 
                         f"High heart disease risk detected: {risk_percentage}%. Please consult your doctor immediately.")
        
        # Get updated assessments list
        patient = db.patients.find_one({'_id': ObjectId(current_user.id)})
        assessments = patient.get('assessments', [])[::-1]  # Reverse to show newest first
        
        return render_template('patients/assessment.html',
                             prediction=prediction,
                             risk_percentage=risk_percentage,
                             features=selected_features,
                             assessments=assessments)
                             
    except Exception as e:
        return render_template('patients/assessment.html',
                             error=f"Error during prediction: {str(e)}",
                             features=selected_features)

@app.route('/history')
@login_required
def history():
    db = get_db_connection().heart_db
    assessments = db.assessments.find(
        {'user_id': ObjectId(current_user.id)}
    ).sort('assessed_at', -1)
    return render_template('history.html', assessments=assessments)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/login')

@app.route('/refresh_heart_rate')
@login_required
def refresh_heart_rate():
    try:
        # Run the fetch_data function to get latest values
        timestamp, heart_rate, spo2, temperature = fetch_data()
        
        # Save to CSV
        from fetch_thingSpeak_data import save_to_csv
        save_to_csv()
        
        if heart_rate == '-999':
            return {'success': False, 'message': 'Please enter heart rate manually', 'heart_rate': None}
        
        return {
            'success': True, 
            'heart_rate': heart_rate,
            'spo2': spo2,
            'temperature': temperature
        }
    except Exception as e:
        return {'success': False, 'message': str(e), 'heart_rate': None}

@app.route('/refresh_sensor_data')
@login_required
def refresh_sensor_data():
    try:
        # Run the fetch_data function to get latest values
        timestamp, heart_rate, spo2, temperature = fetch_data()
        
        # Save to CSV
        from fetch_thingSpeak_data import save_to_csv
        save_to_csv()
        
        # Create response data
        response = {'success': True}
        
        # Check heart rate value
        if heart_rate == '-999':
            response['hr_success'] = False
            response['hr_message'] = 'Heart rate reading unavailable'
        else:
            response['hr_success'] = True
            response['heart_rate'] = heart_rate
            
        # Check SpO2 value
        if spo2 == '-999':
            response['spo2_success'] = False
            response['spo2_message'] = 'SpO2 reading unavailable'
        else:
            response['spo2_success'] = True
            response['spo2'] = spo2
            
        # Add temperature (always available in this dataset)
        response['temperature'] = temperature
        
        return response
    except Exception as e:
        return {
            'success': False, 
            'message': str(e),
            'heart_rate': None,
            'spo2': None,
            'temperature': None
        }

# ======================
# DOCTOR ROUTES
# ======================
@app.route('/login/doctor', methods=['GET', 'POST'])
def login_doctor():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        db = get_db_connection().heart_db
        doctor = db.doctors.find_one({'username': username})
        
        if doctor and check_password_hash(doctor['password'], password):
            user = User.get(doctor['_id'])
            login_user(user)
            return redirect(url_for('doctor_dashboard'))
            
        return render_template('doctors/login_doctor.html', error='Invalid credentials')
        
    return render_template('doctors/login_doctor.html')

@app.route('/register/doctor', methods=['GET', 'POST'])
def register_doctor():
    if request.method == 'POST':
        db = get_db_connection().heart_db
        hashed_pw = generate_password_hash(request.form['password'])
        
        db.doctors.insert_one({
            'username': request.form['username'],
            'password': hashed_pw,
            'email': request.form['email'],
            'phone': request.form['phone'],
            'role': 'doctor',
            'created_at': datetime.datetime.utcnow()
        })
        return redirect('/login/doctor')
    return render_template('doctors/register_doctor.html')

@app.route('/doctor/dashboard')
@login_required
def doctor_dashboard():
    if current_user.role != 'doctor':
        abort(403)
        
    db = get_db_connection().heart_db
    patients = list(db.patients.find())
    return render_template('doctors/doctor_dashboard.html', patients=patients)

@app.route('/doctor/patient/<patient_id>')
@login_required
def patient_details(patient_id):
    if current_user.role != 'doctor':
        abort(403)
    
    db = get_db_connection().heart_db
    patient = db.patients.find_one({'_id': ObjectId(patient_id)})
    
    if not patient:
        abort(404)
        
    return render_template('doctors/patient_details.html', 
                         patient=patient,
                         medical=patient.get('medical_profile', {}),
                         assessments=patient.get('assessments', [])[::-1])

@app.route('/doctor/update_medical/<patient_id>', methods=['GET', 'POST'])
@login_required
def update_medical(patient_id):
    if current_user.role != 'doctor':
        abort(403)
    
    db = get_db_connection().heart_db
    patient = db.patients.find_one({'_id': ObjectId(patient_id)})
    
    if request.method == 'POST':
        medical_data = {
            'chest_pain_type': int(request.form['Chest pain type']),
            'cholesterol': int(request.form['Cholesterol']),
            'fbs': int(request.form['FBS over 120']),
            'ekg': int(request.form['EKG results']),
            'exercise_angina': int(request.form['Exercise angina']),
            'st_depression': float(request.form['ST depression']),
            'slope_st': int(request.form['Slope of ST']),
            'vessels_fluro': int(request.form['Number of vessels fluro']),
            'thallium': int(request.form['Thallium']),
            'bp': int(request.form['BP'])  # Blood Pressure
        }
        
        db.patients.update_one(
            {'_id': ObjectId(patient_id)},
            {'$set': {'medical_profile': medical_data}}
        )
        
        return redirect(url_for('doctor_dashboard'))
    
    return render_template('doctors/doctor_medical_edit.html', 
                         patient=patient,
                         medical=patient.get('medical_profile', {}))

# ======================
# COMMON ROUTES
# ======================
@app.route('/')
def home():
    # If user is already logged in, redirect to appropriate dashboard
    if current_user.is_authenticated:
        if current_user.role == 'doctor':
            return redirect(url_for('doctor_dashboard'))
        else:
            return redirect(url_for('patient_dashboard'))
    
    # Redirect to login page for non-authenticated users
    return redirect(url_for('login'))

def send_sms_alert(phone, message):
    try:
        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        twilio_phone = os.getenv('TWILIO_PHONE')
        
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=message,
            from_=twilio_phone,
            to=phone
        )
        return True
    except Exception as e:
        print(f"SMS sending failed: {str(e)}")
        return False

if __name__ == '__main__':
    app.run(debug=True)