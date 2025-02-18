from flask import Flask, render_template, request, redirect
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

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# Load model artifacts
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
columns = joblib.load('columns.joblib')

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

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.username = user_data['username']

@login_manager.user_loader
def load_user(user_id):
    db = get_db_connection().heart_db
    user_data = db.patients.find_one({'_id': ObjectId(user_id)})
    return User(user_data) if user_data else None

# Routes
@app.route('/')
def home():
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            db = get_db_connection().heart_db
            user_data = db.patients.find_one({'username': request.form['username'].strip()})
            
            if user_data and check_password_hash(user_data['password'], request.form['password']):
                user = User(user_data)
                login_user(user)
                return redirect('/assessment')
            return render_template('login.html', error='Invalid credentials')
        except Exception as e:
            print(f"Login error: {str(e)}")
            return render_template('login.html', error='Database error')
    return render_template('login.html')

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
                'created_at': datetime.datetime.utcnow()
            })
            
            return redirect('/login')
        except pymongo.errors.DuplicateKeyError:
            return render_template('register.html', error='Username/Email already exists')
    return render_template('register.html')

@app.route('/assessment')
@login_required
def assessment():
    # Exclude 'Sex' from the displayed features since we have a dedicated gender dropdown
    filtered_features = [col for col in columns if col != 'Sex']
    return render_template('index.html', features=filtered_features)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Convert gender to numerical value
        gender = request.form['Gender']
        sex_value = 0 if gender == 'Male' else 1
        
        # Create user input with converted gender value
        user_input = {
            col: float(request.form[col]) 
            for col in columns 
            if col != 'Sex'
        }
        user_input['Sex'] = sex_value
        
        input_df = pd.DataFrame([user_input], columns=columns)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        result = "High Risk" if prediction == 1 else "Low Risk"
        
        if prediction == 1:
            # Get patient's phone number
            db = get_db_connection().heart_db
            patient = db.patients.find_one({'_id': ObjectId(current_user.id)})
            
            if patient and 'phone' in patient:
                message = f"ALERT: High heart attack risk detected. Please consult a doctor immediately."
                send_sms_alert(patient['phone'], message)
        
        return render_template('result.html', result=result, prediction=prediction)
    except Exception as e:
        return render_template('index.html', error="Invalid input! Please check all values", features=columns)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/login')

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