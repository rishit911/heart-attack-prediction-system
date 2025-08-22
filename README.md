HeartCare: IoT and ML-Based Heart Attack Detection and Alarm System
📌 Overview

Heart disease remains one of the leading causes of mortality worldwide. HeartCare is an intelligent, real-time heart disease prediction and alert system that combines IoT sensors with an ensemble machine learning model to assess cardiovascular risk efficiently and accurately.

The system collects real-time patient data via a CLI-based interface (with future scope for web/mobile deployment) and predicts the likelihood of heart disease using a Voting Classifier that combines:

Random Forest

Support Vector Machine (SVM)

Logistic Regression

K-Nearest Neighbors (KNN)

This project is designed as a proof-of-concept for a scalable diagnostic assistant in both urban and rural healthcare environments.

🚀 Key Features

Real-time data input from patients or medical staff

Ensemble ML approach (soft voting) to improve prediction accuracy

Feature selection (SelectKBest) for dimensionality reduction

Prediction confidence scores for transparent decision-making

IoT readiness for integration with wearable sensors

High accuracy (89%) outperforming individual models

Command-line interface with planned upgrade to GUI/web app

🏗️ Project Architecture

Data Acquisition – IoT sensors or user input collect parameters such as:

Age, sex, chest pain type

Blood pressure, cholesterol, heart rate

ECG results, ST slope, SpO₂, temperature

Data Preprocessing – normalization, label encoding, missing value handling

Feature Selection – using SelectKBest(f_classif) to select top 10 features

Model Building – individual classifiers trained and combined via VotingClassifier

Prediction Interface – real-time CLI that displays results and confidence level

📊 Performance

Ensemble Model Accuracy: ~89%

Metrics Used: Accuracy, Precision, Recall, F1-score, Confusion Matrix

Result: Ensemble classifier consistently outperforms individual models.
