Here is your **README.md content in a clean ready-to-paste format**.
Just copy everything below and paste it into your `README.md` file.

---

# Gym Willingness Prediction System

## Overview

This project predicts whether a university student is willing to go to the gym on a given day based on behavioral and lifestyle factors such as sleep patterns, stress levels, academic workload, and previous gym activity.

The goal of this project is to demonstrate how a machine learning model can be engineered into a usable system rather than remaining inside a notebook. The project includes data generation, SQL storage, model training, prediction interface, and a dashboard.

---

## Problem Statement

University students often struggle with maintaining a consistent fitness routine due to academic workload, stress levels, sleep patterns, and energy fluctuations. This project aims to build a predictive model that determines whether a student is likely to want to go to the gym on a given day (Yes/No) based on lifestyle and behavioral factors.

---
## System Architecture

Data Generation → SQL Database → Model Training → Model Selection → Saved Model → Streamlit Dashboard → Prediction → Retraining

---
## How It Works

1. Synthetic data is generated using Python scripts  
2. Data is stored in a SQL database (SQLite)  
3. Multiple ML models are trained and evaluated  
4. Best model is selected and saved  
5. Streamlit dashboard allows user interaction  
6. Retraining pipeline updates the model

---
## Model Lifecycle

The system supports continuous improvement:

- New data is added to the database  
- Models are retrained  
- New versions are saved  
- Latest model is used in dashboard
  
---
## Dashboard Preview

The project includes an interactive Streamlit dashboard designed to provide a user-friendly interface for prediction and analysis.

### Features

- User input panel to enter lifestyle and behavioral factors such as study hours, sleep, stress level, and energy level  
- Predicts whether a student is willing to go to the gym (Yes/No)  
- Displays probability score for better interpretation  
- Clean and intuitive UI for easy interaction  
- Interactive visualizations including 3D plots to analyze relationships between features  
- Project timeline section for understanding development stages  

### Purpose

The dashboard makes the machine learning model accessible to users, allowing real-time predictions without requiring technical knowledge. It focuses on usability, visualization, and interaction rather than raw data display.

---

## Features

* Synthetic dataset generation
* SQL database storage
* Multiple machine learning models
* Automatic model comparison
* Model versioning system
* Streamlit interactive dashboard
* Model lifecycle with retraining pipeline
* Git version control

---

## Project Structure

```
Hackathon3_Project
│
├── data
│   └── generate_data.py
│
├── database
│   └── gym_data.db
│
├── models
│   ├── train_model.py
│   ├── latest_model.pkl
│   └── saved_models
│
├── app
│   └── dashboard.py
│
├── retraining
│   └── retrain_model.py
│
├── requirements.txt
└── README.md
```

---

## Dataset Description

The dataset simulates behavioral and lifestyle factors affecting a student's willingness to go to the gym.

Features used in the dataset:

* **STUDY_HOURS** – Number of hours spent studying
* **EXAM_SOON** – Whether an exam is approaching
* **SLEEP_HOURS** – Hours of sleep
* **STRESS_LEVEL** – Stress level from 1 to 10
* **ENERGY_LEVEL** – Energy level from 1 to 10
* **PAST_WEEK_GYM_VISITS** – Number of gym visits in the past week
* **GYM_WILLING** – Target variable (1 = willing to go, 0 = not willing)

---

## Machine Learning Models Used

Three traditional machine learning models are used for comparison:

1. Logistic Regression
2. Random Forest Classifier
3. Support Vector Machine (SVM)

The model with the best performance is automatically selected and saved as the production model.

---

## Model Evaluation

Each model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

The best model is selected and stored as the **latest model** for prediction.

---

## Running the Project

### 1 Generate the Dataset

```
python data/generate_data.py
```

This script generates synthetic data and stores it in the SQL database.

---

### 2 Train the Models

```
python models/train_model.py
```

This script:

* Loads data from the SQL database
* Trains three machine learning models
* Compares their performance
* Saves the best model

---

### 3 Run the Dashboard

```
streamlit run app/dashboard.py
```

This launches the Streamlit dashboard where users can input data and receive predictions.

---

### 4 Retrain the Model

```
python retraining/retrain_model.py
```

This simulates a model lifecycle by retraining models with updated data and saving a new version.

---

## Dashboard Features

The Streamlit dashboard provides:

* Input interface for prediction
* Gym willingness prediction output
* Probability score
* Dataset preview
* Dataset statistics

---

## Model Lifecycle

The system supports continuous model improvement:

1. New data can be generated
2. Data is stored in the SQL database
3. Models are retrained using updated data
4. New model versions are saved
5. The dashboard always loads the latest trained model

---

## Technologies Used

* Python
* Scikit-learn
* SQLite
* Pandas
* Streamlit
* Joblib
* Git

---

## Final Outcome

This project demonstrates a complete machine learning pipeline including:

* Data generation
* Database integration
* Model training and evaluation
* Model deployment
* Interactive dashboard
* Model lifecycle management

The system shows how machine learning models can be engineered into practical applications rather than remaining experimental code.
