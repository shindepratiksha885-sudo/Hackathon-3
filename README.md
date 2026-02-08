Student Personality Enhancement Program
ML Model Deployment & Dashboard Project
Project Overview

This project focuses on building and deploying a machine learning–based personality enhancement system for university students. The goal is to understand how academic performance, fitness habits, sports participation, and lifestyle factors contribute to overall personality development.

The project goes beyond model training and emphasizes deployment, usability, and maintainability, in line with the objectives of Hackathon 3: Development of Pipelines and Maintenance of Models.

An interactive dashboard is created to:

Analyze student data

Evaluate trained models

Allow real-time personality predictions through user input

Problem Statement

University students often struggle to balance academics, fitness, sports, and mental well-being. While universities provide facilities such as gyms and sports infrastructure, students lack a structured way to understand how these factors influence their overall personality development.

This project aims to design a data-driven system that predicts a student’s personality development level based on academic, fitness, and lifestyle indicators, helping students make informed decisions to improve their overall well-being.

📊 Dataset Description

The dataset used in this project represents student-level information, either synthetically generated or curated to reflect a realistic university environment.

Key Features Include:

Academic Factors: Attendance percentage, GPA, study hours

Fitness & Sports: Gym visits, workout duration, preferred sport, sports practice hours

Lifestyle Indicators: Sleep hours, stress level, time management score

Target Variable:

OVERALL_PERSONALITY_SCORE (continuous score from 1–10)

A binary personality category is also derived:

Good Personality Development

Needs Improvement

🤖 Models Used

Up to three traditional machine learning models were explored (excluding neural networks, as per instructions):

Linear Regression

Used as a baseline model for predicting personality score

Random Forest Regressor

Used to predict the continuous personality score

Chosen for its ability to capture non-linear relationships

Random Forest Classifier

Used for binary classification (Good vs Needs Improvement)

Evaluated using confusion matrix and classification metrics

Model performance was tested using standard evaluation metrics, and the best-performing models were saved using .pkl files for deployment.

📈 Exploratory Data Analysis (EDA)

EDA was conducted to understand patterns and relationships within the data. Key analyses include:

Correlation heatmap of numerical features

Relationship between fitness level and personality score

Distribution of academic and lifestyle indicators

These analyses helped guide feature selection and model choice.

🖥️ Dashboard Description

A Gradio-based interactive dashboard was developed to demonstrate the deployed system.

Dashboard Features:

Personality Prediction Tab

User inputs student details

Outputs predicted personality score and category

Provides a short, human-readable recommendation

EDA Tab

Visualizes correlations and key data insights

Model Evaluation Tab

Displays a 2×2 confusion matrix

Shows classification report for the Random Forest Classifier

Gradio was chosen for its simplicity, clean UI, and ease of deployment with a public link.

🔮 Prediction Capability

The deployed system allows users to:

Enter new student data

Instantly receive predictions from trained models

Interpret results through both numeric outputs and qualitative feedback

This ensures the model is prediction-ready, as required.

🔄 Model Updation & Maintenance Plan

To ensure long-term relevance:

New student data can be collected each academic semester

Models can be retrained using updated datasets

Performance of new models can be compared with previous versions

Updated model files can replace older versions in the dashboard

All updates are tracked using Git version control.

📁 Repository Structure (Suggested)
student-personality-enhancement/
│
├── notebooks/
│   └── personality_model.ipynb
│
├── dashboard/
│   └── gradio_app.py
│
├── models/
│   ├── rf_regressor.pkl
│   ├── rf_classifier.pkl
│   └── sport_encoder.pkl
│
├── data/
│   └── student_personality_fitness_data.csv
│
└── README.md

🛠️ Technologies Used

Programming Language: Python

Environment: Google Colab

Machine Learning: scikit-learn

Data Handling: pandas, NumPy

Visualization: matplotlib, seaborn

Dashboard & Deployment: Gradio

Version Control: Git & GitHub

🎓 Conclusion

This project demonstrates the complete lifecycle of a machine learning system — from data understanding and model training to deployment and maintenance planning. By combining academics, fitness, and lifestyle data, the system provides meaningful insights into student personality development and highlights the importance of a balanced university life.
