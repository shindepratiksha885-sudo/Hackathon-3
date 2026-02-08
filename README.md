# 🎯 Student Personality Enhancement Program  
### Machine Learning Model Deployment & Dashboard

---

## 📌 Project Overview

This project focuses on building and deploying a **machine learning–based personality enhancement system for university students**. The objective is to analyze how **academic performance, fitness habits, sports participation, and lifestyle factors** contribute to overall personality development.

The project aligns with **Hackathon 3: Development of Pipelines and Maintenance of Models**, emphasizing not only model training but also **deployment, usability, and long-term maintainability**.

An interactive dashboard is developed to:
- Analyze student data
- Evaluate trained machine learning models
- Enable real-time personality predictions through user input

---

## 🧠 Problem Statement

University students often find it difficult to maintain a healthy balance between academics, physical fitness, sports activities, and mental well-being. Although universities provide facilities such as gyms and sports infrastructure, students lack a structured and data-driven way to understand how these factors influence their personality development.

This project aims to design a **predictive system** that estimates a student’s personality development level based on academic, fitness, and lifestyle indicators, helping students make informed decisions to improve their overall well-being.

---

## 📊 Dataset Description

The dataset represents **student-level data** relevant to a university environment. The data is either synthetically generated or curated to simulate realistic student behavior.

### 🔹 Key Features
- **Academic Factors**
  - Attendance Percentage  
  - Current GPA  
  - Study Hours per Day  

- **Fitness & Sports Factors**
  - Gym Visits per Week  
  - Workout Duration  
  - Preferred Sport  
  - Sports Practice Hours per Week  

- **Lifestyle Indicators**
  - Sleep Hours  
  - Stress Level  
  - Time Management Score  

### 🎯 Target Variable
- `OVERALL_PERSONALITY_SCORE` (continuous score between 1–10)

A binary category is derived from the score:
- **Good Personality Development**
- **Needs Improvement**

---

## 🤖 Models Used

As per the project guidelines (excluding neural networks), up to **three traditional machine learning models** were explored:

### 1️⃣ Linear Regression
- Used as a baseline model
- Predicts the overall personality score

### 2️⃣ Random Forest Regressor
- Predicts the continuous personality score
- Captures non-linear relationships between features

### 3️⃣ Random Forest Classifier
- Classifies students into:
  - Good Personality Development
  - Needs Improvement
- Evaluated using classification metrics and confusion matrix

Trained models were serialized using `.pkl` files for deployment.

---

## 📈 Exploratory Data Analysis (EDA)

EDA was conducted to understand patterns and relationships in the dataset. Key analyses include:
- Correlation heatmap of numerical features
- Relationship between fitness level and personality score
- Distribution of academic and lifestyle indicators

These insights guided feature selection and model choice.

---

## 🖥️ Dashboard Description

A **Gradio-based interactive dashboard** was developed to demonstrate the deployed system.

### 🔹 Dashboard Features

#### 🔮 Personality Prediction
- Accepts student input through an interactive form
- Predicts:
  - Personality score (regression)
  - Personality category (classification)
- Provides short, human-readable recommendations

#### 📊 Exploratory Data Analysis
- Displays correlation heatmaps
- Shows key data insights related to personality development

#### 📉 Model Evaluation
- 2×2 Confusion Matrix for the Random Forest Classifier
- Classification report with precision, recall, and F1-score

Gradio was chosen for its clean UI, ease of use, and ability to provide a public deployment link.

---

## 🔮 Prediction Capability

The deployed system is **prediction-ready**, allowing users to:
- Input new student data
- Receive instant predictions from trained models
- Interpret results using both numerical outputs and qualitative feedback

---

## 🔄 Model Updation & Maintenance Plan

To ensure long-term relevance and accuracy:

- New student data can be collected **at the end of each academic semester**
- Models can be retrained using updated datasets
- Performance of new models can be compared with previous versions
- Updated models can replace older versions in the dashboard

All updates and changes are tracked using **Git version control**.

---

## 🛠️ Technologies Used

- **Programming Language**: Python  
- **Development Environment**: Google Colab  
- **Machine Learning**: scikit-learn  
- **Data Handling**: pandas, NumPy  
- **Visualization**: matplotlib, seaborn  
- **Dashboard & Deployment**: Gradio  
- **Version Control**: Git & GitHub  

---

## 🎓 Conclusion

This project demonstrates the complete lifecycle of a machine learning system — from data understanding and model training to deployment and maintenance planning. By integrating academic, fitness, and lifestyle factors, the system provides meaningful insights into student personality development and highlights the importance of a balanced university life.

---
