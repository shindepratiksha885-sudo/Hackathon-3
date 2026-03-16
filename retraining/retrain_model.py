import sqlite3
import pandas as pd
import os
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Data from SQL
# -----------------------------
conn = sqlite3.connect("database/gym_data.db")

df = pd.read_sql("SELECT * FROM gym_willingness", conn)

conn.close()

print("Data loaded for retraining")
print("Dataset size:", df.shape)

# -----------------------------
# Prepare Features
# -----------------------------
X = df.drop("GYM_WILLING", axis=1)
y = df["GYM_WILLING"]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Define Models
# -----------------------------
models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000),
    "Random_Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}

results = {}

# -----------------------------
# Train Models
# -----------------------------
for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    results[name] = acc

    print(name, "accuracy:", acc)

# -----------------------------
# Select Best Model
# -----------------------------
best_model_name = max(results, key=results.get)

best_model = models[best_model_name]

print("Best model after retraining:", best_model_name)

# -----------------------------
# Save Model Version
# -----------------------------
os.makedirs("models/saved_models", exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

model_path = f"models/saved_models/{best_model_name}_{timestamp}.pkl"

joblib.dump(best_model, model_path)

# Update latest model
joblib.dump(best_model, "models/latest_model.pkl")

print("New model saved:", model_path)
print("latest_model.pkl updated")