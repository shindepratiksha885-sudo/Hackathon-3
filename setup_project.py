import os

# Project folders
folders = [
    "data",
    "database",
    "models",
    "models/saved_models",
    "app",
    "retraining"
]

# Project files
files = [
    "data/generate_data.py",
    "models/train_model.py",
    "models/latest_model.pkl",
    "app/dashboard.py",
    "retraining/retrain_model.py",
    "requirements.txt",
    "README.md"
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for file in files:
    if not os.path.exists(file):
        open(file, "w").close()

print("Project folder structure created successfully!")