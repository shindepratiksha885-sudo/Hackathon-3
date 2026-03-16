import numpy as np
import pandas as pd
import random
import sqlite3
import os

np.random.seed(42)
random.seed(42)

NUM_RECORDS = 2000

data = []

for _ in range(NUM_RECORDS):

    study_hours = round(np.random.uniform(1, 8), 2)
    exam_soon = np.random.choice([0,1], p=[0.7,0.3])
    sleep_hours = round(np.random.uniform(4,9),2)
    stress_level = np.random.randint(1,11)
    energy_level = np.random.randint(1,11)
    past_week_gym = np.random.randint(0,8)

    probability = 0.5

    probability += energy_level * 0.03
    probability += past_week_gym * 0.05

    probability -= stress_level * 0.03
    probability -= exam_soon * 0.2

    if sleep_hours < 5:
        probability -= 0.1

    probability = max(0, min(1, probability))

    gym_willing = 1 if random.random() < probability else 0

    data.append([
        study_hours,
        exam_soon,
        sleep_hours,
        stress_level,
        energy_level,
        past_week_gym,
        gym_willing
    ])

columns = [
    "STUDY_HOURS",
    "EXAM_SOON",
    "SLEEP_HOURS",
    "STRESS_LEVEL",
    "ENERGY_LEVEL",
    "PAST_WEEK_GYM_VISITS",
    "GYM_WILLING"
]

df = pd.DataFrame(data, columns=columns)

os.makedirs("database", exist_ok=True)

conn = sqlite3.connect("database/gym_data.db")

df.to_sql("gym_willingness", conn, if_exists="replace", index=False)

conn.close()

print("Dataset generated and stored in SQL database")