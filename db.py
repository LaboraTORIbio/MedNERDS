import sqlite3
import json
from datetime import datetime

DB_PATH = "mednerds.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            age INTEGER,
            sex TEXT,
            history TEXT,
            symptoms TEXT,
            medication TEXT,
            created_at DATETIME
        )
    """)

    conn.commit()
    conn.close()

def save_record(patient_id, final_df):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    row = final_df.iloc[0]  # solo hay una fila

    c.execute("""
        INSERT INTO records (
            patient_id, age, sex, history, symptoms, medication, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_id,
        row["Age"],
        row["Sex"],
        json.dumps(row["History"]),
        json.dumps(row["Symptoms"]),
        json.dumps(row["Medication"]),
        datetime.now()
    ))

    conn.commit()
    conn.close()