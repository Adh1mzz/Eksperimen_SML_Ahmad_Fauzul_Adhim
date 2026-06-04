# test_predict.py
import requests
import json

# Menembak ke Prometheus Exporter di port 5002
url = "http://127.0.0.1:5002/predict"

data = {
    "dataframe_split": {
        "columns": [
            "Age", "Sex", "RestingBP", "Cholesterol", "FastingBS",
            "MaxHR", "ExerciseAngina", "Oldpeak",
            "ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA",
            "RestingECG_Normal", "RestingECG_ST",
            "ST_Slope_Flat", "ST_Slope_Up"
        ],
        "data": [
            [0.5, 1, -0.3, 0.2, 0, 0.8, 0, -0.5, 0, 0, 1, 1, 0, 0, 1]
        ]
    }
}

try:
    response = requests.post(url, json=data)
    print(f"Status: {response.status_code}")
    print(f"Result: {response.json()}")
except requests.exceptions.ConnectionError:
    print("ERROR: Exporter di port 5002 belum menyala. Cek Terminal 2!")