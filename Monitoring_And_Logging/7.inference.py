import requests
import json
import pandas as pd

# ============================
# Konfigurasi
# ============================
SERVING_URL = "http://localhost:5001/invocations"

# ============================
# Contoh data untuk prediksi
# ============================
# Data sudah dalam format preprocessed (sesuai pipeline preprocessing)
sample_data = {
    "dataframe_split": {
        "columns": [
            "Age", "Sex", "RestingBP", "Cholesterol", "FastingBS",
            "MaxHR", "ExerciseAngina", "Oldpeak",
            "ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA",
            "RestingECG_Normal", "RestingECG_ST",
            "ST_Slope_Flat", "ST_Slope_Up"
        ],
        "data": [
            [0.5, 1, -0.3, 0.2, 0, 0.8, 0, -0.5, 0, 0, 1, 1, 0, 0, 1],
            [-1.2, 0, 0.5, -0.8, 1, -0.3, 1, 1.2, 1, 0, 0, 0, 1, 1, 0]
        ]
    }
}

# ============================
# Kirim request
# ============================
try:
    response = requests.post(
        SERVING_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(sample_data)
    )
    
    if response.status_code == 200:
        predictions = response.json()
        print("=" * 50)
        print("HASIL PREDIKSI")
        print("=" * 50)
        print(f"Predictions: {predictions}")
        for i, pred in enumerate(predictions.get('predictions', predictions)):
            label = "Heart Disease" if pred == 1 else "Normal"
            print(f"  Pasien {i+1}: {label} ({pred})")
    else:
        print(f"Error {response.status_code}: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("Error: Model server belum berjalan!")
    print("Jalankan dulu: mlflow models serve -m <model_path> -p 5001 --no-conda")