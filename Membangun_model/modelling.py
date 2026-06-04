import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
import mlflow
import mlflow.sklearn
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

def main():
    # ==========================================
    # 1. SETUP MLFLOW
    # ==========================================
    # Set tracking URI menggunakan folder mlruns lokal
    tracking_uri = "file:" + os.path.join(os.getcwd(), "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Eksperimen_SML_Ahmad_Fauzul_Adhim")

    # ==========================================
    # 2. DATA LOADING & SPLITTING
    # ==========================================
    # Load data hasil preprocessing
    data_path = os.path.join(os.path.dirname(__file__), 'heart_preprocessing.csv')
    df = pd.read_csv(data_path)
    
    # Split data (Target = HeartDisease)
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Ukuran X_train : {X_train.shape}")
    print(f"Ukuran X_test  : {X_test.shape}")
    print(f"\nDistribusi y_train:\n{y_train.value_counts().to_string()}")
    print(f"\nDistribusi y_test:\n{y_test.value_counts().to_string()}")

    # ==========================================
    # 3. TRAINING & MLFLOW TRACKING
    # ==========================================
    with mlflow.start_run(run_name="RandomForest_ManualLog"):
        # Training model
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)

        # Hitung metrik evaluasi
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary')
        rec  = recall_score(y_test, y_pred, average='binary')
        f1   = f1_score(y_test, y_pred, average='binary')

        # Log parameter manual
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("random_state", 42)

        # Log metrik manual ke MLflow
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall",    rec)
        mlflow.log_metric("f1_score",  f1)

        # Simpan Model Manual
        mlflow.sklearn.log_model(model, "model")

        # ==========================================
        # 4. LOG ARTEFAK (VISUALISASI)
        # ==========================================
        
        # Artefak 1: Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal (0)', 'Heart Disease (1)'],
            yticklabels=['Normal (0)', 'Heart Disease (1)']
        )
        plt.title('Confusion Matrix — Heart Failure')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')

        # Artefak 2: Feature Importance
        importances = model.feature_importances_
        feature_names = X_train.columns  # Mengambil nama fitur otomatis dari dataframe
        
        feat_importances = pd.Series(importances, index=feature_names)
        plt.figure(figsize=(10, 6))
        feat_importances.nlargest(10).plot(kind='barh', color='#3498db')
        plt.title('Top 10 Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        mlflow.log_artifact('feature_importance.png')

        # Hapus file gambar lokal setelah dikirim ke MLflow
        os.remove('confusion_matrix.png')
        os.remove('feature_importance.png')

        # ==========================================
        # 5. RINGKASAN HASIL TERMINAL
        # ==========================================
        print("\n" + "="*45)
        print("   HASIL EVALUASI — Random Forest")
        print("="*45)
        print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print("="*45)

    print("\n✅ Training Selesai!")
    print("👉 Cek MLflow UI dengan menjalankan 'mlflow ui' di terminal")

if __name__ == "__main__":
    main()