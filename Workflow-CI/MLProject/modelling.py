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
import argparse
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

def main():
    # ==========================================
    # 0. SETUP ARGPARSE (MENERIMA PARAMETER DARI MLPROJECT)
    # ==========================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="heart_preprocessing.csv")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    # ==========================================
    # 1. SETUP MLFLOW
    # ==========================================
    # Untuk CI/CD, kita biarkan tracking uri default (bisa diatur lewat env di github actions)
    mlflow.set_experiment("Heart-failure-modelling-CI")

    # ==========================================
    # 2. DATA LOADING & SPLITTING
    # ==========================================
    # Load data dari argumen --data_path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(current_dir, args.data_path)
    df = pd.read_csv(data_file_path)
    
    # Split data (Target = HeartDisease)
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================
    # 3. TRAINING & MLFLOW TRACKING
    # ==========================================
    with mlflow.start_run(run_name="RandomForest_CI_Pipeline"):
        # Training model MENGGUNAKAN ARGUMEN
        model = RandomForestClassifier(
            n_estimators=args.n_estimators, 
            max_depth=args.max_depth, 
            random_state=42
        )
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)

        # Hitung metrik evaluasi
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary')
        rec  = recall_score(y_test, y_pred, average='binary')
        f1   = f1_score(y_test, y_pred, average='binary')

        # Log parameter dari argparse
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("data_path", args.data_path)
        mlflow.log_param("random_state", 42)

        # Log metrik
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall",    rec)
        mlflow.log_metric("f1_score",  f1)

        # Simpan Model
        mlflow.sklearn.log_model(model, "model")

        # ==========================================
        # 4. LOG ARTEFAK (VISUALISASI)
        # ==========================================
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal (0)', 'Heart Disease (1)'],
            yticklabels=['Normal (0)', 'Heart Disease (1)']
        )
        plt.title('Confusion Matrix')
        plt.tight_layout()
        cm_path = os.path.join(current_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)

        importances = model.feature_importances_
        feature_names = X_train.columns
        feat_importances = pd.Series(importances, index=feature_names)
        plt.figure(figsize=(10, 6))
        feat_importances.nlargest(10).plot(kind='barh', color='#3498db')
        plt.title('Top 10 Feature Importances')
        plt.tight_layout()
        fi_path = os.path.join(current_dir, 'feature_importance.png')
        plt.savefig(fi_path)
        mlflow.log_artifact(fi_path)

        # Hapus file gambar setelah dikirim ke MLflow
        if os.path.exists(cm_path): os.remove(cm_path)
        if os.path.exists(fi_path): os.remove(fi_path)

if __name__ == "__main__":
    main()