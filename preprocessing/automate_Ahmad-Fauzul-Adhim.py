# automate_Ahmad-Fauzul-Adhim.py
# File ini melakukan preprocessing data Heart Failure secara otomatis
# Jalankan: python automate_Ahmad-Fauzul-Adhim.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"✅ Data loaded: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df

def handle_invalid_data(df):
    df = df.copy()
    
    # Handle Cholesterol = 0 (impute dengan median)
    cholesterol_zero = (df['Cholesterol'] == 0).sum()
    if cholesterol_zero > 0:
        median_val = df[df['Cholesterol'] > 0]['Cholesterol'].median()
        df.loc[df['Cholesterol'] == 0, 'Cholesterol'] = median_val
        print(f"✅ Setelah impute Cholesterol=0: {cholesterol_zero} nilai diganti")
    
    # Handle RestingBP = 0 (impute dengan median)
    bp_zero = (df['RestingBP'] == 0).sum()
    if bp_zero > 0:
        median_val = df[df['RestingBP'] > 0]['RestingBP'].median()
        df.loc[df['RestingBP'] == 0, 'RestingBP'] = median_val
        print(f"✅ Setelah impute RestingBP=0: {bp_zero} nilai diganti")
    
    # Handle duplikat
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"✅ Setelah drop duplikat: {duplicates} baris dihapus")
    
    return df

def handle_outliers(df):
    df = df.copy()
    
    # IQR method untuk outlier capping
    columns = ['RestingBP', 'Cholesterol', 'Oldpeak', 'MaxHR']
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers_count = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"✅ Outlier {col} di-cap: {outliers_count} outlier ditemukan")
    
    return df

def encode_categorical_features(df):
    df = df.copy()
    
    # Binary Encoding: Sex
    le_sex = LabelEncoder()
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    print(f"✅ Encoding Sex: M→0, F→1")
    
    # Binary Encoding: ExerciseAngina
    le_angina = LabelEncoder()
    df['ExerciseAngina'] = le_angina.fit_transform(df['ExerciseAngina'])
    print(f"✅ Encoding ExerciseAngina: N→0, Y→1")
    
    # One-Hot Encoding: ChestPainType, RestingECG, ST_Slope
    df = pd.get_dummies(
        df, 
        columns=['ChestPainType', 'RestingECG', 'ST_Slope'],
        drop_first=True,
        dtype=int
    )
    print(f"✅ One-Hot Encoding selesai: {df.shape[1]} kolom total")
    
    return df

def normalize_features(df):
    df = df.copy()
    
    # StandardScaler untuk fitur numerik
    scaler = StandardScaler()
    num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print(f"✅ Normalisasi fitur numerik selesai: {num_cols}")
    
    return df

def save_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset disimpan ke: {output_path}")
    print(f"✅ Shape final: {df.shape}")

def preprocess(input_path, output_path):
    print("="*60)
    print("🚀 PREPROCESSING OTOMATIS - Heart Failure Prediction")
    print("="*60)
    df = load_data(input_path)
    df = handle_invalid_data(df)
    df = handle_outliers(df)
    df = encode_categorical_features(df)
    df = normalize_features(df)
    save_data(df, output_path)
    print("="*60)
    print("✅ PREPROCESSING SELESAI!")
    print("="*60)
    return df

if __name__ == "__main__":
    preprocess(
        input_path='data.csv',
        output_path='heartfailure_preprocessing.csv'
    )