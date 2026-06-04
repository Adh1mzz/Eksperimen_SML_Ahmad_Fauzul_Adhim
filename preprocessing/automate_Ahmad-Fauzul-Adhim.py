import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """
    Tahap 1: Load dataset dari file CSV
    
    Args:
        filepath (str): Path ke file CSV
    Returns:
        pd.DataFrame: DataFrame hasil loading
    """
    print("[1/6] Loading data...")
    df = pd.read_csv(filepath)
    print(f"  ✓ Data loaded: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def handle_anomalies(df):
    """
    Tahap 2: Handle anomali pada data (nilai 0 yang tidak valid)
    
    Args:
        df (pd.DataFrame): DataFrame input
    Returns:
        pd.DataFrame: DataFrame tanpa anomali
    """
    print("[2/6] Handling anomalies...")
    df_clean = df.copy()
    
    # Handle Cholesterol = 0
    cholesterol_zero = (df_clean['Cholesterol'] == 0).sum()
    if cholesterol_zero > 0:
        median_val = df_clean[df_clean['Cholesterol'] > 0]['Cholesterol'].median()
        df_clean.loc[df_clean['Cholesterol'] == 0, 'Cholesterol'] = median_val
        print(f"  ✓ {cholesterol_zero} nilai Cholesterol=0 diganti dengan median ({median_val})")
    
    # Handle RestingBP = 0
    bp_zero = (df_clean['RestingBP'] == 0).sum()
    if bp_zero > 0:
        median_val = df_clean[df_clean['RestingBP'] > 0]['RestingBP'].median()
        df_clean.loc[df_clean['RestingBP'] == 0, 'RestingBP'] = median_val
        print(f"  ✓ {bp_zero} nilai RestingBP=0 diganti dengan median ({median_val})")
    
    return df_clean


def handle_outliers(df, columns=None, factor=1.5):
    """
    Tahap 3: Handle outlier menggunakan metode IQR (capping)
    
    Args:
        df (pd.DataFrame): DataFrame input
        columns (list): Kolom yang akan di-cap outliernya
        factor (float): Faktor IQR (default 1.5)
    Returns:
        pd.DataFrame: DataFrame dengan outlier yang sudah di-cap
    """
    print("[3/6] Handling outliers (IQR capping)...")
    df_clean = df.copy()
    
    if columns is None:
        columns = ['RestingBP', 'Cholesterol', 'Oldpeak']
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        
        outliers_count = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
        df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
        print(f"  ✓ {col}: {outliers_count} outlier di-cap (range: {lower:.2f} - {upper:.2f})")
    
    return df_clean


def encode_features(df):
    """
    Tahap 4: Encoding fitur kategorikal
    
    Args:
        df (pd.DataFrame): DataFrame input
    Returns:
        pd.DataFrame: DataFrame dengan fitur yang sudah di-encode
    """
    print("[4/6] Encoding features...")
    df_encoded = df.copy()
    
    # Label Encoding - binary features
    le_sex = LabelEncoder()
    df_encoded['Sex'] = le_sex.fit_transform(df_encoded['Sex'])
    print(f"  ✓ Sex encoded: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")
    
    le_angina = LabelEncoder()
    df_encoded['ExerciseAngina'] = le_angina.fit_transform(df_encoded['ExerciseAngina'])
    print(f"  ✓ ExerciseAngina encoded: {dict(zip(le_angina.classes_, le_angina.transform(le_angina.classes_)))}")
    
    # One-Hot Encoding - multi-category features
    df_encoded = pd.get_dummies(
        df_encoded, 
        columns=['ChestPainType', 'RestingECG', 'ST_Slope'],
        drop_first=True,
        dtype=int
    )
    print(f"  ✓ One-Hot Encoding selesai. Kolom baru: {df_encoded.shape[1]} kolom")
    
    return df_encoded


def scale_features(df, numerical_features=None):
    """
    Tahap 5: Scaling fitur numerik dengan StandardScaler
    
    Args:
        df (pd.DataFrame): DataFrame input
        numerical_features (list): Kolom numerik yang akan di-scale
    Returns:
        pd.DataFrame: DataFrame dengan fitur yang sudah di-scale
    """
    print("[5/6] Scaling features...")
    df_scaled = df.copy()
    
    if numerical_features is None:
        numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    
    scaler = StandardScaler()
    df_scaled[numerical_features] = scaler.fit_transform(df_scaled[numerical_features])
    print(f"  ✓ StandardScaler diterapkan pada: {numerical_features}")
    
    return df_scaled


def split_data(df, target_col='HeartDisease', test_size=0.2, random_state=42):
    """
    Tahap 6: Split data menjadi train dan test set
    
    Args:
        df (pd.DataFrame): DataFrame input
        target_col (str): Nama kolom target
        test_size (float): Proporsi test set
        random_state (int): Random state untuk reproduktivitas
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("[6/6] Splitting data...")
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"  ✓ Train set: {X_train.shape[0]} sampel")
    print(f"  ✓ Test set: {X_test.shape[0]} sampel")
    
    return X_train, X_test, y_train, y_test


def preprocess_pipeline(filepath, output_path=None):
    """
    Pipeline utama: menjalankan seluruh tahapan preprocessing
    
    Args:
        filepath (str): Path ke file CSV mentah
        output_path (str): Path untuk menyimpan hasil preprocessing
    Returns:
        tuple: (df_preprocessed, X_train, X_test, y_train, y_test)
    """
    print("=" * 60)
    print("PREPROCESSING PIPELINE - Heart Failure Prediction")
    print("=" * 60)
    
    # Jalankan pipeline
    df = load_data(filepath)
    df = handle_anomalies(df)
    df = handle_outliers(df)
    df = encode_features(df)
    df = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Simpan hasil
    if output_path is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(output_dir, 'heart_preprocessing.csv')
    
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print(f"✅ PREPROCESSING SELESAI!")
    print(f"📁 Output disimpan di: {output_path}")
    print(f"📊 Shape akhir: {df.shape}")
    print("=" * 60)
    
    return df, X_train, X_test, y_train, y_test


# ============================
# MAIN EXECUTION
# ============================
if __name__ == "__main__":
    # Tentukan path
    raw_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'heart_raw.csv')
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'heart_preprocessing.csv')
    
    # Jalankan pipeline
    df_processed, X_train, X_test, y_train, y_test = preprocess_pipeline(
        filepath=raw_data_path,
        output_path=output_path
    )