import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

print("Starting script to generate training_medians.pkl...")

try:
    # --- Konfigurasi Path ---
    BASE_DIR = Path(__file__).parent.resolve()
    DATA_PATH = BASE_DIR / "integrated_data" / "merged_ckd_data.csv"
    MODEL_DIR = BASE_DIR / "models"
    
    # Buat folder models jika belum ada
    MODEL_DIR.mkdir(exist_ok=True)

    # --- Muat Data ---
    print(f"Loading data from {DATA_PATH}...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}. Please ensure the integrated data exists.")
    
    df = pd.read_csv(DATA_PATH)
    print("Data loaded successfully.")

    # --- [PERBAIKAN 1] Bersihkan data dari target yang kosong ---
    print(f"Original data shape: {df.shape}")
    df.dropna(subset=['target'], inplace=True)
    print(f"Data shape after removing rows with missing target: {df.shape}")

    # --- [PERBAIKAN 2] Hapus kelas yang terlalu sedikit anggotanya ---
    print("Checking for rare classes...")
    class_counts = df['target'].value_counts()
    # Tentukan kelas mana yang memiliki anggota kurang dari 2
    classes_to_remove = class_counts[class_counts < 2].index
    
    if not classes_to_remove.empty:
        print(f"Found rare classes with < 2 members: {list(classes_to_remove)}. Removing them.")
        # Buang baris yang termasuk dalam kelas langka tersebut
        df = df[~df['target'].isin(classes_to_remove)]
        print(f"Data shape after removing rare classes: {df.shape}")
    else:
        print("No rare classes found. Data is suitable for stratification.")

    # --- Pisahkan Fitur dan Target ---
    if 'target' not in df.columns:
        raise ValueError("'target' column not found in the dataset.")
    
    X = df.drop('target', axis=1)
    y = df['target']

    # --- Lakukan Train-Test Split (SAMA PERSIS seperti di preprocessing.py) ---
    # Ini penting agar median dihitung HANYA dari data training
    print("Performing train-test split...")
    X_train, _, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Train-test split complete.")

    # --- Hitung Median dari Data Training ---
    print("Calculating medians from the training set...")
    training_medians = X_train.median().to_dict()

    # --- Simpan File Median ---
    median_file_path = MODEL_DIR / "training_medians.pkl"
    print(f"Saving medians to {median_file_path}...")
    joblib.dump(training_medians, median_file_path)

    print("\n✅ SUCCESS! File 'training_medians.pkl' has been created in the 'models' folder.")
    print("You can now push this file to your GitHub repository.")

except Exception as e:
    print(f"\n❌ ERROR: An error occurred: {e}")