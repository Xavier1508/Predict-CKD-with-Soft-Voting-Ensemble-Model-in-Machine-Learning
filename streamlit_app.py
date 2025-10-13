import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# --- KONFIGURASI DAN PEMUATAN MODEL ---
st.set_page_config(page_title="Prediksi Penyakit Ginjal Kronis", layout="wide")

# Fungsi untuk memuat model dengan caching agar lebih cepat
@st.cache_resource
def load_model_artifacts():
    try:
        base_dir = Path(__file__).parent.resolve()
        model_dir = base_dir / "models"
        
        model = joblib.load(model_dir / 'stackingensemble.pkl')
        feature_names = joblib.load(model_dir / 'feature_names.pkl')
        scaler = joblib.load(model_dir / 'scaler.pkl')
        training_medians = joblib.load(model_dir / 'training_medians.pkl')
        
        return model, feature_names, scaler, training_medians
    except FileNotFoundError as e:
        st.error(f"Error: Salah satu file model tidak ditemukan di folder 'models'. Pastikan semua file .pkl ada. Detail: {e}")
        return None, None, None, None

model, feature_names, scaler, training_medians = load_model_artifacts()

# --- FUNGSI BANTUAN DARI FLASK APP (diadaptasi) ---
def calculate_bmi(weight, height):
    if height > 0 and weight > 0:
        return round(weight / ((height / 100) ** 2), 1)
    return None

def calculate_bun_creatinine_ratio(bu, sc):
    if sc > 0 and bu > 0:
        bun = bu * 0.467  # Konversi Blood Urea ke BUN
        return round(bun / sc, 2)
    return None

def preprocess_input(data, feature_names, scaler, training_medians):
    """
    Fungsi ini mengambil input dari user, memprosesnya, dan menyiapkannya untuk model.
    Logikanya diambil langsung dari app.py Anda.
    """
    input_df = pd.DataFrame(columns=feature_names)
    input_df.loc[0] = np.nan # Inisialisasi dengan NaN

    # Isi nilai yang diberikan
    for feature, value in data.items():
        if feature in feature_names:
            input_df.at[0, feature] = float(value)

    # Hitung fitur turunan
    if 'bun_to_creatinine_ratio' in feature_names:
        ratio = calculate_bun_creatinine_ratio(data.get('bu', 0), data.get('sc', 0))
        input_df.at[0, 'bun_to_creatinine_ratio'] = ratio

    # Isi nilai yang kosong dengan median dari data training
    for col in input_df.columns:
        if pd.isna(input_df.loc[0, col]):
            input_df.at[0, col] = float(training_medians.get(col, 0))

    # Scaling data
    try:
        scaled_values = scaler.transform(input_df)
        return scaled_values
    except Exception as e:
        st.error(f"Terjadi kesalahan saat scaling data: {e}")
        return None


# --- ANTARMUKA PENGGUNA (UI) STREAMLIT ---

st.title("🩺 Prediksi Dini Penyakit Ginjal Kronis (CKD)")
st.markdown("Masukkan data pasien di bawah ini untuk mendapatkan prediksi berbasis model *Stacking Ensemble*.")

if model is None:
    st.stop() # Hentikan aplikasi jika model gagal dimuat

# Gunakan kolom agar lebih rapi
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Data Demografi")
    age = st.number_input("Usia (Tahun)", min_value=1, max_value=120, value=50, step=1)
    htn = st.selectbox("Riwayat Hipertensi?", (0, 1), format_func=lambda x: "Ya" if x == 1 else "Tidak")
    dm = st.selectbox("Riwayat Diabetes Mellitus?", (0, 1), format_func=lambda x: "Ya" if x == 1 else "Tidak")
    appetite = st.selectbox("Nafsu Makan?", (0, 1), format_func=lambda x: "Baik" if x == 0 else "Buruk")
    
    st.subheader("Pengukuran Fisik (Opsional)")
    weight = st.number_input("Berat Badan (kg)", min_value=10.0, max_value=200.0, value=70.0, step=0.5)
    height = st.number_input("Tinggi Badan (cm)", min_value=50.0, max_value=250.0, value=165.0, step=0.5)
    
with col2:
    st.subheader("Hasil Tes Darah")
    bp = st.number_input("Tekanan Darah (mm/Hg)", min_value=40.0, max_value=250.0, value=80.0, step=1.0)
    bgr = st.number_input("Glukosa Darah Acak (mg/dL)", min_value=30.0, max_value=600.0, value=120.0, step=1.0)
    bu = st.number_input("Blood Urea (mg/dL)", min_value=2.0, max_value=400.0, value=40.0, step=1.0)
    sc = st.number_input("Serum Creatinine (mg/dL)", min_value=0.1, max_value=20.0, value=1.2, step=0.1)
    sod = st.number_input("Sodium (mEq/L)", min_value=110.0, max_value=180.0, value=138.0, step=1.0)
    pot = st.number_input("Potassium (mEq/L)", min_value=2.0, max_value=9.0, value=4.5, step=0.1)
    hemo = st.number_input("Hemoglobin (g/dL)", min_value=2.0, max_value=25.0, value=14.0, step=0.1)
    pcv = st.number_input("Packed Cell Volume (%)", min_value=15.0, max_value=65.0, value=42.0, step=1.0)
    wc = st.number_input("White Blood Cell Count (cells/cumm)", min_value=1500.0, max_value=25000.0, value=7500.0, step=100.0)
    rc = st.number_input("Red Blood Cell Count (millions/cmm)", min_value=1.5, max_value=9.0, value=4.8, step=0.1)
    
with col3:
    st.subheader("Hasil Tes Urin")
    sg = st.selectbox("Specific Gravity", (1.005, 1.010, 1.015, 1.020, 1.025), index=2)
    al = st.selectbox("Albumin (skala 0-5)", (0, 1, 2, 3, 4, 5))
    su = st.selectbox("Gula (skala 0-5)", (0, 1, 2, 3, 4, 5))
    rbc = st.selectbox("Sel Darah Merah di Urin?", (0, 1), format_func=lambda x: "Normal" if x == 0 else "Abnormal")
    pc = st.selectbox("Pus Cell di Urin?", (0, 1), format_func=lambda x: "Normal" if x == 0 else "Abnormal")
    pcc = st.selectbox("Pus Cell Clumps?", (0, 1), format_func=lambda x: "Tidak Ada" if x == 0 else "Ada")
    ba = st.selectbox("Bakteri di Urin?", (0, 1), format_func=lambda x: "Tidak Ada" if x == 0 else "Ada")


# Tombol Prediksi dan Hasil
if st.button("Buat Prediksi", type="primary"):
    # Kumpulkan semua data input ke dalam dictionary
    user_data = {
        'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su, 'rbc': rbc, 'pc': pc,
        'pcc': pcc, 'ba': ba, 'bgr': bgr, 'bu': bu, 'sc': sc, 'sod': sod, 'pot': pot,
        'hemo': hemo, 'pcv': pcv, 'wc': wc, 'rc': rc, 'htn': htn, 'dm': dm, 'appetite': appetite
    }
    
    # Hitung BMI jika ada di daftar fitur model
    if 'bmi' in feature_names:
        bmi = calculate_bmi(weight, height)
        # Jika BMI tidak bisa dihitung, gunakan median. Jika bisa, gunakan hasil hitungan.
        user_data['bmi'] = bmi if bmi else training_medians.get('bmi', 25.0)

    # Preprocess data
    processed_data = preprocess_input(user_data, feature_names, scaler, training_medians)
    
    if processed_data is not None:
        # Lakukan prediksi
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1] # Probabilitas kelas 1 (CKD)

        st.subheader("Hasil Prediksi")
        
        if prediction == 1:
            st.error(f"**Risiko Tinggi Terkena Penyakit Ginjal Kronis (CKD)**")
            st.progress(probability)
            st.metric(label="Tingkat Keyakinan Prediksi (Probabilitas CKD)", value=f"{probability*100:.2f}%")
        else:
            st.success(f"**Risiko Rendah Terkena Penyakit Ginjal Kronis (CKD)**")
            st.progress(1 - probability)
            st.metric(label="Tingkat Keyakinan Prediksi (Probabilitas Non-CKD)", value=f"{(1-probability)*100:.2f}%")
        
        # Tampilkan BMI dan Rasio BUN/Creatinine jika dihitung
        with st.expander("Lihat Detail Input yang Diproses"):
            display_data = user_data.copy()
            if 'bmi' in display_data:
                st.write(f"**Body Mass Index (BMI) yang Digunakan:** {display_data['bmi']:.1f}")
            if 'bun_to_creatinine_ratio' in feature_names:
                ratio = calculate_bun_creatinine_ratio(user_data.get('bu'), user_data.get('sc'))
                if ratio:
                    st.write(f"**Rasio BUN/Creatinine yang Dihitung:** {ratio:.2f}")
            st.json(display_data)