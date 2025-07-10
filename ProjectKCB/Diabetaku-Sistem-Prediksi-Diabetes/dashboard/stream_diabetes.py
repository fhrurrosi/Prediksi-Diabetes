import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load model dan dataset
diabetes_model = joblib.load("../model/diabetes_model.sav")
df = pd.read_csv('diabets_dataset_clean.csv')
X = df.drop(columns='diabetes', axis=1)

# Scaling
scaler = StandardScaler()
scaler.fit(X)

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="DiabetaKu",
    layout="wide",
    page_icon="asset/diabetes_icon.png"
)

st.title('Aplikasi Prediksi Diabetes')

col1, col2 = st.columns(2)

with col1:
    Gender = st.text_input('Jenis Kelamin (0 : Perempuan, 1 : Laki-laki)')
    Age = st.text_input("Usia Anda")
    Hipertension = st.text_input('Apakah Anda memiliki tekanan darah tinggi? (1 : Ya, 0 : Tidak)')
    Heart_disease = st.text_input('Apakah Anda memiliki penyakit jantung? (1 : Ya, 0 : Tidak)')

with col2:
    Smoking_history = st.text_input("Riwayat Merokok (-1 : Tidak Diketahui, 0 : Tidak Pernah, 1 : Pernah, 2 : Saat Ini, 3 : Dulu Tapi Tidak Sekarang, 4 : Pernah Merokok)")
    bmi = st.text_input('Indeks Massa Tubuh (BMI) Anda')
    HbA1c_level = st.text_input("Level Hemoglobin A1c")
    Blood_glucose = st.text_input('Kadar Glukosa Darah')

# Proses input
user_inputs = [Gender, Age, Hipertension, Heart_disease, Smoking_history, bmi, HbA1c_level, Blood_glucose]
input_data = pd.to_numeric(user_inputs, errors='coerce')

diabetes_diagnosis = "BELUM MELENGKAPI DATA"

if st.button('Prediksi'):
    if np.isnan(input_data).any():
        st.warning("❗ Input tidak lengkap atau salah. Mohon isi semua kolom dengan angka yang valid.")
    else:
        input_array = np.array(input_data).reshape(1, -1)
        std_data = scaler.transform(input_array)
        prediction = diabetes_model.predict(std_data)

        if prediction[0] == 0:
            diabetes_diagnosis = "✅ PASIEN TIDAK TERKENA DIABETES"
        else:
            diabetes_diagnosis = "⚠️ PASIEN TERKENA DIABETES"

st.subheader("Hasil Prediksi:")
st.success(diabetes_diagnosis)
