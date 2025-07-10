import streamlit as st
import pandas as pd
import pickle

# Load model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load fitur model yang disimpan saat training
with open('model_features.pkl', 'rb') as f:
    model_features = pickle.load(f)

st.set_page_config(page_title="Prediksi Resign atau Bertahan", layout="centered")
st.title("💼 Prediksi Resign Karyawan")

# Input dari user (fitur-fitur utama yang umum digunakan)
age = st.slider("Usia", 18, 60, 30)
monthly_income = st.number_input("Gaji Bulanan", min_value=1000, value=5000, step=500)
job_satisfaction = st.selectbox("Kepuasan Kerja (1=sangat tidak puas, 4=sangat puas)", [1, 2, 3, 4])
overtime = st.selectbox("Lembur?", ["Ya", "Tidak"])
distance_from_home = st.slider("Jarak dari Rumah ke Kantor (km)", 1, 50, 10)
years_at_company = st.slider("Lama Bekerja (Tahun)", 0, 40, 5)
education = st.selectbox("Pendidikan (1=SD, 5=S3)", [1, 2, 3, 4, 5])
gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])

# Encode
overtime_encoded = 1 if overtime == "Ya" else 0
gender_encoded = 1 if gender == "Laki-laki" else 0

# Buat dictionary input
input_raw = {
    'Age': age,
    'MonthlyIncome': monthly_income,
    'JobSatisfaction': job_satisfaction,
    'OverTime': overtime_encoded,
    'DistanceFromHome': distance_from_home,
    'YearsAtCompany': years_at_company,
    'Education': education,
    'Gender': gender_encoded,
}

# Inisialisasi DataFrame dengan kolom model_features
input_data = pd.DataFrame(columns=model_features)

# Isi nilai input ke kolom yang sesuai
for col in input_raw:
    if col in input_data.columns:
        input_data.at[0, col] = input_raw[col]

# Kolom lain yang tidak diisi → set 0
input_data = input_data.fillna(0)

# Tampilkan data yang akan diprediksi
st.subheader("📦 Data yang dikirim ke model:")
st.dataframe(input_data)

# Tombol prediksi
if st.button("Prediksi"):
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0]  # probabilitas bertahan & resign

    st.markdown(f"🟢 **Probabilitas Bertahan:** `{proba[0]*100:.2f}%`")
    st.markdown(f"🔴 **Probabilitas Resign:** `{proba[1]*100:.2f}%`")

    if prediction[0] == 1:
        st.error("⚠️ Karyawan berpotensi **RESIGN**.")
    else:
        st.success("✅ Karyawan cenderung **BERTAHAN**.")
