import streamlit as st
import joblib

# Load model dan vectorizer
model = joblib.load('./src/sentiment_model.pkl')
vectorizer = joblib.load('./src/tfidf_vectorizer.pkl')

# Judul aplikasi
st.title("Sentiment Analysis - Olist Reviews")

st.markdown("""
            Aplikasi ini memprediksi **sentimen** dari teks ulasan pelanggan
            """)

# Input dari user
user_input = st.text_area("Masukkan ulasan pelanggan (Bahasa Portugis):", height=150)

if st.button("Prediksi"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        X_input = vectorizer.transform([user_input])
        prediction = model.predict(X_input)[0]
        label = "Positif :)" if prediction == 1 else "negatif :("
        st.success(f"Hasil Prediksi: **{label}**")