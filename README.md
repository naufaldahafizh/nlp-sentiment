# NLP Sentiment Analysis - Olist Review

Proyek ini melakukan analisis sentimen pada data ulasan pelanggan dari platform e-commerce Olist menggunakan model machine learning. Hasil akhir berupa dashboard interaktif berbasis Streamlit untuk prediksi sentimen dari teks input.

---

## Dataset

Dataset utama: `olist_order_reviews_dataset.csv`  
Dataset tersedia di [Kaggle: Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

Letakkan file CSV asli di:
`data/olist_order_reviews_dataset.csv`


## Tujuan

> *Predict behavior to retain customers.*  
Menggunakan teks ulasan pelanggan untuk mengidentifikasi apakah review bersifat **positif** atau **negatif**, sebagai dasar program loyalitas dan peningkatan layanan.

## Tahapan Proyek

`notebooks/sentiment_analysis.ipynb`
### 1. EDA & Preprocessing 
- Labeling sentimen (positive/negative)
- Cleaning teks ulasan (stopwords, huruf kecil)
- Visualisasi distribusi sentimen
- TF-IDF vectorization

### 2. Modeling
- Logistic Regression dan Random Forest
- Evaluasi dengan classification report
- Save model dan vectorizer dengan `joblib`

`dashboard/app.py`
### 3. Dashboard Interaktif 
- Input teks ulasan oleh user
- Output prediksi sentimen dengan model yang dilatih
- Dibangun dengan **Streamlit**

---

## Tools

- Python 3.10+
- Pandas, Scikit-Learn, NLTK
- Streamlit
- Jupyter Notebook

---

## Menjalankan Streamlit App

```bash
streamlit run dashboard/app.py
```

## Insight
- Sentimen pelanggan dapat diidentifikasi cukup akurat hanya dari teks ulasan.
- Model sederhana seperti Logistic Regression cukup efektif untuk baseline.
- Proyek ini bisa dikembangkan lebih lanjut ke multi-label classification atau transformer-based models.