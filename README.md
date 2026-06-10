# Loan Approval Prediction

Aplikasi prediksi persetujuan pinjaman berbasis Streamlit. Aplikasi ini menggunakan model Random Forest yang sudah dilatih dan scaler untuk melakukan preprocessing fitur input sebelum prediksi.

## Fitur

- Input data pemohon pinjaman melalui antarmuka web.
- Prediksi status pinjaman: disetujui atau ditolak.
- Menampilkan probabilitas hasil prediksi.
- Menggunakan model `model_rf.pkl` dan scaler `scaler.pkl`.

## Struktur File Utama

```text
.
├── app.py
├── requirements.txt
├── model_rf.pkl
├── scaler.pkl
└── code_rf.ipynb
```

Keterangan:

- `app.py`: file utama aplikasi Streamlit.
- `requirements.txt`: daftar library yang dibutuhkan.
- `model_rf.pkl`: model Random Forest untuk prediksi.
- `scaler.pkl`: scaler untuk preprocessing data input.
- `code_rf.ipynb`: notebook training dan evaluasi model.

## Cara Menjalankan Lokal

1. Clone repository:

```bash
git clone https://github.com/ZahranTnc/loan-approval-prediction.git
cd loan-approval-prediction
```

2. Install dependency:

```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi:

```bash
streamlit run app.py
```

4. Buka URL yang muncul di terminal, biasanya:

```text
http://localhost:8501
```

## Deploy ke Streamlit Cloud

1. Buka Streamlit Cloud.
2. Pilih repository `ZahranTnc/loan-approval-prediction`.
3. Pilih branch `main`.
4. Isi main file path dengan:

```text
app.py
```

5. Klik deploy.

## Dependency

Project ini menggunakan:

- Streamlit
- NumPy
- scikit-learn
- Joblib

Semua dependency tersedia di `requirements.txt`.

## Catatan

Pastikan file `model_rf.pkl` dan `scaler.pkl` berada di folder yang sama dengan `app.py`, karena aplikasi memuat kedua file tersebut secara langsung.
