from pathlib import Path

import joblib
import numpy as np
import streamlit as st

@st.cache_resource
def get_artifacts():
    base_path = Path(__file__).resolve().parent
    return (
        joblib.load(base_path / 'model_rf.pkl'),
        joblib.load(base_path / 'scaler.pkl'),
    )


model, scaler = get_artifacts()

tipe_produk_map = {'Kartu Kredit': 0, 'Kredit Berjalan': 1, 'Pinjaman Pribadi': 2}
tujuan_pinjaman_map = {'Bisnis': 0, 'Konsolidasi Hutang': 1, 'Medis': 2, 'Pendidikan': 3, 'Pribadi': 4, 'Renovasi Rumah': 5}

st.title('Loan Approval Prediction')

col1, col2 = st.columns(2)

with col1:
    usia = st.number_input('Usia', min_value=18, max_value=70, value=30)
    lama_bekerja_tahun = st.number_input('Lama Bekerja (tahun)', min_value=0, max_value=40, value=5, step=1)
    pendapatan_tahunan = st.number_input('Pendapatan Tahunan (Rp)', min_value=15000, max_value=250000, value=50000)
    skor_kredit = st.number_input('Skor Kredit', min_value=348, max_value=850, value=650)
    lama_riwayat_kredit_tahun = st.number_input('Lama Riwayat Kredit (tahun)', min_value=0, max_value=30, value=5, step=1)
    aset_tabungan = st.number_input('Aset Tabungan (Rp)', min_value=0, max_value=300000, value=1000)
    hutang_saat_ini = st.number_input('Hutang Saat Ini (Rp)', min_value=0, max_value=200000, value=10000)

with col2:
    gagal_bayar_tercatat = st.selectbox('Gagal Bayar Tercatat', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
    tunggakan_2thn_terakhir = st.number_input('Tunggakan 2 Tahun Terakhir', min_value=0, max_value=9, value=0)
    catatan_negatif = st.number_input('Catatan Negatif', min_value=0, max_value=4, value=0)
    tipe_produk = st.selectbox('Tipe Produk', list(tipe_produk_map.keys()))
    tujuan_pinjaman = st.selectbox('Tujuan Pinjaman', list(tujuan_pinjaman_map.keys()))
    suku_bunga = st.number_input('Suku Bunga (%)', min_value=6.0, max_value=23.0, value=15.0, step=0.1)
    rasio_hutang = st.number_input('Rasio Hutang thd Pendapatan', min_value=0.0, max_value=0.8, value=0.3, step=0.01)
    rasio_pinjaman = st.number_input('Rasio Pinjaman thd Pendapatan', min_value=0.0, max_value=2.0, value=0.5, step=0.01)
    rasio_pembayaran = st.number_input('Rasio Pembayaran thd Pendapatan', min_value=0.0, max_value=0.7, value=0.2, step=0.01)

if st.button('Prediksi'):
    features = np.array([[
        usia,
        lama_bekerja_tahun,
        pendapatan_tahunan,
        skor_kredit,
        lama_riwayat_kredit_tahun,
        aset_tabungan,
        hutang_saat_ini,
        gagal_bayar_tercatat,
        tunggakan_2thn_terakhir,
        catatan_negatif,
        tipe_produk_map[tipe_produk],
        tujuan_pinjaman_map[tujuan_pinjaman],
        suku_bunga,
        rasio_hutang,
        rasio_pinjaman,
        rasio_pembayaran,
    ]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    st.divider()
    if prediction == 1:
        st.success(f'DISETUJUI -- Probabilitas: {probability[1]*100:.1f}%')
    else:
        st.error(f'DITOLAK -- Probabilitas: {probability[0]*100:.1f}%')
