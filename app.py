import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from io import BytesIO

st.title("Ekstraksi Garis Tekanan Barograph dengan Pra-pemrosesan Citra")

uploaded_file = st.file_uploader("Upload gambar barogram", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Baca dan decode gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.subheader("1. Gambar Asli")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")

    # 1. Konversi ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Peningkatan kualitas citra: equalize + Gaussian Blur
    equalized = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

    # 3. Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # 4. Thresholding biner
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

    st.subheader("2. Hasil Pra-pemrosesan Citra")
    col1, col2 = st.columns(2)
    with col1:
        st.image(edges, caption="Edge Detection (Canny)", use_column_width=True)
    with col2:
        st.image(thresh, caption="Thresholding (Binary Inverted)", use_column_width=True)

    # Deteksi garis tekanan dari threshold (anggap sebagai area terang)
    ys, xs = np.where(thresh > 0)
    line_profile = {}
    for x, y in zip(xs, ys):
        if x not in line_profile:
            line_profile[x] = []
        line_profile[x].append(y)

    x_vals = sorted(line_profile.keys())
    y_vals = [np.mean(line_profile[x]) for x in x_vals]

    # Kalibrasi manual (dari kamu)
    def y_to_pressure(y):
        y1, p1 = 1015, 945
        y2, p2 = 0, 1052
        return p1 + (y - y1) * (p2 - p1) / (y2 - y1)

    pressures = [y_to_pressure(y) for y in y_vals]

    # Hitung tekanan per jam berdasarkan x (130px per jam dari x=0, mulai jam 08.00)
    start_hour = 8
    x_start = 0
    pixels_per_hour = 130
    data = []

    for i in range(24):
        x_target = x_start + i * pixels_per_hour
        nearest_x_idx = min(range(len(x_vals)), key=lambda j: abs(x_vals[j] - x_target))
        y_at_hour = y_vals[nearest_x_idx]
        pressure_at_hour = y_to_pressure(y_at_hour)
        time_str = (datetime.strptime("08:00", "%H:%M") + timedelta(hours=i)).strftime("%H:%M")
        data.append((time_str, round(pressure_at_hour, 2)))

    # Tampilkan tabel
    df = pd.DataFrame(data, columns=["Waktu", "Tekanan (hPa)"])
    st.subheader("3. Tabel Tekanan per Jam")
    st.dataframe(df)

    # Grafik tekanan
    st.subheader("4. Grafik Tekanan")
    fig, ax = plt.subplots()
    ax.plot(df["Waktu"], df["Tekanan (hPa)"], marker='o', color='purple')
    ax.set_xlabel("Waktu")
    ax.set_ylabel("Tekanan (hPa)")
    ax.set_title("Profil Tekanan Harian")
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Gambar dengan garis hasil deteksi (opsional)
    output_img = img.copy()
    for x, y in zip(x_vals, y_vals):
        cv2.circle(output_img, (int(x), int(y)), 1, (0, 255, 0), -1)

        # Tombol download CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Unduh sebagai CSV",
        data=csv,
        file_name='tekanan_per_jam.csv',
        mime='text/csv',
    )



