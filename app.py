import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# Konfigurasi Halaman Web
st.set_page_config(
    page_title="Image Color Extractor & CV Analysis",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 Image Color Extractor & Computer Vision Analysis")
st.markdown(
    "Aplikasi web berbasis *Computer Vision* untuk menganalisis dimensi gambar "
    "dan mengekstrak palet warna dominan menggunakan algoritma *K-Means Clustering*."
)

# Bagian Unggah Gambar
uploaded_file = st.file_uploader("Unggah gambar (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Membuka gambar
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Gambar Asli")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("📊 Informasi Visi Komputer")
        width, height = image.size
        st.write(f"- **Resolusi:** {width} x {height} piksel")
        st.write(f"- **Format:** {image.format}")
        st.write(f"- **Mode Warna:** {image.mode}")
        total_pixels = width * height
        st.write(f"- **Total Piksel:** {total_pixels:,} piksel")

    st.markdown("---")
    st.subheader("🎨 Analisis Palet Warna Dominan")

    with st.spinner("Mengekstrak warna dominan menggunakan Machine Learning..."):
        # Ubah ukuran gambar agar proses ekstraksi lebih cepat dan efisien
        image_resized = image.resize((150, 150))
        img_array = np.array(image_resized)

        # Ubah bentuk array piksel menjadi 2D (jumlah_piksel, 3 saluran warna RGB)
        pixels = img_array.reshape(-1, 3)

        # Jika gambar transparan (RGBA), ambil 3 saluran warna utamanya saja (RGB)
        if pixels.shape[1] == 4:
            pixels = pixels[:, :3]

        # Terapkan K-Means Clustering untuk mencari 5 warna dominan
        num_colors = 5
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)

        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_

        # Hitung persentase kemunculan masing-masing warna
        label_counts = np.bincount(labels)
        percentages = label_counts / len(labels)

        # Tampilkan hasil warna dominan secara visual
        color_cols = st.columns(num_colors)

        for i, col in enumerate(color_cols):
            rgb = colors[i]
            hex_code = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            pct = percentages[i] * 100

            with col:
                # Kotak warna visual
                st.markdown(
                    f"<div style='background-color: {hex_code}; height: 60px; "
                    f"border-radius: 8px; border: 1px solid #ccc;'></div>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**{hex_code}**")
                st.caption(f"{pct:.1f}%")

    st.success("Analisis Computer Vision selesai!")
else:
    st.info("Silakan unggah gambar di atas untuk memulai analisis warna.")