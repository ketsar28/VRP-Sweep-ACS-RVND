# Smart Route Optimization System (Decision Support System)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://harunsatr-rvnd.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Project ini saya kerjakan untuk menjawab tantangan manajemen logistik yang makin kompleks saat ini. Efisiensi adalah kunci, tapi kenyataannya di lapangan seringkali kita bertemu masalah biaya operasional yang membengkak, armada yang terbatas, dan jadwal pengiriman yang sangat ketat.

Saya mengembangkan sistem ini sebagai **Sistem Pendukung Keputusan (Decision Support System)** yang cerdas. Dengan algoritma hibrida yang saya bangun, perencanaan rute yang tadinya manual dan membingungkan bisa berubah jadi otomatis, lebih hemat, dan pastinya optimal.

---

## 📋 Kenapa Proyek Ini Penting?

Waktu saya riset, perencanaan rute manual itu penuh celah. Manusia punya keterbatasan buat ngitung puluhan variabel sekaligus—mulai dari kemacetan, kapasitas muatan tiap armada (yang beda-beda jenisnya), sampai jendela waktu (_time windows_) tiap lokasi.

Dampak negatif yang coba saya selesaikan di sini:

1.  **Biaya yang Boros**: Kalau rute nggak efisien, bensin bakal boros dan jarak tempuh jadi jauh nggak karuan.
2.  **Jadwal yang Berantakan**: Lokasi tujuan punya jam operasional masing-masing. Kalau kita telat, kepuasan pelanggan bakal jatuh.
3.  **Armada Nggak Terpakai Maksimal**: Seringkali ada kendaraan yang muatannya penuh banget, tapi yang lain malah sepi. Saya buat sistem ini biar beban kerja armada lebih seimbang.

---

## 🎯 Solusi yang Saya Tawarkan

Fokus utama saya adalah menyelesaikan masalah **Vehicle Routing Problem with Time Windows (VRPTW)**. Jadi rute yang dihasilkan bukan cuma yang terpendek, tapi juga yang paling masuk akal secara operasional.

### Fitur-Fitur Unggulan

**1. Optimasi Armada Heterogen**
Saya buat sistem ini fleksibel untuk berbagai jenis kendaraan (kayak truk besar, van, atau motor) secara bersamaan. Algoritmanya otomatis milihin kombinasi kendaraan paling pas buat nganter barang, sesuai kapasitas dan biaya bensin masing-masing.

**2. Transparansi Algoritma (Academic Replay)**
Ini fitur yang paling saya banggakan. Saya nggak mau hasil rutenya keluar gitu aja tanpa penjelasan. Lewat _Academic Replay_, kita bisa lihat gimana algoritma ini mikir langkah demi langkah sampai ketemu rute terbaik. Fitur ini sangat berguna untuk validasi akademis.

**3. Visualisasi Peta & Analisis**
Hasilnya saya tampilin di peta interaktif. Kita bisa lihat jelas jalur yang dilewati, barang yang dibawa tiap kendaraan, sampai status tiap titik. Kalau ada titik yang nggak bisa dilayani, sistem bakal jelasin alasannya secara jujur.

**4. Input Data Simpel**
Data bisa dimasukkan lewat Excel atau tinggal klik-klik aja di peta. Saya bikin se-praktis mungkin buat kebutuhan di lapangan.

**5. Stabilitas & Keandalan Tinggi (NEW!)**
Sistem ini dilengkapi mekanisme **Multi-Start Optimization**. Jika algoritma menemui rute yang sangat rumit atau muatan yang hampir melampaui kapasitas, sistem akan otomatis melakukan pencarian ulang (retries) dengan strategi yang berbeda sampai menemukan solusi yang paling stabil.

---

## ⚙️ Cara Kerja Algoritma (Urusan Teknis)

Di balik layarnya, saya gabungin beberapa metode metaheuristik yang cukup kompleks:

1.  **Inisialisasi (Sweep & Nearest Neighbor)**: Digunakan untuk membuat kerangka rute awal berdasarkan posisi geografis dan urutan titik terdekat dengan batasan waktu.
2.  **Eksplorasi Global (Ant Colony System)**: Terinspirasi dari cara kerja semut mencari jalan, algoritma ini menelusuri berbagai kemungkinan jalur untuk menghindari jebakan solusi lokal.
3.  **Perbaikan Lokal (RVND)**: Hasil rute dipoles lagi menggunakan _Randomized Variable Neighborhood Descent_. Titik-titik lokasi ditukar-tukar (Swap, Shift, 2-Opt) sampai ketemu biaya paling minimum.

---

## 🛠️ Persiapan & Instalasi Lokal

Panduan ini buat yang mau nyoba jalanin project saya di komputer sendiri (Windows/macOS/Linux).

### 1. Minimal Persyaratan

- **Python 3.9+**
- **RAM Minimal 4GB** (Biar visualisasi petanya lancar)

### 2. Langkah Instalasi

**A. Ambil Source Code**
Download file ZIP project saya atau pakai git:

```bash
git clone https://github.com/ketsar28/Smart-Route-Optimization.git
cd Smart-Route-Optimization
```

**B. Setup Virtual Environment**

```bash
# Windows:
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux:
python3 -m venv venv
source venv/bin/activate
```

**C. Install Library**

```bash
pip install -r requirements.txt
```

### 3. Cara Menjalankan Aplikasi

Pindah ke folder utama project, lalu ketik perintah ini:

```bash
streamlit run Program/gui/app.py
```

Tunggu sebentar, browser bakal terbuka di `http://localhost:8501`.

---

## 📁 Struktur Project

```
Program/
├── 📄 acs_solver.py            # Ant Colony System Optimizer
├── 📄 rvnd.py                  # RVND Optimizer (Stability v2.0)
├── 📄 academic_replay.py       # Engine untuk Academic Replay
├── 📄 distance_time.py         # Matriks Jarak & Waktu
├── 📄 sweep_nn.py              # Initial Route Generator
└── 📁 gui/                     # Tampilan Dashboard Streamlit
```

---

_Project ini saya kembangkan dengan serius untuk memberikan solusi logistik yang nyata dan efisien._

**Author: Harunsatr** - [GitHub](https://github.com/Harunsatr)
⭐ **Jika project ini membantu, jangan lupa berikan star di GitHub!**
