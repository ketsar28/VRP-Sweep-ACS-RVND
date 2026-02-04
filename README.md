# MFVRPTW Route Optimization - Aplikasi Optimasi Rute Distribusi

Aplikasi interaktif untuk optimasi distribusi barang menggunakan **Multi-Fleet Vehicle Routing Problem with Time Windows (MFVRPTW)** dengan antarmuka GUI berbasis Streamlit.

## 📋 Daftar Isi
- [Tentang Proyek](#tentang-proyek)
- [Prasyarat & Instalasi](#prasyarat--instalasi)
- [Cara Menjalankan Program](#cara-menjalankan-program)
- [Panduan Penggunaan GUI](#panduan-penggunaan-gui)
- [Struktur Proyek](#struktur-proyek)
- [Metode Optimasi](#metode-optimasi)

---

## 📌 Tentang Proyek

Proyek ini mengimplementasikan solusi **Multi-Fleet Vehicle Routing Problem with Time Windows (MFVRPTW)** untuk optimasi rute distribusi barang (obat-obatan) ke berbagai lokasi tujuan dengan kendala:

- **Armada Heterogen**: Kendaraan dengan kapasitas berbeda
- **Time Windows**: Setiap lokasi memiliki jam pelayanan tertentu
- **Demand Customer**: Setiap pelanggan memiliki permintaan barang yang berbeda
- **Optimasi Biaya**: Meminimalkan jarak tempuh, waktu perjalanan, dan biaya keterlambatan

---

## 💻 Prasyarat & Instalasi

### Prasyarat Sistem

1. **Python 3.8+**
   - Download dari: https://www.python.org/downloads/
   - Pastikan "Add Python to PATH" dicentang saat instalasi

2. **Git** (opsional, untuk clone repo)
   - Download dari: https://git-scm.com/

### Langkah Instalasi

#### 1. Clone Repository (atau Download ZIP)

```powershell
# Via Git
git clone https://github.com/Harunsatr/Route-Optimization.git
cd "Route-Optimization"

# Atau download ZIP dan ekstrak folder
```

#### 2. Buat Virtual Environment

```powershell
# Di direktori proyek
python -m venv .venv

# Aktivasi virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt:
.venv\Scripts\activate.bat

# Linux/MacOS:
source .venv/bin/activate
```

#### 3. Install Dependencies

```powershell
# Install semua requirements
pip install -r requirements.txt

# Atau install manual jika requirements.txt tidak ada:
pip install streamlit==1.52.2
pip install plotly==6.5.0
pip install pandas==2.3.3
pip install numpy
```

Jika ada error, jalankan:
```powershell
pip install --upgrade pip
pip install streamlit plotly pandas numpy --force-reinstall
```

---

## 🚀 Cara Menjalankan Program

### Menjalankan Aplikasi Streamlit (GUI Interaktif)

#### Windows PowerShell:
```powershell
cd "E:\Kerja Remote\Jokian\Joki Matematika (exe)"

# Aktivasi virtual environment
.\.venv\Scripts\Activate.ps1

# Jalankan aplikasi
streamlit run Program\gui\app.py
```

#### Windows Command Prompt:
```cmd
cd "E:\Kerja Remote\Jokian\Joki Matematika (exe)"
.venv\Scripts\activate.bat
streamlit run Program\gui\app.py
```

#### Linux/MacOS:
```bash
cd "Joki Matematika (exe)"
source .venv/bin/activate
streamlit run Program/gui/app.py
```

**Output:**
```
Streamlit app running on: http://localhost:8504
```

Buka browser dan akses: **http://localhost:8504**

---

## 📖 Panduan Penggunaan GUI

Aplikasi terbagi menjadi **4 Tab Utama**:

### **Tab 1️⃣ - Input Titik (Koordinat)**

Fungsi: Menambahkan lokasi Depot dan Customer dengan koordinat

**Cara Menggunakan:**
1. Pilih tipe titik: **Depot** atau **Customer**
2. **Klik pada Canvas** untuk menambah titik (atau gunakan Input Manual)
   - Sumbu X: 0-100 (Barat-Timur)
   - Sumbu Y: 0-100 (Selatan-Utara)
3. **Input Koordinat Manual** (opsional):
   - Masukkan nilai X dan Y
   - Klik tombol "Tambah Titik"
4. **Lihat Daftar Titik** di bagian bawah (Depot dan Customer)
5. **Hapus Titik**: Klik ikon 🗑️ di samping nama titik
6. **Reset Semua**: Klik "🔄 Reset Semua Titik" untuk menghapus semua

**Catatan:**
- Minimal perlu 1 Depot dan 2 Customer untuk proses selanjutnya
- Depot biasanya adalah pusat distribusi/gudang
- Titik akan ter-simpan di session state

---

### **Tab 2️⃣ - Input Data (Parameter & Jarak)**

Fungsi: Mengatur parameter dan matriks jarak antar titik

#### **Bagian 1: Kapasitas Kendaraan**
- Input kapasitas maksimal kendaraan (satuan unit)
- Contoh: 100 unit per kendaraan

#### **Bagian 2: Jumlah Iterasi**
- Jumlah iterasi untuk algoritma optimasi
- Rekomendasi: 2-5 iterasi (semakin tinggi = hasil lebih baik tapi lebih lama)

#### **Bagian 3: Permintaan Customer**
Masukkan berapa banyak barang yang diminta setiap customer:

| Customer | Permintaan |
|----------|-----------|
| Customer 1 | 10 |
| Customer 2 | 20 |
| ... | ... |

**Cara input:**
1. Klik pada kolom "Permintaan" untuk setiap baris
2. Masukkan angka permintaan
3. Tekan Enter

#### **Bagian 4: Tabel Jarak Antar Titik**
Masukkan matriks jarak (dalam km) antar lokasi:

**Format:**
```
     0    1    2
0 [  0   10   20 ]    ← Jarak Depot ke Depot=0, Depot→C1=10, Depot→C2=20
1 [ 10    0   15 ]    ← Jarak C1→Depot=10, C1→C1=0, C1→C2=15
2 [ 20   15    0 ]    ← Jarak C2→Depot=20, C2→C1=15, C2→C2=0
```

**PENTING: Matriks Simetris**
- Jarak A→B **HARUS SAMA** dengan B→A
- Contoh: Jika Depot→C1 = 10, maka C1→Depot juga harus 10
- **Auto-Sync**: Jika Anda edit satu cell, cell pasangannya akan otomatis ter-update!

**Cara input:**
1. Klik pada cell (selain diagonal yang berwarna abu-abu)
2. Masukkan nilai jarak
3. Cell symmetric-nya akan otomatis ter-update dengan nilai yang sama
4. Tekan Tab atau klik cell lain untuk melanjutkan

**Tombol Aksi:**
- 💾 **Simpan Progres**: Menyimpan semua data input (bisa di-download sebagai JSON)
- 📤 **Muat Progres**: Upload file JSON yang sudah disimpan sebelumnya
- 🚀 **Lanjutkan Proses**: Validasi dan jalankan algoritma optimasi

---

### **Tab 3️⃣ - Hasil (Output Algoritma)**

Menampilkan hasil optimasi dalam bentuk teks:
- Rute setiap kendaraan
- Total jarak tempuh
- Total permintaan per kendaraan
- Informasi waktu layanan dan keterlambatan (jika ada)

---

### **Tab 4️⃣ - Visualisasi Rute (Graph)**

Menampilkan visualisasi interaktif rute distribusi:
- 🟨 **Kuning**: Depot (pusat distribusi)
- 🔴 **Merah**: Customer (lokasi tujuan)
- 🔵 **Biru**: Rute kendaraan

**Interaksi:**
- Hover ke atas garis untuk melihat informasi rute
- Scroll untuk zoom in/out
- Drag untuk pan

---

## 🗂️ Struktur Proyek

```
Route-Optimization/
│
├── README.md                          # File ini
├── requirements.txt                   # Dependencies Python
│
└── Program/
    ├── gui/                           # Aplikasi Streamlit
    │   ├── app.py                     # Main entry point
    │   ├── agents.py                  # Validasi & pipeline
    │   └── tabs/                      # Komponen tab
    │       ├── input_titik.py         # Input koordinat
    │       ├── input_data.py          # Input parameter & jarak
    │       ├── hasil.py               # Tampilkan hasil
    │       └── graph_hasil.py         # Visualisasi rute
    │
    ├── data/
    │   └── processed/                 # Output JSON dari proses
    │       ├── final_solution.json
    │       ├── acs_routes.json
    │       ├── rvnd_routes.json
    │       └── ... (file lainnya)
    │
    ├── docs/                          # Dokumentasi
    │   ├── dokumentasi_id.md          # Penjelasan algoritma (ID)
    │   └── final_summary.md           # Ringkasan hasil
    │
    ├── acs_solver.py                  # Algoritma Ant Colony System
    ├── distance_time.py               # Matriks jarak & waktu
    ├── rvnd.py                        # Random VND (optimasi)
    ├── sweep_nn.py                    # Sweep + Nearest Neighbor
    └── final_integration.py           # Integrasi final
```

---

## 🔍 Metode Optimasi

### Pipeline Algoritma:

1. **Input User**
   - Koordinat depot dan customer
   - Kapasitas kendaraan
   - Permintaan customer
   - Matriks jarak antar titik

2. **Sweep Algorithm**
   - Mengurutkan customer berdasarkan sudut polar dari depot
   - Membentuk cluster sesuai kapasitas kendaraan

3. **Nearest Neighbor (NN)**
   - Membuat rute awal untuk setiap cluster

4. **Ant Colony System (ACS)**
   - Optimasi rute menggunakan algoritma semut
   - Parameter: m=2, α=1, β=2, ρ=0.2, q₀=0.85

5. **RVND (Random Variable Neighborhood Descent)**
   - Penyempurnaan rute dengan 2-opt, swap, relocate
   - Menghasilkan rute yang lebih optimal

6. **Final Output**
   - Rute distribusi final
   - Metrik: jarak total, waktu, biaya, dll

---

## ⚙️ Troubleshooting

### Error: "ModuleNotFoundError: No module named 'streamlit'"

**Solusi:**
```powershell
# Pastikan virtual environment aktif
.\.venv\Scripts\Activate.ps1

# Install ulang streamlit
pip install --upgrade streamlit
```

### Error: "Port 8504 already in use"

**Solusi:**
```powershell
# Gunakan port berbeda
streamlit run Program\gui\app.py --server.port 8505
```

### Error: "Matriks tidak simetris"

**Solusi:**
- Pastikan jarak A→B sama dengan B→A
- Gunakan fitur Auto-Sync (nilai akan ter-update otomatis)
- Contoh: Jika D→C1 = 10, maka C1→D juga harus 10

### Error: "Validasi gagal"

**Periksa:**
- Sudah input minimal 1 Depot dan 2 Customer? ✓
- Semua permintaan customer sudah diisi? ✓
- Matriks jarak simetris dan tidak ada nilai negatif? ✓

---

## 📝 Catatan Penting

- **Data di-save otomatis** di session state Streamlit
- **Gunakan "💾 Simpan Progres"** untuk menyimpan ke file JSON yang bisa di-load ulang
- **Matriks jarak harus simetris** (jarak dua arah sama)
- **Kapasitas harus > 0** dan nilai integer
- Gunakan **"🚀 Lanjutkan Proses"** hanya jika semua data sudah benar

---

## 📚 Referensi & Dokumentasi

- Penjelasan detail algoritma: [dokumentasi_id.md](Program/docs/dokumentasi_id.md)
- Ringkasan hasil optimasi: [final_summary.md](Program/docs/final_summary.md)
- Contoh output: [final_solution.json](Program/data/processed/final_solution.json)

---

## 📧 Support

Untuk pertanyaan atau laporan bug, silakan buat issue di repository GitHub.

---

**Versi**: 1.0 | **Last Updated**: Desember 2025 | **Language**: Bahasa Indonesia
