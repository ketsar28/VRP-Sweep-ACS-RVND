# Dokumentasi Program MFVRPTW

## Ringkasan Proyek
Program ini menyelesaikan masalah **Multi-Fleet Vehicle Routing Problem with Time Windows (MFVRPTW)** untuk distribusi obat dari satu gudang ke 10 pelanggan (rumah sakit, klinik, puskesmas). Solusi yang dihasilkan sepenuhnya mengikuti spesifikasi pada dokumen referensi (DOCX), baik dari sisi data, parameter algoritma, maupun alur perhitungan.

Pipeline optimasi:
1. **Perhitungan Matriks Jarak & Waktu** – Jarak Euclidean dari koordinat, waktu tempuh 1 km = 1 menit.
2. **Sweep Algorithm** – Pengurutan pelanggan berdasarkan sudut polar, dilanjutkan pembentukan cluster kapasitas (1 cluster = 1 kendaraan).
3. **Nearest Neighbor (NN)** – Inisialisasi rute awal per cluster.
4. **Ant Colony System (ACS)** – Optimasi rute per cluster (m=2, α=1, β=2, ρ=0,2, q₀=0,85, iterasi=2).
5. **RVND** – Perbaikan akhir dengan 2-opt, swap, dan relocate.

Seluruh hasil akhir dibekukan dalam `data/processed/final_solution.json` dan rangkuman tekstual `docs/final_summary.md`. GUI hanya membaca data tersebut tanpa menghitung ulang.

## Struktur Folder Utama
```
Program/
├─ data/processed/           # Artefak hasil parsing dan optimasi
├─ docs/                     # Dokumentasi (ringkasan dan dokumentasi ini)
├─ gui/                      # Aplikasi Streamlit untuk presentasi
├─ distance_time.py          # Modul pembuatan matriks jarak/waktu
├─ sweep_nn.py               # Sweep + inisialisasi NN
├─ acs_solver.py             # Implementasi ACS per cluster
├─ rvnd.py                   # Perbaikan RVND
├─ final_integration.py      # Integrasi pipeline dan pembuatan final_solution
└─ gui/app.py                # GUI Streamlit
```

## Cara Menjalankan Pipeline (Opsional)
> Semua artefak final sudah tersedia. Jalankan perintah berikut hanya bila perlu regenerasi data.

1. Aktivasi virtual environment (PowerShell):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2. Pastikan ketiga skrip dieksekusi berurutan:
   ```powershell
   python distance_time.py
   python sweep_nn.py
   python acs_solver.py
   python rvnd.py
   python final_integration.py
   ```

## Menjalankan Dashboard Streamlit
1. Pastikan dependensi telah terpasang:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install streamlit plotly pandas
   ```
2. Jalankan GUI:
   ```powershell
   .\.venv\Scripts\streamlit.exe run Program\gui\app.py
   ```
3. Buka browser ke `http://localhost:8501`.

## Panduan Upload ke GitHub
1. Buat file `.gitignore` bila belum ada (opsional) dan inisialisasi git:
   ```powershell
   git init
   git remote add origin https://github.com/Harunsatr/Route-Optimization.git
   ```
2. Tambahkan file, commit, dan push:
   ```powershell
   git add .
   git commit -m "Initial MFVRPTW solution"
   git branch -M main
   git push -u origin main
   ```

## Catatan Penting
- Jangan ubah angka dalam `final_solution.json` agar konsisten dengan dokumen.
- GUI hanya membaca hasil yang sudah jadi; tidak ada perhitungan ulang di sisi front-end.
- Semua deskripsi, parameter, dan asumsi sesuai dokumen DOCX asli.
