# 📋 Laporan Analisis & Perbaikan Aplikasi Route Optimization

**Tanggal Review:** 4 Februari 2026  
**Status:** Sudah Diperbaiki (Sebagian) + Ada Fitur yang Perlu Ditambahkan

---

## 🔴 Bug/Error yang SUDAH DIPERBAIKI

| No  | Error                          | Penyebab                                                                         | Status       |
| --- | ------------------------------ | -------------------------------------------------------------------------------- | ------------ |
| 1   | `KeyError: 'A'`                | Ketidakcocokan ID kendaraan ("A" vs "Vehicle A") di fungsi `academic_rvnd_inter` | ✅ **FIXED** |
| 2   | `KeyError: 'B'`                | Bug yang sama di fungsi `compute_costs`                                          | ✅ **FIXED** |
| 3   | Import Error `academic_replay` | Typo pada import statement                                                       | ✅ **FIXED** |

**Solusi yang Diimplementasikan:**  
Membuat fungsi helper global `get_vehicle_data()` yang bisa menangani berbagai format ID kendaraan secara otomatis.

---

## ⚠️ Temuan dari Analisis Screenshot

### A. Yang SUDAH BERJALAN DENGAN BENAR:

- ✅ Sweep Algorithm (Polar Angle Calculation & Clustering)
- ✅ Nearest Neighbor dengan Time Window Awareness
- ✅ ACS Objective Function (`Z = αD + βT + γTW`)
- ✅ RVND Intra-Route (3 iterasi per cluster)
- ✅ RVND Inter-Route (50 iterasi, `swap_1_1` neighborhood)
- ✅ Time Window Compliance: **4/4 routes compliant**
- ✅ Route Structure Validation: **Semua valid**
- ✅ Cost Calculation: **Rp 273,900**

### B. Yang PERLU DIPERHATIKAN:

| Temuan                                             | Penjelasan                                | Dampak                                |
| -------------------------------------------------- | ----------------------------------------- | ------------------------------------- |
| **Distance Mismatch**                              | Expected Dist ≠ Actual Dist pada validasi | 🟡 Perlu konfirmasi sumber data jarak |
| **Vehicle Reassignment gagal** untuk Cluster 2 & 4 | Kapasitas/ketersediaan terbatas           | 🟡 Hasil akhir tetap jalan (fallback) |

---

## 🟠 Fitur yang BELUM ADA (Sesuai Keluhan Klien)

### 1. Input Time Window Customer dari UI

**Kondisi Saat Ini:**  
Time Window (TW Start - TW End) customer di-hardcode di file `academic_replay.py`, bukan dari input user di dashboard.

**Yang Perlu Ditambahkan:**

- Kolom "Jam Buka" dan "Jam Tutup" di tabel Permintaan Customer
- Validasi input (format waktu)
- Integrasi ke algoritma

---

### 2. Input Parameter ACS dari UI

**Kondisi Saat Ini:**  
Parameter ACS (α, β, γ, ρ, jumlah semut, iterasi) menggunakan nilai default di kode.

**Yang Perlu Ditambahkan:**

- Slider/input untuk: Alpha (α), Beta (β), Gamma (γ), Rho (ρ)
- Input jumlah semut (num_ants)
- Input jumlah iterasi (max_iterations)

---

### 3. Mode Optimasi Dinamis (Opsional)

**Kondisi Saat Ini:**  
Academic Replay menggunakan data baku (10 customer dari dokumen Word), **BUKAN** data yang di-input user di canvas.

**Yang Perlu Ditambahkan (jika diminta):**

- Mode "Real Optimization" yang menghitung dari data input user
- Toggle/pilihan antara "Academic Replay" vs "Custom Data"

---

## 💰 Estimasi Harga Perbaikan (Untuk Mahasiswa/i)

| Fitur                               | Kompleksitas | Estimasi Harga           |
| ----------------------------------- | ------------ | ------------------------ |
| Input Time Window Customer          | Sedang       | **Rp 150.000 - 200.000** |
| Input Parameter ACS                 | Mudah-Sedang | **Rp 100.000 - 150.000** |
| Mode Optimasi Dinamis (Custom Data) | Tinggi       | **Rp 250.000 - 350.000** |
| **PAKET LENGKAP (Semua Fitur)**     | -            | **Rp 400.000 - 500.000** |

> **Catatan:** Harga sudah memperhitungkan tarif mahasiswa. Termasuk testing & dokumentasi singkat.

---

## ❓ Pertanyaan Klarifikasi untuk Klien

Sebelum melanjutkan, mohon konfirmasi hal-hal berikut:

### Prioritas Fitur:

1. **Fitur mana yang paling dibutuhkan?**
   - [ ] Input Time Window Customer saja
   - [ ] Input Parameter ACS saja
   - [ ] Keduanya
   - [ ] Semua + Mode Optimasi Dinamis

### Spesifikasi Time Window:

2. **Format waktu yang diinginkan?**
   - [ ] Jam:Menit (contoh: 08:00 - 17:00)
   - [ ] Menit dari tengah malam (contoh: 480 - 1020)

3. **Apakah Time Window wajib diisi untuk semua customer, atau boleh kosong (default)?**

### Spesifikasi Parameter ACS:

4. **Apakah perlu ada tombol "Reset ke Default" untuk parameter ACS?**

5. **Apakah perlu ada penjelasan/tooltip untuk setiap parameter?** (agar user paham fungsinya)

### Data Jarak:

6. **Sumber data jarak (Distance Matrix) yang benar:**
   - [ ] Dari dokumen Word (hitung manual)
   - [ ] Dari rumus Euclidean berdasarkan koordinat
   - [ ] Keduanya harus sama (perlu sinkronisasi)

### Deadline:

7. **Kapan deadline pengerjaan?**

---

## 📌 Kesimpulan

| Aspek                | Status                                                                 |
| -------------------- | ---------------------------------------------------------------------- |
| **Apakah parah?**    | ❌ **TIDAK FATAL.** Aplikasi bisa jalan dan menghasilkan output valid. |
| **Apa masalahnya?**  | Kurang fleksibel - user tidak bisa input TW dan parameter ACS sendiri. |
| **Solusi?**          | Tambahkan fitur input di UI sesuai kebutuhan klien.                    |
| **Perkiraan waktu?** | 1-3 hari kerja tergantung scope.                                       |

---

_Dokumen ini dibuat untuk keperluan komunikasi dengan klien._
