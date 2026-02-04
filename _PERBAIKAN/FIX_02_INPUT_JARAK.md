# 🔧 FIX #02: Input Jarak Antar Titik

## 📍 Masalah

Dari screenshot, tabel jarak menunjukkan **semua nilai 0.00**. Ini menyebabkan algoritma tidak bisa menghitung rute dengan benar.

---

## 📍 Lokasi Input Jarak

1. Buka aplikasi di browser: `http://localhost:8501`
2. Klik tab **"Input Data"**
3. Scroll ke bawah sampai menemukan **"4️⃣ Tabel Jarak Antar Titik"**
4. Di situ ada tabel matriks dengan baris dan kolom (0, 1, 2, 3, dst)
5. **Klik sel** dan ketik jarak dalam satuan KM

---

## 🗺️ Pemetaan Node

| Node | Titik      |
| ---- | ---------- |
| 0    | Depot      |
| 1    | Customer 1 |
| 2    | Customer 2 |
| 3    | Customer 3 |

---

## 📊 DATA TEST SIMULASI

Berikut data test yang bisa kamu pakai untuk simulasi:

### Koordinat Titik (Input di Tab "Input Titik")

| Titik          | X   | Y   |
| -------------- | --- | --- |
| Depot (Node 0) | 10  | 20  |
| Customer 1     | 50  | 45  |
| Customer 2     | 25  | 70  |
| Customer 3     | 80  | 70  |

### Rumus Jarak Euclidean

```
Jarak(A, B) = √((X_B - X_A)² + (Y_B - Y_A)²)
```

### Tabel Jarak yang Harus Diinput

Berdasarkan koordinat di atas, berikut jarak yang harus diinput:

| From\To       | 0 (Depot) | 1 (C1)    | 2 (C2)    | 3 (C3)    |
| ------------- | --------- | --------- | --------- | --------- |
| **0 (Depot)** | 0.00      | **47.17** | **52.20** | **86.02** |
| **1 (C1)**    | **47.17** | 0.00      | **35.36** | **39.05** |
| **2 (C2)**    | **52.20** | **35.36** | 0.00      | **55.00** |
| **3 (C3)**    | **86.02** | **39.05** | **55.00** | 0.00      |

> **Catatan:** Diagonal (0→0, 1→1, dst) selalu 0. Matriks simetris (jarak A→B = jarak B→A).

---

## 📝 Perhitungan Detail

### Jarak Depot (0) ke Customer 1 (1)

```
= √((50-10)² + (45-20)²)
= √(40² + 25²)
= √(1600 + 625)
= √2225
= 47.17 km
```

### Jarak Depot (0) ke Customer 2 (2)

```
= √((25-10)² + (70-20)²)
= √(15² + 50²)
= √(225 + 2500)
= √2725
= 52.20 km
```

### Jarak Depot (0) ke Customer 3 (3)

```
= √((80-10)² + (70-20)²)
= √(70² + 50²)
= √(4900 + 2500)
= √7400
= 86.02 km
```

### Jarak Customer 1 (1) ke Customer 2 (2)

```
= √((25-50)² + (70-45)²)
= √((-25)² + 25²)
= √(625 + 625)
= √1250
= 35.36 km
```

### Jarak Customer 1 (1) ke Customer 3 (3)

```
= √((80-50)² + (70-45)²)
= √(30² + 25²)
= √(900 + 625)
= √1525
= 39.05 km
```

### Jarak Customer 2 (2) ke Customer 3 (3)

```
= √((80-25)² + (70-70)²)
= √(55² + 0²)
= √3025
= 55.00 km
```

---

## 🚗 Data Kendaraan (Input di Section 1 Tab "Input Data")

| Kendaraan | Kapasitas | Unit | Jam Mulai | Jam Selesai |
| --------- | --------- | ---- | --------- | ----------- |
| Vehicle A | 60        | 2    | 08:00     | 15:00       |
| Vehicle B | 100       | 2    | 09:00     | 16:00       |
| Vehicle C | 150       | 1    | 10:00     | 17:00       |

---

## 📦 Data Permintaan Customer (Input di Section 3 Tab "Input Data")

| Customer   | Permintaan (Demand) |
| ---------- | ------------------- |
| Customer 1 | 30                  |
| Customer 2 | 45                  |
| Customer 3 | 55                  |

---

## ✅ Langkah Input Lengkap

### Step 1: Input Titik (Tab "Input Titik")

1. Pilih "Depot" di sidebar
2. Masukkan X=10, Y=20, klik "Tambah Titik"
3. Pilih "Customer" di sidebar
4. Masukkan Customer 1: X=50, Y=45, klik "Tambah Titik"
5. Masukkan Customer 2: X=25, Y=70, klik "Tambah Titik"
6. Masukkan Customer 3: X=80, Y=70, klik "Tambah Titik"

### Step 2: Input Kendaraan (Tab "Input Data" Section 1)

1. Klik "➕ Tambah Kendaraan Baru" 3x
2. Isi sesuai tabel kendaraan di atas

### Step 3: Input Permintaan (Tab "Input Data" Section 3)

1. Isi tabel Permintaan sesuai data di atas

### Step 4: Input Jarak (Tab "Input Data" Section 4)

1. Klik sel tabel dan isi sesuai matriks jarak di atas
2. Pastikan simetris (atas dan bawah diagonal sama)

### Step 5: Proses

1. Klik tombol **"🚀 Lanjutkan Proses"**
2. Lihat hasil di tab **"Hasil"** dan **"Graph Hasil"**

---

## ⚠️ Masalah UX yang Perlu Diketahui

Saat ini, jarak HARUS diinput manual. Idealnya, jarak bisa dihitung otomatis dari koordinat. Ini adalah salah satu **perbaikan yang bisa dikerjakan** nanti.

---

_Fix #02 selesai. Lanjut ke CHECKLIST_TESTING untuk verifikasi._
