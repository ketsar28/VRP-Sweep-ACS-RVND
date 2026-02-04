# 📦 Optimisasi Rute Distribusi Obat (MF-VRPTW)
## Multi-Fleet Vehicle Routing Problem with Time Windows

---

## 🎯 Ringkasan Project

Project ini adalah **alat bantu hitung** untuk mengoptimalkan rute distribusi obat dari **gudang (depot)** ke berbagai **customer** (rumah sakit, klinik, apotek). Tujuannya adalah menemukan rute terpendek/termurah dengan mempertimbangkan:

1. **Kapasitas Kendaraan** - Ada beberapa jenis kendaraan dengan kapasitas berbeda
2. **Time Window** - Setiap customer punya jam operasional tertentu untuk menerima paket
3. **Demand** - Setiap customer punya jumlah pesanan yang berbeda

### Analogi Sederhana 🏪

> Bayangkan kamu adalah **manajer logistik GoFood** yang harus mengantar pesanan ke 20 restoran berbeda. Kamu punya 3 jenis motor dengan kapasitas tas berbeda, dan setiap restoran cuma buka jam tertentu. Gimana caranya supaya semua pesanan terkirim tepat waktu dengan biaya bensin seminim mungkin?

---

## 📚 Istilah Penting (Glosarium)

| Istilah | Penjelasan | Analogi |
|---------|------------|---------|
| **Depot** | Gudang pusat, titik awal & akhir distribusi | Kantor pusat ojol |
| **Customer** | Tujuan pengiriman (RS, Klinik, Apotek) | Restoran yang pesan |
| **Demand** | Jumlah pesanan/barang yang diminta customer | Jumlah porsi makanan |
| **Time Window** | Rentang waktu customer bisa terima barang | Jam buka restoran |
| **Fleet** | Armada kendaraan | Motor, mobil, truk |
| **Cluster** | Kelompok customer yang dilayani 1 kendaraan | Zona pengantaran |
| **Route** | Urutan kunjungan customer | Rute navigasi Google Maps |

---

## 🚗 Detail Kendaraan (Fleet)

Berdasarkan penjelasan, gudang memiliki **3 jenis kendaraan**:

| Jenis | Kapasitas | Jumlah Unit |
|-------|-----------|-------------|
| Kendaraan A | (sesuai data) | 2 unit |
| Kendaraan B | (sesuai data) | 2 unit |
| Kendaraan C | (sesuai data) | 1 unit |

> **Total armada**: 5 unit kendaraan

---

## 🔄 Alur Proses Algoritma

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ALUR OPTIMISASI RUTE                            │
└─────────────────────────────────────────────────────────────────────────┘

         INPUT DATA
              │
              ▼
    ┌─────────────────┐
    │  1. SWEEP       │     Urutkan customer berdasarkan sudut polar
    │  ALGORITHM      │     (seperti jarum jam berputar dari depot)
    └────────┬────────┘
              │
              ▼
    ┌─────────────────┐
    │  2. CLUSTERING  │     Kelompokkan customer ke kendaraan
    │  + Nearest      │     berdasarkan kapasitas (pakai metode NN)
    │  Neighbor       │
    └────────┬────────┘
              │
              ▼
    ┌─────────────────┐
    │  3. ACS         │     Optimasi rute tiap cluster dengan
    │  (Ant Colony    │     algoritma semut + cek Time Window
    │  System)        │
    └────────┬────────┘
              │
              ▼
    ┌─────────────────┐
    │  4. RVND        │     Fine-tuning rute dengan local search
    │  (Local Search) │     (tukar-tukar posisi customer)
    └────────┬────────┘
              │
              ▼
         OUTPUT RUTE
         OPTIMAL
```

---

## 📐 TAHAP 1: Sweep Algorithm (Inisialisasi)

### Apa itu Sweep Algorithm?

**Sweep Algorithm** adalah metode untuk mengurutkan customer berdasarkan **sudut polar** terhadap depot.

### Analogi 🕐

> Bayangkan kamu berdiri di tengah lapangan (depot), dan ada 10 orang (customer) tersebar di sekitarmu. Kamu mau menghampiri mereka semua dengan **berputar searah jarum jam** mulai dari jam 12. Sweep Algorithm mengurutkan orang-orang itu berdasarkan posisi jam mereka.

### Rumus Sudut Polar

```
θ = atan2(y_customer - y_depot, x_customer - x_depot)

Keterangan:
- θ (theta) = sudut polar dalam radian
- (x_depot, y_depot) = koordinat depot
- (x_customer, y_customer) = koordinat customer
- atan2 = fungsi arctangent dua argumen
```

### Contoh Perhitungan

```
Depot: (0, 0)
Customer A: (3, 4)  → θ = atan2(4, 3) = 53.13°
Customer B: (-2, 2) → θ = atan2(2, -2) = 135°
Customer C: (1, -1) → θ = atan2(-1, 1) = -45° → 315°

Urutan setelah sweep: A → B → C
```

### Output

Setelah Sweep, kita dapat **satu rute panjang** dengan semua customer terurut:

```
Depot → Customer_1 → Customer_2 → ... → Customer_n → Depot
```

---

## 📦 TAHAP 2: Clustering (Pengelompokan)

### Tujuan

Membagi customer ke beberapa kendaraan berdasarkan **kapasitas**.

### Algoritma

```
1. Ambil daftar customer yang sudah terurut (dari Sweep)
2. Mulai dari kendaraan pertama
3. Masukkan customer satu per satu ke kendaraan:
   - Jika (demand customer + total demand di kendaraan) <= kapasitas kendaraan:
     → Tambahkan ke kendaraan
   - Jika tidak muat:
     → Buka kendaraan baru
4. Ulangi sampai semua customer masuk
5. Gunakan Nearest Neighbor untuk urutkan customer dalam tiap cluster
```

### Analogi 🎒

> Seperti mengisi tas belanja. Kamu punya 3 tas dengan ukuran berbeda. Belanjaan dimasukkan satu per satu sampai tas penuh, baru pindah ke tas berikutnya.

### Contoh Clustering

```
Kapasitas: Kendaraan A = 100 kg, Kendaraan B = 80 kg

Customer (urut dari Sweep):
- C1: 30 kg
- C2: 40 kg  
- C3: 25 kg
- C4: 50 kg
- C5: 35 kg

Hasil Clustering:
- Kendaraan A: C1 (30) + C2 (40) + C3 (25) = 95 kg ✓
- Kendaraan B: C4 (50) + C5 (35) = 85 kg ✗ TIDAK MUAT!
  → C4 masuk Kendaraan B (50 kg)
  → C5 masuk Kendaraan baru C (35 kg)
```

### Output

**Cluster-cluster** berisi daftar customer untuk masing-masing kendaraan.

---

## 🐜 TAHAP 3: Ant Colony System (ACS)

### Apa itu ACS?

**Ant Colony System** adalah algoritma optimisasi yang terinspirasi dari **perilaku semut mencari makanan**. Semut meninggalkan jejak aroma (**pheromone**) di jalur yang mereka lewati. Semakin banyak semut lewat jalur tertentu, semakin kuat aromanya.

### Analogi 🐜

> Bayangkan ada 100 semut yang harus menemukan jalan terpendek dari sarang ke makanan. Awalnya mereka random. Tapi semut yang menemukan jalan pendek akan lebih cepat bolak-balik, jadi jejaknya lebih kuat. Lama-lama semut lain ikut jejak yang kuat itu.

### Parameter ACS

| Parameter | Simbol | Fungsi |
|-----------|--------|--------|
| Alpha (α) | `alpha` | Pengaruh pheromone terhadap pemilihan jalur |
| Beta (β) | `beta` | Pengaruh jarak terhadap pemilihan jalur |
| Rho (ρ) | `rho` | Tingkat penguapan pheromone |
| Q | `Q` | Konstanta pheromone |
| Jumlah Semut | `n_ants` | Banyaknya semut yang mencari |
| Iterasi | `iterations` | Berapa kali proses diulang |

### Cara Kerja ACS

```
┌──────────────────────────────────────────────────────────────┐
│  PROSES ACS UNTUK SETIAP CLUSTER                             │
└──────────────────────────────────────────────────────────────┘

1. INISIALISASI PHEROMONE
   - Semua jalur diberi nilai pheromone awal yang sama
   - τ₀ = 1 / (n × L_nn)  dimana L_nn = panjang rute Nearest Neighbor

2. KONSTRUKSI RUTE (untuk setiap semut)
   
   Loop: Mulai dari depot, pilih customer berikutnya
   │
   ├─→ Hitung probabilitas ke setiap customer yang belum dikunjungi
   │   
   │   P(i→j) = [τᵢⱼ]^α × [ηᵢⱼ]^β / Σ[τᵢₖ]^α × [ηᵢₖ]^β
   │   
   │   dimana:
   │   - τᵢⱼ = pheromone dari i ke j
   │   - ηᵢⱼ = 1/jarak (visibility)
   │   - α = pengaruh pheromone
   │   - β = pengaruh jarak
   │
   ├─→ Pilih customer dengan probabilitas tertinggi
   │
   ├─→ LOCAL UPDATE PHEROMONE
   │   τᵢⱼ = (1-ρ) × τᵢⱼ + ρ × τ₀
   │
   └─→ Ulangi sampai semua customer terkunjungi

3. CEK TIME WINDOW
   - Pastikan rute yang terbentuk memenuhi batasan waktu
   - Jika terlambat → rute tidak valid

4. GLOBAL UPDATE PHEROMONE
   - Hanya rute terbaik yang di-update
   - τᵢⱼ = (1-ρ) × τᵢⱼ + ρ × (1/L_best)
   - L_best = panjang rute terbaik

5. ULANGI dari langkah 2 sampai iterasi selesai
```

### Penjelasan Rumus Probabilitas

```
P(i→j) = [τᵢⱼ]^α × [ηᵢⱼ]^β / Σ[τᵢₖ]^α × [ηᵢₖ]^β

Contoh:
- Dari depot, ada 3 pilihan: A, B, C
- Pheromone: τ_A = 2, τ_B = 1, τ_C = 3
- Jarak: d_A = 5, d_B = 10, d_C = 2 → η_A = 0.2, η_B = 0.1, η_C = 0.5
- α = 1, β = 2

Perhitungan:
- Nilai A = 2¹ × 0.2² = 2 × 0.04 = 0.08
- Nilai B = 1¹ × 0.1² = 1 × 0.01 = 0.01
- Nilai C = 3¹ × 0.5² = 3 × 0.25 = 0.75
- Total = 0.84

Probabilitas:
- P(A) = 0.08 / 0.84 = 9.5%
- P(B) = 0.01 / 0.84 = 1.2%
- P(C) = 0.75 / 0.84 = 89.3% ← TERPILIH!
```

### Output

**Rute optimal** untuk setiap cluster yang sudah melalui proses ACS.

---

## 🔧 TAHAP 4: RVND (Randomized Variable Neighborhood Descent)

### Apa itu RVND?

**RVND** adalah metode **local search** untuk "menyempurnakan" rute yang sudah dibentuk ACS. Ia mencoba berbagai **operasi swap/move** secara acak untuk mencari perbaikan.

### Analogi 🔀

> Seperti menyusun ulang barang di lemari. Kadang kalau kita tukar posisi 2 barang, jadi lebih rapi. RVND mencoba-coba tukar posisi customer untuk cari rute lebih pendek.

### Operasi dalam RVND

```
1. SWAP (Pertukaran)
   Sebelum: A → B → C → D → E
   Tukar B dan D
   Sesudah: A → D → C → B → E

2. 2-OPT (Reverse Segment)
   Sebelum: A → B → C → D → E
   Balik segmen B-C-D
   Sesudah: A → D → C → B → E

3. OR-OPT (Relocate)
   Sebelum: A → B → C → D → E
   Pindahkan C ke sebelum B
   Sesudah: A → C → B → D → E

4. CROSS (Inter-Route Exchange)
   Rute 1: A → B → C
   Rute 2: X → Y → Z
   Tukar C dan Y
   Rute 1: A → B → Y
   Rute 2: X → C → Z
```

### Proses RVND

```
1. Buat daftar neighborhood operators (SWAP, 2-OPT, OR-OPT, dll)
2. Acak urutan operators
3. Untuk setiap operator:
   - Terapkan operator ke rute
   - Jika hasilnya lebih baik → terima, kembali ke step 2
   - Jika tidak → lanjut ke operator berikutnya
4. Selesai jika semua operator sudah dicoba tanpa perbaikan
```

### Output

**Rute final yang sudah dioptimasi** untuk setiap kendaraan.

---

## ⏰ Pengecekan Time Window

### Apa itu Time Window?

Setiap customer punya **jam operasional** untuk menerima pengiriman.

```
Customer A: 08:00 - 12:00 (pagi)
Customer B: 13:00 - 17:00 (siang)
Customer C: 09:00 - 15:00 (fleksibel)
```

### Cara Pengecekan

```python
def cek_time_window(rute, waktu_berangkat):
    waktu_sekarang = waktu_berangkat
    
    for customer in rute:
        # Waktu perjalanan dari titik sebelumnya
        waktu_perjalanan = hitung_jarak(...) / kecepatan
        waktu_tiba = waktu_sekarang + waktu_perjalanan
        
        # Cek apakah tiba dalam time window
        if waktu_tiba < customer.buka:
            # Datang terlalu awal, tunggu
            waktu_sekarang = customer.buka
        elif waktu_tiba > customer.tutup:
            # Terlambat! Rute tidak valid
            return False
        else:
            waktu_sekarang = waktu_tiba
        
        # Tambah waktu pelayanan
        waktu_sekarang += customer.waktu_pelayanan
    
    return True
```

---

## 📊 INPUT yang Diperlukan

### 1. Data Koordinat

| Lokasi | X | Y |
|--------|---|---|
| Depot | 0 | 0 |
| Customer 1 | 5 | 3 |
| Customer 2 | -2 | 7 |
| ... | ... | ... |

### 2. Data Customer

| Customer | Demand (kg) | TW Buka | TW Tutup |
|----------|-------------|---------|----------|
| C1 | 50 | 08:00 | 12:00 |
| C2 | 30 | 09:00 | 14:00 |
| ... | ... | ... | ... |

### 3. Data Kendaraan

| Jenis | Kapasitas | Unit | Kecepatan |
|-------|-----------|------|-----------|
| Motor | 50 kg | 2 | 40 km/h |
| Mobil | 150 kg | 2 | 60 km/h |
| Truk | 300 kg | 1 | 50 km/h |

### 4. Parameter ACS

| Parameter | Nilai Default |
|-----------|---------------|
| α (alpha) | 1 |
| β (beta) | 2-5 |
| ρ (rho) | 0.1 |
| Q | 100 |
| Jumlah semut | 10-50 |
| Iterasi | 100 |

### 5. Time Window Depot

| | Buka | Tutup |
|---|------|-------|
| Depot | 07:00 | 18:00 |

---

## 🧮 Rumus Perhitungan Jarak

Jarak antara dua titik (Euclidean Distance):

```
d(i,j) = √[(xⱼ - xᵢ)² + (yⱼ - yᵢ)²]

Contoh:
Depot = (0, 0)
Customer A = (3, 4)

d = √[(3-0)² + (4-0)²]
d = √[9 + 16]
d = √25
d = 5 km
```

> **Catatan**: Jarak bisa dihitung otomatis dari koordinat, tidak perlu input manual jika ada rumus.

---

## 📤 OUTPUT yang Diharapkan

### 1. Rute per Kendaraan

```
Kendaraan 1 (Motor - Kapasitas 50kg):
  Depot → C3 → C7 → C1 → Depot
  Total demand: 45 kg
  Total jarak: 23.5 km
  Waktu: 08:00 - 11:30

Kendaraan 2 (Mobil - Kapasitas 150kg):
  Depot → C2 → C4 → C8 → C5 → Depot
  Total demand: 140 kg
  Total jarak: 45.2 km
  Waktu: 08:00 - 14:15

...
```

### 2. Ringkasan

```
Total kendaraan terpakai: 4 dari 5
Total jarak tempuh: 125.3 km
Total waktu operasi: 8 jam
Semua time window: TERPENUHI ✓
```

### 3. Visualisasi Rute (Opsional)

Peta dengan rute berbeda warna untuk setiap kendaraan.

---

## ⚠️ Potensi Masalah (Berdasarkan Keluhan)

Berdasarkan keluhan "hasil tidak sesuai", kemungkinan masalah ada di:

### 1. Perhitungan Jarak
- ❌ Jarak tidak dihitung dengan benar dari koordinat
- ✅ Solusi: Pastikan rumus Euclidean Distance benar

### 2. Kapasitas Kendaraan
- ❌ Clustering tidak memperhatikan kapasitas maksimal
- ✅ Solusi: Validasi demand ≤ kapasitas sebelum assign

### 3. Time Window
- ❌ Pengecekan waktu tidak akurat (tidak ada waktu pelayanan, dll)
- ✅ Solusi: Tambahkan service time di setiap customer

### 4. Update Pheromone
- ❌ Formula update salah atau tidak konsisten
- ✅ Solusi: Review rumus local & global update

### 5. RVND Tidak Berjalan
- ❌ Local search tidak improve karena neighborhood terlalu kecil
- ✅ Solusi: Tambah variasi operator (2-opt, 3-opt, dll)

---

## 🛠️ Langkah Selanjutnya

1. **Dapatkan Source Code** dari website Streamlit yang sudah ada
2. **Review Kode** untuk identifikasi bug
3. **Tes dengan Data Kecil** (3-5 customer) untuk validasi manual
4. **Bandingkan Hasil** perhitungan manual vs program
5. **Perbaiki Bug** yang ditemukan

---

## 📚 Referensi

- Dorigo, M., & Gambardella, L. M. (1997). Ant colony system: A cooperative learning approach to the TSP. IEEE.
- Solomon, M. M. (1987). Algorithms for the vehicle routing and scheduling problems with time window constraints.
- Hansen, P., & Mladenović, N. (2001). Variable neighborhood search.
- Sweep Algorithm untuk VRP

---

## 📝 Catatan

> Dokumen ini dibuat untuk membantu memahami project skripsi teman. Jika ada pertanyaan atau butuh penjelasan lebih lanjut tentang bagian tertentu, silakan tanyakan!

**Dibuat oleh**: AI Assistant  
**Tanggal**: 4 Februari 2026
