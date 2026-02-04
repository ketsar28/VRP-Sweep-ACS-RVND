# 🔍 Analisis Problem & Rekomendasi

## Catatan untuk Pengembang

---

## 📋 Ringkasan Masalah

Berdasarkan percakapan:

> "Sebenernya aku udah nyoba, gabisa kan dan capek. Aku dah joki ke temenku, tapi kenapa **ga sesuai hasilnyaaa**"

Ini menunjukkan bahwa:

1. ✅ Aplikasi Streamlit sudah jadi dan bisa diakses
2. ❌ Output/hasil perhitungan tidak sesuai harapan
3. ❓ Belum jelas apa yang "tidak sesuai" (jarak? rute? waktu?)

---

## 🔬 Hal yang Perlu Dicek

### 1. Validasi Input Data

```
[ ] Koordinat depot dan customer sudah benar?
[ ] Format time window konsisten (HH:MM atau menit)?
[ ] Demand customer dalam satuan yang sama dengan kapasitas?
[ ] Kecepatan kendaraan realistis?
```

### 2. Validasi Algoritma Sweep

```
[ ] Rumus sudut polar benar (atan2)?
[ ] Urutan customer sesuai sudut (ascending/descending)?
[ ] Depot dijadikan titik pusat (origin)?
```

### 3. Validasi Clustering

```
[ ] Kapasitas kendaraan tidak terlewati?
[ ] Customer yang tidak muat masuk ke kendaraan baru?
[ ] Semua customer masuk ke cluster?
```

### 4. Validasi ACS

```
[ ] Inisialisasi pheromone benar?
[ ] Rumus probabilitas transisi benar?
[ ] Local update dilakukan setelah setiap transisi?
[ ] Global update dilakukan untuk rute terbaik?
[ ] Time window dicek setelah rute terbentuk?
```

### 5. Validasi RVND

```
[ ] Operator neighborhood diimplementasi dengan benar?
[ ] Improvement diterima jika jarak lebih pendek?
[ ] Time window tetap valid setelah swap?
```

---

## 🧪 Test Case Manual

### Skenario Sederhana (3 Customer)

```
Depot: (0, 0)
Time Window Depot: 08:00 - 18:00

Customer 1: (3, 4)  - Demand: 20 - TW: 09:00-12:00
Customer 2: (6, 0)  - Demand: 30 - TW: 10:00-14:00
Customer 3: (3, -4) - Demand: 25 - TW: 11:00-16:00

Kendaraan: Kapasitas 80, Kecepatan 30 km/jam
```

### Perhitungan Manual

**Sudut Polar:**

- C1: atan2(4, 3) = 53.13°
- C2: atan2(0, 6) = 0°
- C3: atan2(-4, 3) = -53.13° = 306.87°

**Urutan Sweep:** C2 → C1 → C3

**Jarak:**

- D-C2: 6 km
- C2-C1: √[(3-6)² + (4-0)²] = √25 = 5 km
- C1-C3: √[(3-3)² + (-4-4)²] = 8 km
- C3-D: 5 km
- **Total: 24 km**

**Cek Kapasitas:**
20 + 30 + 25 = 75 < 80 ✅

**Cek Time Window:**

- Berangkat 08:00
- Tiba C2: 08:00 + (6/30\*60) = 08:12 → Tunggu sampai 10:00
- Tiba C1: 10:00 + (5/30\*60) = 10:10 ✅ (09:00-12:00)
- Tiba C3: 10:10 + (8/30\*60) = 10:26 ✅ (11:00-16:00) → Tunggu 11:00
- Kembali: 11:00 + (5/30\*60) = 11:10 ✅

---

## 📝 Pertanyaan untuk Klarifikasi

Untuk membantu memperbaiki program, perlu dijawab:

1. **Apa yang "tidak sesuai"?**
   - [ ] Total jarak terlalu jauh/dekat?
   - [ ] Rute tidak masuk akal?
   - [ ] Ada customer yang tidak terlayani?
   - [ ] Time window dilanggar?
   - [ ] Kendaraan over capacity?

2. **Apakah ada data contoh?**
   - Data customer (koordinat, demand, time window)
   - Data kendaraan (kapasitas, jumlah unit)
   - Parameter ACS yang digunakan

3. **Apakah ada hasil yang diharapkan (expected output)?**
   - Bisa bandingkan output program vs expected

4. **Source code program ada?**
   - Perlu review untuk identifikasi bug

---

## 💡 Rekomendasi Perbaikan

### Quick Wins

1. **Print/Log setiap tahap** - Tampilkan hasil Sweep, Clustering, ACS, RVND
2. **Validasi input** - Pastikan data tidak ada yang negatif/invalid
3. **Test case kecil** - Mulai dari 3-5 customer untuk validasi manual

### Medium Effort

1. **Visualisasi rute** - Plot koordinat dan rute pakai Matplotlib
2. **Unit test** - Test setiap fungsi (sudut polar, jarak, dll)
3. **Logging** - Catat pheromone update, probabilitas, dll

### Full Rewrite (jika perlu)

1. **Modularisasi kode** - Pisahkan sweep, clustering, ACS, RVND
2. **Dokumentasi** - Tambahkan docstring di setiap fungsi
3. **Config file** - Parameter ACS di file terpisah

---

## 📞 Next Steps

1. ✅ Dokumentasi sudah dibuat (README.md)
2. ⏳ Dapatkan source code program
3. ⏳ Dapatkan data contoh
4. ⏳ Review dan identifikasi bug
5. ⏳ Perbaiki dan test

---

_Dokumen ini adalah catatan analisis untuk membantu proses debugging._
