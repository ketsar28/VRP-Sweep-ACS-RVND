# 📝 Rangkuman Requirement & Catatan Meeting (DETAIL V2.0)

File ini berisi rangkuman kebutuhan sistem yang SANGAT MENDETAIL berdasarkan diskusi dengan klien. Dokumen ini akan menjadi acuan saat pengembangan fitur agar tidak ada salah paham.

---

## 🚀 BAGIAN 1: Requirement Fitur Baru (Wajib)

### 1. Input Time Window Customer (UI)

- **Kondisi Saat Ini:**
  - Time Window (Jam Buka - Tutup) customer tertulis baku (hardcoded) di dalam script Python (`academic_replay.py`).
  - User tidak punya cara untuk mengubah jam buka toko customer lewat Dashboard Streamlit.
- **Masalah:**
  - Jika user ingin mensimulasikan kasus nyata di mana toko tutup lebih cepat atau buka lebih siang, user tidak bisa melakukannya.
- **Solusi Detail:**
  - Di Tab **"Input Data"** -> Tabel **"Permintaan Customer"**, tambahkan 2 kolom baru:
    1.  **`Jam Buka`** (Format: Time Picker / Text jam:menit, misal `08:00`). Default: `08:00`.
    2.  **`Jam Tutup`** (Format: Time Picker / Text jam:menit, misal `17:00`). Default: `17:00`.
  - **Validasi:** Pastikan `Jam Buka` < `Jam Tutup`.
- **Tujuan:** Agar algoritma bisa menghitung penalti keterlambatan berdasarkan inputan user yang spesifik.

### 2. Input Parameter ACS (UI)

- **Kondisi Saat Ini:**
  - Parameter penting algoritma ACS (Ant Colony System) disembunyikan di dalam kode Python.
  - Nilai default: `Alpha=0.1`, `Beta=2.0`, `Rho=0.1`, `Semut=10`, `Iterasi=50`.
- **Masalah:**
  - Klien minta parameter ini bisa "di-adjust" (diubah-ubah) untuk melihat pengaruhnya terhadap hasil rute.
- **Solusi Detail:**
  - Tambahkan panel **"Konfigurasi Parameter ACS"** (bisa di Sidebar atau di atas tombol Run).
  - Isi Panel:
    1.  **`Alpha (α)`**: Slider 0.0 - 1.0 (Mengatur seberapa penting "Jejak Pheromone").
    2.  **`Beta (β)`**: Slider 1.0 - 5.0 (Mengatur seberapa penting "Jarak/Waktu").
    3.  **`Gamma (γ)`**: Slider 0.0 - 100.0 (Mengatur denda pelanggaran Time Window).
    4.  **`Rho (ρ)`**: Slider 0.0 - 1.0 (Kecepatan penguapan pheromone).
    5.  **`Jumlah Semut`**: Input Angka (Min 1, Rekomendasi 10).
    6.  **`Jumlah Iterasi`**: Input Angka (Min 1, Rekomendasi 50-100).
  - Sediakan tombol **"Reset ke Default"** agar user bisa balik ke settingan awal dengan mudah.

### 3. Mode Optimasi Dinamis (Dynamic Calculation)

- **Kondisi Saat Ini:**
  - Mode "Academic Replay" memaksakan penggunaan 10 data customer dari dokumen skripsi, meskipun user menginput 3 customer, 5 customer, atau 100 customer.
  - Hasil clustering dan rute selalu sama (statis).
- **Masalah:**
  - Aplikasi tidak bisa dipakai untuk kasus nyata atau data eksperimen lain. Klien bingung kenapa inputannya diabaikan.
- **Solusi Detail:**
  - Ubah logika tombol **"Run Optimization"**:
    - **HAPUS** ketergantungan pada variabel hardcoded `WORD_CLUSTERS` dan `WORD_CUSTOMERS`.
    - **GUNAKAN** data real-time dari tabel input user.
    - Jalankan algoritma **Sweep Clustering** secara _real_ (hitung ulang sudut polar setiap titik koordinat user -> urutkan -> bagi cluster berdasarkan kapasitas).
    - Jalankan algoritma **ACS** pada _setiap cluster_ yang terbentuk secara dinamis.

---

## 🛠️ BAGIAN 2: Perbaikan Logika

### 4. Isu Perhitungan Jarak (Distance Matrix)

- **Kondisi Saat Ini:**
  - Sistem menghitung jarak menggunakan rumus Matematika (Euclidean Distance) yang akurat.
  - Dokumen Skripsi menggunakan jarak input manual (yang mungkin hasil pembulatan/ukur kasar), sehingga ada selisih angka (misal 16.51 vs 13.35).
- **Solusi Detail:**
  - Sistem akan menjadikan **Tabel "Jarak Antar Titik"** di UI sebagai sumber kebenaran mutlak (Source of Truth).
  - Jika user menginput angka jarak manual di tabel itu, sistem **AKAN MEMAKAI ANGKA ITU**, dan mengabaikan rumus matematika.
  - Ini menjawab kebutuhan klien: _"Jarak itu udah ada input di awal kan, matriksnya."_

### 5. Logika "No Vehicle" (Kapasitas vs Waktu)

- **Kondisi Saat Ini:**
  - Jika muatan truk cukup, TETAPI waktu pelayanan agak lama (lembur sedikit lewat jam kerja supir), sistem langsung memvonis **"Gagal / No Vehicle"**.
- **Masalah:**
  - Terlalu ketat. Klien ingin prioritasnya adalah **KAPASITAS** dulu. Asal muat, jalan dulu. Urusan lembur belakangan.
- **Solusi Detail:**
  - Ubah logika filter kendaraan:
    - **Cek 1 (Wajib):** Kapasitas muat? Jika TIDAK -> Gagal. Jika YA -> Lanjut.
    - **Cek 2 (Opsional/Soft):** Waktu cukup? Jika Cukup -> Status "OK". Jika Lembur -> Status "Warning: Lembur" (Tapi **TETAP DITERIMA/ASSIGNED**, jangan ditolak).

---

## ❓ BAGIAN 3: Pertanyaan Google Meet (Konteks Lengkap)

Gunakan naskah ini agar pertanyaan terdengar cerdas, terstruktur, dan beralasan.

### 1. Konteks TIME WINDOW (Jam Operasional)

**Latar Belakang:** Saat ini jam buka toko customer dikunci di sistem. User tidak bisa ubah.
**Pertanyaan ke Klien:**

> "Kak, nanti di tabel input customer, saya tambahkan kolom 'Jam Buka' dan 'Jam Tutup' Toko ya.
> **Pertanyaannya:** Apakah setiap customer WAJIB diisi jam bukanya?
>
> - Kalau **WAJIB**: Berarti nanti saya set default '08:00 - 17:00' biar user gak capek ngetik.
> - Kalau **TIDAK WAJIB**: Berarti kalau dikosongin, sistem akan anggap tokonya buka 24 jam (bebas kirim kapan aja).
>   Kakak preferensi yang mana?"

### 2. Konteks NO VEHICLE (Prioritas Pemuatan)

**Latar Belakang:** Sistem sekarang menolak muatan kalau waktunya mepet jam pulang supir, padahal truk masih kosong.
**Pertanyaan ke Klien:**

> "Kak, sistem sekarang itu sangat ketat soal waktu. Kalau muatan masih cukup, tapi pengirimannya bikin supir lembur dikit (misal pulang jam 17:15, padahal jam kerja sampai 17:00), sistem langsung menolak (Status: No Vehicle).
> **Pertanyaannya:** Apakah boleh saya longgarkan aturannya?
> Jadi prioritas utama adalah **'YANG PENTING MUAT'** (Kapasitas). Urusan waktu kalau lewat dikit-dikit, sistem akan tetap izinkan jalan (statusnya 'Warning/Lembur' saja, bukan 'Gagal').
> Setuju tidak kak kalau dilonggarkan begitu?"

### 3. Konteks SWEEP DINAMIS (Beda Hasil dengan Skripsi)

**Latar Belakang:** Jika user input data baru (beda lokasi), hasil clustering pasti beda dengan skripsi.
**Pertanyaan ke Klien:**

> "Kak, nanti kan user bisa input titik customer sendiri secara bebas.
> Kalau user menggeser titik lokasi customer (beda koordinat dengan skripsi), otomatis hasil pembagian wilayah (Cluster)-nya akan berubah dan **BEDA** dengan hasil yang di skripsi.
> **Konfirmasi:** Itu wajar dan boleh kan kak? Karena sistem kan menghitung ulang secara real-time berdasarkan lokasi baru, bukan 'mencocokkan paksa' dengan kunci jawaban skripsi lagi."

### 4. Konteks TABEL VALIDASI (Hapus/Simpan?)

**Latar Belakang:** Tabel yang isinya centang hijau/merah itu fungsinya mencocokkan hasil codingan dengan hasil manual skripsi.
**Pertanyaan ke Klien:**

> "Kak, tabel 'Validasi vs Dokumen Word' (yang isinya centang hijau/merah) itu kan gunanya khusus buat ngecek data skripsi (Replay Mode).
> **Pertanyaannya:** Nanti kalau user pakai Mode Dinamis (Input Data Sendiri), tabel validasi ini sebaiknya **SAYA SEMBUNYIKAN** saja ya?
> Soalnya kan kalau datanya baru, nggak ada 'Kunci Jawaban Manual'-nya buat divalidasi. Validasi merah-merah itu nanti malah bikin bingung user kalau muncul terus."

---

_Dibuat otomatis oleh Assistant - 4 Feb 2026_
