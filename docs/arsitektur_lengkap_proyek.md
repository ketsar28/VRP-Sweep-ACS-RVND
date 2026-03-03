# 🗺️ Peta Navigasi & Bedah Kode: MFVRPTW Route Optimization

Halo! Seperti yang Paman janjikan, di dokumen ini kita akan **membongkar seluruh arsitektur proyek ini dari hulu ke hilir**. Paman akan pandu kamu langkah demi langkah: dari mana aplikasi ini mulai menyala, bagaimana ia melempar data, hingga bagaimana mesin algoritma di belakang layar berputar.

Mari kita mulai penjelajahannya!

---

## 🚪 1. Pintu Masuk (Entry Point): Dimana Semua Bermula?

Jika pengguna menekan _shortcut_ `.exe` atau menjalankan perintah Terminal, eksekusi kode bermula dari **`run_streamlit_app.py`** yang ada di folder terluar (root).

**Fokus Logika `run_streamlit_app.py`:**
Fungsi utamanya adalah menjembatani (membuat _wrapper_) antara antarmuka Desktop (`pywebview`) dengan _server_ Web (`streamlit`).

```python
100: st_thread = threading.Thread(target=run_streamlit, ...)
105: st_thread.start()
```

- **Cara kerja:** Program ini "membelah diri" (_threading_). Satu jalur kerja (di _background_) menyalakan _server_ Streamlit (algoritma UI kita), sedangkan jalur kerja utama menunggu 5 detik, kemudian membuka _Window Desktop_ bergaya aplikasi asli menggunakan `webview`.

Setelah Streamlit menyala, ia diarahkan untuk membaca **`Program/gui/app.py`**. File ini memuat struktur tab yang terlihat oleh _user_: `input_titik`, `input_data`, hasil visualisasi, dsb. Di dalam _logic_ tombol UI inilah, mesin algoritma dijalankan.

---

## ⚙️ 2. Pabrik Data (Data Preparation)

Sebelum mencari rute tercepat, aplikasi butuh tahu seberapa jauh jarak antar pelanggan.

### File: `distance_time.py`

Tugasnya simpel tapi krusial: mengubah titik koordinat spasial (X, Y) menjadi matriks jarak (seperti tabel tarif Gojek).

- **Looping Jarak (`compute_distance_matrix`)**
  ```python
  44: for i in range(size):
  45:     for j in range(i + 1, size):
  46:         distance = euclidean_distance(nodes[i], nodes[j])
  47:         matrix[i][j] = distance
  48:         matrix[j][i] = distance
  ```

  - **Cara membacanya:** Ini adalah _Nested For-Loop_ khas untuk membuat Matriks Segitiga (_Triangular Matrix_). Kenapa `j` dimulai dari `i + 1`? Supaya kode tidak repot-repot menghitung Jarak dari A ke A (pasti 0), dan Jarak A ke B sama persis dengan jarak B ke A. Perulangan dibuat efisien (setengah segitiga saja), lalu disalin secara cermin (_mirrored_) ke `matrix[j][i]`.

---

## 🎯 3. Mesin Logika Tahap 1: Pengelompokan & Rute Buta

### File: `sweep_nn.py`

Algoritma ini menjawab pertanyaan: "Pelanggan mana masuk truk yang mana?" (Sweep) lalu "Urutan pelanggannya di-gimana-in biar lumayan masuk akal?" (Nearest Neighbor).

#### A. Algoritma Sweep (Sapu Melingkar)

- **Logika Penyapuan:**
  Pelanggan diurutkan berdasarkan **sudut polar** (`math.atan2`) dari koordinat markas (Depot). Bayangkan piringan jam dinding tersapu jarum dari arah jam 12 ke arah jam 12 lagi.
- **Proses Pemuatan Kendaraan (_While Loop_)**
  ```python
  85: while unassigned:
  94:     while i < len(unassigned):
  98:         if current_demand + customer["demand"] <= max_capacity:
  99:             current_customers.append(customer)
  100:             current_demand += customer["demand"]
                  # ... pelanggan dihapus dari list unassigned ...
  127:         else:
  128:             i += 1
  ```

  - **Cara membacanya:** Program ini bagaikan sopir yang keliling searah jarum jam (_while_ terluar) berhenti di depan rumah tiap pelanggan (_while/for_ terdalam). Di sebuah persimpangan (`if`), sopir bertanya: "Apakah muatan kardus barusan bikin kapasitas truk saya blenk? (`<= max_capacity`)". Jika MASIH MUAT, masukkan! Jika PENUH (`else`), program melompat mengabaikan rumah itu dan bergerak melihat rumah tetangganya yang muatannya muat. Jika sama sekali tidak ada yang muat, truk berangkat, dan proses _looping_ menunjuk mobil kosong _(fleet)_ selanjutnya!

#### B. Nearest Neighbor (Cari Tetangga Terdekat)

Setelah tiap truk tahu pelanggan siapa saja yang harus dihampiri (dari Algoritma Sweep), truk akan membuat rute awal yang agak "bodoh" tapi logis.

- Program berputar di `while unvisited:`
- Selama belum semua tempat dijenguk, program melakukan _For-Loop_ mencari `min_dist` dari titik ia berada sekarang.
- Begitu ketemu, titik itu digabungkan ke catatan rute (`route_sequence.append(nearest)`), lalu pencarian dimulai lagi dari titik yang baru dituju.

---

## 🐜 4. Mesin Logika Tahap 2: Algoritma Semut (Ant Colony System)

### File: `acs_solver.py`

_(Dokumen detail secara spesifik mengenai ACS ini sudah Paman buatkan terpisah di `penjelasan_acs_solver.md`, namun ini ringkasannya):_

Rute dari `sweep_nn` (Tahap 1) tadi masih jelek nilainya. Kita letakkan ribuan semut simulasi dalam `acs_solver.py` dengan _Nested Loops_: `Generate` > `Semut per Gen` > `Pencarian Titik` menggunakan perhitungan _pheromone_.

Jika jarak tempuh bagus, jejak pheromone (_array update_) ditebalkan lewat _Global Update_. Jika buruk, jejak dibiarkan menguap begitu saja.

---

## 🔬 5. Mesin Logika Akhir: Evaluasi & Pertukaran (Local Search)

### File: `rvnd.py` (Random Variable Neighborhood Descent)

Ini file terbesar di mana program mencoba bertingkah seperti mandor jenius yang mencoba "Menukar-Nukar Urutan" demi memangkas sedikit detik keterlambatan terakhir tanpa menambah biaya.

Terdapat fungsi `evaluate_route` yang mirip dengan yang ada di ACS, di mana ia menelusuri tiap array lokasi menggunakan `for` loop untuk membandingkan Arrival Time (Kapan tiba) vs Time Window (Syarat jam buka).

```python
# Potongan logika di dalam RVND: (Jika pelanggan tidak dilayani karena Telat Parah)
wait_time += max(0, tw_start - (current_time + travel_matrix))
violation += max(0, (current_time + travel_matrix) - tw_end)
```

RVND menggunakan puluhan operator pencarian (misal `two_opt` (memutar balik sub-rute), `reinsertion` (mencabut satu pelanggan dipindah ke depan)). Di dalam metode ini terdapat banyak sekali validasi `if-else` untuk mengecek _feasibility_, alias "Apakah pertukaran urutan ini malah nabrak jam operasional secara fatal atau melebihi kapasitas?"

---

## 📊 6. Simpulan Pelaporan

### File: `final_integration.py` / `academic_replay.py`

Tahap terakhir proyek ini tidak lagi pusing soal pengoptimalan. Perannya bertindak murni sebagai "Kurator Laporan Keuangan & Teknikal".

- Ia menarik _array routes_ JSON akhir yang dimuntahkan `rvnd.py`.
- Menjalankan validasi ketat melintasi sebuah `for-loop` (Apakah _constraint_ kapasitas benar-benar terpenuhi? Apakah Matriks Jarak tersimpan utuh dan nol di urutan diagonal?)
- Mengkalkulasi nilai rupiah (`fixed_cost` dan variabel Rp/km).
- Menyimpan semua ini di folder `docs/final_summary.md` dan `final_solution.json` yang mana nanti isinya dibaca kembali oleh antar-muka _Graphical User Interface_ (GUI) Streamlit di tab paling akhir.

---

### Meringkas Filosofi Kode Ini (Penting Untuk Presentasi):

1. **Kenapa ada banyak file Python?** Ini namanya prinsip _Modularity_ di Software Engineering. Alih-alih satu file 10.000 baris yang memusingkan, kamu memisahkan fungsi spesifik ke modulnya masing-masing.
2. **Semua diikat lewat manipulasi List/Array dan Dictionary (JSON).** Output dari program A akan selalu menjadi Dictionary input untuk program B. File-file `.json` yang bertebaran di folder `data/` adalah "darah" dari aplikasi ini yang membawakan pesan antar mesin.
3. **If-Else = Keputusan, For-Loop = Usaha.** Perhatikan bahwa _for/while logic_ selalu mencerminkan kerja keras menguli data (mencari rute terkecil, mengeksekusi semut), tapi _if-else_ adalah yang mengambil vonis cerdas (apakah jarak ini lebih kecil riilnya? Apakah jam ini melanggar _time window_?).

_Selamat membedah kode! Jika Paman dosen pengujinya, Paman akan sangat bersemangat melihat mahasiswanya memahami alur yang sangat dalam ini._ 👨‍💻🎓
