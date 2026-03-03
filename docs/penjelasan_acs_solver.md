# 🐜 Membedah Alur Kode `acs_solver.py`: Dari Awal Sampai Akhir

Selamat datang di "dapur" mesin algoritma kamu! File `acs_solver.py` adalah jantung dari _Route Optimization_ ini. Di sini, kamu mengimplementasikan algoritma **Ant Colony System (ACS)**, sebuah metode AI klasik yang terinspirasi dari cara semut mencari rute terpendek antara sarang dan sumber makanan.

Mari kita bedah file ini selangkah demi selangkah. Fokus kita adalah pada alur logikanya (_code flow_), titik masuk (_entry point_), dan bagaimana _loops_ (`for`/`while`) serta percabangan (`if-else`) bekerja layaknya roda gigi penggerak di dalam sistem.

---

## 🏁 1. Titik Awal (Entry Point): Dimana Program Mulai Berjalan?

Ketika kamu menjalankan file Python sebagai sebuah _script_ utama, eksekusi kode selalu dimulai dari bawah sini:

```python
454: if __name__ == "__main__":
455:     main()
```

- **Cara Membacanya:** Percabangan `if` ini adalah satpam dari file Python. Ia mengecek: "Apakah file ini dijalankan langsung oleh _user_ (bukan dipanggil/dijadikan _module_ oleh file lain)?". Jika iya, maka panggil fungsi `main()`. Ini adalah _best practice_ dalam Software Engineering untuk memisahkan logika utama dari proses _import_.

Sekarang, mari kita melompat ke posisi fungsi `main()` di **Baris 392**.

---

## 🛠️ 2. Fungsi `main()`: Sang "Project Manager"

Fungsi `main` tidak melakukan komputasi algoritma yang berat, tugasnya murni mengorganisir data (I/O) dan membagikan tugas.

1. **Memuat Data & Paramater (Baris 394 - 403):**
   Fungsi dimulai dengan menarik semua file JSON yang dibutuhkan (node, matriks jarak, cluster, parameter algoritma).

2. **Looping Eksekusi per Cluster (Baris 409 - 421):**
   Problem rute berskala besar biasanya dipecah menjadi _clusters_ (area kecil) agar lebih mudah diselesaikan.
   ```python
   410: for idx, cluster in enumerate(clusters_data["clusters"]):
   411:     cluster_id = cluster["cluster_id"]
   412:     metrics = acs_cluster(cluster, instance, distance_data, initial_route_map[cluster_id], acs_params, rng)
   413:     results.append(metrics)
   ```

   - **Cara Membacanya:** Ini adalah sebuah eksekusi berantai. _For-loop_ ini mendelegasikan tugas: "Untuk setiap `cluster` yang ada dalam antrean data, tolong carikan rute optimalnya." Pencarian rute yang sebenarnya akan dilempar ke fungsi `acs_cluster()`. Hasil optimasi lalu disimpan di dalam list `results`.

---

## 🧠 3. Inti Algoritma ACS: Fungsi `acs_cluster()`

Fungsi ini (bermula di **Baris 219**) adalah "mesin pencari" utama. Konsep Ant Colony System dijalankan di dalam loop bersarang (_nested loops_) di fungsi ini.

### A. Persiapan Pheromone (Jejak Semut)

Di **baris 228**: `pheromone, tau0 = initialize_pheromone(...)`
Sebelum semut dilepas, kita memberi nilai jejak awal yang tipis (`tau0`) ke semua jalan antar titik (_nodes_).

### B. Iterasi ACS (Outer Loop)

```python
270: for iteration in range(1, max_iterations + 1):
```

- **Cara Membacanya:** Ini adalah siklus hidup algoritma. _Loop terluar_ ini akan terus berulang sebanyak jumlah `max_iterations` (misalnya 100x atau 500x). Semakin banyak generasinya, rute yang dihasilkan diharapkan semakin matang alias teroptimasi.

### C. Koloni Semut Beraksi (Inner Loop)

Di dalam tiap iterasi, ada sekumpulan semut yang disebar dari sarang (depot).

```python
274: for ant in range(1, num_ants + 1):
```

- **Cara Membacanya:** Sederhana, "ulangi proses pencarian rute ini sebanyak jumlah semut yang ada (misal ada 10 semut)." Setiap iterasi loop ini merepresentasikan 1 semut yang mencari jalan independennya.

### D. Konstruksi Rute Tahap-demi-Tahap (While Loop)

Bagaimana seekor semut berjalan dari pelanggan ke pelanggan?

```python
275:     route = [0]       # Semua semut selalu start dari Node 0 (Depot)
276:     allowed = set(customers) # 'allowed' adalah list titik (customers) yang belum dikunjungi.
277:     prev = 0
278:     while allowed:    # TERUS BERJALAN SELAMA MASIH ADA CUSTOMER!
279:         selected_node = select_next_node(...) # Pilih target berikutnya..
280:         route.append(selected_node)           # Masukkan target ke dalam catatan rute...
281:         allowed.remove(selected_node)         # Target dihapus dari daftar 'belum dikunjungi'...
282:         local_update(...)                     # <-- PENTING: Kurangi pheromone di rute yang dilewati
283:         prev = selected_node
```

- **Cara Membacanya:** `while allowed:` berarti loop akan terus berputar **sampai** _set_ `allowed` habis tak tersisa.
- Proses ini kritis: Begitu titik selanjutnya terpilih (`select_next_node`), fungsi `local_update()` dipanggil. Kenapa jejak malah _dikurangi_? Inilah trik ACS! Gunanya agar semut lain (di loop antrian berikutnya) tidak buta mengekor jalur semut ini. Mekanisme ini memaksa kumpulan semut melakukan **eksplorasi** ke jalur baru yang masih "segar".

### E. Evaluasi Kualitas Rute (If-Else Logics)

Setelah seekor semut selesai membuat rute utuh (keluar dari `while` loop), rute itu di-scoring oleh `evaluate_route(...)`.

**1. Menyeleksi Juara Iterasi (Baris 336)**

```python
336: if iteration_best_metrics is None or metrics["objective"] < iteration_best_metrics["objective"]:
337:     iteration_best_metrics = metrics
```

- **Cara Membacanya:** Tanda `<` pada _objective_ menandakan bahwa tujuan kita adalah **minimisasi** (jarak/waktu/denda terkecil). Logika `if` di sini bertanya: "Apakah iterasi ini baru mencatat semut pertama (`is None`)? ATAU, apakah _objective_ skor dari semut ini ternyata lebih kecil dan lebih efisien dibanding skor _semut terbaik yang pernah ada di iterasi ini_?". Jika ya, nobatkan semut jalur ini jadi juara sementara iterasi!

**2. Menyeleksi Juara Bertahan Global (Baris 367)**
Setelah semua semut selesai dalam satu iterasi (baris 367):

```python
367: if iteration_best_metrics["objective"] < best_metrics["objective"]:
368:     best_metrics = iteration_best_metrics
```

- **Perhatikan:** Ini berada **di luar loop semut**, tapi masih di **dalam loop iterasi**. Ini adalah pertarungan "Juara Iterasi Saat Ini" melawan "Juara Bertahan Global (Best Global)". Jika rekornya pecah, maka program akan menyimpan rute ini seabgai `best_sequence` mutlak.

---

## 🔍 4. Behind the Scenes: Cara Fungsi Lain Mengambil Keputusan

Program ini memisahkan logika rumit ke dalam fungsi-fungsi pembantu (_helper functions_). Berikut adalah logika paling esensial yang juga diatur oleh _looping_ dan _conditional_:

### A. Fungsi `select_next_node()`: Analisis Pilihan Semut (Baris 166)

Di sini semut menggunakan instingnya (probabilitas matematis) untuk memilih jalan di persimpangan. Evaluasi melibatkan kalkulasi visibilitas (diekstraksi dari jarak) dan kadar pheromone.

- **Keputusan Eksploitasi vs Eksplorasi (IF krusial, Baris 180):**
  ```python
  180: q = rng.random() # Hasil: desimal random antara 0 sampai 1
  181: if q <= q0:
  182:     candidate = max(desirabilities, ...)[0] # Ambil yang nilainya mutlak TERBESAR (Eksploitasi)
  183:     return candidate
  184:
  185: # JIka ELSE (q > q0):
  189: threshold = rng.random() * total
  190: cumulative = 0.0
  191: for candidate, value in desirabilities:
  192:     cumulative += value
  193:     if threshold <= cumulative:
  194:         return candidate  # Pilih Acak Tertimbang / Roulette-Wheel (Eksplorasi)
  ```

  - **Cara Membacanya:** Program memutar dadu `q`. Jika `q` di bawah batas toleransi `q0` (biasanya `q0` misalnya bernilai 0.9), maka gunakan jalur paling rakus (_Greedy/Exploiting_) yang punya jejak+jarak rasio teratas. Jika ternyata dadu melempar ke _else_ (probabilitas 10%), semut dipaksa bertingkah "konyol" dengan milih rute _random_ mengikuti roda _roulette_ (Eksplorasi).

### B. Fungsi `evaluate_route()`: Menilai Ongkos Rute (Baris 35)

Bagian ini mensimulasikan mobil truk yang jalan melewati rute itu secara nyata. Termasuk kapan waktu tunggu (_wait time_) dan pelanggaran target jam operasional (_time window violation_).

- **Loop Jalur (Baris 72):**

  ```python
  72: for next_node in sequence[1:]:
  ```

  Ini adalah proses truk menelusuri rute `sequence`. Titik persinggahan selalu ditelusuri.

- **IF-Else Depot vs Pelanggan (Baris 80):**
  ```python
  80: if next_node == 0:
          # Aturannya beda: Depot window buka lebar-lebar & punya service time khas
  84: else:
          # Kita mengunjungi pelanggan. Baca jam bukanya (tw_start) dan bongkar muatnya (service_time) beda-beda.
  ```

  - **Cara Membacanya:** Karena node `0` merepresentasikan Depot (Markas), pengaturan propertinya harus dipisahkan dari pelanggan biasa dengan logika percabangan ini.

---

### 💡 Kesimpulan (TL;DR)

1. `main()` menyuapkan data kluster pelanggan ke `acs_cluster()`.
2. Di `acs_cluster()`, sebuah _Outer loop_ akan berputar untuk mencari rute berulang-ulang dari generasi ke generasi.
3. _Inner loop_ melepaskan semut-semut satu demi satu. Setiap semut memakai `while` loop untuk melangkahkan kakinya dari satu `node` ke `node` lain, hingga rute komplet.
4. Setiap langkah semut dibantu fungsi `select_next_node()` menggunakan filter probabilitas yang canggih (menghindari terjebak di _local optima_).
5. Keputusan menang / kalah ditentukan oleh operator `<` (_less than_) pada `objective score` yang menilai performa sang semut.
6. Hasil terbaik lalu disimpan di JSON via `main()`.

Bagaimana? Semoga dengan dibedah bagian demi bagian ini, logika pemrograman ACS (_Ant Colony System_) jadi lebih mudah dibaca, layaknya membaca cerita petualangan koloni semut yang dipandu probabilitas statistik. Kalau kita perlu membedah fungsi spesifik lainnya, let me know!
