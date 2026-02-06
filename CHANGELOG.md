# 📋 Changelog - Route Optimization Dashboard

Dokumentasi perubahan yang dilakukan pada proyek.

---

## [2026-02-04] UI Beautification & Bug Fixes

### 🎨 UI Beautification (app.py)

**Waktu**: 2026-02-04 ~23:26 WIB

| Perubahan                | Detail                                                                                                                                  |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| Header Baru              | Judul "🚚 Sistem Optimasi Rute Pengiriman" dengan subtitle MFVRPTW dan pipeline algoritma                                               |
| Tab dengan Emoji         | 📍 Input Titik, 📋 Input Data, 📊 Hasil Optimasi, 🗺️ Visualisasi Rute, 🔬 Proses Optimasi                                               |
| Sidebar Bahasa Indonesia | Panel Kontrol, Validasi Input, Jalankan Optimasi, Status, Tentang Aplikasi                                                              |
| Footer Copyright         | "© 2025 Nabilah Eva Nurhayati \| Mahasiswa Program Studi Matematika, Universitas Negeri Malang \| Tugas Akhir Optimasi Rute Pengiriman" |

---

### 📊 Sample Data (sample_data.json)

**Waktu**: 2026-02-04 ~23:33 WIB

**File**: `gui/data/sample_data.json`

**Isi**: 1 Depot, 10 Customer (dari dokumen Word), 3 Fleet (A=60, B=100, C=150)

---

### 🐛 Bug Fix: Distance Matrix Decimal Places

**Waktu**: 2026-02-04 ~23:47 WIB

**Masalah**: Distance matrix hanya 2 digit desimal, butuh 5.

**Fix**: `format="%.5f"`, `step=0.00001`

---

### 🐛 Bug Fix: Column 5 Tidak Bisa Diedit

**Waktu**: 2026-02-04 ~23:52 WIB

**Masalah**: Kolom terakhir disabled karena loop nested yang salah.

**Root Cause**: Loop `for i, j` meng-overwrite column_config. Kolom 5 tertimpa dengan "disabled" karena iterasi terakhir i=j=5.

**Fix**: Ubah ke single loop `for j` saja. Diagonal di-handle di extraction logic.

---

### 🐛 Bug Fix: Odd-Edit-Rejection (Edit Ganjil Ditolak)

**Waktu**: 2026-02-04 ~23:56 WIB

**Masalah**: Edit ke-1, 3, 5, ... selalu ditolak/hilang. Hanya edit genap yang diterima.

**Root Cause**: DataFrame di-recreate dari `inputData` setiap Streamlit rerun, menyebabkan konflik dengan widget state.

**Fix**:

```python
# Cek dulu apakah key sudah ada di session_state
if editor_key not in st.session_state:
    # HANYA inisialisasi jika belum ada
    initial_df = pd.DataFrame(...)
else:
    # Gunakan data EXISTING dari session state!
    initial_df = st.session_state[editor_key]

edited = st.data_editor(initial_df, key=editor_key, ...)
```

---

### 🎨 Layout Fix: Section Aksi

**Waktu**: 2026-02-04 ~23:56 WIB

**Masalah**: File uploader menampilkan area drop besar yang mengganggu.

**Fix**:

- 2 tombol (Simpan Progres, Lanjutkan Proses) dalam 2 kolom
- File uploader dipindah ke expander "📂 Muat Progres dari File"
- Setelah load file, clear editor keys lalu rerun

---

## [2026-02-05] Optimization Bug Fixes & Cleanup

### 🧹 Cleanup: **pycache** Folders

**Waktu**: 2026-02-05 ~08:24 WIB

**Aksi**: Hapus semua folder `__pycache__` di:

- `Program/__pycache__`
- `Program/gui/__pycache__`
- `Program/gui/tabs/__pycache__`

---

### 🐛 Bug Fix: Data Editor Session State

**Waktu**: 2026-02-05 ~08:24 WIB

**Masalah**: Error `'list' object has no attribute 'items'` saat menggunakan editor.

**Root Cause**: `st.session_state[editor_key]` menyimpan widget state internal (dict/list), BUKAN DataFrame.

**Fix**:

- Pisahkan key untuk DATA storage (`customer_tw_data_{n}`, `distance_matrix_data_{size}`)
- Buat DataFrame dari data tersebut setiap render
- Update data dan inputData bersamaan setelah edit

---

### 🐛 Bug Fix: IndexError pada Optimization

**Waktu**: 2026-02-05 ~08:24 WIB

**Masalah**: `IndexError: list index out of range` di fungsi `academic_nearest_neighbor`.

**Root Cause**: Distance matrix menggunakan index posisi array (0=depot, 1=customer pertama), tapi kode mengakses dengan customer ID (bisa 1, 2, 10, dll).

**Fix**: Tambahkan mapping `id_to_idx` di semua fungsi yang mengakses distance_matrix:

- `academic_nearest_neighbor()` - lines 489-493 dan 527-534
- `evaluate_route()` - lines 937-940 dan 968-971

```python
# Build ID to matrix index mapping
id_to_idx = {0: 0}  # Depot is always at index 0
for idx, c in enumerate(dataset["customers"]):
    id_to_idx[c["id"]] = idx + 1

# Usage
current_idx = id_to_idx.get(current, current)
cid_idx = id_to_idx.get(cid, cid)
dist = distance_matrix[current_idx][cid_idx]
```

---

### 🐛 Bug Fix: IndexError in evaluate_route (Critical)

**Waktu**: 2026-02-05 ~09:15 WIB

**Masalah**: `IndexError: list index out of range` saat menjalankan optimisasi.

**Root Cause**: Customer ID tidak sama dengan matrix index. Matrix menggunakan posisi dalam list (depot=0, customer0=1, customer1=2), tapi kode menggunakan customer ID langsung.

**Fix di** `academic_replay.py`:

1. Tambah fungsi `build_id_to_idx_mapping()` untuk mapping ID → index secara konsisten
2. Update `evaluate_route` menggunakan mapping centralized
3. Update `academic_nearest_neighbor` menggunakan mapping centralized
4. Hapus semua `.get()` fallback yang tidak aman

---

### 🐛 Bug Fix: Upload JSON Tidak Mengisi Distance Matrix

**Waktu**: 2026-02-05 ~08:56 WIB

**Masalah**: Upload file JSON tidak menimpa distance matrix - hanya baris/kolom 0 yang terisi.

**Root Cause**: Pattern clear key di file uploader tidak match dengan pattern data key baru.

**Fix**: Update `keys_to_clear` untuk include SEMUA pattern:

- `customer_tw_data_*`
- `distance_matrix_data_*`
- `customer_tw_editor_*`
- `distance_matrix_editor_*`

---

### 🎨 Improvement: Tampilan Tabel ACS/RVND

**Waktu**: 2026-02-05 ~08:56 WIB

**Masalah**: Kolom-kolom tidak sesuai format skripsi:

- Service Time selalu 0.00
- Kolom "Improved" dengan icon ❌ tidak informatif
- Kolom "Neighborhood" menunjukkan "none"

**Fix**: Redesign tabel di `hasil.py` dan `academic_replay_tab.py`:

- Hapus kolom Service Time, Improved, Neighborhood
- Tambah kolom sesuai skripsi: Iterasi, Cluster, Jarak (km), Waktu Perjalanan, Kendaraan, Fungsi Objektif (Z)
- Semua label dalam Bahasa Indonesia

---

## File yang Dimodifikasi

| File                              | Perubahan                                                    |
| --------------------------------- | ------------------------------------------------------------ |
| `gui/app.py`                      | Header, tabs, sidebar, footer, CSS                           |
| `gui/tabs/input_data.py`          | Decimal places, column fix, odd-edit fix, layout, upload fix |
| `gui/tabs/hasil.py`               | Tabel ACS/RVND format skripsi                                |
| `gui/tabs/academic_replay_tab.py` | Tabel RVND inter/intra format skripsi                        |
| `gui/data/sample_data.json`       | Data sampel dari dokumen Word                                |
| `academic_replay.py`              | ID-to-index mapping untuk distance matrix                    |

### 🚀 Refactor & Final Polish (Dynamic Logic & Localization)

**Waktu**: 2026-02-05 ~10:00 WIB

**Fitur Utama**:

- **Dynamic Calculation**: Menghapus semua hardcoded values (Word doc). Clustering, ACS, RVND, dan Validasi sekarang 100% dinamis berdasarkan input user.
- **Merge Tab**: Visualisasi Rute digabung ke tab Hasil Akhir.
- **Localization**: UI full Bahasa Indonesia (Tabel, Alert, Status).
- **Vehicle Reassignment**: Fitur tetap ada tapi pesan diperjelas (Alasan kendaraan diganti jika kapasitas kurang).
- **Validasi Matematis**: Validasi rute memastikan constraint MFVRP terpenuhi (bukan mencocokkan kunci jawaban).

---

```

```
