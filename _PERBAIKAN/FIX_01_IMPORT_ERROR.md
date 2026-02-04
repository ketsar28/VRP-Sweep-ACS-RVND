# 🔧 FIX #01: Import Error - `run_academic_replay`

## 📍 Lokasi Bug

| Item      | Detail                                                  |
| --------- | ------------------------------------------------------- |
| **File**  | `Program/gui/tabs/academic_replay.py`                   |
| **Baris** | 714-715                                                 |
| **Error** | `ImportError: cannot import name 'run_academic_replay'` |

---

## 🔍 Penyebab Bug

**Clash nama modul!**

Python bingung karena ada 2 file dengan nama mirip:

1. `Program/academic_replay.py` ← File yang punya function `run_academic_replay`
2. `Program/gui/tabs/academic_replay.py` ← File yang mau import function itu

Ketika kode mencoba `from academic_replay import run_academic_replay`, Python malah import dari file **dirinya sendiri** (tabs/academic_replay.py), bukan dari file yang benar (Program/academic_replay.py).

---

## 🛠️ Cara Perbaiki

### Langkah 1: Buka File

```
Program/gui/tabs/academic_replay.py
```

### Langkah 2: Cari Baris 713-716

**KODE LAMA (SALAH):**

```python
# Line 713-716 (sekitar sini)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from academic_replay import run_academic_replay
```

### Langkah 3: Ganti dengan Kode Baru

**KODE BARU (BENAR):**

```python
# Line 713-716 (ganti dengan ini)
import sys
import importlib.util

# Load module dengan path eksplisit untuk hindari clash nama
_academic_module_path = Path(__file__).resolve().parent.parent.parent / "academic_replay.py"
_spec = importlib.util.spec_from_file_location("academic_replay_main", _academic_module_path)
_academic_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_academic_module)
run_academic_replay = _academic_module.run_academic_replay
```

---

## 📝 Penjelasan Kode Baru

| Baris                       | Fungsi                                                    |
| --------------------------- | --------------------------------------------------------- |
| `importlib.util`            | Library Python untuk load modul secara dinamis            |
| `spec_from_file_location`   | Buat spesifikasi modul dari path file eksplisit           |
| `"academic_replay_main"`    | Nama alias modul (beda dari "academic_replay" yang clash) |
| `_academic_module_path`     | Path absolut ke file `Program/academic_replay.py`         |
| `run_academic_replay = ...` | Ambil function dari modul yang sudah di-load              |

---

## ✅ Setelah Perbaiki

1. Save file
2. Refresh browser Streamlit (F5)
3. Buka tab **Academic Replay**
4. Klik tombol **🚀 Run Academic Replay**
5. Seharusnya sudah tidak ada error import

---

## ⚠️ Jika Masih Error

Kalau masih error, coba cara alternatif ini:

**ALTERNATIF (lebih simple tapi kurang elegan):**

Rename file `Program/gui/tabs/academic_replay.py` menjadi `Program/gui/tabs/academic_replay_tab.py`

Lalu update import di `Program/gui/app.py` line 66-70.

---

_Fix #01 selesai. Lanjut ke FIX_02 untuk input jarak._
