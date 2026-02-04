# 🔧 FIX #03: KeyError 'A' - Vehicle ID Mismatch

## 📍 Lokasi Bug

| Item            | Detail                       |
| --------------- | ---------------------------- |
| **File**        | `Program/academic_replay.py` |
| **Baris Error** | 1099 dan 1147                |
| **Error**       | `KeyError: 'A'`              |

---

## 🔍 Penyebab Bug

**Mismatch antara format ID kendaraan!**

### Di Input Data (Tab UI):

User memasukkan:

- "Vehicle A" (kapasitas 60)
- "Vehicle B" (kapasitas 100)
- "Vehicle C" (kapasitas 150)

### Di Kode `academic_replay.py`:

Kode mencari:

```python
fleet = {f["id"]: f for f in dataset["fleet"]}
# Ini membuat dictionary: {"Vehicle A": {...}, "Vehicle B": {...}, ...}

cap_a = fleet[route_a["vehicle_type"]]["capacity"]
# Ini mencari fleet["A"] ← TIDAK ADA! Yang ada fleet["Vehicle A"]
```

### Masalahnya:

- `route["vehicle_type"]` = `"A"` (dari cluster result)
- Tapi `fleet` dictionary key-nya = `"Vehicle A"`
- Mismatch! → KeyError

---

## 🔎 Root Cause Analysis

Bug ini terjadi karena **2 tempat berbeda menggunakan format ID berbeda**:

| Komponen                    | Format ID                             |
| --------------------------- | ------------------------------------- |
| `input_data.py` (UI)        | "Vehicle A", "Vehicle B", "Vehicle C" |
| `sweep_nn.py` (Clustering)  | "A", "B", "C" (kemungkinan hardcoded) |
| `academic_replay.py` (RVND) | Expect match dengan sweep result      |

---

## 🛠️ Ada 2 Opsi Fix

### OPSI A: Ubah Nama Kendaraan di UI (SIMPEL)

Di tab **Input Data**, ganti nama kendaraan:

| Lama      | Baru |
| --------- | ---- |
| Vehicle A | A    |
| Vehicle B | B    |
| Vehicle C | C    |

**Langkah:**

1. Buka tab "Input Data"
2. Hapus kendaraan yang ada (klik tombol hapus 🗑️)
3. Tambah kendaraan baru dengan nama:
   - Nama: `A`, Kapasitas: 60, Unit: 2, dll
   - Nama: `B`, Kapasitas: 100, Unit: 2, dll
   - Nama: `C`, Kapasitas: 150, Unit: 1, dll

---

### OPSI B: Ubah Kode (PROPER FIX)

Jika ingin tetap pakai nama "Vehicle A", "Vehicle B", dll, ubah kode berikut:

**File:** `Program/academic_replay.py`

**Baris 1099 (dan sekitarnya):**

```python
# KODE LAMA:
fleet = {f["id"]: f for f in dataset["fleet"]}

# KODE BARU (normalize ID):
def normalize_vehicle_id(vid):
    """Normalize vehicle ID to match between components."""
    if vid.startswith("Vehicle "):
        return vid.replace("Vehicle ", "")
    return vid

fleet = {}
for f in dataset["fleet"]:
    # Simpan dengan KEDUA format ID
    fleet[f["id"]] = f
    fleet[normalize_vehicle_id(f["id"])] = f
```

**Baris 1147-1148 (dalam function `academic_rvnd_inter`):**

```python
# KODE LAMA:
cap_a = fleet[route_a["vehicle_type"]]["capacity"]
cap_b = fleet[route_b["vehicle_type"]]["capacity"]

# KODE BARU (dengan fallback):
def get_vehicle_capacity(fleet, vehicle_type):
    """Get capacity with fallback for different ID formats."""
    if vehicle_type in fleet:
        return fleet[vehicle_type]["capacity"]
    # Try with "Vehicle " prefix
    full_id = f"Vehicle {vehicle_type}"
    if full_id in fleet:
        return fleet[full_id]["capacity"]
    # Try without "Vehicle " prefix
    short_id = vehicle_type.replace("Vehicle ", "")
    if short_id in fleet:
        return fleet[short_id]["capacity"]
    raise KeyError(f"Vehicle type '{vehicle_type}' not found in fleet")

cap_a = get_vehicle_capacity(fleet, route_a["vehicle_type"])
cap_b = get_vehicle_capacity(fleet, route_b["vehicle_type"])
```

---

## ✅ Rekomendasi

**Untuk testing cepat:** Gunakan **OPSI A** (ganti nama kendaraan di UI jadi "A", "B", "C")

**Untuk fix permanent:** Gunakan **OPSI B** (ubah kode agar handle kedua format)

---

## 📝 Catatan Penting

Masalah ini menunjukkan **inkonsistensi naming convention** di project:

- Ada tempat yang expect `"A"`, `"B"`, `"C"`
- Ada tempat yang expect `"Vehicle A"`, `"Vehicle B"`, `"Vehicle C"`

Ini harus di-standardisasi di seluruh codebase agar tidak muncul error serupa di masa depan.

---

## 🔗 File Terkait yang Mungkin Perlu Dicek

1. `Program/sweep_nn.py` - Function `build_clusters()` yang set `vehicle_type`
2. `Program/gui/tabs/input_data.py` - Cara vehicle disimpan ke session state
3. `Program/academic_replay.py` - Multiple functions yang akses `fleet[vehicle_type]`

---

_Fix #03 selesai. Coba test dengan OPSI A dulu untuk memastikan fix bekerja._
