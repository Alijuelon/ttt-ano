# 🎙️ Image-to-Voices TTS v2.0

Sistem pembaca teks dari gambar kamera yang diucapkan dengan suara Bahasa Indonesia.

---

## 📹 Cara Memilih Kamera (Tidak Menggunakan Default)

Jika perangkat laptop/PC memiliki lebih dari satu kamera (misalnya webcam bawaan laptop dan webcam USB eksternal seperti Logitech Brio):

### 1. 🔍 Menu Pilihan Interaktif (`--select-cam` atau `-s`)
Menampilkan daftar kamera yang aktif dan meminta Anda memilih nomor kamera:
```bash
python main.py --select-cam
# atau dengan live preview terminal:
python main.py --select-cam --terminal
```
*Setelah memilih, sistem akan menawarkan opsi untuk menyimpan pilihan tersebut sebagai default permanen.*

### 2. 🎯 Menentukan Kamera Langsung (`--cam <nomor>` atau `-c <nomor>`)
Gunakan nomor port kamera tertentu untuk sesi ini:
```bash
# Gunakan kamera port 1 di terminal:
python main.py --cam 1 --terminal

# Gunakan kamera port 1 di jendela GUI:
python main.py --cam 1 --preview

# Sekali jepret dengan kamera 1:
python main.py --cam 1
```

### 3. 💾 Mengubah Kamera Default Permanen (`--set-default-cam <nomor>`)
Jika Anda selalu ingin menggunakan webcam USB (misal port 1) tanpa perlu mengetikkan `--cam 1` setiap kali:
```bash
python main.py --set-default-cam 1
```
*Pengaturan ini otomatis tersimpan di file `camera_config.json`.*

### 4. 📋 Melihat Daftar Kamera yang Terhubung (`--list-cams` atau `-l`)
Untuk memeriksa port berapa saja yang aktif:
```bash
python main.py --list-cams
```

### 5. 🔄 Beralih Kamera Saat Live Preview
Saat sedang berada di mode **`--terminal`** maupun **`--preview`**, Anda cukup menekan tombol:
- **`[N]`** : Beralih langsung ke kamera berikutnya tanpa perlu mematikan aplikasi.

---

## 🚀 Pilihan Tampilan Kamera

### 1. 🎥 Preview Langsung di Layar Terminal (Teks / ANSI TrueColor)
Feed kamera ditampilkan langsung di dalam terminal console:
```bash
python main.py --terminal
# atau shortcut:
python main.py -t
```
* Kontrol: `[SPACE]/[ENTER]` = Jepret & Baca, `[N]` = Ganti Kamera, `[C]` = Mode Warna, `[Q]` = Keluar.

### 2. 🪟 Preview Jendela GUI OpenCV
Membuka jendela preview beresolusi tinggi dengan garis panduan (*guide box*) dan meteran fokus:
```bash
python main.py --preview
# atau:
python main.py -p
```
* Kontrol: `[SPACE]` = Jepret & Baca, `[N]` = Ganti Kamera, `[F]` = Refocus, `[Q]` = Keluar.

### 3. 📸 Sekali Jepret Otomatis (Default)
Hitungan mundur 2 detik lalu membaca teks:
```bash
python main.py
```

### 4. 📂 Membaca dari File Gambar
```bash
python main.py --image foto.jpg
```
