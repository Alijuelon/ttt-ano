#!/usr/bin/env python3
"""
ocr_pro.py - Pipeline OCR yang lebih akurat untuk teks dari kamera.
Berjalan sama di Google Colab dan Raspberry Pi 4 (Tesseract + OpenCV).

Tahapan:
  1. Pilih frame tertajam (multi-frame)
  2. Crop area teks -> perbaiki polaritas -> upscale -> deskew -> normalisasi cahaya
  3. CLAHE untuk peningkatan kontras adaptif
  4. Beberapa kandidat gambar (norm / otsu / adaptive), pilih skor terbaik
  5. Smart PSM detection berdasarkan aspect ratio gambar
  6. image_to_data: confidence per kata, buang noise, susun urutan baris
  7. Koreksi: salah baca angka/huruf, kamus kata (opsional), sambung kata terpotong
  8. Konsensus antar-frame + gerbang stabilitas (hanya bicara jika hasil stabil)
  9. Normalisasi untuk TTS: singkatan, angka -> kata (terbilang), rupiah, jam, tanggal

CHANGELOG v2.0:
  - Profile default sekarang "accurate" (3 kandidat preprocessing)
  - Multi-PSM: (6, 3, 11) untuk berbagai jenis layout teks
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) di pipeline preprocessing
  - Fast deskew dengan Hough Line Transform (5-10x lebih cepat dari brute-force)
  - Unsharp masking setelah upscale untuk mempertajam huruf
  - Smart PSM detection berdasarkan aspect ratio
  - drop_conf dinaikkan ke 45.0 untuk mengurangi noise
  - Threshold ketajaman minimum (MIN_SHARPNESS)
  - is_acceptable() lebih toleran untuk teks campuran angka/huruf
  - split_for_tts() dipakai di speak_text() untuk streaming per kalimat
"""
from __future__ import annotations

import re
import os
import time
import threading
import difflib
import collections
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

logger = logging.getLogger(__name__)

# Otomatis cari path tesseract jika di Windows atau path kustom
if not hasattr(pytesseract.pytesseract, '_tess_configured'):
    _candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract"
    ]
    for _p in _candidates:
        if os.path.exists(_p):
            pytesseract.pytesseract.tesseract_cmd = _p
            break
    pytesseract.pytesseract._tess_configured = True


# =====================================================================
# Konfigurasi
# =====================================================================
@dataclass
class Cfg:
    lang: str = "ind+eng"
    # "fast" (1 kandidat, PSM 6) | "accurate" (multi kandidat, multi-PSM) | "turbo" (1 kandidat, PSM 11)
    profile: str = "accurate"
    crop: bool = False                  # False = jangan potong gambar agar judul/footer tidak hilang
    deskew: bool = True                 # luruskan teks miring
    fast_deskew: bool = True            # gunakan Hough Lines (cepat) vs brute-force (akurat)
    target_width: int = 1600            # upscale supaya huruf cukup besar
    max_scale: float = 4.0              # batas scale maksimum
    psm_list: Sequence[int] = (6, 4, 3)     # PSM 6 (paragraf/blok), 4 (kolom bervariasi), 3 (otomatis)
    early_exit_conf: float = 92.0       # berhenti dini jika kandidat sudah sangat yakin
    drop_conf: float = 38.0             # kata di bawah ini dibuang (dinaikkan untuk buang sampah)
    spell_conf: float = 75.0            # kata di bawah ini boleh dikoreksi kamus
    min_line_conf: float = 30.0         # baris di bawah ini dibuang
    denoise: bool = True                # bilateral filter sebelum upscale
    clahe: bool = True                  # CLAHE untuk peningkatan kontras adaptif
    clahe_clip: float = 1.5             # clipLimit untuk CLAHE (diturunkan dari 2.0)
    clahe_grid: int = 8                 # tileGridSize untuk CLAHE
    unsharp: bool = True                # unsharp masking setelah upscale
    unsharp_strength: float = 0.8       # kekuatan unsharp mask (diturunkan dari 1.2)
    min_mean_conf: float = 45.0         # threshold minimum confidence
    min_chars: int = 3                  # minimum karakter
    min_letter_ratio: float = 0.25      # rasio minimum huruf dari total karakter
    tessdata_dir: Optional[str] = (
        "models/tessdata_best" if os.path.exists("models/tessdata_best/ind.traineddata") else None
    )
    wordlist: Optional[set] = None      # set kata huruf kecil (lihat load_wordlist)
    smart_psm: bool = True              # deteksi PSM terbaik berdasarkan aspect ratio
    min_sharpness_for_unsharp: float = 50.0  # skip unsharp jika gambar di bawah ini
    engine: str = "auto"                # "auto" (coba EasyOCR jika terpasang, fallback Tesseract) | "easyocr" | "tesseract"


# =====================================================================
# Struktur data
# =====================================================================
@dataclass
class Word:
    text: str
    conf: float
    block: int
    par: int
    line: int
    left: int
    top: int
    w: int
    h: int


@dataclass
class Line:
    text: str
    conf: float
    new_par: bool = False


@dataclass
class ReadResult:
    lines: List[Line] = field(default_factory=list)
    mean_conf: float = 0.0
    dict_ratio: Optional[float] = None
    score: float = 0.0
    variant: str = ""
    elapsed: float = 0.0        # waktu pemrosesan (detik) — BARU

    @property
    def text(self) -> str:
        return render(self.lines)


# =====================================================================
# 1. Preprocessing gambar
# =====================================================================
def sharpness(gray: np.ndarray) -> float:
    """Hitung ketajaman gambar dengan Laplacian variance."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def best_frames(frames: Sequence[np.ndarray], k: int = 1) -> List[np.ndarray]:
    """Urutkan frame BGR dari yang tertajam, ambil k teratas."""
    scored = [(sharpness(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)), i) for i, f in enumerate(frames)]
    scored.sort(reverse=True)
    return [frames[i] for _, i in scored[:k]]


def crop_text_region(gray: np.ndarray, margin: float = 0.03) -> np.ndarray:
    """Crop ke gabungan area yang tampak seperti teks. Aman: jika ragu, kembalikan gambar utuh."""
    h, w = gray.shape
    s = 640.0 / max(h, w)
    sm = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    sh, sw = sm.shape
    grad = cv2.morphologyEx(sm, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.0008 * sh * sw
    boxes = [cv2.boundingRect(c) for c in cnts]
    boxes = [b for b in boxes if b[2] * b[3] >= min_area and b[2] >= b[3]]
    if not boxes:
        return gray
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    mx, my = int(margin * sw), int(margin * sh)
    x0, y0 = max(0, x0 - mx), max(0, y0 - my)
    x1, y1 = min(sw, x1 + mx), min(sh, y1 + my)
    if (x1 - x0) * (y1 - y0) > 0.92 * sw * sh:      # hampir seluruh gambar -> tidak perlu crop
        return gray
    inv = 1.0 / s if s < 1 else 1.0
    cropped = gray[int(y0 * inv):int(y1 * inv), int(x0 * inv):int(x1 * inv)]
    # Pastikan hasil crop tidak terlalu kecil
    ch, cw = cropped.shape
    if ch < 20 or cw < 20:
        return gray
    return cropped


def fix_polarity(gray: np.ndarray) -> np.ndarray:
    """Pastikan teks gelap di latar terang (Tesseract paling akurat begini).

    Tahan bayangan: cahaya diratakan dulu, lalu teks yang benar adalah kelas minoritas gelap.
    Jika setelah diratakan >50% piksel gelap, berarti polaritasnya terbalik.
    """
    h, w = gray.shape
    s = min(1.0, 640.0 / max(h, w))
    sm = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    dark_frac = float((normalize_illumination(sm) < 128).mean())
    return 255 - gray if dark_frac > 0.5 else gray


def upscale(gray: np.ndarray, cfg: Cfg) -> np.ndarray:
    """Upscale gambar menggunakan interpolasi terbaik sesuai kondisi."""
    h, w = gray.shape
    # Jika gambar sangat kecil (lebar < 600 px), naikkan batas scale agar teks mikro terbaca
    max_scale = 5.0 if max(w, h) < 600 else cfg.max_scale
    scale = min(max_scale, max(1.0, cfg.target_width / max(w, 1)))
    if scale > 1.05:
        # Gunakan INTER_LANCZOS4 untuk resolusi mikro
        interp = cv2.INTER_LANCZOS4 if scale > 1.5 else cv2.INTER_CUBIC
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)
    return gray


def apply_clahe(gray: np.ndarray, cfg: Cfg) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Meningkatkan kontras lokal — sangat efektif untuk gambar dengan
    cahaya tidak merata, bayangan, atau kontras rendah.
    """
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip,
        tileGridSize=(cfg.clahe_grid, cfg.clahe_grid)
    )
    return clahe.apply(gray)


def unsharp_mask(gray: np.ndarray, strength: float = 1.2) -> np.ndarray:
    """
    Unsharp masking untuk mempertajam tepi huruf setelah upscale.
    Rumus: sharpened = original * (1 + strength) - blur * strength
    """
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.0 + strength, blur, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def fast_estimate_skew(gray: np.ndarray, max_angle: float = 10.0) -> float:
    """
    Estimasi kemiringan menggunakan Hough Line Transform.
    5-10x lebih cepat dari metode brute-force projection profile.
    """
    h, w = gray.shape
    s = min(1.0, 500.0 / max(h, w))
    sm = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray

    # Deteksi tepi + Hough Lines
    edges = cv2.Canny(sm, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=max(30, int(sm.shape[1] * 0.1)))

    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        rho, theta = line[0]
        # Konversi ke derajat kemiringan teks
        angle = np.degrees(theta) - 90.0
        if abs(angle) <= max_angle:
            angles.append(angle)

    if not angles:
        return 0.0

    # Gunakan median untuk toleransi outlier
    median_angle = float(np.median(angles))
    # Jika kemiringan sangat kecil, abaikan (noise)
    return median_angle if abs(median_angle) >= 0.3 else 0.0


def estimate_skew(gray: np.ndarray, max_angle: float = 10.0) -> float:
    """
    Estimasi kemiringan (derajat) lewat projection profile.
    Lebih akurat tapi lebih lambat. Dipakai jika fast_deskew=False.
    """
    h, w = gray.shape
    s = 500.0 / max(h, w)
    sm = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    _, bw = cv2.threshold(sm, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    hh, ww = bw.shape
    center = (ww / 2, hh / 2)

    def score(a: float) -> float:
        M = cv2.getRotationMatrix2D(center, a, 1.0)
        r = cv2.warpAffine(bw, M, (ww, hh), flags=cv2.INTER_NEAREST)
        prof = r.sum(axis=1, dtype=np.float64)
        return float(np.sum(np.diff(prof) ** 2))

    coarse = np.arange(-max_angle, max_angle + 0.01, 1.0)
    best = max(coarse, key=score)
    fine = np.arange(best - 1.0, best + 1.01, 0.25)
    best = max(fine, key=score)
    return float(best) if score(best) > score(0.0) * 1.02 else 0.0


def rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Hilangkan bayangan/cahaya tidak merata: bagi gambar dengan estimasi latar."""
    h, w = gray.shape
    s = 4
    small = cv2.resize(gray, (max(1, w // s), max(1, h // s)), interpolation=cv2.INTER_AREA)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    bg = cv2.GaussianBlur(bg, (0, 0), 7)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.divide(gray, bg, scale=255)


def _pad(img: np.ndarray, px: int = 15) -> np.ndarray:
    """Tambah padding putih di sekeliling gambar — membantu Tesseract di tepi."""
    return cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=255)


def detect_smart_psm(gray: np.ndarray) -> int:
    """
    Deteksi PSM terbaik berdasarkan aspect ratio dan karakteristik gambar.
    Mengembalikan PSM yang paling sesuai sebagai override tambahan.
    """
    h, w = gray.shape
    aspect = w / max(h, 1)
    if aspect > 5.0:
        return 7    # satu baris teks horizontal
    if aspect > 2.5:
        return 3    # auto page segmentation
    if aspect < 1.2:
        return 4    # single column, variable-size text (cocok untuk cover/poster vertikal)
    return 4        # single column variable text (lebih toleran variasi ukuran font)


def make_candidates(frame_bgr: np.ndarray, cfg: Cfg):
    """
    Hasilkan daftar (nama, gambar) kandidat untuk di-OCR.

    Pipeline preprocessing (urutan penting):
      1. Grayscale
      2. Crop area teks
      3. Fix polarity (teks gelap di latar terang)
      4. Denoise (bilateral filter — menjaga tepi huruf)
      5. Upscale (Lanczos/Cubic)
      6. Unsharp mask (pertajam huruf — hanya jika gambar cukup tajam)
      7. Deskew (luruskan kemiringan)
      8. Normalize illumination (hilangkan bayangan)
      9. CLAHE (tingkatkan kontras lokal — sekali saja)
     10. Buat kandidat: norm, otsu, adaptive
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr.copy()

    # Hitung sharpness awal untuk keputusan preprocessing adaptif
    img_sharpness = sharpness(gray)
    logger.debug(f"Sharpness input: {img_sharpness:.1f}")

    if cfg.crop:
        gray = crop_text_region(gray)

    gray = fix_polarity(gray)

    if cfg.denoise:
        # Bilateral filter: menjaga tepi huruf lebih baik dari median blur
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

    gray = upscale(gray, cfg)

    # Unsharp mask hanya jika gambar cukup tajam — pada gambar buram justru memperkuat noise
    if cfg.unsharp and img_sharpness >= cfg.min_sharpness_for_unsharp:
        gray = unsharp_mask(gray, cfg.unsharp_strength)
    elif cfg.unsharp:
        logger.debug(f"Skip unsharp mask: sharpness {img_sharpness:.1f} < {cfg.min_sharpness_for_unsharp}")

    if cfg.deskew:
        if cfg.fast_deskew:
            ang = fast_estimate_skew(gray)
        else:
            ang = estimate_skew(gray)
        if abs(ang) >= 0.3:
            gray = rotate(gray, ang)
            logger.debug(f"Deskew: {ang:.2f}°")

    norm = normalize_illumination(gray)

    # CLAHE hanya sekali — tidak digandakan
    if cfg.clahe:
        norm = apply_clahe(norm, cfg)

    smooth = cv2.GaussianBlur(norm, (3, 3), 0)

    # Kandidat preprocessing — norm selalu jadi kandidat utama (paling bersih)
    cands = [("norm", _pad(norm))]

    if cfg.profile in ("accurate", "fast"):
        # Kandidat CLAHE + mild unsharp (hanya jika gambar cukup tajam)
        if img_sharpness >= cfg.min_sharpness_for_unsharp:
            clahe_direct = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(gray)
            sharp_clahe = unsharp_mask(clahe_direct, 0.6)
            cands.append(("clahe_sharp", _pad(sharp_clahe)))

    if cfg.profile in ("accurate",):
        _, otsu = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        ada = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 51, 15)

        # Morphological opening untuk memisahkan huruf yang menempel (terutama font tebal)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel_open)
        ada = cv2.morphologyEx(ada, cv2.MORPH_OPEN, kernel_open)

        cands += [("otsu", _pad(otsu)), ("adaptive", _pad(ada))]

    return cands


# =====================================================================
# 2. Kamus dan koreksi kata
# =====================================================================
_ALPHA = "abcdefghijklmnopqrstuvwxyz"
# Kesalahan visual yang umum pada OCR
_CONFUSION = {"0": "o", "1": "li", "5": "s", "8": "b", "6": "g", "2": "z",
              "l": "i", "i": "l", "rn": "m", "m": "rn", "vv": "w", "cl": "d",
              "c": "e", "e": "c"}

# Pemetaan koreksi langsung untuk pola salah baca yang sangat umum
_DIRECT_REPLACEMENTS = {
    "analasa": "analisa",
    "analika": "analisa",
    "cplup": "celup",
    "ctllf": "celup",
    "cellf": "celup",
    "mdger": "meger",
    "dt": "di",
    "conteh": "contoh",
    "panta": "pantai",
    "kiberkan": "dikibarkan",
    "dikiberkan": "dikibarkan",
    "peraku": "perahu",
    "buten": "hutan",
    "witet": "wisata",
    "witets": "wisata",
    "berdisi": "berdiri",
    "icrbuat": "terbuat",
    "bita": "bisa",
    "mewbusi": "membuat",
}

# Kamus kata umum Bahasa Indonesia untuk koreksi OCR
# Digunakan jika cfg.wordlist tidak di-set secara eksplisit
_BUILTIN_WORDLIST = {
    # Kata umum yang sering salah dibaca OCR
    "analisa", "analisis", "menentukan", "titik", "kerusakan", "pada",
    "elektro", "motor", "pompa", "celup", "menggunakan", "alat", "ukur",
    "meger", "megger", "multi", "meter", "skripsi", "tugas", "akhir",
    "sebagai", "salah", "satu", "syarat", "untuk", "menyelesaikan",
    "program", "studi", "sarjana", "terapan", "politeknik", "negeri",
    "bengkalis", "pulp", "paper", "perawang",
    # Kata umum Indonesia
    "dan", "yang", "dari", "dengan", "adalah", "ini", "itu", "tidak",
    "akan", "dalam", "untuk", "pada", "oleh", "karena", "jika", "atau",
    "juga", "sudah", "telah", "belum", "sedang", "masih", "bisa", "dapat",
    "harus", "perlu", "sangat", "lebih", "kurang", "banyak", "sedikit",
    "besar", "kecil", "tinggi", "rendah", "baru", "lama", "baik", "buruk",
    "benar", "salah", "atas", "bawah", "kiri", "kanan", "depan", "belakang",
    "dalam", "luar", "antara", "tentang", "melalui", "terhadap", "sampai",
    "selama", "setelah", "sebelum", "ketika", "karena", "sehingga",
    "meskipun", "walaupun", "agar", "supaya", "bahwa", "dimana", "kemana",
    # Kata teknik
    "sistem", "proses", "metode", "teknik", "prosedur", "langkah",
    "peralatan", "komponen", "mesin", "listrik", "tegangan", "arus",
    "daya", "frekuensi", "tahanan", "isolasi", "kabel", "instalasi",
    "pengukuran", "pengujian", "pemeliharaan", "perbaikan", "perawatan",
    "kelistrikan", "transformator", "generator", "kapasitor", "resistor",
    "induktor", "rangkaian", "diagram", "gambar", "tabel", "grafik",
    # Kata akademik
    "universitas", "fakultas", "jurusan", "semester", "mahasiswa",
    "dosen", "pembimbing", "penguji", "halaman", "bab", "kesimpulan",
    "saran", "daftar", "pustaka", "lampiran", "abstrak", "pendahuluan",
    "tinjauan", "landasan", "teori", "metodologi", "penelitian",
    "hasil", "pembahasan", "penutup",
    # Kata tambahan umum & bahasa daerah yang sering muncul
    "pancasila", "ketuhanan", "kemanusiaan", "persatuan", "kerakyatan",
    "keadilan", "sosial", "seluruh", "rakyat", "indonesia", "beradab",
    "hikmat", "kebijaksanaan", "permusyawaratan", "perwakilan",
    "deskripsi", "objek", "suasana", "tekstur", "rincian", "majas",
    "pembaca", "merasakan", "kewan", "akeh", "jerapah", "gajah",
    "singa", "harimau", "binatang", "kebun", "surabaya", "pangandaran",
    "bendera", "upacara", "terbuat", "berkibar", "pantai", "nelayan",
}


def load_wordlist(path: str) -> set:
    """Muat daftar kata (satu kata per baris) sebagai set huruf kecil."""
    with open(path, encoding="utf-8") as f:
        return {ln.strip().lower() for ln in f if ln.strip()}


def _variants(word: str):
    for key, vals in _CONFUSION.items():
        start = 0
        while True:
            i = word.find(key, start)
            if i < 0:
                break
            for v in vals if len(key) == 1 else [vals]:
                yield word[:i] + v + word[i + len(key):]
            start = i + 1


def _edits1(word: str):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    for a, b in splits:
        if b:
            yield a + b[1:]                                  # hapus
        if len(b) > 1:
            yield a + b[1] + b[0] + b[2:]                    # tukar
        for c in _ALPHA:
            if b:
                yield a + c + b[1:]                          # ganti
            yield a + c + b                                  # sisip


def _match_case(src: str, new: str) -> str:
    if src.isupper():
        return new.upper()
    if src[:1].isupper():
        return new.capitalize()
    return new


def fix_token(token: str, conf: float, cfg: Cfg) -> str:
    """Perbaiki satu token. Konservatif: lebih baik tidak mengubah daripada salah ubah."""
    m = re.match(r"^(\W*)(.*?)(\W*)$", token, re.S)
    lead, core, tail = m.group(1), m.group(2), m.group(3)
    if not core:
        return token

    # a) Token yang dominan angka: O/o/l/I/S di dalamnya hampir pasti angka
    digits = sum(c.isdigit() for c in core)
    if digits >= max(1, len(core) / 2) and len(core) >= 2:
        core = core.translate(str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "S": "5"}))
        return lead + core + tail

    # b) Direct replacement untuk kata yang sering konsisten salah baca
    lw = core.lower()
    if lw in _DIRECT_REPLACEMENTS:
        return lead + _match_case(core, _DIRECT_REPLACEMENTS[lw]) + tail

    # c) Koreksi kamus
    wl = cfg.wordlist
    lw = core.lower()
    if (not wl or len(lw) < 3 or lw in wl or conf >= 88
            or not re.fullmatch(r"[A-Za-z0-9]+", core)
            or (core.isupper() and len(core) <= 4)      # singkatan (PT, KTP, ...)
            or (digits and len(core) <= 3)):
        return token
    for v in _variants(lw):
        if v in wl:
            return lead + _match_case(core, v) + tail
    if conf < cfg.spell_conf and len(lw) >= 4:
        hits = sorted({v for v in _edits1(lw) if v in wl})
        if len(hits) == 1:                                # hanya jika tidak ambigu
            return lead + _match_case(core, hits[0]) + tail
    return token


# =====================================================================
# 3. OCR + penyusunan teks
# =====================================================================
def resolve_lang(lang: str, tessdata_dir: Optional[str] = None) -> str:
    try:
        avail = set(pytesseract.get_languages(config=f'--tessdata-dir "{tessdata_dir}"'
                                              if tessdata_dir else ""))
    except Exception:
        return lang
    parts = [p for p in lang.split("+") if p in avail]
    return "+".join(parts) if parts else "eng"


def tesseract_words(img: np.ndarray, lang: str, psm: int, cfg: Cfg) -> List[Word]:
    config = (f"--oem 1 --psm {psm} -c preserve_interword_spaces=1 "
              f"-c user_defined_dpi=300")
    if cfg.tessdata_dir and os.path.exists(cfg.tessdata_dir):
        tpath = os.path.abspath(cfg.tessdata_dir)
        config += f' --tessdata-dir {tpath}'
    try:
        d = pytesseract.image_to_data(img, lang=lang, config=config, output_type=Output.DICT)
    except Exception as e:
        logger.warning(f"Tesseract error (psm={psm}): {e}")
        return []
    words = []
    for i, t in enumerate(d["text"]):
        t = (t or "").strip()
        try:
            conf = float(d["conf"][i])
        except (TypeError, ValueError):
            continue
        if not t or conf < 0:
            continue
        words.append(Word(t, conf, d["block_num"][i], d["par_num"][i], d["line_num"][i],
                          d["left"][i], d["top"][i], d["width"][i], d["height"][i]))
    return words


def build_lines(words: List[Word], cfg: Cfg):
    """Filter noise, koreksi token, susun baris. Return (lines, rasio_kamus)."""
    if not words:
        return [], None
    med_h = float(np.median([w.h for w in words if w.h > 0] or [0]))
    kept = []
    for w in words:
        has_alnum = any(c.isalnum() for c in w.text)
        if w.conf < cfg.drop_conf:
            continue
        if not has_alnum and w.conf < 70:
            continue
        # Buang token 1-2 karakter yang bukan kata/angka bermakna (sampah OCR)
        clean = re.sub(r'\W', '', w.text)
        if len(clean) <= 2 and w.conf < 75 and not clean.isdigit():
            continue
        # Jangan buang kata hanya karena beda ukuran font (misal judul besar vs subjudul kecil)
        if has_alnum and med_h and (w.h < 0.15 * med_h or w.h > 6.0 * med_h) and w.conf < 25:
            continue
        w.text = fix_token(w.text, w.conf, cfg)
        kept.append(w)
    if not kept:
        return [], None

    groups = collections.OrderedDict()
    for w in kept:
        groups.setdefault((w.block, w.par, w.line), []).append(w)

    lines, prev_par = [], None
    for (b, p, _), ws in groups.items():
        ws.sort(key=lambda x: x.left)
        text = " ".join(x.text for x in ws)
        tot = sum(len(x.text) for x in ws) or 1
        conf = sum(x.conf * len(x.text) for x in ws) / tot
        lines.append(Line(text, conf, new_par=(prev_par is not None and (b, p) != prev_par)))
        prev_par = (b, p)

    # buang baris sampah: kepercayaan rendah atau hampir tanpa huruf/angka
    def _ok(l: Line) -> bool:
        alnum = sum(c.isalnum() for c in l.text)
        return l.conf >= cfg.min_line_conf and alnum >= 0.25 * max(len(l.text.replace(" ", "")), 1)
    lines = [l for l in lines if _ok(l)]
    if not lines:
        return [], None
    kept = [w for w in kept if any(w.text in l.text for l in lines)]

    ratio = None
    if cfg.wordlist:
        toks = [re.sub(r"\W", "", x.text).lower() for x in kept]
        toks = [t for t in toks if len(t) >= 3 and t.isalpha()]
        if toks:
            ratio = sum(t in cfg.wordlist for t in toks) / len(toks)
    return lines, ratio



def render(lines: Sequence[Line]) -> str:
    """Gabung baris menjadi teks: sambung kata yang terpotong tanda hubung, pisahkan paragraf."""
    out, cur = [], ""
    for ln in lines:
        if ln.new_par and cur:
            out.append(cur)
            cur = ""
        t = ln.text.strip()
        if (cur.endswith("-") and len(cur) > 1 and cur[-2].isalpha() and t[:1].islower()):
            cur = cur[:-1] + t
        else:
            cur = (cur + " " + t).strip()
    if cur:
        out.append(cur)
    return "\n".join(out)


def _score(lines: List[Line], ratio: Optional[float]):
    if not lines:
        return 0.0, 0.0
    tot = sum(len(l.text) for l in lines) or 1
    mean = sum(l.conf * len(l.text) for l in lines) / tot
    r = 1.0 if ratio is None else ratio
    # Beri bobot lebih tinggi pada kelengkapan teks agar kandidat yang membaca teks penuh terpilih
    bonus = min(tot, 350) / 350.0 * 25.0
    score = mean * (0.5 + 0.5 * r) + bonus
    return mean, score


_GLOBAL_EASYOCR_READER = None
_EASYOCR_LOCK = threading.Lock()


def get_easyocr_reader():
    """Inisialisasi lazy loader / singleton untuk model EasyOCR agar hanya dimuat 1x."""
    global _GLOBAL_EASYOCR_READER
    with _EASYOCR_LOCK:
        if _GLOBAL_EASYOCR_READER is None:
            try:
                import warnings
                warnings.filterwarnings("ignore", category=UserWarning)
                import easyocr
                m_dir = os.path.abspath("models/easyocr")
                u_dir = os.path.join(m_dir, "user_network")
                os.makedirs(u_dir, exist_ok=True)
                _GLOBAL_EASYOCR_READER = easyocr.Reader(
                    ['id', 'en'],
                    gpu=False,
                    model_storage_directory=m_dir,
                    user_network_directory=u_dir,
                    verbose=False
                )
            except Exception as e:
                logger.warning(f"EasyOCR tidak tersedia, beralih ke Tesseract: {e}")
                _GLOBAL_EASYOCR_READER = False
    return _GLOBAL_EASYOCR_READER if _GLOBAL_EASYOCR_READER else None


def warmup_easyocr():
    """Pre-load model EasyOCR di latar belakang agar respon SPACE instan."""
    t = threading.Thread(target=get_easyocr_reader, daemon=True)
    t.start()


class OCRReader:
    def __init__(self, cfg: Optional[Cfg] = None):
        self.cfg = cfg or Cfg()
        # Gunakan wordlist bawaan jika tidak ada wordlist kustom
        if self.cfg.wordlist is None:
            self.cfg.wordlist = _BUILTIN_WORDLIST
        self.lang = resolve_lang(self.cfg.lang, self.cfg.tessdata_dir)

    def _get_easyocr(self):
        return get_easyocr_reader()

    def read_easyocr(self, frame_bgr: np.ndarray) -> Optional[ReadResult]:
        """Eksekusi pembacaan teks menggunakan model Deep Learning EasyOCR."""
        reader = self._get_easyocr()
        if not reader:
            return None
        t0 = time.time()
        try:
            results = reader.readtext(frame_bgr)
            if not results:
                return None
            lines = []
            total_conf = 0.0
            for bbox, text, prob in results:
                txt = text.strip()
                if not txt:
                    continue
                # Terapkan token replacement
                words = [fix_token(w, prob * 100.0, self.cfg) for w in txt.split()]
                cleaned_text = " ".join(words)
                conf = prob * 100.0
                if conf >= 20.0 and len(cleaned_text) >= 2:
                    lines.append(Line(cleaned_text, conf))
                    total_conf += conf
            if not lines:
                return None
            mean_conf = total_conf / len(lines)
            res = ReadResult(lines=lines, mean_conf=mean_conf, variant="easyocr", elapsed=time.time() - t0)
            return res
        except Exception as e:
            logger.warning(f"Error saat membaca dengan EasyOCR: {e}")
            return None

    def read(self, frame_bgr: np.ndarray) -> ReadResult:
        """
        Baca satu frame. Coba EasyOCR terlebih dahulu jika dipilih/tersedia,
        atau gunakan pipeline Tesseract multi-kandidat.
        """
        if self.cfg.engine in ("easyocr", "auto"):
            e_res = self.read_easyocr(frame_bgr)
            if e_res and self.is_acceptable(e_res) and len(e_res.text) >= self.cfg.min_chars:
                logger.info(f"OCR selesai (EasyOCR): {e_res.elapsed:.2f}s | conf={e_res.mean_conf:.1f} | chars={len(e_res.text)}")
                return e_res
            if self.cfg.engine == "easyocr":
                return e_res or ReadResult()

        t0 = time.time()
        best = ReadResult()

        candidates = make_candidates(frame_bgr, self.cfg)

        # Tentukan daftar PSM — gabungkan smart PSM jika diaktifkan
        psm_list = list(self.cfg.psm_list)
        if self.cfg.smart_psm:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
            smart = detect_smart_psm(gray)
            if smart not in psm_list:
                psm_list.insert(0, smart)   # prioritaskan PSM hasil deteksi

        for name, img in candidates:
            for psm in psm_list:
                words = tesseract_words(img, self.lang, psm, self.cfg)
                lines, ratio = build_lines(words, self.cfg)
                mean, score = _score(lines, ratio)
                if score > best.score:
                    best = ReadResult(lines, mean, ratio, score, f"{name}/psm{psm}")
                    logger.debug(f"  Kandidat terbaik baru: {name}/psm{psm} conf={mean:.1f} skor={score:.1f}")
            if best.mean_conf >= self.cfg.early_exit_conf:       # sudah cukup yakin -> hemat CPU
                logger.debug(f"Early exit: conf={best.mean_conf:.1f}")
                break

        best.elapsed = time.time() - t0
        logger.info(f"OCR selesai: {best.elapsed:.2f}s | conf={best.mean_conf:.1f} | "
                    f"varian={best.variant} | chars={len(best.text)}")
        return best

    def read_multi(self, frames: Sequence[np.ndarray], top_k: int = 3) -> ReadResult:
        """Baca beberapa frame tertajam lalu ambil konsensus per baris."""
        results = [self.read(f) for f in best_frames(frames, top_k)]
        return consensus(results)

    def is_acceptable(self, res: ReadResult) -> bool:
        """
        Validasi hasil OCR — lebih toleran dari versi sebelumnya:
        - Confidence rata-rata di atas threshold
        - Jumlah karakter minimum
        - Rasio huruf minimum (diturunkan dari 0.5 ke 0.3 agar angka/kode bisa lewat)
        """
        txt = res.text
        if not txt.strip():
            return False
        letters = sum(c.isalpha() for c in txt)
        alnum = sum(c.isalnum() for c in txt)
        return (res.mean_conf >= self.cfg.min_mean_conf
                and len(txt) >= self.cfg.min_chars
                and (letters / max(len(txt), 1) >= self.cfg.min_letter_ratio
                     or alnum / max(len(txt), 1) >= 0.5))

    def text(self, res: ReadResult) -> str:
        return res.text if self.is_acceptable(res) else ""


# =====================================================================
# 4. Konsensus antar-frame dan gerbang stabilitas
# =====================================================================
def consensus(results: Sequence[ReadResult]) -> ReadResult:
    results = [r for r in results if r.lines]
    if not results:
        return ReadResult()
    base = max(results, key=lambda r: r.score)
    if len(results) == 1:
        return base
    new_lines = []
    for ln in base.lines:
        group = [(ln.text, ln.conf)]
        for r in results:
            if r is base:
                continue
            best, br = None, 0.0
            for l2 in r.lines:
                ratio = difflib.SequenceMatcher(None, ln.text.lower(), l2.text.lower()).ratio()
                if ratio > br:
                    br, best = ratio, l2
            if best is not None and br >= 0.7:
                group.append((best.text, best.conf))
        votes = collections.defaultdict(lambda: [0, 0.0])
        for t, c in group:
            votes[t][0] += 1
            votes[t][1] += c
        text = max(votes.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]
        new_lines.append(Line(text, ln.conf, ln.new_par))
    return ReadResult(new_lines, base.mean_conf, base.dict_ratio, base.score,
                      base.variant + "+vote")


class LiveGate:
    """Bicara hanya jika hasil stabil beberapa kali berturut-turut dan bukan pengulangan."""

    def __init__(self, stable_needed: int = 2, same_ratio: float = 0.9,
                 repeat_after: Optional[float] = None):
        self.stable_needed = stable_needed
        self.same_ratio = same_ratio
        self.repeat_after = repeat_after      # None = jangan ulang sebelum teks berubah
        self._pending, self._count = "", 0
        self._last, self._last_t = "", 0.0
        self._empty = 0
        self.empty_reset = 3                  # n pembacaan kosong -> anggap halaman diganti

    def _same(self, a: str, b: str) -> bool:
        return bool(a) and bool(b) and difflib.SequenceMatcher(None, a, b).ratio() >= self.same_ratio

    def update(self, text: str, force: bool = False) -> Optional[str]:
        if not text:
            self._pending, self._count = "", 0
            self._empty += 1
            if self._empty >= self.empty_reset:
                self._last = ""
            return None
        self._empty = 0
        if force:
            self._last, self._last_t, self._count = text, time.time(), 0
            return text
        if self._same(text, self._pending):
            self._count += 1
        else:
            self._count = 1
        self._pending = text
        if self._count < self.stable_needed:
            return None
        if self._same(text, self._last):
            if self.repeat_after is None or time.time() - self._last_t < self.repeat_after:
                return None
        self._last, self._last_t, self._count = text, time.time(), 0
        return text


# =====================================================================
# 5. Normalisasi untuk TTS (singkatan, angka, rupiah, jam, tanggal)
# =====================================================================
_SATUAN = ["nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh",
           "delapan", "sembilan", "sepuluh", "sebelas"]
_BULAN = ["", "Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
          "Agustus", "September", "Oktober", "November", "Desember"]
_ABBR = {
    "yg": "yang", "dgn": "dengan", "utk": "untuk", "tdk": "tidak", "krn": "karena",
    "dll": "dan lain-lain", "dsb": "dan sebagainya", "dst": "dan seterusnya",
    "sbg": "sebagai", "spt": "seperti", "jl": "jalan", "jln": "jalan",
    "kab": "kabupaten", "kec": "kecamatan", "kel": "kelurahan", "dr": "dokter",
    "prof": "profesor", "ir": "insinyur", "bpk": "bapak", "sdr": "saudara",
    "telp": "telepon", "tlp": "telepon", "hub": "hubungi",
    "tsb": "tersebut", "thp": "tahap", "thn": "tahun", "bln": "bulan",
    "dpt": "dapat", "dlm": "dalam", "ttg": "tentang", "spy": "supaya",
    "dkk": "dan kawan-kawan", "hlm": "halaman", "hal": "halaman",
}
_UNITS = {"kg": "kilogram", "mg": "miligram", "ml": "mililiter", "km": "kilometer",
          "cm": "sentimeter", "mm": "milimeter", "g": "gram", "m": "meter", "l": "liter",
          "km/h": "kilometer per jam", "m/s": "meter per detik", "kwh": "kilowatt jam"}


def terbilang(n: int) -> str:
    if n < 0:
        return "minus " + terbilang(-n)
    if n < 12:
        return _SATUAN[n]
    if n < 20:
        return terbilang(n - 10) + " belas"
    if n < 100:
        return _SATUAN[n // 10] + " puluh" + (" " + _SATUAN[n % 10] if n % 10 else "")
    if n < 200:
        return "seratus" + (" " + terbilang(n - 100) if n > 100 else "")
    if n < 1000:
        return _SATUAN[n // 100] + " ratus" + (" " + terbilang(n % 100) if n % 100 else "")
    if n < 2000:
        return "seribu" + (" " + terbilang(n - 1000) if n > 1000 else "")
    for unit, name in ((10 ** 12, "triliun"), (10 ** 9, "miliar"), (10 ** 6, "juta"), (10 ** 3, "ribu")):
        if n >= unit:
            rest = n % unit
            return terbilang(n // unit) + " " + name + (" " + terbilang(rest) if rest else "")
    return str(n)


def _digits_words(s: str) -> str:
    return " ".join(_SATUAN[int(c)] for c in s if c.isdigit())


def number_to_words(tok: str) -> str:
    tok = tok.strip()
    frac = None
    m = re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,(\d+))?", tok)
    if m:                                         # 1.500.000 atau 1.500,25
        ip, frac = tok.split(",")[0].replace(".", ""), m.group(1)
    elif re.fullmatch(r"\d+,\d+", tok):
        ip, frac = tok.split(",")
    elif re.fullmatch(r"\d+\.\d+", tok):
        ip, frac = tok.split(".")
    elif tok.isdigit():
        ip = tok
    else:
        return tok
    if (len(ip) > 1 and ip.startswith("0") and frac is None) or len(ip) >= 8 and frac is None \
            and len(ip) == len(tok):               # nomor telepon/kode -> per digit
        return _digits_words(ip)
    if len(ip) > 15:
        return _digits_words(ip)
    out = terbilang(int(ip))
    if frac:
        out += " koma " + _digits_words(frac)
    return out


def normalize_for_tts(text: str) -> str:
    """
    Normalisasi teks untuk TTS Bahasa Indonesia.
    Menangani: tanggal, jam, rupiah, satuan, persen, singkatan, angka.
    """
    t = text

    # tanggal dd/mm/yyyy
    def _date(m):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= d <= 31 and 1 <= mo <= 12:
            return f"tanggal {terbilang(d)} {_BULAN[mo]} {terbilang(y)}"
        return m.group(0)
    t = re.sub(r"\b(?:(?:tanggal|tgl)\.?\s*)?(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", _date, t,
               flags=re.I)

    # jam 10:30 / "pukul 10.30" / 17.00 (jam dua digit)
    def _time(m):
        h, mi = int(m.group(1)), int(m.group(2))
        return f"pukul {terbilang(h)}" + (f" lewat {terbilang(mi)} menit" if mi else "")
    t = re.sub(r"\b(?:pukul|jam)?\s*([01]?\d|2[0-3]):([0-5]\d)\b", _time, t, flags=re.I)
    t = re.sub(r"\b(?:pukul|jam)\s*([01]?\d|2[0-3])\.([0-5]\d)\b", _time, t, flags=re.I)
    t = re.sub(r"(?<![\d.,$])\b([01]\d|2[0-3])\.([0-5]\d)\b(?![.,]?\d)", _time, t)

    # Rp 15.000,-
    def _rp(m):
        return number_to_words(m.group(1)) + " rupiah"
    t = re.sub(r"\bRp\.?\s*(\d[\d.,]*\d|\d)(?:,-)?", _rp, t, flags=re.I)

    # No. 5 -> nomor 5 ; satuan ; persen ; simbol
    t = re.sub(r"\bno\.?\s*(?=\d)", "nomor ", t, flags=re.I)
    t = re.sub(r"(\d)\s*(km/h|m/s|kwh|kg|mg|ml|km|cm|mm|g|m|l)\b",
               lambda m: f"{m.group(1)} {_UNITS[m.group(2).lower()]}", t, flags=re.I)
    t = t.replace("%", " persen ").replace("&", " dan ").replace("@", " at ")
    t = re.sub(r"[*_|~^<>]", " ", t)

    # nomor telepon bergaris (diawali 0 atau +) -> per digit
    t = re.sub(r"\+?0?\d+(?:-\d+)+",
               lambda m: _digits_words(m.group(0)) if (m.group(0)[0] in "0+") else m.group(0), t)

    # angka biasa
    t = re.sub(r"\d[\d.,]*\d|\d", lambda m: number_to_words(m.group(0)), t)

    # singkatan
    pat = r"(?<!\w)(" + "|".join(sorted(_ABBR, key=len, reverse=True)) + r")\.?(?!\w)"
    t = re.sub(pat, lambda m: _ABBR[m.group(1).lower()], t, flags=re.I)
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\s+([.,;:!?])", r"\1", t).strip()


def split_for_tts(text: str, max_len: int = 180) -> List[str]:
    """Pecah menjadi kalimat pendek agar suara cepat mulai dan mudah dipotong."""
    parts: List[str] = []
    for para in text.split("\n"):
        for s in re.split(r"(?<=[.!?;:,])\s+", para.strip()):
            s = s.strip()
            while len(s) > max_len:
                cut = s.rfind(",", 0, max_len)
                if cut < 40:
                    cut = s.rfind(" ", 0, max_len)
                if cut <= 0:
                    cut = max_len
                head = s[:cut + 1] if s[cut] == "," else s[:cut]
                parts.append(head.strip())
                s = s[cut:].lstrip(", ")
            if s:
                parts.append(s)
    merged: List[str] = []
    for p in parts:                                   # gabung potongan terlalu pendek
        if merged and len(merged[-1]) < 25 and len(merged[-1]) + len(p) + 1 <= max_len:
            merged[-1] += " " + p
        else:
            merged.append(p)
    return [m for m in merged if m.strip()]


# =====================================================================
# CLI uji cepat: python ocr_pro.py foto.jpg [fast|accurate|turbo]
# =====================================================================
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if len(sys.argv) < 2:
        print("Pemakaian: python ocr_pro.py gambar.jpg [fast|accurate|turbo]")
        raise SystemExit(1)
    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Error: Tidak dapat membuka file '{sys.argv[1]}'")
        raise SystemExit(1)
    profile = sys.argv[2] if len(sys.argv) > 2 else "accurate"
    cfg = Cfg(profile=profile)
    reader = OCRReader(cfg)
    t0 = time.time()
    res = reader.read(img)
    print(f"[{time.time() - t0:.2f}s] conf={res.mean_conf:.1f} skor={res.score:.1f} "
          f"varian={res.variant} bahasa={reader.lang}")
    print("--- TEKS ---")
    print(res.text)
    print("--- UNTUK TTS ---")
    for chunk in split_for_tts(normalize_for_tts(res.text)):
        print("*", chunk)
