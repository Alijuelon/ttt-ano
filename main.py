#!/usr/bin/env python3
"""
main.py - Entry point utama Image-to-Voices TTS
Mendukung:
    1. --terminal / -t     : 🎥 Visualisasi kamera LANGSUNG di dalam layar terminal teks (TrueColor/ASCII)
    2. --preview / -p      : 🪟 Preview live kamera di jendela OpenCV GUI terpisah
    3. --select-cam / -s   : 🔍 Menu interaktif untuk memilih kamera dari daftar perangkat terhubung
    4. --list-cams / -l    : 📋 Menampilkan daftar semua kamera yang terdeteksi
    5. --set-default-cam N : 💾 Menyimpan kamera N sebagai kamera default permanen
    6. (default)           : 📸 Sekali jepret langsung setelah hitungan mundur
    7. --image <file>      : 📂 Baca teks dari file gambar tanpa kamera

Pemakaian:
    python main.py --select-cam             # pilih kamera yang terhubung dari daftar
    python main.py --cam 1 --terminal       # pakai kamera 1 langsung di terminal
    python main.py --cam 1 --preview        # pakai kamera 1 di jendela GUI
    python main.py --list-cams              # cek kamera apa saja yang terdeteksi
"""

import sys
import os
import subprocess

# Otomatis beralih ke virtual environment (.venv) jika user menjalankan via python global tanpa modul lengkap
_project_dir = os.path.dirname(os.path.abspath(__file__))
_venv_python = os.path.join(_project_dir, ".venv", "Scripts", "python.exe")
if os.path.exists(_venv_python) and os.path.normcase(sys.executable) != os.path.normcase(_venv_python):
    try:
        import easyocr
    except ImportError:
        target_script = os.path.abspath(__file__)
        args = [_venv_python, target_script] + sys.argv[1:]
        res = subprocess.run(args)
        sys.exit(res.returncode)

import cv2
import time

# Redam internal debug warning dari OpenCV C++
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass
import json
import argparse
import logging
import threading
import platform
from datetime import datetime
import numpy as np
import pytesseract
from pytesseract import Output
from ocr_pro import (
    Cfg, OCRReader, normalize_for_tts, split_for_tts, sharpness, warmup_easyocr
)

# ─────────────────────────────────────────────
# Konfigurasi default
# ─────────────────────────────────────────────
DEFAULT_CAM_INDEX  = 0
CAMERA_CONFIG_FILE = "camera_config.json"
PIPER_MODEL        = "models/piper/id_ID-news_tts-medium.onnx"
MIN_SHARPNESS      = 8.0     # skor ketajaman minimum — turunkan agar webcam fixed-focus tetap bisa capture
WARN_SHARPNESS     = 20.0    # skor di atas ini: hijau (tajam), di bawah: kuning (peringatan)
FRAME_WARMUP       = 10      # frame dibuang untuk pemanasan kamera
FRAME_SAMPLE       = 30      # jumlah frame untuk memilih yang terpajam
FRAME_DELAY        = 0.07    # detik antar frame
COUNTDOWN_DETIK    = 2       # hitungan mundur sebelum jepret
SAVE_DIR           = "hasil" # folder penyimpanan foto

# Warna HUD GUI OpenCV (BGR)
COLOR_GREEN   = (0, 220, 0)
COLOR_YELLOW  = (0, 200, 255)
COLOR_RED     = (0, 0, 230)
COLOR_WHITE   = (255, 255, 255)
COLOR_BLACK   = (0, 0, 0)
COLOR_CYAN    = (230, 220, 0)

# ─────────────────────────────────────────────
# Setup logging
# ─────────────────────────────────────────────
def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("ocr_log.txt", encoding="utf-8", mode="a"),
        ]
    )

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Manajemen Kamera (Scan, Select, Persist)
# ─────────────────────────────────────────────
def get_saved_camera_index() -> int:
    """Membaca indeks kamera default yang tersimpan di camera_config.json."""
    if os.path.exists(CAMERA_CONFIG_FILE):
        try:
            with open(CAMERA_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return int(data.get("camera_index", DEFAULT_CAM_INDEX))
        except Exception:
            pass
    return DEFAULT_CAM_INDEX


def save_camera_index(idx: int):
    """Menyimpan indeks kamera terpilih ke file konfigurasi agar jadi default permanen."""
    try:
        with open(CAMERA_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "camera_index": idx,
                "saved_at": datetime.now().isoformat()
            }, f, indent=2)
        print(f"💾 Kamera [{idx}] berhasil disimpan sebagai default permanen di '{CAMERA_CONFIG_FILE}'.")
    except Exception as e:
        logger.warning(f"Gagal menyimpan config kamera: {e}")


def scan_cameras(max_tested: int = 6) -> list:
    """
    Pindai port/indeks kamera 0..(max_tested-1).
    Mengembalikan daftar info kamera yang berhasil dibuka.
    """
    found = []
    print("🔍 Memindai kamera yang terhubung...", flush=True)
    for idx in range(max_tested):
        # Di Windows coba DSHOW dulu, fallback ke CAP_ANY
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        if not cap.isOpened() and os.name == "nt":
            cap = cv2.VideoCapture(idx)

        if cap.isOpened():
            # Tes apakah bisa membaca frame
            ret, _ = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            found.append({
                "index": idx,
                "width": w,
                "height": h,
                "active": ret
            })
    return found


def interactive_select_camera(cams: list = None) -> int:
    """
    Tampilkan menu interaktif pemilihan kamera ke user.
    User cukup memasukkan nomor kamera (0, 1, 2, ...).
    """
    if cams is None:
        cams = scan_cameras()

    current_default = get_saved_camera_index()

    print("\n" + "═" * 58)
    print("  📹 DAFTAR KAMERA YANG TERHUBUNG")
    print("═" * 58)

    if not cams:
        print("❌ Tidak ada kamera yang terdeteksi.")
        print("   Pastikan webcam eksternal (USB) terpasang dengan baik.")
        print(f"   Tetap menggunakan indeks kamera {current_default}.")
        print("═" * 58 + "\n")
        return current_default

    for c in cams:
        idx = c["index"]
        status = " (Aktif)" if c.get("active") else " (Tidak merespon)"
        is_cur = " ⭐ [DEFAULT SAAT INI]" if idx == current_default else ""
        print(f"  [{idx}] Port Kamera {idx} — Resolusi: {c['width']}x{c['height']}{status}{is_cur}")

    print("═" * 58)

    try:
        ans = input(f"Pilih nomor kamera [0-{cams[-1]['index']}] (tekan Enter untuk {current_default}): ").strip()
        if not ans:
            return current_default
        chosen = int(ans)
        # Tanyakan apakah ingin dijadikan default permanen
        save_ans = input(f"Jadikan Kamera [{chosen}] sebagai default untuk seterusnya? (Y/n): ").strip().lower()
        if save_ans in ("", "y", "ya", "yes"):
            save_camera_index(chosen)
        return chosen
    except (ValueError, KeyboardInterrupt):
        return current_default


# ─────────────────────────────────────────────
# Audio: cross-platform playback
# ─────────────────────────────────────────────
def play_wav(path: str) -> bool:
    """Putar file WAV. Mendukung Windows (winsound), Linux (pw-play/aplay), macOS (afplay)."""
    if not os.path.exists(path):
        return False
    sys_name = platform.system()
    try:
        if sys_name == "Windows":
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)
        elif sys_name == "Linux":
            if os.system("which pw-play > /dev/null 2>&1") == 0:
                os.system(f"wpctl set-volume @DEFAULT_AUDIO_SINK@ 1.0 2>/dev/null")
                os.system(f"pw-play {path}")
            else:
                os.system(f"aplay {path}")
        elif sys_name == "Darwin":
            os.system(f"afplay {path}")
        return True
    except Exception as e:
        logger.error(f"Gagal memutar audio: {e}")
        return False


def synthesize_wav(text: str, output_path: str) -> bool:
    """Sintesis teks ke file WAV menggunakan Piper TTS."""
    if not os.path.exists(PIPER_MODEL):
        logger.warning(f"Model suara tidak ditemukan: {PIPER_MODEL}")
        return False
    try:
        result = subprocess.run(
            [sys.executable, "-m", "piper", "-m", PIPER_MODEL, "-f", output_path],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=30
        )
        if result.returncode != 0:
            logger.error(f"Piper error: {result.stderr.decode('utf-8', errors='replace')}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("Piper timeout (>30 detik)")
        return False
    except Exception as e:
        logger.error(f"Gagal menjalankan Piper: {e}")
        return False


def speak_text(text: str, blocking: bool = True):
    """
    Ucapkan teks dengan Piper TTS.
    - Memecah teks panjang menjadi kalimat pendek (streaming)
    - blocking=False: jalankan di thread terpisah
    """
    if not text.strip():
        return None

    spoken = normalize_for_tts(text)
    chunks = split_for_tts(spoken, max_len=150)
    logger.info(f"[SUARA] {len(chunks)} bagian: {spoken[:80]}{'...' if len(spoken)>80 else ''}")

    def _run():
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            wav_path = f"temp_suara_{i}.wav"
            if synthesize_wav(chunk, wav_path):
                play_wav(wav_path)
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

    if blocking:
        _run()
    else:
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t


# ─────────────────────────────────────────────
# Kamera helpers
# ─────────────────────────────────────────────
def open_camera(cam_index: int) -> cv2.VideoCapture:
    """Buka kamera dengan resolusi tinggi dan autofocus."""
    # Coba DSHOW di Windows dulu agar lebih stabil
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(cam_index, backend)
    if not cap.isOpened() and os.name == "nt":
        cap = cv2.VideoCapture(cam_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    return cap


def warmup_camera(cap: cv2.VideoCapture, n: int = FRAME_WARMUP):
    for _ in range(n):
        cap.read()
        time.sleep(0.04)


def get_roi_sharpness(gray: np.ndarray) -> float:
    """Hitung ketajaman khusus pada area tengah (tempat teks berada), bukan kertas putih kosong di pinggir."""
    h, w = gray.shape
    roi = gray[int(h * 0.15):int(h * 0.85), int(w * 0.12):int(w * 0.88)]
    return sharpness(roi)


def capture_best_frame(cap: cv2.VideoCapture, n: int = FRAME_SAMPLE,
                        min_sharp: float = MIN_SHARPNESS,
                        delay: float = FRAME_DELAY) -> tuple:
    """Ambil frame terbaik dari sejumlah frame berdasarkan Laplacian variance pada area teks."""
    best_frame, best_score = None, 0.0
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            continue
        sc = get_roi_sharpness(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if sc > best_score:
            best_score, best_frame = sc, frame.copy()
        if best_score >= min_sharp * 1.5:
            break
        time.sleep(delay)
    return best_frame, best_score


def save_image(frame: np.ndarray, prefix: str = "hasil") -> str:
    """Simpan frame dengan nama berdasarkan timestamp."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SAVE_DIR, f"{prefix}_{ts}.jpg")
    cv2.imwrite(path, frame)
    logger.info(f"Gambar disimpan: {path}")
    return path


def check_framing(frame: np.ndarray) -> list:
    """Cek apakah teks menyentuh pinggir frame (terpotong)."""
    h, w = frame.shape[:2]
    scale = 640.0 / w
    small = cv2.resize(frame, (640, int(h * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    d = pytesseract.image_to_data(
        gray, lang="ind+eng", config="--oem 1 --psm 11",
        output_type=Output.DICT
    )
    margin = 25
    warnings = []
    x_min, y_min = 640, int(h * scale)
    x_max, y_max = 0, 0
    found = False
    for i in range(len(d["text"])):
        try:
            conf = int(d["conf"][i])
        except (TypeError, ValueError):
            continue
        if conf > 40 and d["text"][i].strip():
            x, y, ww, hh = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x + ww), max(y_max, y + hh)
            found = True
    if not found:
        return ["Tidak ada teks terdeteksi. Pastikan teks jelas dan terang."]
    if x_min < margin:                  warnings.append("Teks terpotong di kiri. Geser kamera ke kiri.")
    if x_max > 640 - margin:            warnings.append("Teks terpotong di kanan. Geser kamera ke kanan.")
    if y_min < margin:                  warnings.append("Teks terpotong di atas. Geser kamera ke atas.")
    if y_max > int(h * scale) - margin: warnings.append("Teks terpotong di bawah. Geser kamera ke bawah.")
    return warnings


def process_frame(frame: np.ndarray, args, check_frame: bool = True) -> str:
    """
    Jalankan OCR → validasi → TTS pada sebuah frame.
    Return: teks hasil OCR, atau string kosong jika gagal.
    """
    if check_frame and not args.no_framing_check:
        warnings = check_framing(frame)
        if warnings:
            msg = ". ".join(warnings)
            logger.warning(f"[Framing] {msg}")
            print(f"\n⚠  {msg}")
            speak_text(msg, blocking=False)
            return ""

    reader = OCRReader(Cfg(
        lang="ind+eng",
        profile=args.profile,
        psm_list=(6, 4, 3),
        drop_conf=38.0,
        min_mean_conf=50.0,
        min_chars=3,
        min_letter_ratio=0.25,
        clahe_clip=1.5,
        unsharp_strength=0.8,
    ))
    res = reader.read(frame)
    logger.info(f"OCR: conf={res.mean_conf:.1f} chars={len(res.text)} waktu={res.elapsed:.2f}s varian={res.variant}")

    if res.text.strip() and reader.is_acceptable(res):
        print(f"\n📄 TEKS:\n{res.text}\n")
        speak_text(res.text, blocking=False)
        return res.text
    else:
        msg = "Teks tidak dapat dibaca. Perbaiki pencahayaan atau jarak kamera."
        print(f"\n⚠  {msg}")
        speak_text("Teks tidak dapat dibaca dengan jelas.", blocking=False)
        return ""


# ─────────────────────────────────────────────
# MODE 1: Preview LANGSUNG DI TERMINAL (ASCII / TrueColor)
# ─────────────────────────────────────────────
def get_terminal_key_nonblocking() -> str:
    """Membaca input keyboard non-blocking di terminal Windows & Linux."""
    if platform.system() == "Windows":
        import msvcrt
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch in (b'\x00', b'\xe0'):
                msvcrt.getch()
                return ""
            try:
                return ch.decode('utf-8', errors='ignore')
            except Exception:
                return ""
    else:
        import select
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            return sys.stdin.read(1)
    return ""


def render_terminal_art(frame: np.ndarray, cols: int = 68, color_mode: bool = True) -> str:
    """Mengubah frame kamera menjadi gambar visual di dalam terminal teks."""
    h, w = frame.shape[:2]
    if color_mode:
        rows = int(cols * (h / w))
        if rows % 2 != 0:
            rows += 1
        resized = cv2.resize(frame, (cols, rows), interpolation=cv2.INTER_AREA)
        lines = []
        for y in range(0, rows, 2):
            parts = []
            for x in range(cols):
                tb, tg, tr = resized[y, x]
                if y + 1 < rows:
                    bb, bg, br = resized[y + 1, x]
                    parts.append(f"\033[38;2;{tr};{tg};{tb}m\033[48;2;{br};{bg};{bb}m▀")
                else:
                    parts.append(f"\033[38;2;{tr};{tg};{tb}m▀")
            parts.append("\033[0m")
            lines.append("".join(parts))
        return "\n".join(lines)
    else:
        rows = int(cols * (h / w) * 0.46)
        resized = cv2.resize(frame, (cols, rows), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        ramp = " .:-=+*#%@"
        ramp_len = len(ramp)
        lines = []
        for y in range(rows):
            line = "".join(ramp[min(ramp_len - 1, int(gray[y, x] / 255 * ramp_len))] for x in range(cols))
            lines.append(line)
        return "\n".join(lines)


def run_terminal_preview(args):
    """
    Menampilkan visual stream kamera LANGSUNG di dalam layar terminal console!
    Tekan [N] untuk beralih kamera secara langsung tanpa keluar aplikasi.
    """
    os.system("") # Aktifkan ANSI escape di Windows Console
    cam_idx = args.cam
    cap = open_camera(cam_idx)
    if not cap.isOpened():
        print(f"❌ Kamera [{cam_idx}] tidak terdeteksi! Periksa koneksi.")
        return

    print(f"⏳ Pemanasan kamera [{cam_idx}]...")
    warmup_camera(cap, FRAME_WARMUP)
    warmup_easyocr()

    color_mode = True
    os.system("cls" if os.name == "nt" else "clear")

    fps_time = time.time()
    fps_count = 0
    fps_val = 0.0
    status_msg = f"Kamera [{cam_idx}]. SPACE/ENTER: Jepret, [N]: Ganti Kamera"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"\n❌ Gagal membaca frame dari kamera [{cam_idx}].")
                break

            fps_count += 1
            if time.time() - fps_time >= 1.0:
                fps_val = fps_count / (time.time() - fps_time)
                fps_count = 0
                fps_time = time.time()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp_val = get_roi_sharpness(gray)

            art = render_terminal_art(frame, cols=68, color_mode=color_mode)

            if sharp_val >= WARN_SHARPNESS:
                sharp_tag = f"\033[92m{sharp_val:5.0f} (TAJAM / SIAP)\033[0m"
            elif sharp_val >= MIN_SHARPNESS:
                sharp_tag = f"\033[93m{sharp_val:5.0f} (SEDANG - OK)\033[0m"
            else:
                sharp_tag = f"\033[91m{sharp_val:5.0f} (BURAM)\033[0m"

            mode_tag = "\033[96mTrueColor 24-bit\033[0m" if color_mode else "\033[97mASCII Grayscale\033[0m"

            sys.stdout.write("\033[H")
            sys.stdout.write(
                f"┌──────────────────────────────────────────────────────────────────────┐\n"
                f"│ 🎥 LIVE TERMINAL │ Cam: [{cam_idx}] │ FPS: {fps_val:4.1f} │ Mode: {mode_tag:<18} │\n"
                f"│ Fokus: {sharp_tag:<32}                              │\n"
                f"├──────────────────────────────────────────────────────────────────────┤\n"
            )
            for line in art.split("\n"):
                sys.stdout.write(f"│ {line} │\n")
            sys.stdout.write(
                f"├──────────────────────────────────────────────────────────────────────┤\n"
                f"│ [SPACE / ENTER] : Jepret & Baca OCR  │ [N] Ganti Kamera Berikutnya   │\n"
                f"│ [C] : Ganti Warna (Color/ASCII)      │ [F] Refocus   [Q] Keluar      │\n"
                f"├──────────────────────────────────────────────────────────────────────┤\n"
                f"│ Status: {status_msg:<60} │\n"
                f"└──────────────────────────────────────────────────────────────────────┘\n"
            )
            sys.stdout.flush()

            k = get_terminal_key_nonblocking()
            if k in (" ", "\r", "\n"):
                sys.stdout.write(f"\n\n📸 JEPRET! Mengambil frame terbaik...\n")
                sys.stdout.flush()

                best_frame, score = capture_best_frame(cap, n=12, min_sharp=MIN_SHARPNESS, delay=0.04)
                if best_frame is None:
                    best_frame = frame.copy()
                    score = sharp_val

                cv2.imwrite("hasil_main.jpg", best_frame)
                if args.save:
                    save_image(best_frame)

                print(f"Ketajaman foto: {score:.1f}")
                if score < MIN_SHARPNESS:
                    print(f"⚠  Gambar agak buram (skor {score:.0f}), tetap mencoba OCR...")
                    speak_text("Gambar agak buram, tetap mencoba membaca.", blocking=False)
                
                # Terapkan Document Filter sebelum OCR
                ocr_frame = apply_document_filter(best_frame, "doc")
                print("🔄 Memproses OCR & Suara...")
                process_frame(ocr_frame, args, check_frame=False)

                print("\n[Tekan tombol apa saja untuk kembali ke preview kamera...]")
                while not get_terminal_key_nonblocking():
                    time.sleep(0.05)
                os.system("cls" if os.name == "nt" else "clear")
                status_msg = f"Kamera [{cam_idx}]. Siap membaca teks."

            elif k in ("n", "N"):
                # Beralih ke kamera berikutnya
                cap.release()
                next_cam = (cam_idx + 1) % 4
                status_msg = f"🔄 Beralih ke kamera [{next_cam}]..."
                sys.stdout.write(f"\n{status_msg}\n")
                sys.stdout.flush()
                new_cap = open_camera(next_cam)
                if new_cap.isOpened():
                    cam_idx = next_cam
                    cap = new_cap
                    warmup_camera(cap, 8)
                    status_msg = f"✅ Menggunakan Kamera [{cam_idx}]."
                else:
                    new_cap.release()
                    cap = open_camera(cam_idx)
                    status_msg = f"⚠ Kamera [{next_cam}] tidak tersedia. Tetap di [{cam_idx}]."
                os.system("cls" if os.name == "nt" else "clear")

            elif k in ("c", "C"):
                color_mode = not color_mode
                os.system("cls" if os.name == "nt" else "clear")
            elif k in ("f", "F"):
                status_msg = f"🔄 Refocusing kamera [{cam_idx}]..."
                warmup_camera(cap, 15)
                status_msg = "✅ Refocus selesai."
            elif k in ("q", "Q", "\x1b"):
                break

            time.sleep(0.03)

    finally:
        cap.release()
        print("\n\033[0m👋 Selesai preview kamera.")


# ─────────────────────────────────────────────
# MODE 2: Preview GUI OpenCV
# ─────────────────────────────────────────────
def apply_document_filter(frame: np.ndarray, mode: str = "doc") -> np.ndarray:
    """
    Filter Mode Dokumen Real-time:
    - 'normal' : Gambar warna asli kamera
    - 'doc'    : Tingkatkan kontras (CLAHE) & ratakan pencahayaan (putihkan kertas, hitamkan tinta)
    - 'bw'     : Binarisasi Hitam-Putih murni (Adaptive B&W)
    """
    if mode == "normal":
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Hilangkan bayangan kertas dengan background division
    small = cv2.resize(gray, (max(1, gray.shape[1] // 4), max(1, gray.shape[0] // 4)), interpolation=cv2.INTER_AREA)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    bg = cv2.GaussianBlur(bg, (0, 0), 7)
    bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    norm = cv2.divide(gray, bg, scale=255)

    # 2. Tingkatkan kontras lokal dengan CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)

    if mode == "doc":
        # Mode Dokumen: Kertas putih bersih, tinta hitam pekat
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    elif mode == "bw":
        # Mode Hitam-Putih Murni (Binary Threshold)
        smooth = cv2.GaussianBlur(enhanced, (3, 3), 0)
        bw = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11)
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    return frame


def draw_overlay(frame: np.ndarray, sharp: float, cam_idx: int = 0, hint: str = "",
                 status: str = "", capture_flash: bool = False, view_mode: str = "doc") -> np.ndarray:
    """Gambar HUD overlay di atas frame kamera."""
    out = frame.copy()
    h, w = out.shape[:2]

    if capture_flash:
        alpha = 0.5
        white = np.ones_like(out) * 255
        cv2.addWeighted(white, alpha, out, 1 - alpha, 0, out)

    bx1, by1 = int(w * 0.08), int(h * 0.12)
    bx2, by2 = int(w * 0.92), int(h * 0.88)
    cv2.rectangle(out, (bx1 + 2, by1 + 2), (bx2 + 2, by2 + 2), COLOR_BLACK, 2)
    cv2.rectangle(out, (bx1, by1), (bx2, by2), COLOR_GREEN, 2)

    corner = 22
    thick = 3
    for cx, cy, dx, dy in [(bx1, by1, 1, 1), (bx2, by1, -1, 1),
                            (bx1, by2, 1, -1), (bx2, by2, -1, -1)]:
        cv2.line(out, (cx, cy), (cx + dx * corner, cy), COLOR_CYAN, thick)
        cv2.line(out, (cx, cy), (cx + dy * corner, cy), COLOR_CYAN, thick)

    panel_h = 42
    overlay_top = out[0:panel_h, 0:w].copy()
    cv2.rectangle(out, (0, 0), (w, panel_h), COLOR_BLACK, -1)
    cv2.addWeighted(overlay_top, 0.35, out[0:panel_h, 0:w], 0.65, 0, out[0:panel_h, 0:w])

    cv2.putText(out, f"TTS CAM [{cam_idx}]", (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_CYAN, 2)

    bar_x, bar_y = w - 220, 10
    bar_w, bar_h = 200, 22
    pct = min(1.0, sharp / 50.0)
    bar_color = COLOR_GREEN if sharp >= WARN_SHARPNESS else (COLOR_YELLOW if sharp >= MIN_SHARPNESS else COLOR_RED)
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + int(bar_w * pct), bar_y + bar_h), bar_color, -1)
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_WHITE, 1)
    cv2.putText(out, f"Fokus: {sharp:.0f}", (bar_x + 4, bar_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

    bot_y = h - 44
    overlay_bot = out[bot_y:h, 0:w].copy()
    cv2.rectangle(out, (0, bot_y), (w, h), COLOR_BLACK, -1)
    cv2.addWeighted(overlay_bot, 0.3, out[bot_y:h, 0:w], 0.7, 0, out[bot_y:h, 0:w])

    ctrl_text = "[SPACE] Jepret  |  [D] Doc Mode  |  [N] Ganti Kamera  |  [Q] Keluar"
    cv2.putText(out, ctrl_text, (10, bot_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_WHITE, 1)

    # Tampilkan mode view aktif di panel atas
    mode_labels = {"normal": "RAW", "doc": "DOC", "bw": "B&W"}
    mode_label = mode_labels.get(view_mode, "?")
    mode_color = COLOR_CYAN if view_mode == "normal" else (COLOR_GREEN if view_mode == "doc" else COLOR_YELLOW)
    cv2.putText(out, f"[{mode_label}]", (200, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

    if hint:
        cv2.putText(out, hint, (10, bot_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_YELLOW, 1)

    if status:
        lines = status.split("\n")
        for li, line in enumerate(lines[:3]):
            ts = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
            tx = max(10, (w - ts[0]) // 2)
            ty = by2 + 22 + li * 22
            if ty < bot_y - 5:
                cv2.putText(out, line, (tx + 1, ty + 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_BLACK, 2)
                cv2.putText(out, line, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_GREEN, 1)

    cv2.putText(out, f"Posisikan teks dalam kotak ini (Kamera {cam_idx})", (bx1 + 8, by1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_GREEN, 1)

    return out


def run_preview(args):
    """
    Mode live preview: tampilkan feed kamera dengan jendela OpenCV GUI.
    Tekan [N] untuk beralih kamera secara langsung.
    """
    cam_idx = args.cam
    print("\n" + "=" * 58)
    print(f"  🎥  MODE PREVIEW KAMERA (GUI) — Kamera [{cam_idx}]")
    print("  SPACE = Jepret  |  N = Ganti Kamera  |  F = Refocus  |  Q = Keluar")
    print("=" * 58 + "\n")

    cap = open_camera(cam_idx)
    if not cap.isOpened():
        print(f"❌ Kamera [{cam_idx}] tidak terdeteksi! Periksa koneksi kamera.")
        return

    print(f"⏳ Pemanasan kamera [{cam_idx}]...")
    warmup_camera(cap, FRAME_WARMUP)
    warmup_easyocr()
    warmup_easyocr()  # Pre-load model EasyOCR di latar belakang agar tidak lag saat SPACE ditekan
    print("✅ Kamera siap. Jendela preview terbuka.\n")

    win_name = "Image-to-Voices TTS — Preview"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 560)

    last_sharp    = 0.0
    status_text   = ""
    hint_text     = "Arahkan ke teks lalu tekan SPACE untuk membaca"
    flash_until   = 0.0
    view_mode     = "doc"   # Default: Mode Dokumen (kontras tinggi)
    ocr_busy      = False   # Mencegah penumpukan thread OCR bersamaan

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Gagal membaca frame dari kamera [{cam_idx}].")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        last_sharp = get_roi_sharpness(gray)

        if last_sharp < MIN_SHARPNESS:
            hint_text = "⚠ Gambar sangat buram — dekatkan/jauhkan kamera sedikit"
        elif last_sharp < WARN_SHARPNESS:
            hint_text = f"△ Kualitas sedang (skor {last_sharp:.0f}) — tekan SPACE untuk membaca"
        else:
            hint_text = f"✓ Kamera [{cam_idx}] Siap — tekan SPACE untuk capture & OCR"

        now = time.time()
        do_flash = (now < flash_until)

        # Terapkan Document Filter ke tampilan
        display_frame = apply_document_filter(frame, view_mode)
        display = draw_overlay(display_frame, last_sharp, cam_idx, hint_text, status_text, do_flash, view_mode)

        dh, dw = display.shape[:2]
        if dw > 960:
            scale = 960 / dw
            display = cv2.resize(display, (960, int(dh * scale)))

        cv2.imshow(win_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            if ocr_busy:
                status_text = "⏳ Sedang memproses OCR, mohon tunggu sebentar..."
                continue

            flash_until = time.time() + 0.25
            print(f"\n📸 Capture dari Kamera [{cam_idx}]...")
            ocr_busy = True
            best_frame, score = capture_best_frame(cap, n=12, min_sharp=MIN_SHARPNESS, delay=0.04)
            if best_frame is None:
                best_frame = frame.copy()
                score = last_sharp

            # Terapkan Document Filter sebelum simpan & OCR
            ocr_frame = apply_document_filter(best_frame, view_mode)
            cv2.imwrite("hasil_main.jpg", ocr_frame)
            if args.save:
                save_image(ocr_frame)

            if score < MIN_SHARPNESS:
                # Tetap coba OCR walaupun buram, hanya beri peringatan
                status_text = f"⚠ Buram (skor {score:.0f}) — tetap mencoba OCR..."
                print(f"⚠  {status_text}")
                speak_text("Gambar agak buram, tetap mencoba membaca.", blocking=False)

            status_text = "🔄 Memproses OCR..."
            print("🔄 Memproses OCR...")

            def _do_ocr(f, a):
                nonlocal status_text, ocr_busy
                try:
                    txt = process_frame(f, a, check_frame=False)
                    if txt:
                        lines = [l.strip() for l in txt.strip().split("\n") if l.strip()]
                        status_text = " | ".join(lines[:2])[:70]
                    else:
                        status_text = "⚠ Teks tidak terbaca — coba lagi"
                finally:
                    ocr_busy = False

            t = threading.Thread(target=_do_ocr, args=(ocr_frame.copy(), args), daemon=True)
            t.start()

        elif key in (ord("n"), ord("N")):
            # Beralih ke kamera berikutnya
            cap.release()
            next_cam = (cam_idx + 1) % 4
            print(f"🔄 Mencoba beralih ke kamera [{next_cam}]...")
            new_cap = open_camera(next_cam)
            if new_cap.isOpened():
                cam_idx = next_cam
                cap = new_cap
                warmup_camera(cap, 8)
                status_text = f"✅ Beralih ke Kamera [{cam_idx}]"
            else:
                new_cap.release()
                cap = open_camera(cam_idx)
                status_text = f"⚠ Kamera [{next_cam}] tidak ada, kembali ke [{cam_idx}]"

        elif key in (ord("f"), ord("F")):
            print("🔄 Refocus kamera...")
            status_text = "Refocus..."
            warmup_camera(cap, 15)
            status_text = "✅ Refocus selesai"

        elif key in (ord("d"), ord("D")):
            # Ganti mode tampilan: doc → bw → normal → doc
            cycle = {"doc": "bw", "bw": "normal", "normal": "doc"}
            view_mode = cycle.get(view_mode, "doc")
            mode_names = {"doc": "Dokumen (Kontras Tinggi)", "bw": "Hitam-Putih Murni", "normal": "Normal (Raw)"}
            status_text = f"Mode: {mode_names[view_mode]}"
            print(f"🔄 Ganti mode → {mode_names[view_mode]}")

        elif key in (ord("q"), ord("Q"), 27):
            print("\n👋 Keluar dari preview.")
            break

        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# MODE 3: Sekali jepret
# ─────────────────────────────────────────────
def run_once(args):
    """Mode sekali jepret: countdown → jepret terbaik → OCR → TTS."""
    cam_idx = args.cam
    cap = open_camera(cam_idx)
    if not cap.isOpened():
        print(f"❌ Kamera [{cam_idx}] tidak terdeteksi!")
        speak_text("Kamera tidak terdeteksi.")
        return

    print(f"\n⏳ Pemanasan kamera [{cam_idx}]...")
    speak_text("Bersiap. Arahkan kamera ke teks dengan jelas dan terang.", blocking=False)
    warmup_camera(cap, FRAME_WARMUP)

    for i in range(COUNTDOWN_DETIK, 0, -1):
        print(f"   Jepret dalam {i}...")
        time.sleep(1)

    print("📸 Mengambil gambar...")
    best_frame, best_score = capture_best_frame(cap, FRAME_SAMPLE, MIN_SHARPNESS, FRAME_DELAY)
    cap.release()

    if best_frame is None:
        print("❌ Gagal mengambil gambar.")
        speak_text("Gagal mengambil gambar. Silakan coba lagi.")
        return

    print(f"   Ketajaman: {best_score:.1f}")
    
    # Terapkan Document Filter sebelum simpan & OCR
    ocr_frame = apply_document_filter(best_frame, "doc")
    cv2.imwrite("hasil_main.jpg", ocr_frame)
    if args.save:
        save_image(ocr_frame)

    if best_score < MIN_SHARPNESS:
        print(f"⚠  Gambar agak buram (skor {best_score:.0f}), tetap mencoba OCR...")
        speak_text("Gambar agak buram, tetap mencoba membaca.", blocking=False)

    print("🔄 Memproses OCR...")
    process_frame(ocr_frame, args, check_frame=False)


# ─────────────────────────────────────────────
# MODE 4: Dari file gambar
# ─────────────────────────────────────────────
def run_from_image(args):
    """Baca teks dari file gambar yang sudah ada."""
    img = cv2.imread(args.image)
    if img is None:
        print(f"❌ Tidak dapat membuka gambar: {args.image}")
        speak_text("File gambar tidak ditemukan.")
        return

    print(f"📂 Membaca: {args.image} ({img.shape[1]}×{img.shape[0]})")
    score = sharpness(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    print(f"   Ketajaman: {score:.1f}")

    reader = OCRReader(Cfg(
        lang="ind+eng",
        profile=args.profile,
        psm_list=(6, 4, 3),
        drop_conf=38.0,
        min_mean_conf=50.0,
        min_chars=3,
        min_letter_ratio=0.25,
        clahe_clip=1.5,
        unsharp_strength=0.8,
    ))
    print("🔄 Memproses OCR...")
    res = reader.read(img)

    print(f"\n📊 Confidence: {res.mean_conf:.1f}  |  Waktu: {res.elapsed:.2f}s  |  Varian: {res.variant}")
    print("\n📄 TEKS OCR:")
    print(res.text if res.text.strip() else "(tidak ada teks terdeteksi)")
    print("\n🔊 TEKS TTS:")
    spoken = normalize_for_tts(res.text)
    for chunk in split_for_tts(spoken):
        print(f"   * {chunk}")

    if res.text.strip() and reader.is_acceptable(res):
        speak_text(res.text)
    else:
        print("\n⚠  Teks tidak memenuhi kriteria untuk dibacakan.")
        speak_text("Teks tidak dapat dibaca dengan jelas.")


# ─────────────────────────────────────────────
# Argparse
# ─────────────────────────────────────────────
def parse_args():
    saved_cam = get_saved_camera_index()

    parser = argparse.ArgumentParser(
        description="Image-to-Voices TTS v2.0 — Baca teks dari kamera/gambar lalu diucapkan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
PILIHAN KAMERA:
  python main.py --select-cam             # 🔍 Pilih kamera dari daftar interaktif
  python main.py --list-cams              # 📋 Lihat daftar kamera yang terdeteksi
  python main.py --set-default-cam 1      # 💾 Simpan kamera 1 sebagai default permanen
  python main.py --cam 1                  # 🎯 Gunakan kamera 1 untuk sesi ini

TAMPILAN KAMERA:
  python main.py --terminal               # 🎥 LIHAT KAMERA LANGSUNG DI TERMINAL TEKS
  python main.py --preview                # 🪟 Buka jendela GUI preview kamera
  python main.py                          # 📸 Sekali jepret langsung (default)

Kamera default saat ini: Port [{saved_cam}]
        """
    )
    # Pilihan kamera
    parser.add_argument(
        "--select-cam", "-s", action="store_true",
        help="🔍 Tampilkan menu interaktif untuk memilih kamera dari daftar perangkat yang terhubung"
    )
    parser.add_argument(
        "--list-cams", "-l", action="store_true",
        help="📋 Pindai dan tampilkan daftar semua kamera yang terdeteksi, lalu selesai"
    )
    parser.add_argument(
        "--set-default-cam", type=int, default=None, metavar="N",
        help="💾 Simpan kamera port N sebagai kamera default permanen"
    )
    parser.add_argument(
        "--cam", "-c", type=int, default=None,
        help=f"🎯 Tentukan nomor port kamera yang digunakan (default tersimpan: {saved_cam})"
    )

    # Mode visualisasi
    parser.add_argument(
        "--terminal", "-t", "--ascii", "--cli", action="store_true",
        help="🎥 Tampilkan preview kamera LANGSUNG di dalam layar terminal teks console!"
    )
    parser.add_argument(
        "--preview", "-p", "--gui", action="store_true",
        help="🪟 Buka jendela GUI preview LIVE kamera (OpenCV window)"
    )

    # Opsi OCR & File
    parser.add_argument(
        "--profile", choices=["fast", "accurate", "turbo"], default="accurate",
        help="Profile OCR: 'accurate' (default), 'fast', 'turbo'"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path file gambar (skip kamera, langsung proses gambar)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Simpan foto capture dengan timestamp di folder 'hasil/'"
    )
    parser.add_argument(
        "--no-framing-check", action="store_true",
        help="Lewati pengecekan posisi teks, langsung OCR"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Tampilkan log debug lengkap"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    setup_logging(args.verbose)

    # 1. Jika hanya ingin simpan default camera
    if args.set_default_cam is not None:
        save_camera_index(args.set_default_cam)
        return

    # 2. Jika hanya ingin scan dan list kamera
    if args.list_cams:
        cams = scan_cameras()
        cur_default = get_saved_camera_index()
        print("\n" + "═" * 58)
        print("  📹 DAFTAR KAMERA YANG TERDETEKSI")
        print("═" * 58)
        if not cams:
            print("❌ Tidak ada kamera yang terdeteksi.")
        else:
            for c in cams:
                st = " (Aktif)" if c.get("active") else " (Tidak merespon)"
                df = " ⭐ [DEFAULT SAAT INI]" if c["index"] == cur_default else ""
                print(f"  Port [{c['index']}]: Resolusi {c['width']}x{c['height']}{st}{df}")
        print("═" * 58 + "\n")
        return

    # 3. Tentukan indeks kamera yang akan dipakai
    if args.select_cam:
        args.cam = interactive_select_camera()
    elif args.cam is None:
        args.cam = get_saved_camera_index()

    # Label mode untuk header
    if args.terminal:
        mode_str = "PREVIEW LIVE TERMINAL (ASCII / TrueColor)"
    elif args.preview:
        mode_str = "PREVIEW LIVE GUI (OpenCV Window)"
    elif args.image:
        mode_str = f"FILE ({args.image})"
    else:
        mode_str = "SEKALI JEPRET (Default)"

    print("=" * 60)
    print("  🎙️  Image-to-Voices TTS v2.0")
    print(f"  Mode     : {mode_str}")
    print(f"  Kamera   : Port [{args.cam}]  |  Profile OCR: {args.profile}")
    print(f"  Platform : {platform.system()}")
    print("=" * 60)

    if args.image:
        run_from_image(args)
    elif args.terminal:
        run_terminal_preview(args)
    elif args.preview:
        run_preview(args)
    else:
        run_once(args)

    print("\n✅ Selesai.")


if __name__ == "__main__":
    main()
