#!/usr/bin/env python3
"""
tes_kamera.py - Alat pengujian kamera & OCR yang lebih lengkap.

Pemakaian:
    python tes_kamera.py                # uji kamera + OCR + TTS (flow lengkap)
    python tes_kamera.py --show         # tampilkan preview gambar yang diambil
    python tes_kamera.py --no-tts       # OCR saja tanpa suara
    python tes_kamera.py --cam 1        # gunakan kamera indeks 1
    python tes_kamera.py --repeat 3     # ambil 3 kali dan tampilkan perbandingan
"""

import cv2
import time
import sys
import os
import argparse
import platform
import subprocess
import logging
import numpy as np
from ocr_pro import Cfg, OCRReader, normalize_for_tts, split_for_tts, sharpness

logger = logging.getLogger(__name__)

CAM_INDEX   = 0
PIPER_MODEL = "models/piper/id_ID-news_tts-medium.onnx"


def parse_args():
    parser = argparse.ArgumentParser(description="Alat pengujian kamera & OCR")
    parser.add_argument("--cam",    type=int, default=CAM_INDEX, help="Indeks kamera")
    parser.add_argument("--show",   action="store_true",         help="Tampilkan preview gambar")
    parser.add_argument("--no-tts", action="store_true",         help="Jangan putar suara")
    parser.add_argument("--repeat", type=int, default=1,         help="Jumlah pengulangan tangkapan")
    parser.add_argument("--profile", choices=["fast","accurate","turbo"], default="accurate",
                        help="Profile OCR")
    parser.add_argument("--verbose", "-v", action="store_true",  help="Log debug")
    return parser.parse_args()


def speak(text: str):
    """TTS sederhana tanpa import main.py."""
    if not os.path.exists(PIPER_MODEL):
        print(f"[TTS skip] model tidak ada: {PIPER_MODEL}")
        return
    spoken = normalize_for_tts(text)
    for i, chunk in enumerate(split_for_tts(spoken, max_len=150)):
        wav = f"temp_tes_{i}.wav"
        r = subprocess.run(
            [sys.executable, "-m", "piper", "-m", PIPER_MODEL, "-f", wav],
            input=chunk.encode("utf-8"), capture_output=True, timeout=30
        )
        if r.returncode == 0 and os.path.exists(wav):
            if platform.system() == "Linux":
                os.system(f"pw-play {wav}" if os.system("which pw-play>/dev/null 2>&1")==0
                          else f"aplay {wav}")
            elif platform.system() == "Windows":
                import winsound
                winsound.PlaySound(wav, winsound.SND_FILENAME)
            elif platform.system() == "Darwin":
                os.system(f"afplay {wav}")
            try:
                os.remove(wav)
            except OSError:
                pass


def capture_frame(cam_index: int) -> tuple:
    """Buka kamera, ambil frame terbaik, tutup kamera."""
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    if not cap.isOpened():
        print("ERROR: Kamera tidak terdeteksi!")
        return None, 0.0

    # Warmup
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    best_frame, best_score = None, 0.0
    print("Mengambil 30 frame terbaik...")
    for _ in range(30):
        ret, frame = cap.read()
        if ret:
            score = sharpness(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if score > best_score:
                best_score, best_frame = score, frame.copy()
            if best_score > 300:   # sudah sangat tajam
                break
        time.sleep(0.07)

    cap.release()
    return best_frame, best_score


def run_test(args, iteration: int = 1):
    sep = "─" * 50
    print(f"\n{sep}")
    print(f"  Tangkapan #{iteration}")
    print(sep)

    frame, score = capture_frame(args.cam)
    if frame is None:
        return

    path = f"hasil_tangkapan_{iteration}.jpg"
    cv2.imwrite(path, frame)
    print(f"✓ Frame tersimpan: {path}  (ketajaman: {score:.1f})")

    if score < 80:
        print("⚠  Peringatan: gambar buram — coba dekatkan kamera atau perbaiki cahaya")

    if args.show:
        display = cv2.resize(frame, (960, 540)) if frame.shape[1] > 960 else frame
        cv2.imshow(f"Tangkapan #{iteration}", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # OCR
    print(f"\nMemproses OCR (profile={args.profile})...")
    t0 = time.time()
    reader = OCRReader(Cfg(
        lang="ind+eng",
        profile=args.profile,
        psm_list=(6, 3, 11),
        drop_conf=45.0,
        min_mean_conf=55.0,
        min_chars=3,
    ))
    res = reader.read(frame)
    elapsed = time.time() - t0

    print(f"\n{'─'*40}")
    print(f"  Confidence : {res.mean_conf:.1f}")
    print(f"  Waktu OCR  : {elapsed:.2f}s")
    print(f"  Varian     : {res.variant}")
    print(f"  Karakter   : {len(res.text)}")
    print(f"{'─'*40}")
    print(f"TEKS:\n{res.text}")

    if not args.no_tts:
        if res.text.strip() and reader.is_acceptable(res):
            print("\nMemutar suara...")
            speak(res.text)
        else:
            print("\n[SKIP TTS] Teks tidak memenuhi threshold.")
            print(f"  Alasan: conf={res.mean_conf:.1f} (min 55), chars={len(res.text)} (min 3)")


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    print("=" * 50)
    print("  Tes Kamera & OCR v2.0")
    print(f"  Kamera: {args.cam} | Profile: {args.profile}")
    print("=" * 50)

    for i in range(1, args.repeat + 1):
        run_test(args, i)

    if args.repeat > 1:
        print(f"\n✓ Selesai {args.repeat} tangkapan.")


if __name__ == "__main__":
    main()
