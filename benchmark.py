#!/usr/bin/env python3
"""
benchmark.py - Alat benchmark untuk membandingkan performa OCR.

Gunakan ini untuk mengukur peningkatan sebelum dan sesudah optimasi.

Pemakaian:
    python benchmark.py                         # benchmark semua gambar di folder test_images/
    python benchmark.py --images a.jpg b.jpg    # benchmark gambar tertentu
    python benchmark.py --profile fast          # bandingkan profile fast saja
    python benchmark.py --compare               # bandingkan fast vs accurate
    python benchmark.py --ground-truth gt.txt   # hitung CER jika ada ground truth

Format file ground truth (gt.txt):
    nama_file.jpg|Teks yang seharusnya terbaca di gambar tersebut
    buku.jpg|Ini adalah teks dari buku
"""

import cv2
import time
import sys
import os
import argparse
import difflib
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from ocr_pro import Cfg, OCRReader, normalize_for_tts, sharpness

logger = logging.getLogger(__name__)


@dataclass
class BenchResult:
    path: str
    profile: str
    mean_conf: float
    n_chars: int
    n_words: int
    elapsed: float
    sharpness_score: float
    variant: str
    text: str
    cer: Optional[float] = None     # Character Error Rate (jika ada ground truth)
    acceptable: bool = True


def cer(reference: str, hypothesis: str) -> float:
    """
    Hitung Character Error Rate (CER).
    CER = (edit_distance) / len(reference)
    0.0 = sempurna, 1.0 = salah semua
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    sm = difflib.SequenceMatcher(None, reference.lower(), hypothesis.lower())
    # Edit distance sederhana menggunakan SequenceMatcher
    matches = sm.ratio() * (len(reference) + len(hypothesis)) / 2
    edits = len(reference) + len(hypothesis) - 2 * matches
    return min(1.0, edits / max(len(reference), 1))


def load_ground_truth(path: str) -> Dict[str, str]:
    """Muat file ground truth (format: nama_file.jpg|teks)."""
    gt = {}
    if not os.path.exists(path):
        return gt
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "|" in line:
                fname, text = line.split("|", 1)
                gt[fname.strip()] = text.strip()
    return gt


def benchmark_image(
    path: str,
    profile: str,
    gt_text: Optional[str] = None
) -> BenchResult:
    """Jalankan OCR pada satu gambar dan kembalikan metrik."""
    img = cv2.imread(path)
    if img is None:
        logger.error(f"Tidak dapat membuka: {path}")
        return BenchResult(path, profile, 0, 0, 0, 0, 0, "error", "", None, False)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharp = sharpness(gray)

    reader = OCRReader(Cfg(
        lang="ind+eng",
        profile=profile,
        psm_list=(6, 3, 11) if profile == "accurate" else (6,),
        drop_conf=45.0 if profile == "accurate" else 35.0,
        min_mean_conf=55.0,
        min_chars=3,
    ))

    t0 = time.time()
    res = reader.read(img)
    elapsed = time.time() - t0

    n_words = len(res.text.split()) if res.text.strip() else 0
    acceptable = reader.is_acceptable(res)

    char_err = None
    if gt_text and res.text:
        char_err = cer(gt_text, res.text)

    return BenchResult(
        path=path,
        profile=profile,
        mean_conf=res.mean_conf,
        n_chars=len(res.text),
        n_words=n_words,
        elapsed=elapsed,
        sharpness_score=sharp,
        variant=res.variant,
        text=res.text[:200],
        cer=char_err,
        acceptable=acceptable,
    )


def print_table(results: List[BenchResult], show_text: bool = True):
    """Cetak hasil benchmark dalam format tabel."""
    sep = "─" * 90

    print(f"\n{'=' * 90}")
    print(f"{'HASIL BENCHMARK':^90}")
    print(f"{'=' * 90}")

    # Header
    header = f"{'File':<25} {'Profile':<10} {'Conf':>6} {'Chars':>6} {'Words':>6} {'Waktu(s)':>9} {'Sharp':>7} {'OK':>4}"
    if any(r.cer is not None for r in results):
        header += f" {'CER':>6}"
    print(header)
    print(sep)

    for r in results:
        fname = os.path.basename(r.path)[:24]
        ok_sym = "✓" if r.acceptable else "✗"
        row = (f"{fname:<25} {r.profile:<10} {r.mean_conf:>6.1f} {r.n_chars:>6} "
               f"{r.n_words:>6} {r.elapsed:>9.2f} {r.sharpness_score:>7.0f} {ok_sym:>4}")
        if r.cer is not None:
            row += f" {r.cer:>6.3f}"
        print(row)

    print(sep)

    # Summary statistik
    if results:
        avg_conf   = sum(r.mean_conf for r in results) / len(results)
        avg_time   = sum(r.elapsed for r in results) / len(results)
        avg_chars  = sum(r.n_chars for r in results) / len(results)
        n_ok       = sum(1 for r in results if r.acceptable)
        avg_cer    = (sum(r.cer for r in results if r.cer is not None) /
                      max(1, sum(1 for r in results if r.cer is not None)))

        print(f"\n📊 RINGKASAN ({len(results)} gambar):")
        print(f"   Rata-rata confidence : {avg_conf:.1f}")
        print(f"   Rata-rata waktu      : {avg_time:.2f}s")
        print(f"   Rata-rata karakter   : {avg_chars:.0f}")
        print(f"   Gambar diterima      : {n_ok}/{len(results)} ({n_ok/len(results)*100:.0f}%)")
        if any(r.cer is not None for r in results):
            print(f"   Rata-rata CER        : {avg_cer:.3f} ({avg_cer*100:.1f}%)")

    if show_text:
        print(f"\n{'=' * 90}")
        print("PRATINJAU TEKS (200 karakter pertama):")
        print(f"{'=' * 90}")
        for r in results:
            fname = os.path.basename(r.path)
            status = "✓" if r.acceptable else "✗ [DITOLAK]"
            print(f"\n[{fname}] {status} conf={r.mean_conf:.1f}")
            print(f"  {r.text[:200]!r}")


def save_results_json(results: List[BenchResult], out_path: str = "benchmark_results.json"):
    """Simpan hasil benchmark ke file JSON."""
    data = []
    for r in results:
        data.append({
            "path": r.path,
            "profile": r.profile,
            "mean_conf": round(r.mean_conf, 2),
            "n_chars": r.n_chars,
            "n_words": r.n_words,
            "elapsed_s": round(r.elapsed, 3),
            "sharpness": round(r.sharpness_score, 1),
            "variant": r.variant,
            "acceptable": r.acceptable,
            "cer": round(r.cer, 4) if r.cer is not None else None,
            "text_preview": r.text[:100],
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Hasil disimpan ke: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark OCR — ukur dan bandingkan performa",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--images", nargs="*",
                        help="File gambar untuk di-benchmark (default: semua JPG di test_images/)")
    parser.add_argument("--profile", choices=["fast", "accurate", "turbo", "all"],
                        default="accurate",
                        help="Profile OCR yang diuji. 'all' = bandingkan semua profile")
    parser.add_argument("--ground-truth", default=None,
                        help="File ground truth untuk hitung CER (format: file.jpg|teks)")
    parser.add_argument("--no-text", action="store_true",
                        help="Jangan tampilkan pratinjau teks")
    parser.add_argument("--save-json", action="store_true",
                        help="Simpan hasil ke benchmark_results.json")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Kumpulkan daftar gambar
    if args.images:
        image_paths = [p for p in args.images if os.path.exists(p)]
    else:
        test_dir = "test_images"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
            print(f"📁 Folder '{test_dir}/' dibuat.")
            print(f"   Taruh gambar JPG/PNG ke folder tersebut lalu jalankan ulang.")
            print(f"   Atau gunakan: python benchmark.py --images foto1.jpg foto2.jpg")
            return
        image_paths = [
            os.path.join(test_dir, f)
            for f in sorted(os.listdir(test_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

    if not image_paths:
        print("⚠  Tidak ada gambar yang ditemukan.")
        print("   Gunakan: python benchmark.py --images foto.jpg")
        return

    print(f"\n🔍 Benchmark {len(image_paths)} gambar...")

    # Ground truth
    gt = {}
    if args.ground_truth:
        gt = load_ground_truth(args.ground_truth)
        print(f"   Ground truth dimuat: {len(gt)} entri")

    # Tentukan profile yang diuji
    profiles = ["fast", "accurate"] if args.profile == "all" else [args.profile]

    all_results = []
    for profile in profiles:
        print(f"\n📌 Profile: {profile}")
        for path in image_paths:
            fname = os.path.basename(path)
            gt_text = gt.get(fname)
            print(f"   → {fname}...", end="", flush=True)
            r = benchmark_image(path, profile, gt_text)
            all_results.append(r)
            status = "✓" if r.acceptable else "✗"
            print(f" [{status}] conf={r.mean_conf:.1f} chars={r.n_chars} waktu={r.elapsed:.2f}s")

    print_table(all_results, show_text=not args.no_text)

    if args.save_json:
        save_results_json(all_results)


if __name__ == "__main__":
    main()
