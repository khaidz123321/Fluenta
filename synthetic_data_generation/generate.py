"""
Script chinh: sinh du lieu noi lap gia lap tu VIVOS (Huong A).

Cach chay (vi du sinh thu 30 file) - chay tu bat ky dau, vd tu thu muc goc du an:
    python synthetic_data_generation/generate.py --n 30 --seed 42

Dau ra:
    Dataset/synth_stutter_vn/audio/*.wav
    Dataset/synth_stutter_vn/labels.csv

GIOI HAN DA BIET (xem them insertion.py):
- Chua co "Chem tu" (Interjection) vi can 1 kho am thanh dem ("a", "u")
  rieng, hien chua co nguon sach -> de danh cho ban mo rong sau.
- Diem chen loi la xap xi (do nang luong), khong phai ranh gioi tu chinh xac.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from insertion import ERROR_FUNCS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIVOS_ROOT = PROJECT_ROOT / "Dataset/vivos_data/data/vivos/train"
OUT_DIR = PROJECT_ROOT / "Dataset/synth_stutter_vn"


def load_prompts(prompts_path: Path) -> list[tuple[str, str]]:
    rows = []
    with open(prompts_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id, text = line.split(" ", 1)
            rows.append((utt_id, text))
    return rows


def wav_path_for(utt_id: str) -> Path:
    speaker = utt_id.split("_")[0]
    return VIVOS_ROOT / "waves" / speaker / f"{utt_id}.wav"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="so cau se xu ly")
    ap.add_argument("--fluent_ratio", type=float, default=0.2, help="ty le giu nguyen, khong chen loi (nhan Troi_chay)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_dur_sec", type=float, default=2.0, help="bo qua cau qua ngan, khong du cho de chen loi")
    args = ap.parse_args()

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_out = OUT_DIR / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(VIVOS_ROOT / "prompts.txt")
    random.shuffle(prompts)

    error_types = list(ERROR_FUNCS.keys())
    rows_written = 0
    label_rows = []

    for utt_id, text in prompts:
        if rows_written >= args.n:
            break
        wav_p = wav_path_for(utt_id)
        if not wav_p.exists():
            continue
        audio, sr = sf.read(str(wav_p), dtype="int16")
        if audio.ndim > 1:
            audio = audio[:, 0]
        if len(audio) / sr < args.min_dur_sec:
            continue

        is_fluent = random.random() < args.fluent_ratio
        if is_fluent:
            out_audio = audio
            meta = {"error_type": "troi_chay", "insert_start_sec": "", "insert_end_sec": ""}
        else:
            err = random.choice(error_types)
            func = ERROR_FUNCS[err]
            out_audio, meta = func(audio, sr, rng=rng)

        out_name = f"{utt_id}__{meta['error_type']}.wav"
        sf.write(str(audio_out / out_name), out_audio, sr, subtype="PCM_16")

        row = {
            "output_file": out_name,
            "source_vivos_id": utt_id,
            "source_text": text,
            "error_type": meta["error_type"],
            "insert_start_sec": meta.get("insert_start_sec", ""),
            "insert_end_sec": meta.get("insert_end_sec", ""),
            "params": {k: v for k, v in meta.items() if k not in ("error_type", "insert_start_sec", "insert_end_sec")},
        }
        label_rows.append(row)
        rows_written += 1

    labels_csv = OUT_DIR / "labels.csv"
    write_header = not labels_csv.exists()
    with open(labels_csv, "a", encoding="utf-8", newline="") as f:
        fieldnames = ["output_file", "source_vivos_id", "source_text", "error_type", "insert_start_sec", "insert_end_sec", "params"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for row in label_rows:
            w.writerow(row)

    print(f"Da sinh {rows_written} file vao {audio_out}")
    print(f"Nhan luu tai {labels_csv}")
    from collections import Counter
    c = Counter(r["error_type"] for r in label_rows)
    print("Phan bo loai loi trong lo nay:", dict(c))


if __name__ == "__main__":
    main()
