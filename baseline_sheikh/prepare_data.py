"""Chuyển nhãn SEP-28k thành manifest đơn nhãn R/P/B/I/F.

SEP-28k cho phép nhiều nhãn một clip; bài 2306.00689 không công bố quy tắc
quy đổi. Quy tắc ở đây là giả định tái hiện và được ghi vào metadata JSON.
"""

import argparse
import csv
import json
from collections import Counter

import config

VOTE_COLS = (
    "Unsure", "PoorAudioQuality", "Prolongation", "Block", "SoundRep",
    "WordRep", "DifficultToUnderstand", "Interjection", "NoStutteredWords",
    "NaturalPause", "Music", "NoSpeech",
)
FIELDS = ("path", "show", "ep_id", "clip_id", "group", "label", *VOTE_COLS)


def assign_label(v):
    # Suy ra từ Bảng 1: giữ đúng một loại lỗi có >=2 phiếu; loại đa loại lỗi.
    votes = {
        "R": max(v["SoundRep"], v["WordRep"]),
        "P": v["Prolongation"],
        "B": v["Block"],
        "I": v["Interjection"],
    }
    eligible = [c for c in config.DISFLUENT if votes[c] >= config.MIN_VOTES]
    if len(eligible) == 1:
        return eligible[0]
    if len(eligible) > 1:
        return None
    return "F" if v["NoStutteredWords"] >= config.MIN_VOTES else None


def build_manifest(limit=0, per_class=0):
    if not config.LABELS_CSV.is_file():
        raise FileNotFoundError(f"Thiếu {config.LABELS_CSV}")
    if not config.CLIPS_DIR.is_dir():
        raise FileNotFoundError(f"Thiếu {config.CLIPS_DIR}")

    rows, counts = [], Counter()
    with config.LABELS_CSV.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, skipinitialspace=True):
            show, ep, clip = (row[col].strip() for col in ("Show", "EpId", "ClipId"))
            path = config.CLIPS_DIR / show / ep / f"{show}_{ep}_{clip}.wav"
            counts["total"] += 1
            if not path.is_file():
                counts["missing_audio"] += 1
                continue
            votes = {col: int(row[col]) for col in VOTE_COLS}
            label = assign_label(votes)
            if label is None:
                counts["excluded"] += 1
                continue
            if per_class and counts[label] >= per_class:
                continue
            rows.append({
                "path": str(path.resolve()), "show": show, "ep_id": ep,
                "clip_id": clip, "group": f"{show}_{ep}", "label": label, **votes,
            })
            counts[label] += 1
            if limit and len(rows) >= limit:
                break
            if per_class and all(counts[c] >= per_class for c in config.CLASSES):
                break

    if not rows:
        raise ValueError("Không có clip hợp lệ. Kiểm tra SEP_DIR và cấu trúc clips_output.")
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    with config.MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "dataset": str(config.SEP_DIR.resolve()),
        "label_rule": "exactly one R/P/B/I with >=2 votes; else F if NoStutteredWords>=2; R=max(sound,word)",
        "counts": dict(counts),
        "rows": len(rows),
        "podcast_episodes": len({row["group"] for row in rows}),
    }
    config.MANIFEST.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Manifest: {config.MANIFEST} ({len(rows)} clip)")
    print(f"Thiếu audio: {counts['missing_audio']}; loại theo nhãn: {counts['excluded']}")
    print("Phân bố:", {c: counts[c] for c in config.CLASSES})
    print("Số tập podcast:", metadata["podcast_episodes"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-class", type=int, default=0)
    args = parser.parse_args()
    build_manifest(args.limit, args.per_class)
