"""Đọc nhãn SEP-28k, ghép với các clip đã cắt, gán một nhãn cho mỗi clip và ghi manifest.csv.

Quy tắc gán nhãn (bài không mô tả cụ thể, đây là giả định của nhóm):
  - R = max(SoundRep, WordRep), vì bài gộp hai loại lặp thành một lớp.
  - Trong R, P, B, I, lớp nào có >= MIN_VOTES phiếu thì ứng viên; chọn lớp nhiều phiếu nhất
    (hòa thì theo thứ tự R, P, B, I).
  - Không có lớp lỗi nào đạt ngưỡng mà NoStutteredWords >= MIN_VOTES thì gán F (trôi chảy).
  - Còn lại (không có tiếng nói, nhạc, không rõ...) thì loại, giống bài bỏ các nhãn không phải nói lắp.
Manifest vẫn giữ nguyên số phiếu SoundRep/WordRep riêng để sau này chuyển sang 6 nhãn đa nhãn.
"""
import argparse
import csv
from collections import Counter

import config

VOTE_COLS = ["Unsure", "PoorAudioQuality", "Prolongation", "Block", "SoundRep", "WordRep",
             "DifficultToUnderstand", "Interjection", "NoStutteredWords", "NaturalPause", "Music", "NoSpeech"]


def assign_label(v):
    votes = {"R": max(v["SoundRep"], v["WordRep"]), "P": v["Prolongation"],
             "B": v["Block"], "I": v["Interjection"]}
    passed = [c for c in config.DISFLUENT if votes[c] >= config.MIN_VOTES]
    if passed:
        return max(passed, key=lambda c: votes[c])
    if v["NoStutteredWords"] >= config.MIN_VOTES:
        return "F"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="chỉ lấy N clip đầu tiên có file (để chạy thử)")
    ap.add_argument("--per-class", type=int, default=0,
                    help="chỉ lấy N clip đầu tiên của mỗi lớp (chạy thử có đủ các lớp)")
    args = ap.parse_args()
    taken = Counter()

    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    rows_out, n_missing, n_dropped = [], 0, 0
    with open(config.LABELS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, skipinitialspace=True):
            show, ep, clip = r["Show"].strip(), r["EpId"].strip(), r["ClipId"].strip()
            path = config.CLIPS_DIR / show / ep / f"{show}_{ep}_{clip}.wav"
            if not path.exists():
                n_missing += 1
                continue
            v = {c: int(r[c]) for c in VOTE_COLS}
            label = assign_label(v)
            if label is None:
                n_dropped += 1
                continue
            if args.per_class and taken[label] >= args.per_class:
                continue
            taken[label] += 1
            rows_out.append({"path": str(path), "show": show, "ep_id": ep, "clip_id": clip,
                             "group": f"{show}_{ep}", "label": label, **v})
            if args.limit and len(rows_out) >= args.limit:
                break
            if args.per_class and all(taken[c] >= args.per_class for c in config.CLASSES):
                break

    with open(config.MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    dist = Counter(r["label"] for r in rows_out)
    print(f"Đã ghi {len(rows_out)} clip vào {config.MANIFEST}")
    print(f"Bỏ qua: {n_missing} dòng nhãn không có file clip, {n_dropped} clip không đủ phiếu cho lớp nào")
    print("Phân bố nhãn:", {c: dist.get(c, 0) for c in config.CLASSES})
    print("Số nhóm podcast (tập):", len({r['group'] for r in rows_out}))


if __name__ == "__main__":
    main()
