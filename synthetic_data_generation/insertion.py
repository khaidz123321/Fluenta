"""
Cac ham chen loi noi lap gia lap vao audio sach (VIVOS).

QUAN TRONG - gioi han da biet truoc (Huong A, chua co forced alignment cap tu):
- Khong biet ranh gioi tu chinh xac trong audio, nen diem chen duoc chon
  bang cach do nang luong (RMS) de tim vung "co kha nang la am huu thanh"
  trong 15%-65% dau cua cau, KHONG phai ranh gioi tu that.
- Vi vay nhan gan cho du lieu sinh ra la nhan MUC DICH (biet truoc vi minh
  tu chen), nhung VI TRI trong cau la xap xi, khong chinh xac tuyet doi.
- Cai thien o Huong B: dung forced aligner (vd Montreal Forced Aligner) de
  chen dung ranh gioi tu/am tiet that, hoac dung TTS+LLM (LLM-Dys).

v2 - da them bien thien tu nhien (jitter) sau khi nghe thu ban v1 bi "robot":
- Lap am/Lap tu: moi lan lap deu bien thien nho ve am luong + toc do (resample),
  cach nhau 1 khoang lang cuc ngan, thay vi copy y het bit-by-bit.
- Keo dai am: tang do dai don vi lap + lam muot crossfade hon, giam cam giac
  rung tuan hoan (buzz).
- Block: thay lang tuyet doi bang nhieu tho nhe + fade in/out o bien, tranh
  cat cut kieu ky thuat so.
"""
from __future__ import annotations

import numpy as np


def _rms(seg: np.ndarray) -> float:
    return float(np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-9))


def find_voiced_window(
    audio: np.ndarray,
    sr: int,
    win_ms: float = 120,
    search_start: float = 0.15,
    search_end: float = 0.65,
    top_fraction: float = 0.4,
    rng: np.random.Generator | None = None,
) -> tuple[int, int]:
    """Tim 1 cua so co nang luong cao (xap xi am huu thanh) trong doan
    [search_start, search_end] cua chieu dai cau, tra ve (start_sample, win_len).
    Day la xap xi bang nang luong, KHONG phai ranh gioi tu chinh xac."""
    rng = rng or np.random.default_rng()
    win = max(1, int(sr * win_ms / 1000))
    n = len(audio)
    lo = int(n * search_start)
    hi = int(n * search_end) - win
    if hi <= lo:
        lo, hi = 0, max(1, n - win)
    step = max(1, win // 2)
    candidates = []
    for start in range(lo, max(lo + 1, hi), step):
        seg = audio[start : start + win]
        if len(seg) < win:
            continue
        candidates.append((_rms(seg), start))
    if not candidates:
        return 0, min(win, n)
    candidates.sort(key=lambda c: c[0], reverse=True)
    top_k = max(1, int(len(candidates) * top_fraction))
    idx = int(rng.integers(0, top_k))
    return candidates[idx][1], win


def _resample_linear(seg: np.ndarray, rate: float) -> np.ndarray:
    """Thay doi toc do phat (va ca cao do, giong tua may hat quay nhanh/cham)
    bang noi suy tuyen tinh don gian - dung de tao bien thien nho giua cac
    lan lap, khong can thu vien ngoai (librosa)."""
    if len(seg) < 2:
        return seg
    n_out = max(1, int(round(len(seg) / rate)))
    x_old = np.linspace(0, 1, len(seg))
    x_new = np.linspace(0, 1, n_out)
    return np.interp(x_new, x_old, seg.astype(np.float64))


def _jittered_repeats(unit: np.ndarray, sr: int, n_repeats: int, rng: np.random.Generator,
                       gain_jitter=0.12, rate_jitter=0.05, gap_ms_range=(15, 45)) -> np.ndarray:
    """Ghep n_repeats ban sao cua unit, moi ban co bien thien nho ve am luong
    va toc do, xen ke khoang lang cuc ngan - mo phong viec nguoi that khong
    bao gio lap lai 1 am/tu giong het nhau tung li."""
    pieces = []
    for i in range(n_repeats):
        rate = 1.0 + rng.uniform(-rate_jitter, rate_jitter)
        gain = 1.0 + rng.uniform(-gain_jitter, gain_jitter)
        piece = _resample_linear(unit, rate) * gain
        pieces.append(piece)
        if i < n_repeats - 1:
            gap_ms = rng.uniform(*gap_ms_range)
            pieces.append(np.zeros(int(sr * gap_ms / 1000)))
    out = np.concatenate(pieces)
    return out


def apply_sound_rep(audio, sr, n_repeats=3, unit_ms=90, rng=None):
    """Lap am (SoundRep): lap 1 mau am ngan (~duoi 1 am tiet) N lan, co bien
    thien tu nhien giua cac lan lap."""
    rng = rng or np.random.default_rng()
    start, win = find_voiced_window(audio, sr, win_ms=unit_ms, rng=rng)
    unit = audio[start : start + win].astype(np.float64)
    inserted = _jittered_repeats(unit, sr, n_repeats, rng)
    inserted = np.clip(inserted, -32768, 32767).astype(audio.dtype)
    out = np.concatenate([audio[:start], inserted, audio[start:]])
    return out, {
        "error_type": "lap_am",
        "insert_start_sec": round(start / sr, 3),
        "insert_end_sec": round((start + len(inserted)) / sr, 3),
        "n_repeats": n_repeats,
        "unit_ms": unit_ms,
    }


def apply_word_rep(audio, sr, n_repeats=2, unit_ms=420, rng=None):
    """Lap tu (WordRep): lap 1 doan dai hon (xap xi 1 tu) N lan, co bien thien
    tu nhien giua cac lan lap."""
    rng = rng or np.random.default_rng()
    start, win = find_voiced_window(audio, sr, win_ms=unit_ms, rng=rng)
    unit = audio[start : start + win].astype(np.float64)
    inserted = _jittered_repeats(unit, sr, n_repeats, rng)
    inserted = np.clip(inserted, -32768, 32767).astype(audio.dtype)
    out = np.concatenate([audio[:start], inserted, audio[start:]])
    return out, {
        "error_type": "lap_tu",
        "insert_start_sec": round(start / sr, 3),
        "insert_end_sec": round((start + len(inserted)) / sr, 3),
        "n_repeats": n_repeats,
        "unit_ms": unit_ms,
    }


def apply_prolongation(audio, sr, extend_ms=550, unit_ms=130, rng=None):
    """Keo dai am (Prolongation): lap-crossfade 1 doan (~130ms, dai hon ban v1)
    de giam cam giac rung tuan hoan (buzz), them crossfade rong hon cho muot."""
    rng = rng or np.random.default_rng()
    start, win = find_voiced_window(audio, sr, win_ms=unit_ms, rng=rng)
    unit = audio[start : start + win].astype(np.float64)
    fade = max(4, int(win * 0.4))
    n_loops = max(1, int(extend_ms / unit_ms))
    extended = unit.copy()
    for _ in range(n_loops):
        # moi vong lap bien thien nho toc do de tranh lap y het (giam buzz)
        rate = 1.0 + rng.uniform(-0.03, 0.03)
        seg = _resample_linear(unit, rate)
        if len(seg) < fade + 1:
            seg = unit.copy()
        cross = (
            np.linspace(1, 0, fade) * extended[-fade:]
            + np.linspace(0, 1, fade) * seg[:fade]
        )
        extended = np.concatenate([extended[:-fade], cross, seg[fade:]])
    extended = np.clip(extended, -32768, 32767).astype(audio.dtype)
    out = np.concatenate([audio[:start], extended, audio[start + win :]])
    return out, {
        "error_type": "keo_dai_am",
        "insert_start_sec": round(start / sr, 3),
        "insert_end_sec": round((start + len(extended)) / sr, 3),
        "extend_ms": extend_ms,
        "unit_ms": unit_ms,
    }


def _breath_noise(n_samples: int, sr: int, ref_rms: float, level=0.06, rng=None) -> np.ndarray:
    """Sinh 1 doan nhieu nhe (loc thong thap don gian bang trung binh truot)
    mo phong tieng tho/cang co thanh quan khi khung, thay vi lang tuyet doi."""
    rng = rng or np.random.default_rng()
    noise = rng.normal(0, 1, n_samples)
    # loc thong thap don gian (moving average) de nghe "mem" hon white noise thuan
    k = max(1, int(sr * 0.002))
    kernel = np.ones(k) / k
    noise = np.convolve(noise, kernel, mode="same")
    noise = noise / (np.max(np.abs(noise)) + 1e-9)
    return noise * ref_rms * level


def apply_block(audio, sr, silence_ms=650, rng=None):
    """Block (khung/ngap ngung): chen khoang gan-lang (co tieng tho nhe, khong
    phai lang tuyet doi 0) + fade in/out o bien, tranh cat cut ky thuat so."""
    rng = rng or np.random.default_rng()
    start, probe_win = find_voiced_window(audio, sr, win_ms=50, rng=rng)
    local_rms = _rms(audio[start : start + probe_win])
    n = int(sr * silence_ms / 1000)
    filler = _breath_noise(n, sr, local_rms, rng=rng).astype(np.float64)
    fade = min(n // 4, int(sr * 0.03))
    if fade > 1:
        fade_in = np.linspace(0, 1, fade)
        fade_out = np.linspace(1, 0, fade)
        filler[:fade] *= fade_in
        filler[-fade:] *= fade_out
    filler = np.clip(filler, -32768, 32767).astype(audio.dtype)
    out = np.concatenate([audio[:start], filler, audio[start:]])
    return out, {
        "error_type": "khung_ngap_ngung",
        "insert_start_sec": round(start / sr, 3),
        "insert_end_sec": round((start + len(filler)) / sr, 3),
        "silence_ms": silence_ms,
    }


ERROR_FUNCS = {
    "lap_am": apply_sound_rep,
    "lap_tu": apply_word_rep,
    "keo_dai_am": apply_prolongation,
    "khung_ngap_ngung": apply_block,
}
