"""
Huong B (buoc 1 - thu truoc, chua fine-tune): dung TTS tieng Viet co san
(facebook/mms-tts-vie, mo nguon, tai qua HuggingFace) de sinh am thanh loi
lap TRUC TIEP TU VAN BAN da bi thao tac san (vd "khong khong khong"),
thay vi cat-dan audio nhu Huong A.

Cach chay (vi du sinh thu 10 cau):
    python synthetic_data_generation/tts_synthesis.py --n 10 --seed 42

Dau ra:
    Dataset/synth_stutter_vn_ttsB/audio/*.wav
    Dataset/synth_stutter_vn_ttsB/labels.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIVOS_ROOT = PROJECT_ROOT / "Dataset/vivos_data/data/vivos/train"
OUT_DIR = PROJECT_ROOT / "Dataset/synth_stutter_vn_ttsB"

MODEL_ID = "facebook/mms-tts-vie"


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


def make_disfluent_text(text: str, rng: random.Random) -> tuple[str, dict]:
    """Thao tac van ban de tao lỗi lap, TTS se tu tong hop giong tu nhien
    theo dung van ban nay (khong ghep am thanh thu cong).

    Luu y: day la Huong B don gian (thao tac text truoc TTS), KHONG phai
    port nguyen pipeline LLM-Dys (vi pipeline goc chi ho tro tieng Anh)."""
    words = text.split()
    if len(words) < 2:
        return text, {"error_type": "troi_chay"}

    err = rng.choice(["lap_tu", "lap_am", "keo_dai_am", "khung_ngap_ngung", "troi_chay"])
    idx = rng.randint(0, len(words) - 1)
    w = words[idx]

    if err == "lap_tu":
        n = rng.choice([2, 3])
        new_words = words[:idx] + [w] * n + words[idx + 1 :]
        out_text = " ".join(new_words)
        return out_text, {
            "error_type": "lap_tu", "word": w, "n_repeats": n, "word_index": idx,
            "words_before": words[:idx],
        }

    if err == "lap_am":
        # lap am dau cua tu (xap xi bang 1-2 ky tu dau, vi tieng Viet don am tiet)
        frag = w[: max(1, len(w) // 2)]
        n = rng.choice([2, 3])
        new_word = "-".join([frag] * n) + "-" + w
        new_words = words[:idx] + [new_word] + words[idx + 1 :]
        out_text = " ".join(new_words)
        return out_text, {"error_type": "lap_am", "word": w, "fragment": frag, "n_repeats": n, "word_index": idx}

    if err == "keo_dai_am":
        # v2: KHONG con nhan ban chu cai nguyen am trong van ban nua - da kiem
        # chung truoc do rang thu thuat nay khien TTS phat am RO RANG tung
        # ky tu duoc nhan ban (nghe nhu doc chinh xac 1 tu dai hon), thay vi
        # tao cam giac "keo dai/mo ho" nhu nguoi that. Thay vao do: giu nguyen
        # van ban troi chay, TTS tong hop binh thuong, roi keo dai THOI LUONG
        # THAT cua doan audio chua tu do bang time-stretch (giu nguyen cao do)
        # - ky thuat nay co can cu nghien cuu: WSOLA/Tempo Perturbation duoc
        # AS-70 challenge paper (arXiv 2409.05430, doi T006, trich dan
        # Verhelst & Roelands 1993) dung de mo phong "slower speech cadence
        # observed in PWS". Khac voi Huong A (ghep noi tu 2 nguon am thanh
        # khac nhau, da bi bac bo vi nghe "thoo thien"), o day chi keo dai MOT
        # DOAN LIEN TUC trong CUNG MOT file da tong hop, khong co diem noi
        # giua 2 nguon khac nhau nen it rui ro tao vet ghep.
        words_before = words[:idx]
        return text, {
            "error_type": "keo_dai_am", "word": w, "word_index": idx,
            "words_before": words_before, "stretch_factor": rng.choice([2.2, 2.6, 3.0]),
        }

    if err == "khung_ngap_ngung":
        # v4: sau khi xac dinh lai (nguoi dung dong y) rang cam giac "am mo
        # ho/am ho ậm ừ" thuoc ve Chêm từ (Interjection) chu KHONG phai Block -
        # dung dinh nghia SEP-28k phan biet 2 loai nay - Block chi con nhiem
        # vu la KHOANG LANG, khong can co "am thanh mo ho" nao ca.
        #
        # Doi huong tu trick dau phay (v3, kiem soat gian tiep qua tokenizer,
        # khong chinh duoc chinh xac do dai) sang CHEN KHOANG LANG THAT o tang
        # audio, co kiem soat truc tiep bang giay - dung phuong phap ma ca
        # LLM-Dys (chen lang 0.8-3.5s muc tu, dua tren forced-alignment) va
        # AS-70/T006 (ky thuat "InsertSilence") deu dung cho Block. Van giu
        # van ban TROI CHAY (khong chen text gi), TTS tong hop binh thuong,
        # sau do chen lang ngay TRUOC tu muc tieu.
        #
        # LUU Y phan biet voi that bai truoc day: lan nay LA chen mot doan
        # LANG (val=0) vao GIUA audio cua CHINH 1 clip da tong hop - khac voi
        # that bai da ghi nhan truoc do la ghep 2 NGUON AM THANH KHAC NHAU
        # (TTS + nguoi khac) voi nhau (luon lo diem noi vi 2 giong/prosody
        # khac nhau). O day dung crossfade (nhu da xac nhan hieu qua o Lap tu)
        # de tranh tieng click tai diem chen.
        words_before = words[:idx]
        silence_sec = rng.choice([0.8, 1.2, 1.6, 2.2])
        return text, {
            "error_type": "khung_ngap_ngung", "word_index": idx,
            "words_before": words_before, "silence_sec": silence_sec,
        }

    return text, {"error_type": "troi_chay"}


def _crossfade_concat(segments: list[np.ndarray], sr: int, fade_ms: float = 12.0) -> np.ndarray:
    """Noi nhieu doan audio lai voi CROSSFADE tai moi diem noi, thay vi
    concatenate() thuan tuy. Nguyen nhan can ham nay: nguoi dung phan hoi ban
    v2 cua lap_tu nghe "giong bi lag/cat ghep" - do np.concatenate ghep truc
    tiep 2 mau tin hieu khong lien tuc ve bien do/pha tai diem noi (kho de
    dung dung diem qua-0), tao tieng click/discontinuity nghe nhu vet cat.
    Crossfade tuyen tinh (~12ms, du ngan de khong lam mo tieng noi nhung du
    dai de loai bo click) la ky thuat khu-click tieu chuan trong xu ly am
    thanh, ap dung duoc cho ca doan lang (fade ve 0 roi fade len tu 0, thay vi
    cat cung sang im lang tuyet doi)."""
    if not segments:
        return np.array([], dtype=np.float32)
    out = segments[0].astype(np.float32)
    fade_len_default = max(1, int(sr * fade_ms / 1000))
    for seg in segments[1:]:
        seg = seg.astype(np.float32)
        fade_len = min(fade_len_default, len(out), len(seg))
        if fade_len <= 1:
            out = np.concatenate([out, seg])
            continue
        fade_out = np.linspace(1.0, 0.0, fade_len)
        fade_in = np.linspace(0.0, 1.0, fade_len)
        overlap = out[-fade_len:] * fade_out + seg[:fade_len] * fade_in
        out = np.concatenate([out[:-fade_len], overlap, seg[fade_len:]])
    return out


def apply_lap_tu_emphasis(wav: np.ndarray, disfluent_text: str, word: str, n_repeats: int,
                           word_index: int, words_before: list[str], sr: int,
                           peak_gain: float = 2.1, pause_sec: float = 0.16) -> np.ndarray:
    """v3 - sua tiep theo phan hoi nguoi dung ("van co cam giac bi lag/cat
    ghep"): v2 dung np.concatenate truc tiep giua wav goc / khoang lang /
    doan bat am luong -> tao click/discontinuity ro ret tai tung diem noi.
    v3 giu nguyen y tuong (khoang ngung ngan + bat am luong dot ngot tren lan
    lap cuoi) nhung thay TAT CA cac diem noi bang crossfade (_crossfade_concat)
    de loai bo tieng cat.

    LUU Y RO RANG (giu nguyen tu v2, theo yeu cau kiem chung cua nguoi dung):
    y tuong "khoang ngung truoc tu bat ra" la SUY LUAN CUA TOI, khong trich
    dan tu paper nao - chi la ky thuat khu-click (crossfade) o day co can cu
    ky thuat chuan, con cau truc noi dung (pause + burst) van can nguoi dung
    tu danh gia bang tai."""
    total_chars = len(disfluent_text)
    if total_chars == 0:
        return wav

    prefix = " ".join(words_before)
    chars_before = len(prefix) + (1 if words_before else 0)
    word_len = len(word)

    n = len(wav)
    to_sample = lambda c: int(n * c / total_chars)

    # vi tri (uoc luong ti le ky tu) cua LAN LAP CUOI: word #(n_repeats-1),
    # cach nhau boi 1 dau cach voi cac lan truoc
    last_chars_before = chars_before + (n_repeats - 1) * (word_len + 1)
    last_start = min(to_sample(last_chars_before), n)
    last_end = min(to_sample(last_chars_before + word_len), n)
    if last_end <= last_start:
        return wav

    pause_len = int(sr * pause_sec)
    pause = np.zeros(pause_len, dtype=np.float32)

    burst = np.clip(wav[last_start:last_end].astype(np.float32) * peak_gain, -1.0, 1.0)

    out = _crossfade_concat([wav[:last_start], pause, burst, wav[last_end:]], sr)
    return out.astype(wav.dtype) if wav.dtype.kind != "f" else out.astype(wav.dtype)


def _estimate_noise_floor(wav: np.ndarray, sr: int, window_ms: float = 20.0) -> float:
    """Uoc luong bien do noise floor thuc cua 1 file TTS bang RMS cua 20% cua
    so nho nhat trong cac cua so 20ms (tuc la doan "yen tinh nhat" trong
    chinh clip do, dai dien cho muc nen khi khong co giong noi ro)."""
    win = max(1, int(sr * window_ms / 1000))
    n_win = len(wav) // win
    if n_win == 0:
        return 0.0
    rms = np.array([np.sqrt(np.mean(wav[i * win:(i + 1) * win].astype(np.float64) ** 2)) for i in range(n_win)])
    return float(np.percentile(rms, 10))


def _make_room_tone(n: int, amp: float, sr: int) -> np.ndarray:
    """Sinh 1 doan noise bien do rat thap (room tone), da loc thong thap don
    gian (moving average ~2ms) de nghe "mem" hon white noise thuan, mo phong
    nen am tu nhien thay vi im lang tuyet doi."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    noise = np.random.randn(n).astype(np.float32)
    kernel_len = max(1, int(sr * 0.002))
    if kernel_len > 1:
        kernel = np.ones(kernel_len, dtype=np.float32) / kernel_len
        noise = np.convolve(noise, kernel, mode="same")
    peak = np.max(np.abs(noise)) + 1e-8
    return (noise / peak * amp).astype(np.float32)


def _find_low_energy_cut(wav: np.ndarray, sr: int, target_idx: int,
                          search_ms: float = 120.0, sub_win_ms: float = 8.0) -> int:
    """Quet 1 cua so nho (+-search_ms) quanh target_idx, tim diem co RMS thap
    nhat trong cac cua so con ~8ms - tuc khoang ngat tu nhien giua 2 am tiet
    ma chinh TTS da tu tao ra (khong phai im lang tuyet doi, chi la nang
    luong thap hon). Muc dich: tranh cat vao GIUA luc mot tu dang phat am do
    (nguyen nhan gay cam giac "cat ghep" nguoi dung phat hien), thay vi chi
    dua vao uoc luong ti le ky tu tho. Y tuong tuong tu find_voiced_window da
    dung trong insertion.py (Huong A)."""
    search = int(sr * search_ms / 1000)
    sub = max(1, int(sr * sub_win_ms / 1000))
    lo = max(0, target_idx - search)
    hi = min(len(wav), target_idx + search)
    if hi - lo < sub:
        return target_idx

    best_idx, best_rms = target_idx, None
    i = lo
    step = max(1, sub // 2)
    while i + sub <= hi:
        seg = wav[i:i + sub].astype(np.float64)
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if best_rms is None or rms < best_rms:
            best_rms = rms
            best_idx = i + sub // 2
        i += step
    return best_idx


def apply_block_silence(wav: np.ndarray, text: str, words_before: list[str],
                         sr: int, silence_sec: float = 1.2) -> np.ndarray:
    """Chen 1 khoang NGUNG NGAY TRUOC tu muc tieu, mo phong Block (nghen/ngung
    do nói lap) - co can cu tu LLM-Dys (chen lang 0.8-3.5s muc tu dua tren
    forced-alignment) va AS-70/T006 (ky thuat "InsertSilence"). Vi tri chen
    uoc luong ti le theo so ky tu (nhu cac ham khac trong file nay, do
    MMS-TTS khong tra ve alignment tung tu that).

    v2 - sua theo phan hoi nguoi dung (van nghe "nhu cat ghep/khoang trong"
    du da dung crossfade khu click): PHAN NAY LA SUY LUAN KY THUAT CUA TOI,
    KHONG trich dan tu paper - ca LLM-Dys va AS-70 khi mo ta "chen khoang
    lang" deu khong noi ro co xu ly noise floor hay khong. Nguyen nhan nghi
    ngo: im lang TUYET DOI (gia tri 0) tuong phan manh voi noise floor von co
    cua audio TTS xung quanh, du khong con click nhung van tao cam giac "ho"
    bat thuong. Thay vao do: do noise floor THAT cua chinh clip nay va sinh 1
    doan room-tone bien do thap khop voi no thay cho im lang tuyet doi."""
    total_chars = len(text)
    if total_chars == 0:
        return wav

    prefix = " ".join(words_before)
    chars_before = len(prefix) + (1 if words_before else 0)

    n = len(wav)
    cut_est = min(int(n * chars_before / total_chars), n)
    if cut_est <= 0 or cut_est >= n:
        return wav
    cut = _find_low_energy_cut(wav, sr, cut_est)
    cut = max(1, min(cut, n - 1))

    noise_floor = _estimate_noise_floor(wav, sr)
    room_tone = _make_room_tone(int(sr * silence_sec), max(noise_floor, 1e-4) * 1.4, sr)
    out = _crossfade_concat([wav[:cut], room_tone, wav[cut:]], sr)
    return out.astype(wav.dtype)


def apply_prolongation_stretch(wav: np.ndarray, text: str, word: str, word_index: int,
                                words_before: list[str], stretch_factor: float = 2.4) -> np.ndarray:
    """Keo dai THOI LUONG THAT cua doan audio chua tu muc tieu bang time-stretch
    (WSOLA qua librosa.effects.time_stretch), giu nguyen cao do - khong doi
    van ban, khong ghep tu nguon khac. Vi tri doan can keo dai duoc uoc luong
    ti le theo so ky tu (cung cach uoc luong da dung o apply_lap_tu_emphasis,
    do MMS-TTS khong tra ve alignment tung tu)."""
    import librosa

    total_chars = len(text)
    if total_chars == 0:
        return wav

    prefix = " ".join(words_before)
    chars_before = len(prefix) + (1 if words_before else 0)
    chars_word = len(word)

    n = len(wav)
    start = int(n * chars_before / total_chars)
    end = int(n * (chars_before + chars_word) / total_chars)
    end = min(end, n)
    if end <= start:
        return wav

    segment = wav[start:end].astype(np.float32)
    # rate < 1.0 lam CHAM/KEO DAI doan audio; rate = 1/stretch_factor
    stretched = librosa.effects.time_stretch(segment, rate=1.0 / stretch_factor)

    out = np.concatenate([wav[:start], stretched, wav[end:]]).astype(wav.dtype)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from transformers import VitsModel, AutoTokenizer

    print(f"Dang tai model {MODEL_ID} (lan dau se tai ve, co the mat vai phut)...")
    model = VitsModel.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model.eval()

    rng = random.Random(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_out = OUT_DIR / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(VIVOS_ROOT / "prompts.txt")
    rng.shuffle(prompts)

    label_rows = []
    written = 0
    for utt_id, text in prompts:
        if written >= args.n:
            break
        disfluent_text, meta = make_disfluent_text(text, rng)

        inputs = tokenizer(disfluent_text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform

        wav = output.squeeze().cpu().numpy()
        sr = model.config.sampling_rate

        if meta["error_type"] == "lap_tu":
            wav = apply_lap_tu_emphasis(
                wav, disfluent_text, meta["word"], meta["n_repeats"],
                meta["word_index"], meta["words_before"], sr,
            )
        elif meta["error_type"] == "keo_dai_am":
            wav = apply_prolongation_stretch(
                wav, disfluent_text, meta["word"], meta["word_index"],
                meta["words_before"], meta["stretch_factor"],
            )
        elif meta["error_type"] == "khung_ngap_ngung":
            wav = apply_block_silence(
                wav, disfluent_text, meta["words_before"], sr, meta["silence_sec"],
            )

        out_name = f"{utt_id}__{meta['error_type']}_ttsB.wav"
        import soundfile as sf
        sf.write(str(audio_out / out_name), wav, sr, subtype="PCM_16")

        label_rows.append({
            "output_file": out_name,
            "source_vivos_id": utt_id,
            "source_text": text,
            "disfluent_text": disfluent_text,
            "error_type": meta["error_type"],
            "params": {k: v for k, v in meta.items() if k != "error_type"},
        })
        written += 1
        print(f"[{written}/{args.n}] {utt_id}: '{disfluent_text}' -> {out_name}")

    labels_csv = OUT_DIR / "labels.csv"
    write_header = not labels_csv.exists()
    with open(labels_csv, "a", encoding="utf-8", newline="") as f:
        fieldnames = ["output_file", "source_vivos_id", "source_text", "disfluent_text", "error_type", "params"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for row in label_rows:
            w.writerow(row)

    print(f"\nDa sinh {written} file vao {audio_out}")
    print(f"Nhan luu tai {labels_csv}")


if __name__ == "__main__":
    main()
