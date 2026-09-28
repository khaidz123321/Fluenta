"""Trích đặc trưng wav2vec2 (không finetune) cho từng clip trong manifest.

Mỗi lớp được chọn (L1, L7, L11) cho ra chuỗi T x 768; lấy trung bình và độ lệch chuẩn theo thời gian
(statistical pooling) -> vector 1536 chiều. Kết quả lưu theo từng khối để chạy tiếp được nếu bị ngắt.
"""
import argparse
import csv

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

import config

CHUNK = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_audio(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != config.SAMPLE_RATE:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=config.SAMPLE_RATE)
    return wav


def stat_pool(h):
    return torch.cat([h.mean(dim=1), h.std(dim=1)], dim=-1)


@torch.no_grad()
def embed_batch(model, fe, wavs):
    inputs = fe(wavs, sampling_rate=config.SAMPLE_RATE, return_tensors="pt", padding=True)
    out = model(inputs.input_values.to(DEVICE), output_hidden_states=True)
    return {name: stat_pool(out.hidden_states[idx]).cpu().numpy() for name, idx in config.LAYERS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    with open(config.MANIFEST, encoding="utf-8") as f:
        paths = [r["path"] for r in csv.DictReader(f)]

    config.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    fe = Wav2Vec2FeatureExtractor.from_pretrained(config.MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(config.MODEL_NAME).eval().to(DEVICE)
    torch.set_grad_enabled(False)
    print(f"Thiết bị: {DEVICE}, {len(paths)} clip", flush=True)

    n_chunks = (len(paths) + CHUNK - 1) // CHUNK
    for ci in range(n_chunks):
        chunk_file = config.FEATURE_DIR / f"chunk_{ci:04d}.npz"
        if chunk_file.exists():
            continue
        chunk_paths = paths[ci * CHUNK:(ci + 1) * CHUNK]
        feats = {name: [] for name in config.LAYERS}
        batch, batch_len = [], None
        for p in chunk_paths:
            wav = load_audio(p)
            # wav2vec2-base không dùng attention mask, nên chỉ ghép batch các clip cùng độ dài
            # để phần đệm không làm sai giá trị pooling.
            if batch and (len(wav) != batch_len or len(batch) >= args.batch):
                for k, v in embed_batch(model, fe, batch).items():
                    feats[k].append(v)
                batch = []
            batch.append(wav)
            batch_len = len(wav)
        if batch:
            for k, v in embed_batch(model, fe, batch).items():
                feats[k].append(v)
        np.savez(chunk_file, **{k: np.concatenate(v) for k, v in feats.items()})
        print(f"Khối {ci + 1}/{n_chunks}: {len(chunk_paths)} clip", flush=True)

    merged = {name: [] for name in config.LAYERS}
    for ci in range(n_chunks):
        data = np.load(config.FEATURE_DIR / f"chunk_{ci:04d}.npz")
        for name in config.LAYERS:
            merged[name].append(data[name])
    for name, parts in merged.items():
        arr = np.concatenate(parts)
        assert len(arr) == len(paths), "số vector không khớp manifest, hãy xóa work/features rồi chạy lại"
        np.save(config.FEATURE_DIR / f"{name}.npy", arr)
        print(f"{name}: {arr.shape}")


if __name__ == "__main__":
    main()
