"""Trích Wav2Vec2 (13 tầng) và ECAPA-TDNN (192 chiều), không fine-tune.

Cache theo manifest để chạy tiếp trên Kaggle sau khi bị ngắt. Mặc định trích
L1/L7/L11 và ECAPA, đủ cho thí nghiệm chính trong Bảng 2.
"""

import argparse
import csv
import os
import numpy as np
import soundfile as sf
import torch

import config


def read_paths():
    with config.MANIFEST.open(newline="", encoding="utf-8") as stream:
        paths = [row["path"] for row in csv.DictReader(stream)]
    if not paths:
        raise ValueError("Manifest rỗng.")
    return paths


def load_audio(path):
    wav, rate = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if rate != config.SAMPLE_RATE:
        import librosa
        wav = librosa.resample(wav, orig_sr=rate, target_sr=config.SAMPLE_RATE)
    if not np.isfinite(wav).all() or len(wav) < 400:
        raise ValueError(f"Audio rỗng/không hợp lệ: {path}")
    return np.asarray(wav, dtype=np.float32)


def batches(paths, batch_size):
    """Ghép clip cùng độ dài để padding không làm sai pooling Wav2Vec2."""
    group, length = [], None
    for path in paths:
        wav = load_audio(path)
        if group and (len(group) >= batch_size or len(wav) != length):
            yield group
            group = []
        group.append(wav)
        length = len(wav)
    if group:
        yield group


def stat_pool(hidden):
    return torch.cat((hidden.mean(dim=1), hidden.std(dim=1, unbiased=False)), dim=-1)


def write_chunk(path, arrays):
    temp = path.with_name(path.stem + ".partial" + path.suffix)
    if path.suffix == ".npz":
        np.savez(temp, **arrays)
    else:
        np.save(temp, arrays)
    os.replace(temp, path)


def merge_chunks(paths, destination, key, n_rows, width):
    if destination.exists():
        old = np.load(destination, mmap_mode="r")
        if old.shape == (n_rows, width):
            return
        raise ValueError(f"Cache sai kích thước: {destination}")
    temp = destination.with_name(destination.stem + ".partial.npy")
    out = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32, shape=(n_rows, width))
    pos = 0
    for path in paths:
        data = np.load(path)
        part = data[key] if key is not None else data
        out[pos:pos + len(part)] = part
        pos += len(part)
        if hasattr(data, "close"):
            data.close()
    if pos != n_rows:
        raise ValueError(f"Số hàng đặc trưng sai: {pos} != {n_rows}")
    out.flush()
    del out
    os.replace(temp, destination)


def extract_wav2vec2(paths, layers, cache, batch_size, chunk_size, device):
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    final = [cache / f"{layer}.npy" for layer in layers]
    if all(path.is_file() and np.load(path, mmap_mode="r").shape == (len(paths), 1536) for path in final):
        print("Wav2Vec2: dùng cache đã hoàn chỉnh.", flush=True)
        return
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.W2V_MODEL)
    model = Wav2Vec2Model.from_pretrained(config.W2V_MODEL).eval().to(device)
    assert model.config.num_hidden_layers == 12
    indices = {layer: int(layer[1:]) - 1 for layer in layers}
    tag = "w2v_" + "-".join(layers)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(paths), chunk_size):
            part = paths[start:start + chunk_size]
            ci = start // chunk_size
            chunk = cache / f"{tag}_chunk_{ci:04d}.npz"
            chunks.append(chunk)
            if chunk.is_file():
                with np.load(chunk) as old:
                    if all(k in old and old[k].shape == (len(part), 1536) for k in layers):
                        continue
                raise ValueError(f"Cache không hợp lệ: {chunk}")
            pieces = {layer: [] for layer in layers}
            for wavs in batches(part, batch_size):
                inputs = extractor(wavs, sampling_rate=config.SAMPLE_RATE,
                                   return_tensors="pt", padding=True)
                states = model(inputs.input_values.to(device), output_hidden_states=True).hidden_states
                for layer, index in indices.items():
                    pieces[layer].append(stat_pool(states[index]).cpu().numpy())
            write_chunk(chunk, {key: np.concatenate(value).astype(np.float32)
                                for key, value in pieces.items()})
            print(f"Wav2Vec2: {min(start + chunk_size, len(paths))}/{len(paths)}", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    for layer in layers:
        merge_chunks(chunks, cache / f"{layer}.npy", layer, len(paths), 1536)


def extract_ecapa(paths, cache, batch_size, chunk_size, device):
    destination = cache / "ecapa.npy"
    if destination.is_file() and np.load(destination, mmap_mode="r").shape == (len(paths), 192):
        print("ECAPA: dùng cache đã hoàn chỉnh.", flush=True)
        return
    from speechbrain.inference.speaker import EncoderClassifier

    model = EncoderClassifier.from_hparams(
        source=config.ECAPA_MODEL,
        savedir=str(cache / "ecapa_pretrained"),
        run_opts={"device": device},
    )
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(paths), chunk_size):
            part = paths[start:start + chunk_size]
            ci = start // chunk_size
            chunk = cache / f"ecapa_chunk_{ci:04d}.npy"
            chunks.append(chunk)
            if chunk.is_file():
                if np.load(chunk, mmap_mode="r").shape == (len(part), 192):
                    continue
                raise ValueError(f"Cache không hợp lệ: {chunk}")
            pieces = []
            for wavs in batches(part, batch_size):
                batch = torch.from_numpy(np.stack(wavs)).to(device)
                emb = model.encode_batch(batch, normalize=False)
                pieces.append(emb.reshape(len(wavs), -1).cpu().numpy())
            array = np.concatenate(pieces).astype(np.float32)
            if array.shape != (len(part), 192):
                raise ValueError(f"ECAPA không tạo ra vector 192 chiều: {array.shape}")
            write_chunk(chunk, array)
            print(f"ECAPA: {min(start + chunk_size, len(paths))}/{len(paths)}", flush=True)
    merge_chunks(chunks, destination, None, len(paths), 192)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="wav2vec2,ecapa",
                        help="wav2vec2,ecapa hoặc một trong hai")
    parser.add_argument("--layers", default=",".join(config.FUSION_LAYERS),
                        help="L1,L7,L11 hoặc all (13 tầng)")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=256)
    args = parser.parse_args()
    if args.batch < 1 or args.chunk < 1:
        parser.error("--batch và --chunk phải dương")
    models = {name.strip() for name in args.models.split(",")}
    if not models <= {"wav2vec2", "ecapa"} or not models:
        parser.error("--models chỉ nhận wav2vec2 và ecapa")
    layers = config.ALL_LAYERS if args.layers == "all" else tuple(
        name.strip() for name in args.layers.split(",")
    )
    if not layers or len(set(layers)) != len(layers) or any(layer not in config.ALL_LAYERS for layer in layers):
        parser.error("--layers phải là danh sách L1..L13 không trùng nhau")
    paths = read_paths()
    cache = config.feature_dir()
    cache.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(paths)} clip; thiết bị {device}; cache {cache}", flush=True)
    if "wav2vec2" in models:
        extract_wav2vec2(paths, layers, cache, args.batch, args.chunk, device)
    if "ecapa" in models:
        extract_ecapa(paths, cache, args.batch, args.chunk, device)


if __name__ == "__main__":
    main()
