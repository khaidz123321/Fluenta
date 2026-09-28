"""Huấn luyện và đánh giá theo Sheikh et al. (2023).

Mỗi fold: chia theo tập podcast (không trùng giữa train/val/test) -> LDA 4 chiều cho từng lớp wav2vec2,
fit trên train -> ghép các lớp -> 3 bộ phân loại: mạng hai nhánh (NN), KNN (k=5), Gaussian Naive Bayes.
Báo recall và F1 từng lớp, UAR (trung bình recall 5 lớp) và độ chính xác, lấy trung bình qua các fold.
"""
import argparse
import csv
import json

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

import config
from model import TwoBranchNet, loss_fn, predict

assert config.DISFLUENT == config.CLASSES[:4], "nhánh lỗi dùng chỉ số 0..3 của CLASSES"


def load_data(layers):
    with open(config.MANIFEST, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    y = np.array([config.CLASSES.index(r["label"]) for r in rows])
    groups = np.array([r["group"] for r in rows])
    feats = {name: np.load(config.FEATURE_DIR / f"{name}.npy") for name in layers}
    return feats, y, groups


def make_split(y, groups, fold, mode):
    rng = np.random.default_rng(config.SEED + fold)
    units = np.unique(groups) if mode == "podcast" else np.arange(len(y))
    units = rng.permutation(units)
    n_tr = int(round(len(units) * config.TRAIN_FRAC))
    n_va = int(round(len(units) * config.VAL_FRAC))
    tr_u, va_u, te_u = units[:n_tr], units[n_tr:n_tr + n_va], units[n_tr + n_va:]
    if mode == "podcast":
        return np.isin(groups, tr_u), np.isin(groups, va_u), np.isin(groups, te_u)
    masks = [np.zeros(len(y), bool) for _ in range(3)]
    for m, u in zip(masks, (tr_u, va_u, te_u)):
        m[u] = True
    return masks


def lda_features(feats, y, tr):
    n_comp = min(config.LDA_COMPONENTS, len(np.unique(y[tr])) - 1)
    out = []
    for arr in feats.values():
        lda = LinearDiscriminantAnalysis(n_components=n_comp).fit(arr[tr], y[tr])
        out.append(lda.transform(arr))
    return np.concatenate(out, axis=1).astype(np.float32)


def train_nn(X, y, tr, va):
    torch.manual_seed(config.SEED)
    model = TwoBranchNet(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=config.LR)
    Xtr, ytr = torch.from_numpy(X[tr]), torch.from_numpy(y[tr])
    Xva, yva = torch.from_numpy(X[va]), torch.from_numpy(y[va])
    best, best_state, bad = float("inf"), None, 0
    for _ in range(config.MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), config.BATCH_SIZE):
            idx = perm[i:i + config.BATCH_SIZE]
            if len(idx) < 2:  # BatchNorm cần ít nhất 2 mẫu
                continue
            opt.zero_grad()
            loss_fn(*model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(*model(Xva), yva).item()
        if val_loss < best:
            best, bad = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= config.PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def metrics(y_true, y_pred):
    labels = list(range(len(config.CLASSES)))
    present = [c for c in labels if (y_true == c).any()]
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "recall": {config.CLASSES[c]: float(rec[c]) for c in labels},
        "f1": {config.CLASSES[c]: float(f1[c]) for c in labels},
        "UAR": float(np.mean([rec[c] for c in present])),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classes_in_test": [config.CLASSES[c] for c in present],
    }


def average(fold_results):
    avg = {}
    for key in ("recall", "f1"):
        avg[key] = {c: float(np.mean([r[key][c] for r in fold_results])) for c in config.CLASSES}
    avg["UAR"] = float(np.mean([r["UAR"] for r in fold_results]))
    avg["accuracy"] = float(np.mean([r["accuracy"] for r in fold_results]))
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="L1,L7,L11", help="vd. 'L11' hoặc 'L1,L7,L11'")
    ap.add_argument("--folds", type=int, default=config.N_FOLDS)
    ap.add_argument("--split", choices=["podcast", "random"], default="podcast",
                    help="'random' chỉ dùng khi chạy thử với rất ít clip")
    args = ap.parse_args()

    layers = [l.strip() for l in args.layers.split(",")]
    feats, y, groups = load_data(layers)
    print(f"{len(y)} clip, lớp wav2vec2: {layers}, chia theo: {args.split}, số fold: {args.folds}")

    results = {"NN+LDA": [], "KNN+LDA": [], "NBC+LDA": []}
    for fold in range(args.folds):
        tr, va, te = make_split(y, groups, fold, args.split)
        if len(np.unique(y[tr])) < 2 or va.sum() == 0 or te.sum() == 0:
            print(f"Fold {fold}: bỏ qua vì tập train/val/test quá nhỏ")
            continue
        X = lda_features(feats, y, tr)
        nn_model = train_nn(X, y, tr, va)
        preds = {
            "NN+LDA": predict(nn_model, torch.from_numpy(X[te])).numpy(),
            "KNN+LDA": KNeighborsClassifier(n_neighbors=min(config.KNN_K, int(tr.sum())))
            .fit(X[tr], y[tr]).predict(X[te]),
            "NBC+LDA": GaussianNB().fit(X[tr], y[tr]).predict(X[te]),
        }
        for name, p in preds.items():
            results[name].append(metrics(y[te], p))
        print(f"Fold {fold}: train {tr.sum()}, val {va.sum()}, test {te.sum()} | "
              + ", ".join(f"{n} UAR {results[n][-1]['UAR'] * 100:.1f}%" for n in results))

    if not results["NN+LDA"]:
        print("Không có fold nào chạy được.")
        return

    summary = {name: average(r) for name, r in results.items()}
    print("\nKết quả trung bình (recall % từng lớp như cách trình bày ở Bảng 2 của bài):")
    print(f"{'Mô hình':10s} " + " ".join(f"{c:>6s}" for c in config.CLASSES) + f" {'Acc':>6s} {'UAR':>6s}")
    for name, s in summary.items():
        print(f"{name:10s} " + " ".join(f"{s['recall'][c] * 100:6.1f}" for c in config.CLASSES)
              + f" {s['accuracy'] * 100:6.1f} {s['UAR'] * 100:6.1f}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / f"results_{'-'.join(layers)}_{args.split}_{args.folds}folds.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "summary": summary, "per_fold": results}, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu chi tiết vào {out}")


if __name__ == "__main__":
    main()
