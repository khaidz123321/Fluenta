"""Đánh giá các cấu hình trong Bảng 2 của Sheikh et al. (2023).

Mỗi lần chia giữ nguyên toàn bộ tập podcast giữa train/val/test. LDA và lựa
chọn kích thước NN chỉ dùng train/validation; test dùng đúng một lần.
"""

import argparse
import csv
import json
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

import config
from model import TwoBranchNet, loss_fn, predict, probabilities

CORE_EXPERIMENTS = ("l11", "multilevel", "ecapa", "score_fusion", "embedding_fusion")
OPTIONAL_EXPERIMENTS = ("l11_raw", "ecapa_raw", "layer_scan")
METHODS = ("NN", "KNN", "NBC")


def load_manifest():
    with config.MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Manifest rỗng.")
    y = np.array([config.CLASSES.index(row["label"]) for row in rows], dtype=np.int64)
    groups = np.array([row["group"] for row in rows])
    return y, groups


def make_split(groups, fold, seed):
    """10 lần chia độc lập 80/10/10 theo podcast; protocol gốc chưa phát hành."""
    rng = np.random.default_rng(seed + fold)
    units = rng.permutation(np.unique(groups))
    n_train = round(len(units) * config.TRAIN_FRAC)
    n_val = round(len(units) * config.VAL_FRAC)
    train = np.isin(groups, units[:n_train])
    val = np.isin(groups, units[n_train:n_train + n_val])
    test = np.isin(groups, units[n_train + n_val:])
    return train, val, test


def load_features(names, n_rows):
    cache = config.feature_dir()
    arrays = {}
    for name in sorted(names):
        path = cache / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu {path}; chạy extract_features.py trước.")
        array = np.load(path, mmap_mode="r")
        if array.ndim != 2 or len(array) != n_rows:
            raise ValueError(f"Feature không khớp manifest: {path} {array.shape}")
        arrays[name] = array
    return arrays


def lda_transform(array, y, train):
    if set(np.unique(y[train])) != set(range(len(config.CLASSES))):
        raise ValueError("Tập train cần có đủ cả 5 lớp để LDA ra 4 chiều.")
    lda = LinearDiscriminantAnalysis(n_components=config.LDA_COMPONENTS)
    lda.fit(array[train], y[train])
    result = lda.transform(array).astype(np.float32)
    if result.shape[1] != config.LDA_COMPONENTS:
        raise ValueError(f"LDA ra {result.shape[1]} chiều, cần 4 chiều.")
    return result


def fit_nn(X, y, train, val, fold, device):
    X_train = torch.as_tensor(np.asarray(X[train]), dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y[train], dtype=torch.long, device=device)
    X_val = torch.as_tensor(np.asarray(X[val]), dtype=torch.float32, device=device)
    y_val = torch.as_tensor(y[val], dtype=torch.long, device=device)
    best_choice = None
    for choice, hidden in enumerate(config.HIDDEN_CANDIDATES):
        torch.manual_seed(config.SEED + fold * 100 + choice)
        model = TwoBranchNet(X.shape[1], hidden).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.NN_LR)
        best_loss, best_state, bad, best_epoch = float("inf"), None, 0, 0
        for epoch in range(config.MAX_EPOCHS):
            model.train()
            order = torch.randperm(len(X_train), device=device)
            for start in range(0, len(order), config.NN_BATCH_SIZE):
                ids = order[start:start + config.NN_BATCH_SIZE]
                if len(ids) < 2:
                    continue  # BatchNorm yêu cầu ít nhất hai mẫu.
                optimizer.zero_grad(set_to_none=True)
                loss_fn(*model(X_train[ids]), y_train[ids]).backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(*model(X_val), y_val).item()
            if val_loss < best_loss:
                best_loss, bad, best_epoch = val_loss, 0, epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= config.NN_PATIENCE:
                    break
        if best_choice is None or best_loss < best_choice[0]:
            best_choice = (best_loss, hidden, best_epoch, best_state)
    loss, hidden, epoch, state = best_choice
    model = TwoBranchNet(X.shape[1], hidden).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, {"hidden": list(hidden), "epoch": epoch, "validation_loss": loss}


def aligned_probabilities(raw, classes):
    aligned = np.zeros((len(raw), len(config.CLASSES)), dtype=np.float32)
    aligned[:, classes] = raw
    return aligned


def fit_classifiers(X, y, train, val, test, fold, device, name, checkpoint_dir):
    output, selection = {}, {}
    nn, details = fit_nn(X, y, train, val, fold, device)
    selection["NN"] = details
    state = {k: v.detach().cpu() for k, v in nn.state_dict().items()}
    torch.save({"state_dict": state, "input_dim": X.shape[1],
                "hidden": details["hidden"], "classes": config.CLASSES},
               checkpoint_dir / f"{name}_fold{fold + 1:02d}.pt")
    with torch.no_grad():
        x_test = torch.as_tensor(np.asarray(X[test]), dtype=torch.float32, device=device)
        output["NN"] = {
            "pred": predict(nn, x_test).cpu().numpy(),
            "prob": probabilities(nn, x_test).cpu().numpy(),
        }
    knn = KNeighborsClassifier(n_neighbors=config.KNN_K, metric="minkowski", p=2)
    knn.fit(X[train], y[train])
    output["KNN"] = {
        "pred": knn.predict(X[test]),
        "prob": aligned_probabilities(knn.predict_proba(X[test]), knn.classes_),
    }
    nbc = GaussianNB()
    nbc.fit(X[train], y[train])
    output["NBC"] = {
        "pred": nbc.predict(X[test]),
        "prob": aligned_probabilities(nbc.predict_proba(X[test]), nbc.classes_),
    }
    return output, selection


def metrics(true, predicted):
    labels = list(range(len(config.CLASSES)))
    recall = recall_score(true, predicted, labels=labels, average=None, zero_division=0)
    f1 = f1_score(true, predicted, labels=labels, average=None, zero_division=0)
    return {
        "recall": dict(zip(config.CLASSES, map(float, recall))),
        "f1": dict(zip(config.CLASSES, map(float, f1))),
        "UAR": float(recall.mean()),
        "accuracy": float(accuracy_score(true, predicted)),
        "confusion_matrix": confusion_matrix(true, predicted, labels=labels).tolist(),
        "support": dict(zip(config.CLASSES, map(int, np.bincount(true, minlength=5)))),
    }


def summary(folds):
    return {
        "recall": {c: float(np.mean([item["recall"][c] for item in folds]))
                   for c in config.CLASSES},
        "f1": {c: float(np.mean([item["f1"][c] for item in folds]))
               for c in config.CLASSES},
        "UAR": float(np.mean([item["UAR"] for item in folds])),
        "accuracy": float(np.mean([item["accuracy"] for item in folds])),
    }


def experiment_plan(selected):
    selected = set(selected)
    if "layer_scan" in selected:
        selected.remove("layer_scan")
        selected.update(f"layer_{layer.lower()}" for layer in config.ALL_LAYERS)
    allowed = set(CORE_EXPERIMENTS + OPTIONAL_EXPERIMENTS) | {
        f"layer_{layer.lower()}" for layer in config.ALL_LAYERS
    }
    if not selected or not selected <= allowed:
        raise ValueError(f"Thí nghiệm không hợp lệ: {sorted(selected - allowed)}")
    names = {"L11"} if selected & {"l11", "l11_raw", "score_fusion", "embedding_fusion"} else set()
    if selected & {"multilevel"}:
        names.update(config.FUSION_LAYERS)
    if selected & {"ecapa", "ecapa_raw", "score_fusion", "embedding_fusion"}:
        names.add("ecapa")
    names.update(name[6:].upper() for name in selected if name.startswith("layer_"))
    return sorted(selected), names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default=",".join(CORE_EXPERIMENTS),
                        help="l11,multilevel,ecapa,score_fusion,embedding_fusion; "
                             "thêm l11_raw,ecapa_raw,layer_scan nếu cần")
    parser.add_argument("--folds", type=int, default=config.N_FOLDS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()
    if args.folds < 1:
        parser.error("--folds phải dương")
    selected, needed = experiment_plan([x.strip() for x in args.experiments.split(",")])
    y, groups = load_manifest()
    features = load_features(needed, len(y))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result_dir = config.RESULTS_DIR / config.feature_dir().name / f"seed_{args.seed}"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = result_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    results = {name: {method: [] for method in METHODS} for name in selected}
    splits, tuning = [], []
    print(f"{len(y)} clip; {len(np.unique(groups))} tập; {device}; {selected}", flush=True)

    for fold in range(args.folds):
        train, val, test = make_split(groups, fold, args.seed)
        if not train.any() or not val.any() or not test.any() or len(np.unique(y[train])) != 5:
            raise ValueError(f"Fold {fold + 1} không đủ dữ liệu cho năm lớp và ba tập.")
        split = {"fold": fold + 1, "train": int(train.sum()), "val": int(val.sum()),
                 "test": int(test.sum()), "train_groups": int(len(np.unique(groups[train]))),
                 "val_groups": int(len(np.unique(groups[val]))),
                 "test_groups": int(len(np.unique(groups[test])))}
        splits.append(split)
        representations = {}
        if "L11" in needed:
            representations["l11"] = lda_transform(features["L11"], y, train)
        if "ecapa" in needed:
            representations["ecapa"] = lda_transform(features["ecapa"], y, train)
        if "multilevel" in selected:
            transformed = {layer: (representations["l11"] if layer == "L11"
                                   else lda_transform(features[layer], y, train))
                           for layer in config.FUSION_LAYERS}
            representations["multilevel"] = np.concatenate(
                [transformed[layer] for layer in config.FUSION_LAYERS], axis=1)
        if "embedding_fusion" in selected:
            representations["embedding_fusion"] = np.concatenate(
                [representations["l11"], representations["ecapa"]], axis=1)
        if "l11_raw" in selected:
            representations["l11_raw"] = features["L11"]
        if "ecapa_raw" in selected:
            representations["ecapa_raw"] = features["ecapa"]
        for name in selected:
            if name.startswith("layer_"):
                layer = name[6:].upper()
                representations[name] = (representations["l11"] if layer == "L11"
                                         and "l11" in representations else
                                         lda_transform(features[layer], y, train))

        fitted = {}
        needed_reps = [name for name in representations if name in selected]
        if "score_fusion" in selected:
            needed_reps += [name for name in ("l11", "ecapa") if name not in needed_reps]
        for name in needed_reps:
            fitted[name], chosen = fit_classifiers(
                representations[name], y, train, val, test, fold, device, name, checkpoint_dir)
            tuning.append({"fold": fold + 1, "experiment": name, **chosen["NN"]})

        for name in selected:
            for method in METHODS:
                if name == "score_fusion":
                    score = (config.SCORE_ALPHA * fitted["l11"][method]["prob"]
                             + (1 - config.SCORE_ALPHA) * fitted["ecapa"][method]["prob"])
                    predicted = score.argmax(axis=1)
                else:
                    predicted = fitted[name][method]["pred"]
                results[name][method].append(metrics(y[test], predicted))
        print(f"Fold {fold + 1}/{args.folds}: {split['train']}/{split['val']}/{split['test']} "
              f"| NN L11 UAR {results['l11']['NN'][-1]['UAR']:.3f}"
              if "l11" in results else f"Fold {fold + 1}/{args.folds} xong", flush=True)

    report = {
        "paper": "Sheikh et al. 2023, arXiv:2306.00689v1",
        "manifest": str(config.MANIFEST),
        "feature_cache": str(config.feature_dir()),
        "pretrained_models": {"wav2vec2": config.W2V_MODEL, "ecapa": config.ECAPA_MODEL},
        "assumptions": {
            "label_rule": "exactly one R/P/B/I with >=2 votes; else F if NoStutteredWords>=2",
            "splits": "10 repeated seeded 80/10/10 podcast-level splits; original IDs unavailable",
            "hidden_candidates": config.HIDDEN_CANDIDATES,
            "max_epochs": config.MAX_EPOCHS,
            "score_alpha": config.SCORE_ALPHA,
            "score_alpha_policy": "fixed paper value, not tuned on test",
        },
        "args": vars(args), "splits": splits, "nn_selection": tuning,
        "summary": {name: {method: summary(folds) for method, folds in methods.items()}
                    for name, methods in results.items()},
        "per_fold": results,
    }
    output = result_dir / f"results_{'-'.join(selected)}_{args.folds}folds.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nExperiment / classifier        Accuracy       UAR")
    for name in selected:
        for method in METHODS:
            scores = report["summary"][name][method]
            print(f"{name + ' / ' + method:28s} {scores['accuracy'] * 100:7.2f}%  {scores['UAR'] * 100:7.2f}%")
    print(f"\nChi tiết: {output}")


if __name__ == "__main__":
    main()
