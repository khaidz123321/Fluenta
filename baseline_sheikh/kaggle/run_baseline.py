"""Entry point Kaggle cho baseline Sheikh et al. (2023).

Input Kaggle Dataset có thể chứa SEP-28k_labels.csv và clips_output/ cùng
thư mục, hoặc đặt FLUENTA_LABELS_CSV và FLUENTA_CLIPS_DIR cho hai input riêng.
Repo phải chứa phiên bản code mới này trước khi chạy kernel.
"""

import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_URL = "https://github.com/khaidz123321/Fluenta.git"
WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")


def run(*command, cwd=None):
    print("$", " ".join(map(str, command)), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def find_dataset(root):
    for labels in sorted(root.rglob("SEP-28k_labels.csv")):
        if (labels.parent / "clips_output").is_dir():
            return labels.parent
    return None


def dataset_dir():
    if "FLUENTA_SEP_DIR" in os.environ:
        folder = Path(os.environ["FLUENTA_SEP_DIR"])
        if not (folder / "SEP-28k_labels.csv").is_file() or not (folder / "clips_output").is_dir():
            raise FileNotFoundError(f"FLUENTA_SEP_DIR thiếu nhãn hoặc clips_output: {folder}")
        return folder
    found = find_dataset(INPUT)
    if found:
        return found
    for archive in sorted(INPUT.rglob("*.zip")):
        with zipfile.ZipFile(archive) as source:
            names = source.namelist()
            if not any(name.endswith("SEP-28k_labels.csv") for name in names):
                continue
            target = WORKING / "sep28k_input"
            target.mkdir(parents=True, exist_ok=True)
            for name in names:
                resolved = (target / name).resolve()
                if not resolved.is_relative_to(target.resolve()):
                    raise ValueError(f"ZIP chứa đường dẫn không hợp lệ: {name}")
            print(f"Giải nén {archive} -> {target}", flush=True)
            source.extractall(target)
            found = find_dataset(target)
            if found:
                return found
    raise FileNotFoundError("Kaggle Input cần SEP-28k_labels.csv và clips_output/ cùng thư mục.")


def dataset_paths():
    labels = os.environ.get("FLUENTA_LABELS_CSV")
    clips = os.environ.get("FLUENTA_CLIPS_DIR")
    if labels or clips:
        if not labels or not clips:
            raise ValueError("Cần đặt cả FLUENTA_LABELS_CSV và FLUENTA_CLIPS_DIR.")
        labels_path, clips_path = Path(labels), Path(clips)
        if not labels_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy CSV: {labels_path}")
        if not clips_path.is_dir():
            raise NotADirectoryError(f"Không tìm thấy thư mục clip: {clips_path}")
        return labels_path, clips_path
    folder = dataset_dir()
    return folder / "SEP-28k_labels.csv", folder / "clips_output"


def source_dir():
    def current_code(folder):
        cfg = folder / "config.py"
        return ((folder / "train_eval.py").is_file()
                and (folder / "requirements.txt").is_file()
                and cfg.is_file()
                and "PIPELINE_VERSION = 3" in cfg.read_text(encoding="utf-8"))

    # Nếu upload cả baseline_sheikh như một Kaggle Dataset, dùng code tại đó.
    named = os.environ.get("FLUENTA_BASELINE_DIR")
    if named:
        if not current_code(Path(named)):
            raise FileNotFoundError(f"FLUENTA_BASELINE_DIR chưa có code baseline mới: {named}")
        return Path(named)
    current = Path(__file__).resolve().parents[1]
    if current_code(current):
        return current
    target = WORKING / "Fluenta"
    if not target.is_dir():
        run("git", "clone", "--depth", "1", REPO_URL, target)
    baseline = target / "baseline_sheikh"
    if not current_code(baseline):
        raise RuntimeError("GitHub chưa có baseline mới. Hãy push code hoặc upload "
                           "baseline_sheikh lên Kaggle Input rồi đặt FLUENTA_BASELINE_DIR.")
    return baseline


def main():
    run("nvidia-smi")
    baseline = source_dir()
    labels, clips = dataset_paths()
    os.environ["FLUENTA_LABELS_CSV"] = str(labels)
    os.environ["FLUENTA_CLIPS_DIR"] = str(clips)
    os.environ.setdefault("FLUENTA_WORK_DIR", str(WORKING / "baseline_work"))
    print(f"Code: {baseline}\nCSV: {labels}\nClips: {clips}\n"
          f"Work: {os.environ['FLUENTA_WORK_DIR']}", flush=True)

    run(sys.executable, "-m", "pip", "install", "-q", "-r", baseline / "requirements.txt")
    run(sys.executable, "prepare_data.py", cwd=baseline)
    run(sys.executable, "extract_features.py", "--batch",
        os.environ.get("FLUENTA_BATCH", "8"), "--layers",
        os.environ.get("FLUENTA_LAYERS", "L1,L7,L11"), cwd=baseline)
    run(sys.executable, "train_eval.py", "--folds",
        os.environ.get("FLUENTA_FOLDS", "10"), "--experiments",
        os.environ.get("FLUENTA_EXPERIMENTS",
                       "l11,multilevel,ecapa,score_fusion,embedding_fusion"), cwd=baseline)
    print(f"Kết quả: {Path(os.environ['FLUENTA_WORK_DIR']) / 'results'}", flush=True)


if __name__ == "__main__":
    main()
