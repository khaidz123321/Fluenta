"""Chạy toàn bộ baseline Sheikh et al. (2023) trên Kaggle (GPU)."""
import glob
import os
import shutil
import subprocess
import sys
import zipfile


def sh(cmd):
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh("nvidia-smi || true")
sh("git clone --depth 1 https://github.com/khaidz123321/Fluenta.git /kaggle/working/Fluenta")

labels = glob.glob("/kaggle/input/**/SEP-28k_labels.csv", recursive=True)
if not labels:  # nếu Kaggle không tự giải nén file zip
    archive = glob.glob("/kaggle/input/**/*.zip", recursive=True)[0]
    zipfile.ZipFile(archive).extractall("/kaggle/tmp/sep")
    labels = glob.glob("/kaggle/tmp/sep/**/SEP-28k_labels.csv", recursive=True)
os.environ["FLUENTA_SEP_DIR"] = os.path.dirname(labels[0])
print("Dữ liệu SEP-28k:", os.environ["FLUENTA_SEP_DIR"], flush=True)

os.chdir("/kaggle/working/Fluenta/baseline_sheikh")
py = sys.executable
sh(f"{py} prepare_data.py")
sh(f"{py} extract_features.py --batch 32")
sh(f"{py} train_eval.py")
sh(f"{py} train_eval.py --layers L11")

# Giữ lại manifest, đặc trưng đã gộp và kết quả; bỏ các khối trung gian cho gọn phần output.
for chunk in glob.glob("work/features/chunk_*.npz"):
    os.remove(chunk)
shutil.copytree("work/results", "/kaggle/working/results", dirs_exist_ok=True)
print("\nHoàn tất. Kết quả nằm ở /kaggle/working/results", flush=True)
