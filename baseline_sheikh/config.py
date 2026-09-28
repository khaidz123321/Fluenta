"""Sheikh et al. (2023), arXiv:2306.00689 -- cấu hình tái hiện.

Các thông số bài công bố được giữ nguyên. Lựa chọn khác được ghi trong README.
"""

import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPELINE_VERSION = 3
SEP_DIR = Path(os.environ.get("FLUENTA_SEP_DIR", ROOT / "ml-stuttering-events-dataset"))
LABELS_CSV = Path(os.environ.get("FLUENTA_LABELS_CSV", SEP_DIR / "SEP-28k_labels.csv"))
CLIPS_DIR = Path(os.environ.get("FLUENTA_CLIPS_DIR", SEP_DIR / "clips_output"))

WORK_DIR = Path(os.environ.get("FLUENTA_WORK_DIR", ROOT / "work"))
MANIFEST = WORK_DIR / "manifest.csv"
FEATURE_ROOT = WORK_DIR / "features"
RESULTS_DIR = WORK_DIR / "results"

# Bài: Wav2Vec2 base 960 h LibriSpeech rồi fine-tune ASR; ECAPA trên VoxCeleb.
W2V_MODEL = "facebook/wav2vec2-base-960h"
ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
SAMPLE_RATE = 16_000
ALL_LAYERS = tuple(f"L{i}" for i in range(1, 14))
FUSION_LAYERS = ("L1", "L7", "L11")

# Bài: năm lớp R/P/B/I/F, LDA 4 chiều trên từng embedding.
CLASSES = ("R", "P", "B", "I", "F")
DISFLUENT = CLASSES[:4]
LDA_COMPONENTS = 4
KNN_K = 5
NN_BATCH_SIZE = 128
NN_LR = 1e-2
NN_PATIENCE = 7
NN_DROPOUT = 0.2
N_FOLDS = 10
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1

# Bài không công bố quy tắc quy đổi nhãn đa nhãn và kích thước lớp ẩn.
MIN_VOTES = 2
HIDDEN_CANDIDATES = ((64, 32), (128, 64))
MAX_EPOCHS = 120
SEED = 2023
SCORE_ALPHA = 0.9  # Giá trị bài nêu; giữ cố định, không tối ưu trên test.


def feature_dir():
    """Cache gắn với manifest và phiên bản bộ trích đặc trưng."""
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Chưa có manifest: {MANIFEST}. Chạy prepare_data.py trước.")
    digest = hashlib.sha256()
    digest.update(MANIFEST.read_bytes())
    digest.update(f"v{PIPELINE_VERSION}|{W2V_MODEL}|{ECAPA_MODEL}|{SAMPLE_RATE}".encode())
    return FEATURE_ROOT / digest.hexdigest()[:16]
