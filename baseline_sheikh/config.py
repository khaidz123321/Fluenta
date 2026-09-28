"""Cấu hình chung để tái hiện Sheikh et al. (2023), arXiv:2306.00689, trên SEP-28k."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Trên Kaggle/Modal, đặt FLUENTA_SEP_DIR tới thư mục chứa SEP-28k_labels.csv và clips_output/.
SEP_DIR = Path(os.environ.get(
    "FLUENTA_SEP_DIR",
    ROOT.parent / "Dataset" / "ml-stuttering-events-dataset-main" / "ml-stuttering-events-dataset-main"))
LABELS_CSV = SEP_DIR / "SEP-28k_labels.csv"
CLIPS_DIR = SEP_DIR / "clips_output"

WORK_DIR = Path(os.environ.get("FLUENTA_WORK_DIR", ROOT / "work"))
MANIFEST = WORK_DIR / "manifest.csv"
FEATURE_DIR = WORK_DIR / "features"
RESULTS_DIR = WORK_DIR / "results"

# wav2vec2-base pretrain 960h LibriSpeech rồi finetune ASR (CTC), đúng mô tả của bài.
MODEL_NAME = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000

# Bài đánh số L1 = local encoder, L2..L13 = 12 lớp transformer.
# Với HuggingFace, hidden_states[0] là đầu ra local encoder, hidden_states[k] là lớp transformer thứ k,
# nên L_k tương ứng hidden_states[k - 1].
LAYERS = {"L1": 0, "L7": 6, "L11": 10}

# R = lặp (bài gộp SoundRep + WordRep), P = kéo dài âm, B = khựng, I = chêm từ, F = trôi chảy.
CLASSES = ["R", "P", "B", "I", "F"]
DISFLUENT = ["R", "P", "B", "I"]
MIN_VOTES = 2  # một nhãn được tính khi có từ 2/3 người gán đồng ý

LDA_COMPONENTS = 4
BATCH_SIZE = 128
LR = 1e-2
PATIENCE = 7
MAX_EPOCHS = 200
DROPOUT = 0.2
HIDDEN = (64, 32)  # bài không nêu số nơ-ron lớp ẩn; đây là giả định của nhóm
KNN_K = 5

N_FOLDS = 10
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1
SEED = 0
