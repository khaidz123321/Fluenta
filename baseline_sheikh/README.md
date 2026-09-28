# Tái hiện Sheikh et al. (2023) trên SEP-28k

Nguồn: `2306.00689v1.pdf`, *Stuttering detection using speaker representations and self-supervised contextual embeddings*.

## Đã triển khai

- SEP-28k: một nhãn R/P/B/I/F cho mỗi clip, chia train/validation/test theo **tập podcast** (80/10/10), chạy 10 lần.
- Wav2Vec2 `facebook/wav2vec2-base-960h` đã pretrain và fine-tune ASR; trích L1 đến L13, mean + standard deviation theo thời gian thành 1536 chiều. Mặc định chỉ lưu L1/L7/L11.
- ECAPA-TDNN `speechbrain/spkrec-ecapa-voxceleb` trên VoxCeleb; lấy embedding 192 chiều từ `encode_batch`.
- LDA 4 chiều cho **từng** embedding, chỉ fit trên train. Ghép L1/L7/L11 thành 12 chiều; ghép L11 + ECAPA thành 8 chiều.
- KNN (k=5, Euclidean), GaussianNB, mạng hai nhánh (fluent / R-P-B-I), batch 128, Adam 1e-2, dropout 0.2, patience 7.
- Score fusion của Wav2Vec2 L11 và ECAPA với α=0.9 như bài; α được cố định trước khi xem test.
- Lưu JSON kết quả, các split, kích thước NN được chọn, và checkpoint của NN từng fold.

## Các giả định vì bài không công bố chi tiết

- Không có checkpoint downstream của tác giả, chỉ có các model tiền huấn luyện công khai. NN trong dự án này được **huấn luyện mới**. Kích thước lớp ẩn được chọn giữa `(64,32)` và `(128,64)` theo loss validation. Tối đa 120 epoch.
- CSV có nhãn đa lựa chọn nhưng bài báo báo cáo phân loại đơn nhãn. Quy tắc suy ra từ Bảng 1 là: giữ clip có đúng một loại lỗi R/P/B/I đạt ít nhất 2/3 phiếu; nếu không có lỗi nào đạt ngưỡng thì chọn F khi `NoStutteredWords` đạt 2/3 phiếu. SoundRep/WordRep gộp thành R bằng `max`. Quy tắc này tái tạo **chính xác** số lớp của Bảng 1 trên toàn bộ CSV: R=3.286, P=1.770, B=2.103, I=3.995, F=12.419; tổng 23.573. Bài không ghi thuật toán này, nên đây vẫn là một suy luận từ bảng.
- Bài không phát hành chính xác podcast ID của từng split. Chúng tôi dùng 10 lần chia ngẫu nhiên có seed, không phải 10-fold phân hoạch mỗi mẫu xuất hiện test đúng một lần.
- Score fusion trong bài chọn α trên **test**; bản này dùng α=0.9 đã nêu sẵn để tránh rò rỉ test.
- Các mô hình MFCC/ResNet/StutterNet trong Bảng 2 là baselines từ nghiên cứu trước, không thuộc pipeline embedding được triển khai ở đây.

## Chạy trên Kaggle

Kaggle Dataset Input phải chứa:

```
SEP-28k_labels.csv
clips_output/<Show>/<EpId>/<Show>_<EpId>_<ClipId>.wav
```

`wavs_output` (episode gốc) không cần cho pipeline. Bật GPU và Internet. `kaggle/run_baseline.py` tự tìm dữ liệu trong `/kaggle/input`, cài `requirements.txt`, chạy trích đặc trưng và đánh giá. Khi chạy qua Kaggle metadata, script clone GitHub; **cần đẩy phiên bản code mới lên repo trước** hoặc đặt `FLUENTA_BASELINE_DIR` tới bản code đã upload trong Kaggle Input. Kết quả ở `/kaggle/working/baseline_work/results`.

Nếu CSV và clip nằm ở **hai Kaggle Input riêng**, đặt `FLUENTA_LABELS_CSV` tới file CSV và `FLUENTA_CLIPS_DIR` tới thư mục chứa các thư mục `HeStutters/`, `HVSA/`... trước khi chạy script. Tên file CSV có thể là `sep.csv`; không cần đổi tên hay copy audio. Dùng `find /kaggle/input -maxdepth 4` trong notebook để xem đường dẫn thực tế.

Chạy thủ công trong thư mục `baseline_sheikh`:

```bash
python -m pip install -r requirements.txt
python prepare_data.py
python extract_features.py --batch 8
python train_eval.py
```

Chỉ chạy thử nhanh:

```bash
python prepare_data.py --per-class 100
python extract_features.py --batch 4
python train_eval.py --folds 2 --experiments l11,multilevel
```

Để khảo sát riêng cả 13 tầng, trích tất cả trước:

```bash
python extract_features.py --models wav2vec2 --layers all --batch 4
python train_eval.py --experiments layer_scan
```

Cache ở `work/features/<hash-manifest>/`; chạy lại trong cùng phiên Kaggle sẽ bỏ qua các khối đã xong. Nếu phiên Kaggle bị xóa, cần lưu cache ra một Kaggle Dataset/Output rồi gắn lại thì mới tiếp tục được. Nếu thay dataset hoặc manifest, code tự dùng thư mục cache khác. Có thể đặt `FLUENTA_SEP_DIR` tới thư mục dataset và `FLUENTA_WORK_DIR` tới nơi lưu kết quả.

Lưu ý: bản dữ liệu trong máy hiện có 20.906 clip trên 28.177 hàng nhãn. Với quy tắc trên, 17.358 clip có cả audio lẫn nhãn dùng được; 7.271 hàng thiếu audio và chỉ có 258/385 tập podcast. Kết quả từ bản thiếu audio này không so sánh trực tiếp với số trong Bảng 2.
