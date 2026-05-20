# Project Walkthrough: VSL Recognition

Tài liệu này giúp đọc dự án `vsl-recognition` theo đúng dòng chảy thực tế: từ bài toán, dữ liệu, training trên Colab/Google Drive, đến deploy realtime bằng Streamlit. Mục tiêu không phải chỉ biết file nào nằm ở đâu, mà hiểu vì sao pipeline hiện tại được thiết kế như vậy.

Nếu cần bản giảng sâu hơn theo từng khối code đặc trưng, đọc thêm `PROJECT_DEEP_DIVE.md`.

## 1. Bức tranh tổng thể

Dự án giải bài toán **Vietnamese Sign Language isolated sign recognition**: mỗi video hoặc segment tương ứng một gesture, model dự đoán một trong khoảng 100 class. Đây chưa phải bài toán nhận diện câu dài liên tục.

Pipeline chính:

```text
Video hoặc webcam segment
-> lấy mẫu đều 16 frames
-> resize/crop về 224x224
-> VideoMAE-Small
-> top-1/top-5 prediction
```

Dataset chính hiện tại là **Olympic AI2025 preliminary dataset**. Dự án đã chuyển khỏi hướng Multi-VSL làm dataset train chính vì Olympic AI2025 có cấu trúc sạch hơn, nhiều class/video hơn cho mục tiêu hiện tại, và phù hợp hơn để train baseline 100-class.

Training chạy trên Colab/Kaggle GPU. Google Drive lưu data, metadata, checkpoint và model sau train. Máy local chỉ cần folder model cuối để chạy app, không cần tải toàn bộ dataset về.

## 2. Luồng training trên Colab/Google Drive

Đọc notebook theo thứ tự sau:

```text
notebooks/01_download_and_explore.ipynb
-> notebooks/02_train_videomae.ipynb
-> notebooks/03_inference_and_deploy.ipynb
```

### Notebook 01: Download và EDA

Notebook 01 chịu trách nhiệm tải dataset, giải nén, scan cấu trúc, tạo metadata và đưa ra bằng chứng cho các quyết định modeling.

Cần đọc notebook này với câu hỏi: dữ liệu buộc mình phải xử lý thế nào?

- Dataset có khoảng 100 class và vài nghìn video train.
- Notebook tạo `train.json`, `val.json`, `class_names.json`, `dataset_stats.json`.
- EDA kiểm tra frame count, FPS, duration, resolution, class imbalance, video lỗi, bad frames và duplicate candidates.
- `16 frames` được chọn vì EDA cho thấy video đủ frame và input này cân bằng giữa thông tin thời gian, VRAM, latency.
- `224x224` phù hợp vì dataset gốc/VideoMAE pretrained đều xoay quanh input này.
- Không dùng horizontal flip vì trong sign language, trái/phải có thể mang nghĩa khác nhau.
- Weighted sampling được dùng vì class imbalance tương đối rõ.

Notebook 01 không nên âm thầm xóa dataset. Những phát hiện như bad frames hoặc duplicate candidates chủ yếu dùng để giải thích và cảnh báo.

### Notebook 02: Fine-tune VideoMAE

Notebook 02 đọc metadata từ Drive rồi fine-tune VideoMAE-Small.

- Input metadata: `train.json`, `val.json`, `class_names.json`.
- Loader đọc video, lọc/bù frame khi cần, rồi sample đúng `NUM_FRAMES`.
- Transform train dùng crop/color jitter nhất quán theo video để tránh làm chuyển động bị méo từng frame.
- Model bắt đầu từ `MCG-NJU/videomae-small-finetuned-kinetics`.
- Loss là CrossEntropy với label smoothing.
- Optimizer là AdamW, scheduler cosine warmup.
- Class imbalance được xử lý bằng `WeightedRandomSampler`.

Checkpoint có hai loại:

- `checkpoints/videomae_olympic/last_checkpoint.pt`: dùng để resume training khi Colab ngắt hoặc hết GPU.
- `models/videomae_olympic_best/`: folder model tốt nhất để inference/deploy, chứa `model.safetensors`, `config.json`, `preprocessor_config.json`, `class_names.json`, `training_history.json`.

Khi đánh giá training, không chỉ nhìn accuracy cuối. Nên xem `training_history.json`, `metrics.json`, và training curves để biết model có overfit hay dao động không.

### Notebook 03: Inference và deploy check

Notebook 03 dùng để xác nhận model đã train có thể load lại, predict sample video, benchmark latency và hướng dẫn cấu trúc local deploy.

Điểm cần nhớ: local app không dùng `/content/drive/...`; nó chỉ load:

```text
models/videomae_olympic_best/
```

## 3. Code lõi trong `src/`

Đọc `src/` sau khi đã hiểu notebook.

### `src/models.py`

File này định nghĩa cách tạo VideoMAE classifier:

- `DEFAULT_VIDEOMAE_MODEL = "MCG-NJU/videomae-small-finetuned-kinetics"`.
- `create_videomae_model(...)` load pretrained VideoMAE và thay classification head theo số class.
- `freeze_backbone` có thể dùng để chỉ train head, nhưng pipeline hiện tại fine-tune toàn bộ model.
- `get_model_info(...)` phục vụ báo cáo số parameters và kích thước model.

### `src/dataset.py`

File này gom logic đọc video và transform.

- `load_video_decord(...)` đọc video bằng Decord và sample đều frame.
- `load_video_opencv(...)` là fallback nếu Decord lỗi.
- `VideoTransform(mode="train")` dùng RandomResizedCrop và ColorJitter nhất quán theo toàn video.
- `VideoTransform(mode="eval")` dùng Resize + CenterCrop ổn định.
- `VSLVideoDataset` đọc metadata dạng `{"path": ..., "label": ...}`.
- `make_sample_weights(...)` tạo weight nghịch đảo tần suất class cho balanced sampling.

### `src/inference.py`

File này cung cấp recognizer chuẩn cho inference từ video hoặc frame buffer.

`VSLRecognizer` làm các việc:

- load model từ folder Hugging Face local,
- load class names,
- chọn CPU/CUDA và fp16 nếu có GPU,
- predict từ file video bằng `predict_video(...)`,
- predict từ frame array bằng `predict_frames(...)`.

## 4. App realtime local

App chính là:

```text
app.py
```

App hiện tại có 2 đường dùng:

- Browser webcam bằng `streamlit-webrtc`.
- Upload video để kiểm tra nhanh model/inference với clip có sẵn.

Đường OpenCV webcam trực tiếp đã bị bỏ vì không ổn trong WSL và dễ lỗi `/dev/video0`.

### Load model

App load model từ:

```text
models/videomae_olympic_best/
```

Folder này cần có ít nhất:

```text
config.json
model.safetensors
preprocessor_config.json
class_names.json
training_history.json
```

App normalize class names sang Unicode NFC để hiển thị tiếng Việt tốt hơn.

### Preprocess realtime

App nhận frame BGR từ webcam/browser, sau đó:

- loại bớt border đen nếu có,
- crop về hình vuông vùng trung tâm/active content,
- sample đều 16 frame,
- resize/crop về 224,
- normalize ImageNet mean/std,
- đưa vào VideoMAE.

### Gesture spotting

Điểm quan trọng: model chỉ giỏi classify một clip/segment đã chứa gesture. Webcam realtime là stream liên tục, nên app phải tự cắt segment trước khi classify.

`BrowserVideoProcessor` dùng state machine:

```text
idle -> collecting -> predicting -> cooldown
```

Ý nghĩa:

- `idle`: chờ chuyển động đủ lớn.
- `collecting`: gom frame của gesture.
- `predicting`: classify segment vừa gom.
- `cooldown`: nghỉ ngắn để tránh predict lặp liên tục.

Các tham số chính:

- `START_MOTION_THRESHOLD`: ngưỡng bắt đầu gesture.
- `END_MOTION_THRESHOLD`: ngưỡng xem gesture đã lắng xuống.
- `MIN_SEGMENT_FRAMES`: segment tối thiểu trước khi predict.
- `MAX_SEGMENT_FRAMES`: giới hạn segment quá dài.
- `CONFIDENCE_THRESHOLD`: ngưỡng để hiển thị prediction.
- `RESULT_HOLD_SECONDS`: thời gian giữ kết quả trên overlay.

### Debug realtime

Mỗi segment realtime có thể được lưu vào:

```text
tmp_analysis/realtime_debug/
```

Folder này giúp xem model thực sự được feed gì, rất hữu ích khi webcam nhận diện kém.

## 5. Debug và giới hạn thực tế

Các file trong `tmp_analysis/` là vùng phân tích, không phải core product code.

- `video_report.json`, `992931_contact.png`, `7842195253917_contact.png`: so sánh clip dataset với clip webcam thực tế.
- `realtime_debug/`: segment do app realtime tự lưu.
- `realtime_debug_review/`: contact sheet/report sinh ra khi phân tích lại segment.
- `realtime_prediction_compare.json`: so sánh các đợt test realtime khác nhau.
- `inspect_videos.py`, `inspect_realtime_debug.py`, `predict_realtime_segments.py`: script phụ để debug video/segment.

Giới hạn quan trọng của dự án:

- Upload video có thể đúng hơn realtime vì upload thường là clip đã cắt sẵn.
- Webcam realtime khó hơn vì có idle frame, blur, ánh sáng yếu, framing khác dataset và timing segment không hoàn hảo.
- VideoMAE là RGB model nên nhạy với domain shift: nền, ánh sáng, khoảng cách camera, crop và chất lượng webcam.
- TensorRT chỉ giúp latency/FPS, không tự sửa được sai lệch dữ liệu giữa train và webcam thực tế.

## 6. Câu hỏi tự kiểm tra

Sau khi đọc xong, hãy tự trả lời 10 câu này:

1. Dự án đang dùng dataset nào và vì sao bỏ Multi-VSL làm train chính?
2. Vì sao chọn VideoMAE-Small thay vì model lớn hơn?
3. Vì sao input là 16 frames?
4. Metadata nào được notebook 01 sinh ra để notebook 02 dùng?
5. `last_checkpoint.pt` khác gì với `videomae_olympic_best/`?
6. Vì sao upload video có thể đúng nhưng webcam realtime lại khó hơn?
7. App phát hiện gesture realtime bằng cơ chế nào?
8. `tmp_analysis/realtime_debug` dùng để làm gì?
9. Vì sao class name tiếng Việt từng bị lỗi font?
10. Nếu cần demo local, folder/file tối thiểu nào phải có?

Nếu trả lời trôi chảy các câu trên, bạn đã hiểu dự án đủ để thuyết trình, bảo vệ quyết định kỹ thuật, sửa app và tiếp tục tối ưu.

## 7. Thứ tự đọc khuyến nghị

Đọc nhanh trong 30 phút:

```text
README.md
PROJECT_PLAN.md
PROJECT_WALKTHROUGH.md
PROJECT_DEEP_DIVE.md
app.py
src/dataset.py
src/models.py
src/inference.py
```

Đọc kỹ để báo cáo hoặc bảo vệ:

```text
README.md
PROJECT_PLAN.md
PROJECT_WALKTHROUGH.md
PROJECT_DEEP_DIVE.md
notebooks/01_download_and_explore.ipynb
notebooks/02_train_videomae.ipynb
notebooks/03_inference_and_deploy.ipynb
src/
app.py
tmp_analysis/
```
