# Project Deep Dive: VSL Recognition

Tài liệu này giải thích dự án như thể bạn mới bắt đầu đọc code. Mục tiêu là hiểu sâu các phần quan trọng: EDA, tiền xử lý, kiến trúc VideoMAE, training flow và app realtime.

Nên đọc song song tài liệu này với các file:

```text
notebooks/01_download_and_explore.ipynb
notebooks/02_train_videomae.ipynb
src/dataset.py
src/models.py
src/inference.py
app.py
```

## 1. Dự án đang giải bài toán gì?

Dự án làm **Vietnamese Sign Language isolated sign recognition**. Nghĩa là model nhận một video ngắn hoặc một segment webcam đã chứa một gesture, rồi dự đoán gesture đó thuộc class nào.

Điểm rất quan trọng: model hiện tại không tự hiểu một câu dài gồm nhiều gesture liên tục. Với webcam realtime, app phải tự cắt stream thành từng segment trước, rồi mới đưa segment đó vào model.

Pipeline khái quát:

```text
Video/segment
-> lấy 16 frame đại diện
-> resize/crop/normalize từng frame
-> tensor shape (B, T, C, H, W)
-> VideoMAE-Small
-> logits shape (B, 100)
-> softmax -> top-1/top-5 class
```

Ký hiệu:

- `B`: batch size.
- `T`: số frame, hiện là 16.
- `C`: số channel màu, RGB nên là 3.
- `H`, `W`: chiều cao/rộng frame, hiện là 224x224.
- `100`: số class trong Olympic AI2025 dataset.

## 2. EDA: vì sao phải phân tích dữ liệu trước?

EDA trong `notebooks/01_download_and_explore.ipynb` không phải phần phụ. Nó là bằng chứng cho các quyết định sau:

- Có nên cố định số frame không?
- Cố định bao nhiêu frame?
- Có resize/crop về 224 được không?
- Dataset có mất cân bằng class không?
- Có video lỗi, frame đen/trắng hoặc duplicate không?
- Split train/val có bị leakage do duplicate không?

### 2.1. Cấu hình Drive-first

Notebook 01 bắt đầu bằng các path:

```python
PROJECT_ROOT = "/content/drive/MyDrive/vsl-recognition"
DATA_DIR = f"{PROJECT_ROOT}/data/olympic_ai2025"
METADATA_DIR = f"{PROJECT_ROOT}/metadata"
```

Giải thích:

- `PROJECT_ROOT`: thư mục gốc trên Google Drive.
- `DATA_DIR`: nơi chứa dataset Olympic AI2025 sau khi download/extract.
- `METADATA_DIR`: nơi lưu các file JSON trung gian cho notebook train.

Lý do dùng Drive-first:

- Dataset và checkpoint lớn, không nên phụ thuộc vào ổ tạm của Colab.
- Nếu Colab ngắt, Drive vẫn giữ data/model/checkpoint.
- Local deploy chỉ cần model cuối, không cần dataset.

### 2.2. Chuẩn hóa label mapping

Dataset có `label_mapping.pkl`, nhưng format có thể là dict theo nhiều hướng khác nhau. Notebook 01 có logic normalize để đưa về:

```text
folder/class name -> label index chuẩn 0..99
class_names[index] -> tên class hiển thị
```

Ý nghĩa:

- Model chỉ học số nguyên label, ví dụ `78`.
- App cần đổi `78` lại thành tên class như `Thức ăn`.
- Nếu mapping sai, model vẫn train được nhưng nhãn hiển thị sẽ lệch hoàn toàn.

Đầu ra quan trọng:

```text
metadata/class_names.json
metadata/train.json
metadata/val.json
metadata/dataset_stats.json
```

### 2.3. EDA class imbalance

Notebook đếm số video mỗi class bằng ý tưởng:

```python
class_counts[class_dir.name] = len(videos)
```

Sau đó xem:

- class nhiều nhất có bao nhiêu video,
- class ít nhất có bao nhiêu video,
- tỉ lệ max/min,
- nhóm head/body/tail.

Vì sao cần?

Nếu class `A` có 74 video nhưng class `B` chỉ có 6 video, model rất dễ học tốt class nhiều và bỏ quên class ít. Accuracy tổng có thể vẫn ổn, nhưng Macro-F1 sẽ tệ vì class ít bị sai nhiều.

Quyết định kéo theo:

```python
WeightedRandomSampler(...)
```

Weighted sampler làm class ít xuất hiện nhiều hơn trong quá trình train. Nó không tạo dữ liệu mới, chỉ thay đổi xác suất lấy sample.

### 2.4. EDA thời gian: frame count, FPS, duration

Notebook đọc metadata video bằng OpenCV:

```python
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
duration = frame_count / fps if fps else np.nan
```

Giải thích từng dòng:

- `CAP_PROP_FRAME_COUNT`: tổng số frame của video.
- `CAP_PROP_FPS`: số frame mỗi giây.
- `duration`: thời lượng video, tính bằng frame count chia FPS.

Vì sao quan trọng?

Model VideoMAE cần input có số frame cố định. Dataset thật thì video có thể dài/ngắn khác nhau. Vì vậy ta phải chọn một số `T` cố định.

Notebook kiểm tra coverage cho các mốc:

```text
8 frames
16 frames
24 frames
32 frames
```

Nếu hầu hết video có ít nhất 16 frame, chọn `NUM_FRAMES = 16` là hợp lý. Nếu chọn 32 mà nhiều video không đủ frame, loader phải lặp frame cuối quá nhiều, làm input kém tự nhiên.

### 2.5. EDA không gian: resolution và aspect ratio

Notebook nhóm video theo `(width, height)`:

```python
resolution_counts = video_df.groupby(["width", "height"]).size()
```

Nếu phần lớn video là `224x224`, điều này ủng hộ:

```python
IMAGE_SIZE = 224
```

Vì sao?

- Dataset đã gần đúng kích thước này.
- VideoMAE pretrained cũng dùng 224.
- Không cần thay kiến trúc hoặc train processor mới.

### 2.6. EDA bad frames

Notebook có logic phát hiện frame đen/trắng/solid:

```python
bad_count = sum(is_solid_or_blank(frame) for frame in frames)
bad_ratio = bad_count / len(frames)
```

Ý nghĩa:

- `is_solid_or_blank(frame)`: kiểm tra frame gần như không có thông tin.
- `bad_ratio`: tỉ lệ frame xấu trong video.

Nếu nhiều video có bad frame, training loader nên lọc frame xấu trước khi sample. Nếu không, model có thể học từ frame gần như trống.

### 2.7. EDA duplicate candidates

Notebook tạo signature nhẹ cho video để phát hiện các nhóm nghi duplicate. Mục tiêu không phải xóa duplicate ngay, mà tránh train/val leakage.

Vấn đề leakage:

```text
video_A_copy_1 nằm trong train
video_A_copy_2 nằm trong val
```

Khi đó validation accuracy có thể ảo vì val quá giống train.

Quyết định kéo theo:

- split train/val nên duplicate-aware,
- các video cùng duplicate group nên nằm cùng một phía.

## 3. Tiền xử lý video trong training

Code lõi nằm trong `src/dataset.py`, còn notebook 02 có phiên bản mở rộng hơn cho Colab training.

### 3.1. Load video bằng Decord

Đoạn đặc trưng:

```python
def load_video_decord(video_path: str, num_frames: int = 16) -> np.ndarray:
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    if total_frames <= 0:
        raise ValueError(f"Could not read video: {video_path}")
```

Giải thích:

- `VideoReader(...)`: mở file video.
- `ctx=cpu(0)`: decode video trên CPU.
- `total_frames = len(vr)`: lấy tổng số frame.
- Nếu `total_frames <= 0`, video không đọc được nên phải báo lỗi.

### 3.2. Uniform temporal sampling

Đoạn quan trọng:

```python
if total_frames >= num_frames:
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
else:
    indices = np.concatenate(
        [np.arange(total_frames), np.full(num_frames - total_frames, total_frames - 1, dtype=int)]
    )
```

Giải thích:

- Nếu video đủ dài, `np.linspace` chọn đều `num_frames` vị trí từ đầu đến cuối video.
- Nếu video quá ngắn, lấy hết frame hiện có rồi lặp frame cuối cho đủ.

Ví dụ video có 30 frame, cần 16 frame:

```text
indices ~= [0, 1, 3, 5, 7, ..., 29]
```

Vì sao không lấy 16 frame đầu?

Vì gesture có thể nằm ở giữa hoặc cuối video. Lấy đều giúp giữ thông tin toàn bộ động tác.

Sau đó:

```python
return vr.get_batch(indices).asnumpy()
```

`get_batch` đọc đúng các frame đã chọn và trả về numpy array shape:

```text
(T, H, W, C)
```

Ví dụ:

```text
(16, 224, 224, 3)
```

### 3.3. Fallback OpenCV

Nếu Decord lỗi, code dùng OpenCV:

```python
cap = cv2.VideoCapture(video_path)
...
frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
```

OpenCV đọc frame dạng BGR, nhưng model/torchvision thường dùng RGB. Vì vậy phải đổi:

```text
BGR -> RGB
```

Fallback giúp pipeline bền hơn khi Decord không đọc được một số codec.

## 4. Transform frame thành tensor cho model

Class quan trọng:

```python
class VideoTransform:
    def __init__(self, mode: str = "train", image_size: int = 224):
        self.image_size = image_size
        self.mode = mode
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

Giải thích:

- `mode`: train hay eval.
- `image_size`: kích thước cuối cùng, 224.
- `Normalize`: chuẩn hóa theo ImageNet mean/std vì backbone pretrained được học trên phân phối tương tự.

### 4.1. Đổi numpy frame thành tensor

```python
video = torch.from_numpy(frames).float() / 255.0
video = video.permute(0, 3, 1, 2)
```

Trước dòng này:

```text
frames shape = (T, H, W, C)
value range = 0..255
```

Sau `float() / 255.0`:

```text
value range = 0..1
```

Sau `permute(0, 3, 1, 2)`:

```text
(T, H, W, C) -> (T, C, H, W)
```

PyTorch convolution/transform thường cần channel đứng trước height/width.

### 4.2. Transform khi train

Đoạn đặc trưng:

```python
i, j, h, w = RandomResizedCrop.get_params(video[0], scale=(0.8, 1.0), ratio=(0.9, 1.1))
_, brightness, contrast, saturation, _ = ColorJitter.get_params(
    brightness=(0.9, 1.1),
    contrast=(0.9, 1.1),
    saturation=(0.9, 1.1),
    hue=None,
)
```

Giải thích:

- `RandomResizedCrop.get_params(...)` chọn một crop ngẫu nhiên.
- `scale=(0.8, 1.0)` nghĩa là crop giữ từ 80% đến 100% diện tích ảnh.
- `ratio=(0.9, 1.1)` hạn chế méo tỉ lệ.
- `ColorJitter.get_params(...)` chọn mức chỉnh sáng/tương phản/bão hòa.

Điểm cực quan trọng:

```python
for frame in video:
    frame = frame[:, i : i + h, j : j + w]
```

Tất cả frame dùng chung `i, j, h, w`. Đây gọi là **temporal-consistent augmentation**.

Nếu mỗi frame crop một kiểu khác nhau, chuyển động trong video sẽ bị rung giả. Với video sign language, điều này rất hại vì model cần học quỹ đạo tay.

Tiếp theo:

```python
frame = Resize((self.image_size, self.image_size), antialias=True)(frame)
frame = adjust_brightness(frame, brightness)
frame = adjust_contrast(frame, contrast)
frame = adjust_saturation(frame, saturation)
transformed.append(self.normalize(frame))
```

Giải thích:

- resize crop về 224x224,
- chỉnh sáng/tương phản/bão hòa nhẹ,
- normalize theo ImageNet,
- đưa vào list.

Cuối cùng:

```python
return torch.stack(transformed)
```

Output shape:

```text
(T, C, H, W) = (16, 3, 224, 224)
```

### 4.3. Transform khi eval/inference

Eval không dùng random crop/color jitter:

```python
frame = Resize(self.image_size + 32, antialias=True)(frame)
frame = CenterCrop(self.image_size)(frame)
transformed.append(self.normalize(frame))
```

Giải thích:

- resize cạnh ngắn lên khoảng 256,
- center crop 224,
- normalize.

Lý do eval phải deterministic:

- Cùng một video phải cho kết quả ổn định.
- Validation metric không nên dao động do random augmentation.

## 5. Dataset class và DataLoader

Class chính:

```python
class VSLVideoDataset(Dataset):
    def __init__(self, video_list, num_frames=16, transform=None):
        self.video_list = video_list
        self.num_frames = num_frames
        self.transform = transform or VideoTransform(mode="eval")
```

Ý nghĩa:

- `video_list`: list metadata, mỗi item chứa path và label.
- `num_frames`: số frame cần sample.
- `transform`: train hoặc eval transform.

Khi DataLoader lấy một sample:

```python
def __getitem__(self, idx):
    item = self.video_list[idx]
    frames = load_video(item["path"], self.num_frames)
    return self.transform(frames), int(item["label"])
```

Dòng dữ liệu:

```text
path video
-> load_video(...)
-> frames (16, H, W, 3)
-> transform(...)
-> tensor (16, 3, 224, 224)
-> label int
```

Khi DataLoader gom batch size 8:

```text
video batch shape = (8, 16, 3, 224, 224)
label batch shape = (8,)
```

## 6. Kiến trúc VideoMAE-Small trong dự án

Model deploy hiện tại là `VideoMAEForVideoClassification`. Theo `models/videomae_olympic_best/config.json`, cấu hình chính:

```text
num_frames = 16
image_size = 224
patch_size = 16
tubelet_size = 2
hidden_size = 384
num_hidden_layers = 12
num_attention_heads = 16
intermediate_size = 1536
num_labels = 100
use_mean_pooling = true
```

### 6.1. Input vào model

Input từ DataLoader/app:

```text
(B, T, C, H, W) = (B, 16, 3, 224, 224)
```

VideoMAE sẽ chia video thành các **tubelet patches**.

Với:

```text
tubelet_size = 2
patch_size = 16
```

Ta có:

```text
temporal tokens = 16 / 2 = 8
spatial tokens per frame = 224 / 16 x 224 / 16 = 14 x 14 = 196
total tokens = 8 x 196 = 1568
```

Mỗi token đại diện cho một khối nhỏ:

```text
2 frames x 16 pixels x 16 pixels x 3 channels
```

### 6.2. Patch embedding

Patch embedding biến mỗi tubelet thành vector:

```text
raw tubelet -> hidden vector size 384
```

Sau bước này:

```text
(B, 16, 3, 224, 224)
-> (B, 1568, 384)
```

Nghĩa là mỗi video trở thành chuỗi 1568 token, mỗi token là vector 384 chiều.

### 6.3. Transformer encoder

Model có 12 transformer layers. Mỗi layer gồm:

```text
LayerNorm
-> Multi-Head Self-Attention
-> residual connection
-> LayerNorm
-> MLP / feed-forward
-> residual connection
```

Self-attention giúp mỗi token nhìn các token khác:

- token tay trái có thể liên hệ với tay phải,
- frame đầu có thể liên hệ frame sau,
- model học được chuyển động theo thời gian.

Trong config:

```text
hidden_size = 384
num_attention_heads = 16
intermediate_size = 1536
```

Ý nghĩa:

- vector mỗi token có 384 chiều,
- attention chia thành 16 heads để học nhiều kiểu quan hệ,
- MLP mở rộng lên 1536 chiều rồi nén về 384.

### 6.4. Mean pooling và classification head

Vì `use_mean_pooling = true`, model lấy trung bình các token cuối:

```text
(B, 1568, 384) -> (B, 384)
```

Sau đó classification head:

```text
Linear(384 -> 100)
```

Output:

```text
logits shape = (B, 100)
```

Mỗi số trong 100 logits là điểm thô cho một class. Softmax biến logits thành xác suất:

```python
probs = torch.softmax(outputs.logits[0].float(), dim=0)
```

Top-5:

```python
top5_probs, top5_idx = torch.topk(probs, min(5, len(class_names)))
```

## 7. Training flow chi tiết

Một batch đi qua training như sau:

```text
DataLoader
-> video_tensor (B, 16, 3, 224, 224)
-> labels (B,)
-> model(pixel_values=video_tensor)
-> logits (B, 100)
-> CrossEntropyLoss(logits, labels)
-> backward
-> optimizer.step()
-> scheduler.step()
```

### 7.1. Loss

Loss chính là CrossEntropy:

```text
CrossEntropy = LogSoftmax + Negative Log Likelihood
```

Nó phạt model nếu xác suất class đúng thấp.

Label smoothing `0.1` làm target bớt tuyệt đối. Thay vì ép class đúng là 1.0 và class sai là 0.0, nó phân phối một phần nhỏ xác suất sang class khác. Điều này giúp giảm overconfidence.

### 7.2. Optimizer AdamW

AdamW cập nhật weight dựa trên gradient, đồng thời dùng weight decay để regularize.

Vì fine-tune transformer, AdamW là lựa chọn phổ biến hơn SGD.

### 7.3. Scheduler cosine warmup

Warmup giúp learning rate tăng từ nhỏ lên target trong vài epoch đầu. Cosine scheduler giảm dần learning rate về sau.

Ý nghĩa:

- đầu training tránh update quá mạnh làm hỏng pretrained weights,
- cuối training giảm learning rate để hội tụ mượt hơn.

### 7.4. Mixed precision

Khi có CUDA, training dùng fp16/mixed precision:

```text
model weights/activations một phần ở float16
loss scale để tránh underflow
```

Lợi ích:

- giảm VRAM,
- tăng tốc trên GPU như T4,
- cho phép batch size thực tế hơn.

### 7.5. Resume checkpoint vs best model

Hai folder/file có mục đích khác nhau:

```text
last_checkpoint.pt
```

Dùng để train tiếp. Nó cần lưu:

- model state,
- optimizer state,
- scheduler state,
- scaler state,
- epoch hiện tại,
- history,
- best metrics.

```text
models/videomae_olympic_best/
```

Dùng để deploy. Nó cần:

- `model.safetensors`,
- `config.json`,
- `preprocessor_config.json`,
- `class_names.json`,
- `training_history.json`.

Không dùng `last_checkpoint.pt` để app predict, vì app load theo format Hugging Face folder.

## 8. Inference file/video trong `src/inference.py`

`VSLRecognizer` là wrapper giúp dùng model dễ hơn.

Khởi tạo:

```python
self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
self.use_fp16 = use_fp16 and self.device == "cuda"
self.transform = VideoTransform(mode="eval")
```

Giải thích:

- nếu có GPU thì dùng CUDA,
- fp16 chỉ bật trên CUDA,
- inference dùng eval transform deterministic.

Load model:

```python
self.model = VideoMAEForVideoClassification.from_pretrained(
    str(self.model_path),
    num_labels=len(self.class_names),
)
```

Ý nghĩa:

- đọc `config.json`,
- đọc `model.safetensors`,
- tạo classification head đúng 100 class.

Predict tensor:

```python
video_tensor = video_tensor.unsqueeze(0).to(self.device)
outputs = self.model(pixel_values=video_tensor)
```

`unsqueeze(0)` thêm batch dimension:

```text
(16, 3, 224, 224) -> (1, 16, 3, 224, 224)
```

## 9. App realtime trong `app.py`

App hiện có hai tab:

- `Browser webcam (WSL-friendly)`
- `Upload video`

OpenCV webcam trực tiếp đã bỏ vì WSL thường không có `/dev/video0`.

### 9.1. Load model local

```python
MODEL_PATH = APP_DIR / "models" / "videomae_olympic_best"
```

App chỉ tìm model local ở path này. Nó không dùng Google Drive path.

```python
model = VideoMAEForVideoClassification.from_pretrained(
    str(MODEL_PATH),
    num_labels=len(class_names),
)
```

Nếu có CUDA:

```python
if device == "cuda":
    model.half()
```

Mục đích là giảm VRAM và tăng tốc inference.

### 9.2. Normalize class name

```python
def normalize_label(label: str) -> str:
    return unicodedata.normalize("NFC", label)
```

Dataset lưu tiếng Việt dạng Unicode tổ hợp, ví dụ chữ và dấu tách rời. `NFC` gom về dạng dựng sẵn để hiển thị ổn hơn.

Overlay webcam dùng PIL vì OpenCV `putText` không hỗ trợ tiếng Việt tốt:

```python
draw.text((16, y), text, fill=color, font=font)
```

### 9.3. Preprocess app realtime

Hàm chính:

```python
def preprocess_frames(frames: list[np.ndarray]):
```

Input là list frame BGR từ browser/OpenCV decode.

Chọn 16 frame:

```python
if total >= NUM_FRAMES:
    indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
else:
    indices = np.concatenate([np.arange(total), np.full(NUM_FRAMES - total, total - 1, dtype=int)])
```

Giống training: nếu segment dài thì sample đều; nếu ngắn thì lặp frame cuối.

Xử lý từng frame:

```python
frame_rgb = cv2.cvtColor(center_square_crop(frames[idx]), cv2.COLOR_BGR2RGB)
tensor = torch.from_numpy(frame_rgb).float() / 255.0
tensor = tensor.permute(2, 0, 1)
tensor = Resize(IMAGE_SIZE + 32, antialias=True)(tensor)
tensor = CenterCrop(IMAGE_SIZE)(tensor)
transformed.append(normalize(tensor))
```

Giải thích:

- `center_square_crop`: loại border đen và crop vuông để giống dataset.
- `BGR -> RGB`: đổi format màu.
- `/ 255.0`: đưa pixel về 0..1.
- `permute(2, 0, 1)`: `(H,W,C)` thành `(C,H,W)`.
- `Resize(256) + CenterCrop(224)`: giống eval preprocessing.
- `normalize`: đưa phân phối pixel về đúng kiểu pretrained model mong đợi.

Cuối cùng:

```python
return torch.stack(transformed).unsqueeze(0)
```

Shape:

```text
(16, 3, 224, 224) -> (1, 16, 3, 224, 224)
```

### 9.4. Crop bỏ border đen

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
col_energy = gray.mean(axis=0)
row_energy = gray.mean(axis=1)
active_cols = np.where(col_energy > 8)[0]
active_rows = np.where(row_energy > 8)[0]
```

Giải thích:

- đổi frame sang grayscale,
- tính độ sáng trung bình từng cột/hàng,
- cột/hàng quá tối thường là border đen,
- giữ lại vùng có năng lượng hình ảnh.

Sau đó crop vuông:

```python
size = min(height, width)
top = max((height - size) // 2, 0)
left = max((width - size) // 2, 0)
return frame[top : top + size, left : left + size]
```

Mục tiêu là đưa webcam portrait/letterbox/pillarbox về gần dạng ảnh vuông như dataset train.

### 9.5. Gesture spotting state machine

Class chính:

```python
class BrowserVideoProcessor:
```

State ban đầu:

```python
self.state = "idle"
self.segment_frames = []
self.quiet_frames = 0
```

App không predict mọi frame. Nó làm:

```text
idle -> collecting -> predicting -> cooldown
```

### 9.6. Ước lượng chuyển động

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
score = float(np.mean(cv2.absdiff(small, self.prev_motion_frame)))
```

Giải thích:

- đổi frame hiện tại sang grayscale,
- resize nhỏ để tính nhanh,
- so với frame trước bằng absolute difference,
- lấy trung bình làm motion score.

Nếu motion score lớn, có khả năng người dùng đang làm gesture.

### 9.7. Bắt đầu collect segment

```python
if self.state == "idle" and motion_score >= START_MOTION_THRESHOLD:
    self.state = "collecting"
    self.segment_frames = list(self.raw_buffer)[-PREROLL_FRAMES:]
    self.quiet_frames = 0
```

Giải thích:

- nếu đang idle mà chuyển động đủ lớn, bắt đầu một gesture.
- lấy thêm vài frame trước đó bằng `PREROLL_FRAMES` để không mất phần đầu động tác.
- reset số frame yên lặng.

### 9.8. Kết thúc segment

```python
segment_finished = (
    len(self.segment_frames) >= MAX_SEGMENT_FRAMES
    or (len(self.segment_frames) >= MIN_SEGMENT_FRAMES and self.quiet_frames >= END_SILENCE_FRAMES)
)
```

Segment kết thúc khi:

- quá dài, hoặc
- đủ dài tối thiểu và đã yên lặng vài frame.

Vì sao cần logic này?

Webcam realtime là stream liên tục. Nếu cứ lấy 16 frame gần nhất, model sẽ thấy nhiều frame idle hoặc nửa động tác. State machine giúp app đưa vào model một đoạn giống video train hơn.

### 9.9. Predict segment

```python
result, latency_ms = predict_from_frames(model, class_names, device, frames)
if result["confidence"] >= CONFIDENCE_THRESHOLD:
    self.last_result = result
```

Nếu confidence thấp, app không hiển thị để tránh đoán bừa.

Hiện tại threshold thấp hơn lúc đầu vì debug cho thấy realtime webcam confidence thấp hơn upload video do ánh sáng, blur và domain shift.

### 9.10. Debug segment

```python
save_debug_segment(frames_for_inference)
```

Mỗi segment được lưu vào:

```text
tmp_analysis/realtime_debug/
```

Đây là phần rất quan trọng để debug. Khi app nhận sai, ta xem chính video segment mà model đã thấy, không đoán mò.

## 10. Vì sao upload video đúng hơn webcam realtime?

Upload video thường là clip đã cắt gọn:

```text
toàn bộ clip ~= một gesture hoàn chỉnh
```

Webcam realtime thì khác:

```text
idle -> chuẩn bị tay -> làm gesture -> dừng -> di chuyển người
```

Nếu cắt segment lệch, model sẽ thấy:

- quá nhiều idle frames,
- thiếu phần đầu gesture,
- thiếu phần cuối gesture,
- người ra khỏi khung,
- ánh sáng/motion blur khác dataset.

Vì vậy app realtime cần thêm `gesture spotting`, crop, threshold và debug segment. Đây là lớp logic nằm ngoài model classification.

## 11. Cách giải thích dự án khi thuyết trình

Một cách nói gọn nhưng sâu:

```text
Dự án dùng VideoMAE-Small để classify isolated sign video. 
Notebook 01 làm EDA để chứng minh các quyết định tiền xử lý: 16 frames, 224x224, no flip, weighted sampling.
Notebook 02 fine-tune VideoMAE trên 100 class, lưu best model dạng Hugging Face folder và checkpoint resume riêng.
Khi deploy realtime, webcam không còn là isolated clip nên app thêm gesture spotting state machine để cắt stream thành segment trước khi đưa vào model.
```

Nếu bị hỏi “vì sao realtime khó hơn validation?”, trả lời:

```text
Validation dùng clip đã cô lập gesture và cùng phân phối dataset. Webcam là stream liên tục, có idle frames, ánh sáng khác, blur, crop khác và người dùng khác. Vì vậy vấn đề realtime không chỉ là classification, mà là spotting + classification.
```

Nếu bị hỏi “TensorRT có giúp không?”, trả lời:

```text
TensorRT giúp latency/FPS nhưng không sửa domain shift hoặc segment cắt sai. Nên trước khi tối ưu TensorRT, cần làm đúng preprocessing, gesture spotting và debug segment.
```

## 12. Những file nên đọc khi muốn sửa từng phần

Sửa EDA hoặc giải thích dataset:

```text
notebooks/01_download_and_explore.ipynb
```

Sửa training, checkpoint, hyperparameter:

```text
notebooks/02_train_videomae.ipynb
src/dataset.py
src/models.py
```

Sửa inference file-based:

```text
src/inference.py
notebooks/03_inference_and_deploy.ipynb
```

Sửa webcam realtime:

```text
app.py
tmp_analysis/realtime_debug/
tmp_analysis/inspect_realtime_debug.py
tmp_analysis/predict_realtime_segments.py
```

