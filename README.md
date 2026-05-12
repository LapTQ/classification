# Action Classification with ResNet50 & PyTorch Lightning

Đây là bộ công cụ huấn luyện mô hình phân loại hành động (Action Classification) sử dụng kiến trúc ResNet50 và framework PyTorch Lightning. Bộ công cụ hỗ trợ huấn luyện từ ảnh crop người, quản lý phiên làm việc tự động và hỗ trợ resume training hoàn hảo.

## 📂 Cấu trúc thư mục
```text
classification/
├── configs/
│   └── resnet.yaml          # File cấu hình chính
├── src/
│   ├── core/
│   │   ├── data.py          # Xử lý nạp dữ liệu (DataModule & Dataset)
│   │   ├── model.py         # Kiến trúc ResNet50 LightningModule
│   │   └── utils.py         # Tiện ích visualize và quản lý thư mục
│   └── entrypoints/
│       ├── train.py         # Script huấn luyện chính
│       ├── eval.py          # Script đánh giá model
│       └── predict.py       # Script dự đoán (Inference)
└── requirements.txt         # Các thư viện cần thiết
```

## 🛠 Cài đặt
Yêu cầu Python 3.8+ và đã cài đặt driver NVIDIA (nếu dùng GPU).
```bash
pip install -r requirements.txt
```

## 🚀 Hướng dẫn sử dụng

### 1. Chuẩn bị dữ liệu
Dữ liệu đầu vào là các file `.txt`, mỗi dòng chứa một đường dẫn tuyệt đối tới ảnh.
Ví dụ cấu trúc file `train.txt`:
```text
/path/to/image1.jpg
/path/to/image2.jpg
```

### 2. Cấu hình
Mở file `configs/resnet.yaml` để điều chỉnh:
- `classes`: Danh sách nhãn theo đúng thứ tự.
- `train_data` / `val_data`: Map từng nhãn tới danh sách các file `.txt` tương ứng.
- `output_root`: Nơi lưu kết quả huấn luyện.

### 3. Huấn luyện (Train)
Mở `src/entrypoints/train.py`, chỉnh sửa biến `CONFIG_PATH` nếu cần, sau đó chạy:
```bash
python src/entrypoints/train.py
```
- Nếu muốn **Resume**, hãy sửa `RESUME_PATH = "đường_dẫn_thư_mục_run_cũ"`.

### 4. Đánh giá (Evaluation)
Mở `src/entrypoints/eval.py`, sửa `CKPT_PATH` tới file `.ckpt` tốt nhất:
```bash
python src/entrypoints/eval.py
```

### 5. Dự đoán (Inference)
Mở `src/entrypoints/predict.py`, sửa `CKPT_PATH` và `INPUT_PATH` (có thể là folder ảnh hoặc file .txt):
```bash
python src/entrypoints/predict.py
```

## 📊 Kết quả đầu ra
Mỗi lượt chạy sẽ tạo một thư mục riêng gồm:
- `weights/`: Chứa file `best...ckpt` và `last.ckpt`.
- `logs/`: Chứa log dạng CSV.
- `tensorboard/`: Chứa log để xem bằng TensorBoard.
- `config.yaml`: Bản sao cấu hình đã dùng.
- `train_batch_N.jpg`: Ảnh minh họa dữ liệu đã qua augmentation.
