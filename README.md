# Classification with PyTorch Lightning

## 🛠 Cài đặt
```bash
pip install -r requirements.txt
```

## 🚀 Hướng dẫn sử dụng

### Chuẩn bị dữ liệu
Dữ liệu đầu vào là các file `.txt`, mỗi dòng chứa một đường dẫn tuyệt đối tới ảnh.
Ví dụ cấu trúc file `train.txt`:
```text
/path/to/image1.jpg
/path/to/image2.jpg
```


### Huấn luyện (Train)
Mở `src/entrypoints/train.py`, chỉnh sửa biến `CONFIG_PATH` nếu cần, sau đó chạy:
```bash
python src/entrypoints/train.py
```
- Nếu muốn **Resume**, hãy sửa `RESUME_PATH = "đường_dẫn_thư_mục_run_cũ"`.

### Đánh giá (Evaluation)
Mở `src/entrypoints/eval.py`, sửa `CKPT_PATH` tới file `.ckpt` tốt nhất:
```bash
python src/entrypoints/eval.py
```

### Dự đoán (Inference)
Mở `src/entrypoints/predict.py`, sửa `CKPT_PATH` và `INPUT_PATH` (có thể là folder ảnh hoặc file .txt):
```bash
python src/entrypoints/predict.py
```