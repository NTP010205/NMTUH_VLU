# Chỉ mục lịch sử

Tên đường dẫn cũ được giữ để bảo toàn imports, notebooks và tham chiếu nguồn. V1 là thí nghiệm 5 biến, sampler seed 42, 40 Random và 40 TPE trials. V2 dùng 4 biến theo `configs/final.yaml`.

- [README v1](README_v1.md) lưu hướng dẫn cũ; các đường dẫn tương đối trong đó được hiểu từ gốc repository.
- `configs/experiment.yaml`, `configs/hpo_ap_model_size_v1.yaml`, `src/optimize_ap_model_size.py`, `src/week4_analysis.py` thuộc lịch sử.
- Notebooks 01–05 phân tích v1; notebook EDA được đồng bộ từ script có sẵn vì bản đóng gói thiếu cell/hình mà test yêu cầu. Không sửa manifest kết quả lịch sử để hợp thức hóa notebook.
- `results/hpo/ap_model_size_v1/` và `results/analysis/week4/` là bằng chứng v1, không đổi nhãn thành v2.
- `requirements_v1.txt` giữ nguyên pins lịch sử. V2 đã cài thành công đúng các pins này với Python 3.11.15.

**SQLite lịch sử:** `src.week4_analysis.validate_sources` cần DB từ gói lịch sử, hiện bị Git-ignore. Clone GitHub không đủ để chạy audit v1. Không xóa DB hoặc ghi SQLite PASS khi thiếu DB. V2 phân tích CSV/JSON, kiểm tra hash và fold aggregates; SQLite là audit bổ sung.

`report/Ra_soat_do_an_NMTUH.docx` là tài liệu chuẩn bị, không phải báo cáo nộp v2. Snapshot hash trước/sau nằm ở `docs/review/v1_sha256_*.json`.
