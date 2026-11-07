# Dữ liệu cho protocol v2

Dùng đúng UCI Bank Marketing `bank-full.csv`, 45.211 dòng và 17 cột, phân cách bằng dấu `;`. Nguồn và giấy phép CC BY 4.0 ở [DATASET.md](DATASET.md). Tác giả nguồn: Moro, Rita và Cortez; DOI [10.24432/C5K306](https://doi.org/10.24432/C5K306).

Từ gốc dự án, chạy `.venv/Scripts/python.exe scripts/download_data.py`. Script tải `bank.zip/bank-full.csv` từ archive UCI, kiểm tra checksum và từ chối ghi đè file đã có nếu checksum khác.

Nếu mạng bị chặn, tải tại https://archive.ics.uci.edu/dataset/222/bank+marketing, giải nén `bank.zip` rồi đặt file tại `data/raw/bank-full.csv`. Không thay bằng `bank-additional-full.csv` hoặc `bank.csv`.

SHA-256 đã khóa: `d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d`.

Raw data bị Git-ignore. Phân tích v2 chỉ cần CSV/JSON đã xuất; huấn luyện mới cần raw data. Frozen split và CV folds được giữ trong `results/baseline/split_indices.npz` và `results/hpo/cv_folds_seed42.npz`. Search chỉ materialize train; `duration` bị loại, `unknown` giữ như giá trị categorical. Không chạy audit cũ để ghi đè `results/data_audit.json`.
