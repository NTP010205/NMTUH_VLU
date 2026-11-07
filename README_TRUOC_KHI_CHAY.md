# Bắt đầu từ đây

**Cập nhật triển khai ngày 07/11/2026:** dự án đã có runner, config, analysis và tests cho protocol 4 biến v2. Đọc [README.md](README.md) để xem trạng thái thực nghiệm và đường chạy hiện tại. Nội dung bên dưới mô tả gói bàn giao ban đầu ngày 03/11/2026; không dùng các câu “chưa triển khai” bên dưới như trạng thái hiện tại. V1 vẫn được giữ nguyên bytes và kiểm tra bằng SHA-256. Repository đích: https://github.com/NTP010205/NMTUH_VLU.

Đây là bản đóng gói gọn từ ZIP được cung cấp, rà soát ngày 03/11/2026. Mã và log gốc được giữ nguyên bytes; chưa triển khai protocol 4 biến hoặc chạy thêm seeds. Repo đích: https://github.com/NTP010205/NMTUH_VLU.

1. Đọc `report/Ra_soat_do_an_NMTUH.docx` để xem kết luận và số liệu đã kiểm tra.
2. Mở dự án trong VS Code/Codex, đưa toàn bộ `PROMPT_CODEX_HOAN_THIEN.md` cho Codex thực hiện.
3. Prompt yêu cầu chỉnh 4 biến, chạy nhiều seeds, phân tích và tạo báo cáo Word cuối từ kết quả thật.
4. Chỉ đăng bản hoàn thiện sau khi kiểm tra đường chạy từ clone sạch. Gói này chưa được push lên GitHub.

## Tình trạng hiện tại

- V1: 5 siêu tham số, 1 seed, 40 Random + 40 TPE trials. Log kiểm tra nhất quán; xem `docs/review/audit_existing.json`.
- V2 đề xuất: 4 biến gồm n_estimators, max_depth, min_samples_leaf, max_features; cố định class_weight=None; 5 search seeds × 2 sampler × 40 trials.
- Báo cáo rà soát là tài liệu chuẩn bị hoàn thiện, không phải báo cáo nộp cuối đã có kết quả v2.
- Source `src/`, `configs/`, notebooks và results giữ đường dẫn cũ để không làm hỏng tham chiếu. README.md cũ mô tả lịch sử và có một số lệnh legacy; chưa dùng nó như hướng dẫn final v2.

## Bản đóng gói đã bỏ gì

Không đóng gói môi trường Python, cache, bytecode, và metadata Git. Phải cài môi trường từ requirements trong môi trường phù hợp; không dùng lại venv Windows trên máy khác. Gói vẫn giữ raw data, SQLite và model baseline để bảo toàn bằng chứng cục bộ. `.gitignore` vẫn bỏ các file đó khi commit.

SQLite hiện bị Git-ignore nhưng phân tích Week 4 bắt buộc có nó. Vì vậy gói ZIP có thể kiểm tra log cũ nhưng clone GitHub chưa đủ để chạy toàn bộ đường phân tích; prompt đã yêu cầu xử lý đúng điểm này. Không xóa DB rồi tuyên bố SQLite audit PASS.

## Giữ lịch sử Git thật

Không xóa .git của repository bạn đang làm việc. Nếu có checkout gốc, áp dụng thay đổi trên nhánh riêng và giữ commits hiện tại. Gói này không mang metadata Git và không đại diện cho 5 tuần lịch sử phát triển. Không force push hoặc tạo commit giả lùi ngày.

## Trạng thái kiểm chứng

Đã chạy `src.week4_analysis.validate_sources`: PASS cả 14 kiểm tra nguồn, xác nhận CSV/JSON/SQLite và 13 nghiệm Pareto. Đã kiểm tra cú pháp toàn bộ .py bằng ast.parse; chưa chạy pytest hoặc train lại. Môi trường rà soát không có Optuna/pytest và khác sklearn lịch sử.

Nguồn dữ liệu và attribution: xem `data/DATASET.md`. Các file cũ ghi rằng repository không phân phối raw data; raw CSV ở đây được giữ lại từ gói bạn gửi để làm việc cục bộ và vẫn bị Git-ignore.
