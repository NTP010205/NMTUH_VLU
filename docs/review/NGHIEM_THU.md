# Nghiệm thu v2 ngày 08/11/2026

Repository đích: https://github.com/NTP010205/NMTUH_VLU.

Ngân sách chính thức đã đủ 400 attempted trials: 398 COMPLETE và 2 FAIL, qua 10 studies và 5 search seeds. Có 1197 fold fits có log hoàn tất, trong đó 1194 thuộc các trials COMPLETE. Hai gián đoạn là Random seed 42 trial 20 và TPE seed 72 trial 0; không chạy bù.

| Sampler | Best AP mean | Sample std giữa 5 runs |
|---|---:|---:|
| random | 0.446143 | 0.000503 |
| tpe | 0.445613 | 0.000367 |

Nguồn: [sampler_summary.csv](../../results/analysis/final/sampler_summary.csv), [run_summary.csv](../../results/analysis/final/run_summary.csv).

Nghiệm cuối: `n_estimators=50`, `max_depth=14`, `min_samples_leaf=11`, `max_features=0.6468411998858646`.
CV AP = 0.443294, baseline = 0.414861; max-fold size = 0.735479 MB; full-train = 0.918818 MB. Nghiệm thuộc cả Random và TPE startup seed 62; nguồn Random được ghi theo tie-break.

Sensitivity: 8 cấu hình phân biệt, 24 folds đã đối chiếu aggregates và hashes. Một lân cận trội hơn trung tâm được công bố ở trang 9; không dùng hậu kiểm để đổi nghiệm chính.

## Kiểm chứng thực tế

- Workspace chính: 62 tests PASS; compileall PASS. Xem verification_local_20261107.json.
- Clone tại commit 58f46d6: 60 tests PASS, 2 SKIP vì thiếu SQLite lịch sử; compileall PASS. Xem clean_checkout_acceptance.json và clean_tests_final_20261108.log.
- Clone dùng venv cài riêng từ requirements, raw CSV tải lại từ UCI và kiểm tra checksum. Smoke mới: 4 COMPLETE, 12 fold records, cộng 3 folds baseline; chỉ số và tham số trùng smoke chính thức. Resume không thêm trial.
- Khi raw CSV tạm rời đường dẫn mặc định, không có DB/model full: phân tích 400 trials PASS; 8 CSV và 3 hình dựng lại trùng SHA-256. SQLite ghi not checked cho cả 10 studies.
- Full 400 trials được thực thi ở workspace chính. Clone độc lập đã chạy smoke và toàn bộ phân tích; không chạy thêm một full 400-trial thứ hai. Không gộp smoke tái lập vào kết quả chính thức.
- 111 tệp bằng chứng v1 giữ nguyên SHA-256 trước/sau. SQLite lịch sử vẫn còn cục bộ, không giả PASS khi thiếu trên clone.
- Báo cáo Word: 10 trang nội dung + 2 phụ lục, tóm tắt bảo vệ 1 trang; tất cả 13 trang đã được xem. Dùng Microsoft Word xuất PDF và PyMuPDF tạo ảnh vì renderer chuẩn thiếu soffice.exe. Hash tài liệu và ảnh ở document_qa.json.

## Tệp bàn giao

- [Báo cáo Word](../../report/Bao_cao_NMTUH.docx).
- [Tóm tắt bảo vệ](../../report/Tom_tat_bao_ve.docx).
- [Protocol](../../configs/final.yaml), [nghiệm khóa](../../results/final/selected_candidate.json), [sensitivity CSV](../../results/sensitivity/summary.csv).
- Nhánh Git complete/4hp-v2 bảo toàn ba commits gốc; các commits mới phản ánh công việc thực. Không coi đây là bằng chứng năm tuần phát triển.

## Giới hạn còn lại

Không có đánh giá holdout v2 hoặc external test; holdout từng được baseline v1 xem. Năm search seeds chỉ cho bằng chứng mô tả và có hai trial gián đoạn. Cap 50 MB không binding trên các trials hoàn thành. Lịch sử năm tuần không thể được tạo lại bằng commits lùi ngày. Sinh viên cần đọc báo cáo, hiểu các quyết định và nộp link trên LMS; những thao tác cá nhân đó chưa được thực hiện thay sinh viên. Trạng thái push phải kiểm tra bằng remote GitHub, không suy ra từ việc có commit cục bộ.

## GitHub đã xác nhận

Đã push main và complete/4hp-v2 lên repository public, mặc định main. GitHub xác nhận commit khoa học e655158cb9172368e02b628b822467e551442d17 và báo cáo Word; mô tả About cùng sáu topics đã cập nhật. [Bằng chứng xác nhận](github_publication.json). Các commits sau đó chỉ bổ sung hồ sơ đăng.
