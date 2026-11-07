# Tóm tắt bảo vệ trong 5 đến 7 phút

Nguyễn Thành Phát · MSSV 2474802016639 · https://github.com/NTP010205/NMTUH_VLU

Phút 0–1: Dự đoán y=yes/no trên bank-full.csv; bỏ duration để tránh thông tin sau cuộc gọi. Tối ưu đồng thời chất lượng xếp hạng AP và dung lượng toàn bộ pipeline.

Phút 1–2: x gồm n_estimators, max_depth, min_samples_leaf, max_features. min F(x)=(−mean AP qua 3 folds, max-fold size). Ràng buộc 50 MB, class_weight=None, split/CV/model seed 42.

Phút 2–3: 5 search seeds × 2 sampler × 40 attempted trials; 398 COMPLETE và 2 FAIL. Random và TPE dùng cùng domain/folds/budget; 10 startup trials được kiểm tra trùng cấu hình. Std giữa runs khác std giữa folds.

Phút 3–4: Baseline AP 0.414861, nghiệm chọn AP 0.443294; CV size 0.735 MB. Chỉ so CV với CV, full-train với full-train. Dẫn Hình 1 và Hình 2.

Phút 4–5: Bộ giá trị cuối: n_estimators=50, max_depth=14, min_samples_leaf=11, max_features=0.6468411998858646. Giữ AP cách best không quá 0,005, chọn size nhỏ nhất. Refit full train 0.919 MB, đạt cap.

Phút 5–6: Sensitivity thay một biến quanh nghiệm đã khóa; không chọn lại theo sensitivity. Năm seeds chỉ là bằng chứng mô tả. CV dùng cho tìm kiếm có thiên lệch lựa chọn; holdout từng được baseline v1 đánh giá và không dùng trong tuning v2.

Phút 6–7: Minh họa CSV/JSON, manifest hash và lệnh python -m src.analyze_results. Phân tích clone không cần DB; SQLite thiếu ghi not checked. Không khẳng định tối ưu toàn cục hoặc TPE luôn thắng.
