# Caption và nhận xét biểu đồ EDA train-only

Nguồn: 36.168 frozen training rows; không materialize holdout; `duration` bị loại khi đọc.

## Hình EDA-1. Phân bố biến mục tiêu trên frozen training set

File: `figures/week4/eda/01_target_distribution_train.png`

**Caption:** Phân bố nhãn trên 36.168 mẫu train cố định: `no` = 31.937 (88,30%) và `yes` = 4.231 (11,70%).

**Nhận xét:** Tập train mất cân bằng, với số mẫu `no` xấp xỉ 7,55 lần số mẫu `yes`. Vì vậy Average Precision phù hợp hơn accuracy đơn thuần cho đánh giá xếp hạng lớp dương.

## Hình EDA-2. Phân bố tuổi trên frozen training set

File: `figures/week4/eda/02_age_distribution_train.png`

**Caption:** Phân bố tuổi của 36.168 mẫu train; tuổi trung bình 40,89, trung vị 39, khoảng tứ phân vị 33–48 tuổi.

**Nhận xét:** Phần lớn quan sát tập trung ở nhóm tuổi lao động trung niên; đuôi phải thể hiện một số khách hàng lớn tuổi. Đây là mô tả train-only, không phải bằng chứng về quan hệ nhân quả hoặc khả năng tổng quát ngoài dữ liệu.

## Hình EDA-3. Phân bố số dư tài khoản trên frozen training set

File: `figures/week4/eda/03_balance_distribution_train.png`

**Caption:** Phân bố `balance` của 36.168 mẫu train: trung bình 1365,49, trung vị 451, khoảng tứ phân vị 74–1430. Biểu đồ chính giữ toàn bộ dữ liệu và dùng trục tần số log; ô nhỏ phóng đại vùng P1–P99 (-626 đến 13161).

**Nhận xét:** Số dư lệch phải mạnh và có cả giá trị âm. Không quan sát nào bị xóa khỏi biểu đồ chính; vùng P1–P99 chỉ là cửa sổ phóng đại để nhìn rõ phần mật độ cao.
