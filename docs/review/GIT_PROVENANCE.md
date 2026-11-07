# Nguồn lịch sử và nhánh hoàn thiện

Gói `NMTUH_VLU` ban đầu không có metadata Git. Checkout gốc được tìm thấy tại `D:/Thực hành/NMTUH/project/bank-marketing-pareto-hpo`; có ba commits `4d2904c`, `3b64a83`, `9a8ea87` và các thay đổi chưa commit. Checkout đó được giữ nguyên.

Một clone riêng tên `NMTUH_VLU_git` được tạo từ lịch sử gốc, nhánh `complete/4hp-v2`, remote đích https://github.com/NTP010205/NMTUH_VLU. Nội dung gói rà soát và triển khai mới được áp dụng vào clone riêng. Không reset, force push hoặc tạo commit lùi ngày.

`.gitattributes` giữ bytes nguyên vẹn, bao gồm line endings. Gói rà soát có nhiều logs CRLF trong khi các blobs ở commit gốc dùng LF; vì thế commit nhập bằng chứng có diff lớn về line endings. Nguồn chuẩn để bảo toàn bằng chứng là SHA-256 của gói người dùng cung cấp, không sửa manifest/hash v1 để che việc chuyển đổi newline. `v1_sha256_before.json`, `v1_sha256_after.json` và `v1_integrity.json` kiểm tra điều này trực tiếp.

`original_git_status_20261107.txt` và `original_git_remotes_20261107.txt` ghi trạng thái checkout gốc tại lần tiếp tục ngày 07/11/2026. Các commits mới phản ánh công việc thực, không đại diện cho năm tuần phát triển.

Kiểm tra `git ls-remote --exit-code` ngày 07/11/2026 truy cập được repo đích nhưng không thấy refs (exit 2). Điều này không phải bằng chứng đã push hoặc repo là public. Chỉ cập nhật trạng thái đăng sau khi có xác nhận từ GitHub.
