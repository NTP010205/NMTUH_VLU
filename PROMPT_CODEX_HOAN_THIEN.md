# Yêu cầu hoàn thiện đồ án Nhập môn Tối ưu hóa

Hãy trực tiếp thực hiện yêu cầu trong checkout dự án, không chỉ trả lời bằng kế hoạch. Đích GitHub: https://github.com/NTP010205/NMTUH_VLU. Đọc AGENTS.md nếu có. Giữ cấu trúc và tận dụng mã hiện tại; không viết lại toàn bộ, không thay bằng web app. Dùng tiếng Việt trong tài liệu. Không giả lập kết quả, commit hoặc thời gian phát triển.

## 1. Kiểm tra đầu vào và bảo toàn bằng chứng

- Xem README, configs/experiment.yaml, configs/hpo_ap_model_size_v1.yaml, src/optimize_ap_model_size.py, src/pipeline.py, src/week4_analysis.py, tests/, docs/ và results/hpo/ap_model_size_v1/full_run_01/.
- Nếu dùng gói đã rà soát, đọc README_TRUOC_KHI_CHAY.md và report/Ra_soat_do_an_NMTUH.docx. Nếu có đề bài PDF và báo cáo gốc, đối chiếu cả hai.
- Chụp git status, branch và remote; không ghi đè thay đổi chưa commit. Nếu repo đích có nội dung khác, tích hợp trên nhánh mới, giữ nguyên nội dung không liên quan. Không force push, không ghi đè lịch sử và không tự tạo commit lùi ngày.
- Kết quả v1 là 5 biến, một sampler seed 42, Random 40 + TPE 40 trials, CV 3 fold, 13 nghiệm Pareto. Selected TPE trial 35 có AP 0.44341749539409525 và full-train size 1.289340 MB. Chỉ coi đây là số liệu lịch sử; không đổi nhãn thành kết quả 4 biến hoặc nhiều seed.
- Không sửa bytes của log v1, manifests hoặc hashes để làm kiểm tra PASS. Lưu SHA-256 trước và sau công việc.
- Môi trường đang có thể chứa thư viện khác requirements. Tạo venv đúng Python, thử cài yêu cầu. Nếu pin không cài được, ghi lỗi, tạo lock cho v2 từ môi trường cài thành công, ghi lý do; giữ environment lịch sử v1. Không dùng runtime Windows đi kèm ZIP trên hệ điều hành khác.

## 2. Chốt bài toán 4 siêu tham số

Giữ RandomForestClassifier, học có giám sát, phân loại y=yes/no. Tạo protocol mới ap_model_size_4hp_v2, không đổi protocol v1.

Vector biến quyết định x = (n_estimators, max_depth, min_samples_leaf, max_features):
- n_estimators: int 50..300, bước 25.
- max_depth: int 4..24.
- min_samples_leaf: int 1..12.
- max_features: float 0.4..1.0; tỷ lệ cột sau tiền xử lý xét khi tách nút.

Cố định class_weight=None và ghi trong fixed_parameters, không gọi suggest_categorical cho class_weight. Đây là lựa chọn đơn giản hóa phạm vi, không phải kết luận class_weight không ảnh hưởng. Split seed=42, CV seed=42, model random_state=42, n_jobs=1. Seed thực nghiệm thay đổi sampler, không đồng thời thay data split/CV/model seed. Giải thích rõ 4 siêu tham số cụ thể không phải 4 optimizer SGD, Momentum, RMSProp, Adam của Lab 5. RF không có learning_rate hoặc batch_size trong thí nghiệm này.

Mục tiêu: maximize AP_CV(x)=mean(AP_1, AP_2, AP_3) và minimize S(x)=max(size_bytes_1,size_bytes_2,size_bytes_3)/1e6; có thể viết min(-AP_CV,S). Serialize toàn bộ fitted preprocessing+RF bằng Joblib compress=3. Ràng buộc S<=50 MB, miền biến như trên. Không dùng latency vào mục tiêu hoặc Pareto. Giữ ngưỡng 50 MB và công bố nếu không binding, không siết hậu nghiệm để tạo vi phạm. AP là average_precision_score từ predict_proba, không gọi AP là accuracy hoặc đồng nhất với diện tích PR tính hình thang.

## 3. Dữ liệu và baseline

- bank-full.csv: 45211 hàng, 17 cột; không thay bằng bank-additional-full.csv. Bỏ duration; 15 đặc trưng gốc còn lại. Giữ unknown như quy trình cũ, mô tả số lượng.
- Tái sử dụng frozen 80/20 split và 3 CV folds nếu checksum dữ liệu khớp. Fit imputer/encoder chỉ trên training fold. Không dùng test trong tuning, chọn seed, chọn tolerance hay chọn thuật toán.
- Baseline RF mặc định: n_estimators=100, max_depth=None, min_samples_leaf=1, max_features='sqrt', class_weight=None, random_state=42, n_jobs=1. Baseline có thể ngoài search domain; không ép baseline kém để search thắng.
- Chạy baseline CV trong môi trường v2; lưu AP từng fold, mean, sample std (ddof=1), serialized size từng fold, max-fold size, runtime. Khi so size CV chỉ so CV với CV; full-train chỉ so full-train với full-train.
- Holdout đã từng được đánh giá ở baseline v1: không gọi hoàn toàn chưa từng xem. Sau khi khóa lựa chọn có thể báo một lần đánh giá mô tả cuối, kèm hạn chế, nhưng không đổi cấu hình theo test. Không hứa có external test nếu không có.

## 4. Thực nghiệm nhiều lần

Mặc định chính thức: search_seeds=[42,52,62,72,82], samplers=[random,tpe], trials_per_sampler=40, CV=3. Tổng 400 trials, 1200 fold fits (chưa kể baseline, sensitivity, refit). Đây là 10 studies độc lập, không phải chạy lại cùng lệnh resume seed42. 5 lần là đề xuất cho yêu cầu nhiều lần, không ghi rằng đề PDF bắt buộc đúng 5.

- Mỗi seed một thư mục results/final/seed_XX/{random,tpe}; database và study_name phải khác nhau theo protocol, seed, sampler. Protocol snapshot lưu mọi seed và fixed params.
- RandomSampler(seed=s), TPESampler(seed=s,n_startup_trials=10) và constraints phù hợp với phiên bản Optuna. Dùng cùng domain/folds/budget, không thay tham số TPE sau khi xem kết quả.
- Smoke 2 trials/sampler trên dữ liệu train thật với output riêng; không trộn vào full. Sau smoke và tests PASS, chạy full. Yêu cầu này đã cho phép chạy full, không dừng ở bước hỏi lại về tác vụ tính toán cục bộ thông thường.
- Idempotent resume: đủ 40 trials thì không thêm. Ghi failed trials và lý do; quy định budget theo trial attempted, không âm thầm retry vô hạn để che failure. Từ chối resume nếu fingerprint/protocol thay đổi.
- Ưu tiên tuần tự, giới hạn thread giống nhau để runtime so sánh được. Log phần cứng, OS, Python, packages, run wall time và tổng trial time, không trộn hai loại thời gian.
- Nếu runtime bị giới hạn: lưu checkpoint, số trials thực tế, lệnh resume; đánh dấu chưa hoàn tất, không tự giảm budget rồi gọi là full.

## 5. Phân tích và chọn nghiệm

- Log run_id, protocol, search_seed, model_seed, sampler, trial, 4 biến, fixed class_weight, AP từng fold/mean/std, size bytes từng fold/max, g=S-50, feasible, runtime, status, dataset/split/config hashes.
- Lập bảng một dòng mỗi sampler/seed: best feasible AP, AP và size của selected candidate, số complete/failed, runtime, số nghiệm Pareto distinct. Tổng hợp mean ± sample std GIỮA 5 SEARCH RUNS, tách biệt CV fold std. Baseline cố định không có 5 kết quả độc lập; không sao chép baseline 5 lần để suy ra variance.
- So sánh paired difference TPE-Random theo cùng seed; 5 seeds chỉ cho bằng chứng mô tả hạn chế, không khẳng định thống kê quá mức. Có bảng best-so-far tại budget 10/20/40. Nếu 10 startup trials giống nhau phải nêu rõ, không gọi các cấu hình trùng là khám phá độc lập.
- Tính Pareto từ nghiệm complete+feasible và bỏ cấu hình trùng khi đếm nghiệm. Từng seed có selection riêng; mô hình cuối chọn từ tập cấu hình hợp lệ gộp cả 5 seeds, ghi rõ đã dùng ngân sách gộp, không coi đây là performance trung bình của thuật toán.
- Quy tắc khóa trước chạy: AP_best từ tập khả thi; giữ AP>=AP_best-0.005; chọn size CV nhỏ nhất, hòa thì AP cao hơn, fold std nhỏ hơn, config_hash, sampler rồi trial. Refit full train và kiểm tra full-train size<=50 MB; nếu fail chọn tiếp trong cùng tập near-best, không đổi tolerance.
- Để đáp ứng sensitivity/ablation tuần 4: sau khi khóa lựa chọn, thay từng biến một quanh nghiệm đã chọn, giữ 3 biến khác. Mỗi biến tối đa 3 giá trị hợp lệ, unique; có giá trị selected và các giá trị thấp/cao lân cận. Chạy cùng CV, ghi ra results/sensitivity/ và không dùng kết quả này để sửa lựa chọn chính. Phân tích cục bộ, không khẳng định importance toàn cục hoặc quan hệ nhân quả.
- Sinh tối thiểu: best-so-far AP mean±std theo trial, AP-size Pareto với selected, phân bố best AP qua seeds; thêm sensitivity nếu phù hợp. Mọi hình/bảng phải được dựng từ CSV/JSON, không nhập số bằng tay.

## 6. Sửa đúng những điểm gây rối

- src/optimize_ap_model_size.py đang hardcode protocol v1, 40 trials, output, class_weight suggest và một seed. Giữ runner v1 làm lịch sử. Thêm src/run_experiments.py cho v2, dùng lại các helper đã phù hợp; không phá v1 chỉ để đổi tên.
- src/pipeline.py: thêm fixed_parameters có mặc định tương thích v1; bảo đảm class_weight vẫn là None thật, không chuỗi 'None'.
- src/week4_analysis.py hardcode 80 trials/240 folds/hash selected cũ và bắt buộc SQLite. Giữ phân tích lịch sử đó; thêm src/analyze_results.py cho protocol mới, suy ra counts từ config và log, không hardcode hash người thắng.
- .gitignore đang bỏ *.db nhưng phân tích v1 bắt buộc chúng. Chọn cách clone tái lập: v2 phân tích CSV/JSON có manifest và cross-check fold aggregates; SQLite chỉ dùng resume/audit bổ sung, thiếu DB phải ghi 'not checked', không ghi SQLite PASS. Nếu giữ phân tích v1, tài liệu phải nói cần DB từ gói lịch sử; không quảng cáo clone-only chạy được lệnh đó.
- README mặc định phải chỉ có đường chạy chính v2 và link lịch sử, không lẫn lệnh latency full bị block. Cấu hình baseline mới bỏ mục TODO latency lỗi thời, không sửa snapshot lịch sử.
- Chỉ di chuyển các file legacy khi đã sửa imports, CLI, notebooks, docs, tests và manifest liên quan. Không đơn thuần kéo thư mục khiến đường dẫn chết. Chấp nhận giữ tên đường dẫn cũ và tạo docs/archive index khi đó ít rủi ro hơn.
- Không đưa .venv, .python, .uv-cache, __pycache__, model lớn, cache hoặc token vào Git. Không xóa .git của checkout thật. Gói chuyển giao có thể bỏ .git, nhưng phải bảo toàn lịch sử repo đang làm việc.
- Cấu trúc chính: README.md, requirements.txt, configs/final.yaml, src/, tests/, notebooks/, results/final/, figures/final/, report/, docs/archive/, data/README.md. Nêu rõ phần nào historical và final.

## 7. Kiểm thử và nghiệm thu

Test đúng hành vi: chỉ 4 suggest params; fixed class_weight None; seed khác không ghi đè study; đủ budget resume không thêm trial; từ chối fingerprint mismatch; preprocessing chỉ fit train; AP/size aggregation; Pareto và tie-break; full-train feasibility; không đọc holdout trong search; figure/report lấy đúng run. Không tự bịa 'tests passed'.

Kiểm tra từ checkout sạch với bộ files được Git track, không dựa vào DB/data/model đang bị ignore mà không hướng dẫn cách lấy. Chạy pytest, compileall, smoke, full, analysis theo thứ tự; ghi kết quả thật và các bước chưa chạy.

## 8. Báo cáo Word và GitHub

Sau khi có full runs thật, tạo report/Bao_cao_NMTUH.docx khoảng 8-12 trang nội dung không tính phụ lục, có hình nhúng thật, bảng số liệu và số trang. Bám 7 phần của đề: Đặt vấn đề; Mô hình hóa; Phương pháp; Dữ liệu và thiết kế thực nghiệm; Kết quả; Thảo luận; Kết luận. Phụ lục gồm config, run commands, link repo, AI assistance và nguồn.

Theo báo cáo gốc: sinh viên Nguyễn Thành Phát; MSSV 2474802016639; lớp 261_71ITAI51403_0102; GV Nguyễn Thị Diễm Anh. Nếu hồ sơ đã được người dùng cập nhật khác, hỏi xác nhận riêng phần tên trước bản nộp, không tự sửa.

Báo cáo phải trả lời: tối ưu 4 siêu tham số nào; x là gì; f(x) là gì; ràng buộc gì; đã chạy bao nhiêu seeds/trials; chỉ số có cải thiện so baseline không; nghiệm cuối là bộ giá trị nào; tại sao chọn. Giữ kết quả v1 riêng nếu đưa vào phụ lục. Không khẳng định nghiệm tối ưu toàn cục, TPE thắng, hoặc cải thiện nếu dữ liệu không chứng minh.

Render DOCX thành PDF/ảnh, xem tất cả trang và sửa bảng/hình bị vỡ. Cập nhật README với bảng và tối thiểu 2 hình chính, cách lấy dữ liệu và lệnh chạy. Thêm một trang tóm tắt bảo vệ 5-7 phút. Link repo trong mọi tài liệu phải là https://github.com/NTP010205/NMTUH_VLU.

Không tự mở rộng sang 4-6 mô hình, deep learning, CVXPY hoặc latency; các Lab là kiến thức tham khảo, không phải yêu cầu tích hợp toàn bộ.

Tạo commit theo những thay đổi thực sự, không giả 5 tuần lịch sử. Chuẩn bị nhánh/commit để đăng repo đích; chỉ thông báo đã push/public khi có kết quả xác nhận từ GitHub. Không force push. Nếu chưa có quyền truy cập, bàn giao patch/ZIP, báo rõ chưa đăng.

Cuối cùng bàn giao: danh sách file sửa; bảng kết quả có nguồn CSV; 4 giá trị selected; kết quả test thực; file Word; trạng thái GitHub; các việc còn thiếu. Tiếp tục đến khi hoàn thành hoặc gặp blocker cụ thể, không kết thúc ngay sau kế hoạch.
