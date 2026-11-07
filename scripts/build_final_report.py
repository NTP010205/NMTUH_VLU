"""Build Vietnamese report content from completed, audited v2 exports only.

Default writes Markdown/content JSON/defense notes. --docx additionally requires
python-docx in an authorized document runtime, followed by separate render QA.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AP = 'average_precision_cv_mean'
SIZE = 'model_size_max_mb'
STD = 'average_precision_cv_std'
REPO = 'https://github.com/NTP010205/NMTUH_VLU'
PARAMS = ['n_estimators', 'max_depth', 'min_samples_leaf', 'max_features']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def table(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def fmt(value, digits=6):
    return f'{float(value):.{digits}f}'


def audited_sources(root):
    analysis = root / 'results/analysis/final'
    run = root / 'results/final'
    config = read_json(run / 'protocol_snapshot.json')
    manifest = read_json(analysis / 'analysis_manifest.json')
    expected = len(config['search_seeds']) * len(config['samplers']) * config['trials_per_sampler']
    if manifest['status'] != 'COMPLETE' or manifest['attempted'] != expected:
        raise ValueError('Report requires the complete registered attempted-trial budget')
    if manifest['protocol'] != config['protocol'] or len(manifest['runs']) != len(config['search_seeds']) * len(config['samplers']):
        raise ValueError('Report source protocol/run count mismatch')
    paths = [analysis / 'analysis_manifest.json', run / 'protocol_snapshot.json']
    for item in manifest['runs']:
        path = run / f"seed_{item['search_seed']}" / item['sampler'] / 'manifest.json'
        if digest(path) != item['manifest_sha256']:
            raise ValueError('Analysis is stale relative to experiment manifests')
        study_manifest = read_json(path)
        for name, sha in study_manifest['files'].items():
            if digest(path.parent / name) != sha:
                raise ValueError('Trial exports changed after audit')
        paths.append(path)
    for name, sha in manifest['tables'].items():
        path = analysis / name
        if digest(path) != sha:
            raise ValueError('Analysis table hash mismatch')
        paths.append(path)
    for name, sha in manifest['figures'].items():
        path = root / 'figures/final' / name
        if digest(path) != sha:
            raise ValueError('Figure hash mismatch')
        paths.append(path)
    baseline = read_json(run / 'baseline.json')
    if digest(run / 'baseline.json') != manifest['baseline_sha256']:
        raise ValueError('Baseline hash mismatch')
    selected = read_json(run / 'selected_candidate.json')
    for name, sha in selected['source_manifests'].items():
        if digest(run / name) != sha:
            raise ValueError('Selection source manifest mismatch')
    if selected['fingerprint'] != baseline['fingerprint'] or not selected['full_train']['feasible']:
        raise ValueError('Selected full-train audit invalid')
    sensitivity_manifest = read_json(root / 'results/sensitivity/manifest.json')
    if sensitivity_manifest['selected_hash'] != selected['candidate']['config_hash']:
        raise ValueError('Sensitivity belongs to a different selection')
    sensitivity_path = root / 'results/sensitivity/summary.csv'
    if digest(sensitivity_path) != sensitivity_manifest['summary_sha256']:
        raise ValueError('Sensitivity summary hash mismatch')
    paths.extend([run / 'baseline.json', run / 'baseline_full_train.json', run / 'selected_candidate.json',
                  run / 'context.json', sensitivity_path, root / 'results/sensitivity/manifest.json'])
    tables = {name: table(analysis / f'{name}.csv') for name in
              ['run_summary', 'sampler_summary', 'paired_differences', 'budget_summary', 'startup_duplicates', 'pareto']}
    trials = [row for p in run.glob('seed_*/*/trials.json') for row in read_json(p)]
    if len(trials) != expected:
        raise ValueError('Unexpected report trial count')
    return {'config': config, 'manifest': manifest, 'baseline': baseline,
            'baseline_full': read_json(run / 'baseline_full_train.json'), 'selected': selected,
            'context': read_json(run / 'context.json'), 'sensitivity': table(sensitivity_path),
            'tables': tables, 'trials': trials,
            'sources': {str(p.relative_to(root)): digest(p) for p in paths}}


def build_pages(data):
    cfg, selected, base = data['config'], data['selected'], data['baseline']
    row = selected['candidate']
    params = selected['parameters']
    runs = data['tables']['run_summary']
    sampler = data['tables']['sampler_summary']
    paired = data['tables']['paired_differences']
    differences = [float(r['tpe_minus_random']) for r in paired]
    good = [r for r in data['trials'] if r['status'] == 'COMPLETE' and r['feasible']]
    best = max(r[AP] for r in good)
    improvement = row[AP] - base[AP]
    cv_reduction = 100 * (1 - row[SIZE] / base[SIZE])
    full_reduction = 100 * (1 - selected['full_train']['full_train_size_mb'] / data['baseline_full']['full_train_size_mb'])
    max_size = max(r[SIZE] for r in good)
    failed = [r for r in data['trials'] if r['status'] == 'FAIL']
    actual_folds = sum(3 if r['status'] == 'COMPLETE' else 0 for r in data['trials'])
    unknown = data['context']['train_unknown_counts']
    pages = []
    def page(title, *blocks):
        pages.append({'title': title, 'blocks': list(blocks)})
    def p(text):
        return {'type': 'paragraph', 'text': text}
    def t(headers, rows, caption):
        return {'type': 'table', 'headers': headers, 'rows': rows, 'caption': caption}
    def f(name, caption):
        return {'type': 'figure', 'path': 'figures/final/' + name, 'caption': caption}
    page('Tối ưu bốn siêu tham số Random Forest cho Bank Marketing',
         p('Đồ án học phần Nhập môn Tối ưu hóa'),
         p('Sinh viên Nguyễn Thành Phát · MSSV 2474802016639 · Lớp 261_71ITAI51403_0102'),
         p('Giảng viên Nguyễn Thị Diễm Anh'),
         p(f'Repository: {REPO}'),
         {'type': 'heading', 'text': '1 Đặt vấn đề'},
         p('Bài toán dự đoán một khách hàng có đăng ký tiền gửi có kỳ hạn hay không là bài toán học có giám sát, với nhãn y nhận giá trị yes hoặc no. Mục đích nghiên cứu là tìm một cấu hình Random Forest có chất lượng xếp hạng tốt, đồng thời giảm dung lượng lưu trữ của toàn bộ quy trình dự đoán.'),
         p('Hai tiêu chí có thể xung đột: tăng số cây hoặc độ sâu có thể nâng chất lượng nhưng tạo tệp mô hình lớn hơn. Vì vậy, đồ án đánh giá nhiều cấu hình theo hai mục tiêu, xác định tập Pareto và áp dụng một quy tắc lựa chọn đã khóa trước thực nghiệm. Dung lượng được đo trực tiếp sau khi huấn luyện, không suy đoán từ số cây.'),
         p(f'Thực nghiệm sử dụng {len(cfg["search_seeds"])} search seeds, hai sampler và {len(data["trials"])} trials attempted. Nghiệm cuối có CV AP {fmt(row[AP])}, so với baseline {fmt(base[AP])}; chênh lệch tuyệt đối là {improvement:+.6f}. Dung lượng max-fold của nghiệm chọn là {fmt(row[SIZE], 3)} MB. Đây là so sánh trên các folds đã dùng cho lựa chọn, cần đọc cùng những giới hạn ở phần thảo luận.'),
         p('Phạm vi giữ một họ mô hình RandomForestClassifier. Bốn biến quyết định là bốn siêu tham số cụ thể của mô hình; chúng không phải bốn bộ tối ưu SGD, Momentum, RMSProp và Adam. Thí nghiệm không đưa learning_rate, batch_size, deep learning hay latency vào bài toán.'))
    page('2 Mô hình hóa',
         p('Vector quyết định x = (n_estimators, max_depth, min_samples_leaf, max_features) thuộc miền hỗn hợp số nguyên và số thực. Việc đánh giá một điểm x cần huấn luyện pipeline trên ba training folds và dự đoán xác suất trên các validation folds tương ứng.'),
         t(['Biến', 'Miền', 'Ý nghĩa'], [
             ['n_estimators', '50 đến 300, bước 25', 'Số cây trong rừng'],
             ['max_depth', 'Số nguyên 4 đến 24', 'Độ sâu tối đa của cây'],
             ['min_samples_leaf', 'Số nguyên 1 đến 12', 'Số mẫu tối thiểu tại lá'],
             ['max_features', 'Số thực 0,4 đến 1,0', 'Tỷ lệ cột sau tiền xử lý xét tại phép tách']], 'Bảng 1 Miền tìm kiếm được khóa trong configs/final.yaml'),
         p('AP_CV(x) = [AP₁(x) + AP₂(x) + AP₃(x)] / 3.'),
         p('S(x) = maxₖ size_bytes(Pipelineₖ(x)) / 1 000 000, với k = 1, 2, 3.'),
         p('min F(x) = (−AP_CV(x), S(x)), với g(x) = S(x) − 50 ≤ 0 và x thuộc miền đã nêu.'),
         p('Average Precision được tính bằng average_precision_score từ xác suất predict_proba trên validation. AP tổng hợp precision theo các mức tăng recall; đây không phải accuracy và không đồng nhất với diện tích PR tính theo quy tắc hình thang [2].'),
         p('S là dung lượng tệp Joblib của toàn bộ fitted preprocessing và RF với compress=3, dùng MB thập phân. Đây không phải dung lượng RAM, kích thước riêng các cây hay thời gian suy luận. Mục tiêu dùng max của ba folds để biểu diễn fold có tệp lớn nhất.'),
         p('Cố định class_weight=None là quyết định thu gọn phạm vi, không chứng minh class_weight không ảnh hưởng. Model random_state, split seed và CV seed đều bằng 42; n_jobs=1. Chỉ seed của sampler thay đổi giữa các search runs.'))
    page('3 Phương pháp',
         p('Random Search lấy mẫu từ miền đã định nghĩa mà không sử dụng giá trị mục tiêu của các trial trước để định hướng đề xuất. TPE dùng lịch sử đánh giá để điều chỉnh phân phối đề xuất. Với Optuna 4.9.0, TPESampler được chỉ định tường minh cho bài toán hai mục tiêu và có n_startup_trials=10 [3]. Các thiết lập này không thay đổi sau khi xem kết quả.'),
         p('Mỗi seed s tạo một RandomSampler(seed=s) và một TPESampler(seed=s, n_startup_trials=10, constraints_func=...). Ràng buộc trả về g=S−50; giá trị không dương là khả thi. Cả hai phương pháp dùng chung data split, CV folds, mô hình, miền và ngân sách attempted trials. Trials lỗi được ghi lý do và không được tự động chạy bù vô hạn.'),
         p('Một cấu hình bị trội nếu tồn tại cấu hình khả thi khác có AP không thấp hơn, size không lớn hơn và ít nhất một mục tiêu tốt hơn nghiêm ngặt. Tập Pareto được tính từ COMPLETE+feasible, loại cấu hình trùng khi đếm. Các cấu hình bằng nhau giữa hai sampler không được coi là hai khám phá độc lập.'),
         p('Quy tắc khóa trước chạy: xác định AP_best trong tập khả thi, giữ AP ≥ AP_best−0,005, rồi chọn size CV nhỏ nhất. Thứ tự hòa tiếp theo là AP cao hơn, fold std nhỏ hơn, config_hash, sampler, trial. Các tham số này không được điều chỉnh theo holdout.'),
         p('Mỗi search run có một nghiệm chọn riêng để so sánh giữa runs. Mô hình cuối được chọn từ tập gộp cả năm seeds và hai sampler, tức dùng ngân sách gộp. Sau đó pipeline được refit trên toàn bộ train; nếu size vượt 50 MB thì thử ứng viên kế tiếp trong cùng tập near-best, không mở rộng tolerance.'),
         p('Bản v1 là thí nghiệm lịch sử năm biến với một sampler seed; runner và logs cũ được giữ ở đường dẫn riêng. Không lấy các con số v1 rồi đổi nhãn thành kết quả bốn biến hoặc nhiều seed.'))
    page('4 Dữ liệu và thiết kế thực nghiệm',
         p('Dữ liệu UCI Bank Marketing được tải từ bản công bố của Moro, Rita và Cortez, giấy phép CC BY 4.0 [1]. File sử dụng là bank-full.csv, có 45.211 hàng và 17 cột bao gồm nhãn y. Không thay thế bằng bank-additional-full.csv. Checksum được so với bản audit đã khóa trước khi đọc các hàng train.'),
         t(['Thành phần', 'Thiết lập'], [
             ['Đặc trưng', 'Bỏ duration, còn 15 đặc trưng gốc trước one-hot'],
             ['Train và holdout', '36.168 train và 9.043 holdout, stratified 80/20'],
             ['CV', '3 folds trên train; mỗi fold fit 24.112 và validation 12.056'],
             ['Nhãn train', f"{data['context']['train_positive']} yes trên {data['context']['train_rows']} hàng"],
             ['Search seeds', ', '.join(map(str, cfg['search_seeds']))],
             ['Ngân sách', f"{cfg['trials_per_sampler']} trials attempted mỗi sampler và seed"]], 'Bảng 2 Thiết kế thực nghiệm v2'),
         p('duration chỉ được biết sau cuộc gọi nên bị loại khỏi đầu vào. Các cột campaign, contact, day và month vẫn được giữ theo protocol; tính sẵn có của chúng tại thời điểm triển khai cần được xác nhận riêng. Imputer và OneHotEncoder nằm trong pipeline và chỉ fit trên training fold. Validation dùng handle_unknown=ignore.'),
         p('Giữ chuỗi unknown như một giá trị phân loại. Số lượng trên phần train: ' + '; '.join(f'{k}={v}' for k, v in unknown.items() if v) + '. Không chuyển ngầm các giá trị này thành null.'),
         p('Baseline là RF mặc định: n_estimators=100, max_depth=None, min_samples_leaf=1, max_features=sqrt, class_weight=None, random_state=42, n_jobs=1. Baseline có thể nằm ngoài miền tìm kiếm. Chỉ có một kết quả baseline CV; không sao chép năm lần để ước lượng variance giữa runs.'),
         p('Môi trường huấn luyện dùng Python 3.11.15, scikit-learn 1.5.2, Optuna 4.9.0 và các pins trong requirements.txt. Máy ghi nhận Intel Core i7-14650HX, 16 cores/24 logical processors; thực nghiệm giới hạn một thread. Thời gian dưới đây là số đo trong các phiên chạy, không phải bốn ngày lịch giữa lần bắt đầu và resume.'))
    page('5 Kết quả thực nghiệm',
         p(f'Đã thử {len(data["trials"])} trials: {data["manifest"]["complete"]} COMPLETE và {len(failed)} FAIL. Các trial COMPLETE tương ứng {actual_folds} fold fits đủ bộ; partial folds của trial lỗi được giữ trong folds.csv và không đưa vào tính Pareto. ' + ('Lỗi được ghi trong trials.json, gồm gián đoạn tiến trình khi resume; vẫn tính vào ngân sách attempted.' if failed else 'Không có trial thất bại.')),
         t(['Seed', 'Sampler', 'Best AP', 'AP chọn', 'Size chọn MB', 'OK / lỗi', 'Pareto'], [
             [r['search_seed'], r['sampler'], fmt(r['best_ap']), fmt(r['selected_ap']), fmt(r['selected_size_mb'], 3),
              f"{r['complete']} / {r['failed']}", r['pareto_distinct']] for r in runs],
           'Bảng 3 Một dòng mỗi search run từ results/analysis/final/run_summary.csv'),
         p(f'Baseline CV AP = {fmt(base[AP])} ± {fmt(base[STD])}; size max-fold = {fmt(base[SIZE], 3)} MB. Độ lệch chuẩn ở câu này là sample std giữa ba folds (ddof=1), không phải độ biến động của một thuật toán tìm kiếm qua năm seeds.'),
         p(f'Nghiệm cuối có chênh lệch AP so baseline là {improvement:+.6f}, và thay đổi dung lượng CV là giảm {cv_reduction:.2f}%. So sánh này dùng cùng folds và cùng kiểu đo CV. Kết quả đã qua chọn lọc trên CV nên không phải ước lượng độc lập về mức cải thiện ngoài mẫu.'),
         p('Nguồn kiểm chứng gồm CSV/JSON theo từng study, manifest hash và folds.csv. AP mean/std và max-fold bytes được tính lại từ ba hàng fold; kết quả tổng hợp không nhập tay.'))
    budget_rows = []
    for method in cfg['samplers']:
        for budget in cfg['comparison_budgets']:
            values = [float(r['best_ap']) for r in data['tables']['budget_summary'] if r['sampler'] == method and int(r['budget']) == budget]
            budget_rows.append([method, str(budget), fmt(statistics.mean(values)), fmt(statistics.stdev(values))])
    page('5 Kết quả theo ngân sách tìm kiếm',
         f('convergence.png', 'Hình 1 Best-so-far AP trung bình và sample std giữa các search seeds. Dải biểu diễn độ phân tán giữa runs, không phải khoảng tin cậy.'),
         t(['Sampler', 'Budget', 'AP mean', 'AP std giữa runs'], budget_rows,
           'Bảng 4 Best-so-far tại 10, 20 và 40 attempted trials, tính từ budget_summary.csv'),
         p('Mỗi đường theo seed lấy AP khả thi cao nhất đã quan sát đến ngân sách tương ứng. Trial lỗi vẫn chiếm một đơn vị ngân sách; đường best-so-far không tăng từ trial đó. Việc dùng cùng ngân sách giúp đọc trực tiếp hiệu quả tìm kiếm trong phạm vi thực nghiệm.'),
         p('Số cấu hình startup giống nhau theo từng seed: ' + '; '.join(f"seed {r['search_seed']}: {r['identical_startup_configs']}/{r['compared']}" for r in data['tables']['startup_duplicates']) + '. Khi startup trùng nhau, hai phương pháp dùng cùng các điểm khởi đầu; không diễn giải chúng là các khám phá độc lập.'))
    page('5 Pareto và nghiệm cuối',
         f('pareto.png', 'Hình 2 AP và dung lượng max-fold của các trial khả thi, Pareto gộp và nghiệm cuối theo quy tắc đã khóa.'),
         t(['Siêu tham số', 'Giá trị được chọn'], [[pname, str(params[pname])] for pname in PARAMS], 'Bảng 5 Bốn giá trị của nghiệm cuối'),
         p(f"Nguồn nghiệm: sampler {row['sampler']}, search seed {row['search_seed']}, trial {row['trial']} (đánh số từ 0). Config hash: {row['config_hash']}. CV AP={fmt(row[AP])} ± {fmt(row[STD])}; S={fmt(row[SIZE])} MB."),
         p(f'AP tốt nhất trong tập gộp là {fmt(best)}; nghiệm chọn mất {best-row[AP]:.6f} điểm AP tuyệt đối, trong tolerance 0,005. Tập Pareto có {len(data["tables"]["pareto"])} cấu hình phân biệt. Lý do chọn là dung lượng nhỏ nhất trong vùng chất lượng near-best, không phải AP cao nhất hoặc một nghiệm tối ưu toàn cục.'),
         p(f'Refit full train của nghiệm chọn đạt {fmt(selected["full_train"]["full_train_size_mb"], 3)} MB, baseline full train {fmt(data["baseline_full"]["full_train_size_mb"], 3)} MB, giảm {full_reduction:.2f}%. Đây là so sánh full-train với full-train. Mức lớn nhất trong các trial CV khả thi là {max_size:.3f} MB; ' + ('ngưỡng 50 MB không binding trong các trial hoàn thành.' if all(r['feasible'] for r in data['trials'] if r['status'] == 'COMPLETE') else 'có trial hoàn thành vi phạm ngưỡng 50 MB và đã bị loại khỏi chọn nghiệm.')))
    page('5 Độ ổn định giữa các seeds',
         f('seed_distribution.png', 'Hình 3 Best AP của Random và TPE ghép cặp theo cùng seed. Mỗi đoạn nối là một cặp search runs.'),
         t(['Sampler', 'Best AP mean ± std', 'AP chọn mean ± std', 'Size chọn mean ± std'], [
             [r['sampler'], f"{fmt(r['best_ap_mean'])} ± {fmt(r['best_ap_std'])}",
              f"{fmt(r['selected_ap_mean'])} ± {fmt(r['selected_ap_std'])}",
              f"{fmt(r['selected_size_mean'],3)} ± {fmt(r['selected_size_std'],3)}"] for r in sampler],
           'Bảng 6 Mean và sample std giữa năm search runs, không phải giữa folds'),
         p('Chênh lệch best AP theo seed (TPE−Random): ' + '; '.join(f"{r['search_seed']}: {float(r['tpe_minus_random']):+.6f}" for r in paired) + '.'),
         p(f'Trung bình chênh lệch ghép cặp là {statistics.mean(differences):+.6f}, sample std {statistics.stdev(differences):.6f}. Chỉ năm seeds tạo bằng chứng mô tả hạn chế; không suy ra một sampler luôn tốt hơn trên mọi dữ liệu hoặc ngân sách.'),
         p('Các runs chia sẻ dữ liệu, folds và model seed. Vì vậy std giữa runs đo biến động của quá trình search trên cùng bài toán cố định, không đo toàn bộ bất định do lấy mẫu dữ liệu, chia folds hoặc triển khai thực tế. Các cấu hình giống nhau giữa sampler càng làm giảm cơ sở để coi mọi điểm là quan sát độc lập.'))
    sensitivity_notes = []
    for name in PARAMS:
        values = [r for r in data['sensitivity'] if r['variable'] == name]
        sensitivity_notes.append(f"{name}: AP từ {min(float(r[AP]) for r in values):.6f} đến {max(float(r[AP]) for r in values):.6f}; size từ {min(float(r[SIZE]) for r in values):.3f} đến {max(float(r[SIZE]) for r in values):.3f} MB")
    page('5 Phân tích độ nhạy cục bộ',
         f('sensitivity.png', 'Hình 4 Thay từng biến quanh nghiệm chọn. Thanh sai số là sample std giữa ba folds; đường dọc đánh dấu giá trị selected.'),
         p('Sensitivity được thực hiện sau khi khóa selected_candidate.json. Mỗi lần chỉ thay một biến, giữ ba biến còn lại, cùng train và folds. Mỗi biến có tối đa ba giá trị hợp lệ và không trùng; nếu nghiệm ở biên miền thì số lân cận có thể ít hơn. Kết quả được lưu riêng trong results/sensitivity.'),
         *[p(note + '.') for note in sensitivity_notes],
         p('Những thay đổi quan sát được mô tả độ nhạy quanh một cấu hình cụ thể. Không coi chúng là feature importance, mức quan trọng toàn cục của siêu tham số hoặc quan hệ nhân quả. Nghiệm trung tâm xuất hiện trong bốn lát cắt nhưng được tái sử dụng từ cùng một đánh giá CV; không đếm thành bốn quan sát độc lập.'),
         p('Kể cả nếu một lân cận có chỉ số thuận lợi, nghiên cứu này vẫn giữ lựa chọn chính đã khóa. Muốn đổi nghiệm theo sensitivity cần một protocol mới, ngân sách mới và chiến lược đánh giá tách khỏi kết quả đã dùng để lựa chọn.'))
    page('6 Thảo luận',
         p(f'Nghiệm cuối có AP {fmt(row[AP])}, so với {fmt(base[AP])} của baseline, và có dung lượng sau refit {fmt(selected["full_train"]["full_train_size_mb"],3)} MB. Hai mục tiêu làm rõ lý do chấp nhận một mức mất AP nhỏ so với best để ưu tiên dung lượng. Tolerance 0,005 là lựa chọn thiết kế; thực nghiệm không chứng minh đây là tolerance tốt nhất cho mọi ứng dụng.'),
         p('Holdout đã từng được đánh giá ở giai đoạn baseline v1. V2 không dùng holdout trong search, chọn sampler, chọn seed, chọn tolerance hoặc sensitivity; báo cáo này không bổ sung đánh giá holdout cuối. Không mô tả holdout là hoàn toàn chưa từng xem và không hứa có external test. CV của nghiệm được chọn có thể lạc quan do được dùng lặp lại trong tìm kiếm.'),
         p('Việc cố định class_weight và giữ một họ mô hình giúp so sánh trong phạm vi rõ ràng, nhưng cũng giới hạn kết luận. Nghiên cứu không chứng minh Random Forest tốt hơn các họ mô hình khác. Ràng buộc 50 MB được giữ nguyên; nếu không binding thì chỉ chứng minh các cấu hình đã thử không chạm cap, không chứng minh mọi cấu hình khả dĩ đều như vậy.'),
         p('Runtime phụ thuộc cấu hình, phần cứng và tải hệ thống. Thí nghiệm được tiếp tục từ checkpoint ở nhiều phiên; run wall time là tổng thời gian các phiên thực thi đã ghi, còn tổng trial time chỉ cộng thời gian trial đã có số đo. Không cộng khoảng nghỉ giữa các phiên và không coi hai loại thời gian này là một. Trial bị ngắt có thể thiếu thời gian đầy đủ, nên so sánh tốc độ chỉ mang tính mô tả.'),
         {'type': 'heading', 'text': '7 Kết luận'},
         p(f'Đồ án đã tối ưu đúng bốn siêu tham số trong protocol {cfg["protocol"]}, sử dụng {len(cfg["search_seeds"])} sampler seeds và {len(data["trials"])} attempted trials. Bộ giá trị cuối là ' + ', '.join(f'{k}={v}' for k,v in params.items()) + '. Nghiệm được chọn từ tập gộp theo AP gần tốt nhất và dung lượng nhỏ nhất, sau đó kiểm tra điều kiện full-train size.'),
         p('Các logs cho phép tái tạo bảng và hình từ CSV/JSON, không cần SQLite hoặc model đã bị Git-ignore. Hướng phát triển phù hợp là kiểm chứng trên dữ liệu mới hoặc một thiết kế đánh giá độc lập; không suy diễn kết quả hiện tại thành bảo đảm tối ưu toàn cục hay ưu thế thống kê chắc chắn của TPE.'))
    page('Phụ lục Lệnh chạy và dấu vết tái lập',
         p('Protocol đầy đủ: configs/final.yaml. Frozen data SHA-256: ' + data['context']['hashes']['dataset_sha256'] + '.'),
         p('Bắt đầu từ Python 3.11 và cài requirements.txt. Tải bank-full.csv bằng scripts/download_data.py, sau đó chạy pytest, compileall, smoke, full và analysis theo thứ tự. Không dùng lại venv được sao chép từ máy khác.'),
         {'type': 'code', 'text': '\n'.join([
             'python -m pip install -r requirements.txt', 'python scripts/download_data.py',
             'python -m pytest -q', 'python -m compileall -q src tests scripts',
             'python -m src.run_experiments --mode smoke', 'python -m src.run_experiments --mode full',
             'python -m src.analyze_results', 'python -m src.finalize_experiments',
             'python scripts/build_final_report.py --docx'])},
         p('Resume giữ nguyên protocol, fingerprint và ngân sách attempted. Nếu clone chỉ có logs mà không có DB/checkpoint, phân tích trực tiếp bằng src.analyze_results; muốn huấn luyện lại, dùng --output results/reproduction. Không xóa DB để biến kiểm tra bị thiếu thành PASS.'),
         t(['Seed', 'Sampler', 'Tổng trial giây', 'Wall time giây'], [
             [r['search_seed'], r['sampler'], fmt(r['total_trial_seconds'],1), fmt(r['wall_seconds'],1)] for r in runs],
           'Bảng 7 Thời gian theo run từ run_summary.csv'),
         p('Các thời gian là số đo thực; lỗi và gián đoạn vẫn được giữ trong logs. Kiểm tra SHA-256 lịch sử nằm ở docs/review/v1_integrity.json. Không tạo commits lùi ngày hoặc giả lịch sử năm tuần.'))
    page('Phụ lục Nguồn và hỗ trợ AI',
         p('[1] Moro, S., Rita, P. và Cortez, P. Bank Marketing. UCI Machine Learning Repository. DOI 10.24432/C5K306. https://archive.ics.uci.edu/dataset/222/bank+marketing. Dữ liệu được phân phối theo CC BY 4.0.'),
         p('[2] scikit-learn 1.5.2. average_precision_score. https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.average_precision_score.html'),
         p('[3] Optuna 4.9.0. TPESampler. https://optuna.readthedocs.io/en/v4.9.0/reference/samplers/generated/optuna.samplers.TPESampler.html'),
         p('[4] scikit-learn 1.5.2. RandomForestClassifier. https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html'),
         p(f'[5] Mã, cấu hình và dữ liệu kết quả của đồ án: {REPO}. report/report_sources.json ghi hash các nguồn trực tiếp dùng để sinh tài liệu này.'),
         {'type': 'heading', 'text': 'Hỗ trợ AI'},
         p('Codex hỗ trợ rà soát mã, triển khai protocol v2, chạy lệnh kiểm thử và thực nghiệm, tổng hợp bảng hình, soạn thảo và kiểm tra tài liệu. Các kết quả số trong báo cáo lấy từ logs thực thi; dữ liệu synthetic chỉ dùng cho unit tests và không đưa vào kết quả thực nghiệm.'),
         p('Tác giả chịu trách nhiệm hiểu mô hình hóa, quy tắc chọn nghiệm, giới hạn nghiên cứu và nội dung nộp. Bằng chứng thực nghiệm, các lần lỗi và trạng thái kiểm chứng được lưu để có thể đối chiếu. Sự hỗ trợ AI không thay thế việc đọc và xác nhận tài liệu cuối của tác giả.'),
         {'type': 'heading', 'text': 'Phân biệt với bằng chứng lịch sử'},
         p('V1 là thí nghiệm năm biến, một sampler seed, 40 Random và 40 TPE trials. Các file lịch sử giữ riêng tại results/hpo/ap_model_size_v1 và results/analysis/week4. Các bảng kết quả chính của báo cáo này chỉ lấy từ protocol bốn biến v2; không đổi nhãn kết quả v1.'))
    return pages


def write_markdown(pages, path):
    lines = []
    for page in pages:
        lines += ['# ' + page['title'], '']
        for block in page['blocks']:
            kind = block['type']
            if kind == 'paragraph':
                lines += [block['text'], '']
            elif kind == 'heading':
                lines += ['## ' + block['text'], '']
            elif kind == 'code':
                lines += ['```text', block['text'], '```', '']
            elif kind == 'figure':
                lines += [f"![{block['caption']}](../{block['path']})", '']
            else:
                lines += [block['caption'], '', '| ' + ' | '.join(block['headers']) + ' |',
                          '| ' + ' | '.join('---' for _ in block['headers']) + ' |']
                lines += ['| ' + ' | '.join(map(str, row)) + ' |' for row in block['rows']]
                lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


def write_docx(pages, root, path):
    from docx import Document
    from docx.shared import Cm, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin, section.bottom_margin = Cm(1.8), Cm(1.8)
    section.left_margin, section.right_margin = Cm(2), Cm(2)
    for name in ['Normal', 'Title', 'Heading 1', 'Heading 2', 'Caption']:
        style = document.styles[name]
        style.font.name = 'Calibri'
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.size = Pt({'Normal': 11, 'Title': 21, 'Heading 1': 16, 'Heading 2': 13, 'Caption': 9}[name])
        style.paragraph_format.space_after = Pt(7)
        style.paragraph_format.line_spacing = 1.08
    document.core_properties.author = 'Nguyễn Thành Phát'
    document.core_properties.title = pages[0]['title']
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run('Trang ')
    field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'), 'PAGE'); footer._p.append(field)
    for index, page in enumerate(pages):
        if index:
            document.add_page_break()
        document.add_paragraph(page['title'], 'Title' if index == 0 else 'Heading 1')
        for block in page['blocks']:
            kind = block['type']
            if kind in ['paragraph', 'heading', 'code']:
                p = document.add_paragraph(block['text'], 'Heading 2' if kind == 'heading' else 'Normal')
                if kind == 'code':
                    for run in p.runs:
                        run.font.name, run.font.size = 'Consolas', Pt(9)
            elif kind == 'figure':
                p = document.add_paragraph()
                p.paragraph_format.keep_with_next = True
                p.add_run().add_picture(str(root / block['path']), width=Cm(16.6))
                document.add_paragraph(block['caption'], 'Caption')
            else:
                caption = document.add_paragraph(block['caption'], 'Caption')
                caption.paragraph_format.keep_with_next = True
                tab = document.add_table(rows=1, cols=len(block['headers']))
                tab.alignment = WD_TABLE_ALIGNMENT.CENTER
                tab.autofit = False
                widths = ([2.6, 3.8, 10.2] if len(block['headers']) == 3 else
                          [5.0, 11.6] if len(block['headers']) == 2 else
                          [1.0, 1.6, 2.5, 2.5, 3.0, 2.2, 1.8] if len(block['headers']) == 7 else
                          [16.6 / len(block['headers'])] * len(block['headers']))
                for cell, heading in zip(tab.rows[0].cells, block['headers']):
                    cell.text = heading
                for row in block['rows']:
                    for cell, value in zip(tab.add_row().cells, row):
                        cell.text = str(value)
                borders = OxmlElement('w:tblBorders')
                for edge in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
                    border = OxmlElement('w:' + edge)
                    for k, v in {'val': 'single', 'sz': '4', 'color': 'D9D9D9'}.items():
                        border.set(qn('w:' + k), v)
                    borders.append(border)
                tab._tbl.tblPr.append(borders)
                repeat = OxmlElement('w:tblHeader'); tab.rows[0]._tr.get_or_add_trPr().append(repeat)
                for ri, table_row in enumerate(tab.rows):
                    for ci, cell in enumerate(table_row.cells):
                        cell.width = Cm(widths[ci]); cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                        tcpr = cell._tc.get_or_add_tcPr()
                        shade = OxmlElement('w:shd'); shade.set(qn('w:fill'), 'E4EBF2' if ri == 0 else 'FFFFFF'); tcpr.append(shade)
                        margins = OxmlElement('w:tcMar')
                        for edge in ['top', 'bottom', 'left', 'right']:
                            val = OxmlElement('w:' + edge); val.set(qn('w:w'), '85'); val.set(qn('w:type'), 'dxa'); margins.append(val)
                        tcpr.append(margins)
                        for p in cell.paragraphs:
                            p.paragraph_format.space_after = Pt(2)
                            for run in p.runs:
                                run.font.size = Pt(9 if len(widths) > 3 else 10)
                                run.bold = ri == 0
                document.add_paragraph().paragraph_format.space_after = Pt(1)
    document.save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docx', action='store_true')
    args = parser.parse_args(argv)
    data = audited_sources(ROOT)
    pages = build_pages(data)
    directory = ROOT / 'report'
    directory.mkdir(exist_ok=True)
    write_markdown(pages, directory / 'Bao_cao_NMTUH.md')
    (directory / 'report_content.json').write_text(json.dumps(pages, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (directory / 'report_sources.json').write_text(json.dumps(data['sources'], indent=2) + '\n', encoding='utf-8')
    s, b = data['selected'], data['baseline']
    defense = '\n\n'.join([
        '# Tóm tắt bảo vệ trong 5 đến 7 phút',
        f'Nguyễn Thành Phát · MSSV 2474802016639 · {REPO}',
        'Phút 0–1: Dự đoán y=yes/no trên bank-full.csv; bỏ duration để tránh thông tin sau cuộc gọi. Tối ưu đồng thời chất lượng xếp hạng AP và dung lượng toàn bộ pipeline.',
        'Phút 1–2: x gồm n_estimators, max_depth, min_samples_leaf, max_features. min F(x)=(−mean AP qua 3 folds, max-fold size). Ràng buộc 50 MB, class_weight=None, split/CV/model seed 42.',
        f'Phút 2–3: 5 search seeds × 2 sampler × 40 attempted trials; {data["manifest"]["complete"]} COMPLETE và {data["manifest"]["failed"]} FAIL. Random và TPE dùng cùng domain/folds/budget; 10 startup trials được kiểm tra trùng cấu hình. Std giữa runs khác std giữa folds.',
        f'Phút 3–4: Baseline AP {fmt(b[AP])}, nghiệm chọn AP {fmt(s["candidate"][AP])}; CV size {fmt(s["candidate"][SIZE],3)} MB. Chỉ so CV với CV, full-train với full-train. Dẫn Hình 1 và Hình 2.',
        'Phút 4–5: Bộ giá trị cuối: ' + ', '.join(f'{k}={v}' for k,v in s['parameters'].items()) + f'. Giữ AP cách best không quá 0,005, chọn size nhỏ nhất. Refit full train {fmt(s["full_train"]["full_train_size_mb"],3)} MB, đạt cap.',
        'Phút 5–6: Sensitivity thay một biến quanh nghiệm đã khóa; không chọn lại theo sensitivity. Năm seeds chỉ là bằng chứng mô tả. CV dùng cho tìm kiếm có thiên lệch lựa chọn; holdout từng được baseline v1 đánh giá và không dùng trong tuning v2.',
        'Phút 6–7: Minh họa CSV/JSON, manifest hash và lệnh python -m src.analyze_results. Phân tích clone không cần DB; SQLite thiếu ghi not checked. Không khẳng định tối ưu toàn cục hoặc TPE luôn thắng.',
    ])
    (directory / 'Tom_tat_bao_ve.md').write_text(defense + '\n', encoding='utf-8')
    summary_lines = [
        f"Đã hoàn thành ngân sách {data['manifest']['attempted']} attempted trials: {data['manifest']['complete']} COMPLETE, {data['manifest']['failed']} FAIL. Các lỗi vẫn được tính vào budget và giữ trong logs.",
        '', '| Sampler | Best AP mean ± std giữa 5 runs | AP chọn mean ± std |', '|---|---:|---:|',
    ]
    for item in data['tables']['sampler_summary']:
        summary_lines.append(f"| {item['sampler']} | {fmt(item['best_ap_mean'])} ± {fmt(item['best_ap_std'])} | {fmt(item['selected_ap_mean'])} ± {fmt(item['selected_ap_std'])} |")
    summary_lines += ['', f"Baseline CV AP: {fmt(b[AP])} ± {fmt(b[STD])} (std giữa folds). Nghiệm cuối từ ngân sách gộp: " + ', '.join(f'`{k}={v}`' for k,v in s['parameters'].items()) + '.',
                      '', f"Selected CV AP {fmt(s['candidate'][AP])}, max-fold size {fmt(s['candidate'][SIZE],3)} MB; full-train size {fmt(s['full_train']['full_train_size_mb'],3)} MB.",
                      '', 'Nguồn: [run_summary.csv](results/analysis/final/run_summary.csv), [sampler_summary.csv](results/analysis/final/sampler_summary.csv), [selected_candidate.json](results/final/selected_candidate.json).',
                      '', '![Best-so-far AP giữa seeds](figures/final/convergence.png)', '', '![Pareto và nghiệm chọn](figures/final/pareto.png)',
                      '', 'Báo cáo nội dung: [Bao_cao_NMTUH.md](report/Bao_cao_NMTUH.md). Tóm tắt bảo vệ: [Tom_tat_bao_ve.md](report/Tom_tat_bao_ve.md). File Word chỉ được nghiệm thu sau khi render và kiểm tra từng trang.']
    readme = ROOT / 'README.md'
    text = readme.read_text(encoding='utf-8')
    start, end = '<!-- FINAL_RESULTS_START -->', '<!-- FINAL_RESULTS_END -->'
    if start not in text or end not in text:
        raise ValueError('README result markers missing')
    readme.write_text(text.split(start)[0] + start + '\n' + '\n'.join(summary_lines) + '\n' + end + text.split(end)[1], encoding='utf-8')
    if args.docx:
        write_docx(pages, ROOT, directory / 'Bao_cao_NMTUH.docx')
    print(f'Report sources verified; {len(pages)-2} content pages planned plus 2 appendices. Render QA still required for DOCX.')


if __name__ == '__main__':
    main()
