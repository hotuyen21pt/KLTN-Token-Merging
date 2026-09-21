# Cần bổ sung để nộp tạp chí Q2

**Phạm vi**: toàn repo `KLTN-Token-Merging` · nhánh `tuyen` · rà soát ngày 2026-09-21
**Mục tiêu**: đưa công trình từ mức khoá luận lên mức bài báo Q2 (ESWA / KBS / Applied Intelligence …)
**Căn cứ**: mọi con số trong tài liệu này đọc trực tiếp từ file trong repo, không lấy từ README

---

## Đang tốt — giữ nguyên

Nền tảng kỹ thuật không phải làm lại. Cụ thể:

- **Lưới ablation thiết kế đúng**: 21 biến thể phủ LCF × ToMe × {resize, compact} × {pre, post}-encoder. Đây là thứ khó làm và đã xong.
- **Tách Oracle vs End-to-end** để định vị nút thắt — phân tích sắc, cho thấy giới hạn nằm ở ATE chứ không ở classifier. Giữ nguyên cách trình bày này trong paper.
- **Hạ tầng multi-seed đã dựng sẵn** (`common/run_multiseed.py`): eval inline sau mỗi lượt train, ghi `results_raw.csv` ngay, resume theo `done_keys`. Chỉ còn thiếu việc *chạy* nó.
- **Trung thực học thuật**: README tự sửa tên `attention_weighted`/AWM → `sequential_cosine` khi phát hiện cài đặt không dùng attention score. Reviewer đánh giá cao điều này — giữ ghi chú đó trong paper.
- **Đóng góp thật sự có**: SCM bảo vệ token aspect và CLS/SEP khi gộp (`token_merging/tome_1d.py`) — đây là điểm khác biệt so với ToMe gốc, và là thứ đáng viết.

Tài liệu này liệt kê phần **bằng chứng thực nghiệm** còn thiếu, không phải phần code.

---

## Bảng tổng hợp — theo thứ tự nên làm

| # | Việc | Vùng | Ưu tiên | Trạng thái |
|---|---|---|---|---|
| 1 | Chạy multi-seed, báo mean±std per-class | `run_multiseed.py` | **P1** | Còn treo |
| 2 | Bỏ Macro-F1 làm metric chính, đổi sang weighted-F1 | Giao thức đánh giá | **P1** | Còn treo |
| 3 | k-fold CV thay tập test cố định 312 mẫu | Giao thức đánh giá | **P1** | Còn treo |
| 4 | So với các pooling khác (mean/max/attention/CLS) | `models/` | **P1** | Còn treo |
| 5 | Chốt động cơ: pooling hay efficiency — rồi sửa README cho khớp | Tài liệu | **P1** | Còn treo |
| 6 | Sửa 7 chỗ mô tả dữ liệu là tiếng Việt | Tài liệu | **P2** | Còn treo |
| 7 | Khai báo nguồn gốc + quy trình tạo dữ liệu | Tài liệu | **P2** | Còn treo |
| 8 | Thêm benchmark chuẩn (SemEval-2014 Rest/Laptop) | `dataset/` | **P2** | Còn treo |
| 9 | Chạy lại GAS trên chính dữ liệu này | `gas/` | **P2** | Còn treo |
| 10 | Đo FLOPs / params / throughput | `run_all.py` | **P2** | Còn treo |
| 11 | Chạy nhóm `pretome` (gộp trước encoder) | `run_all.py` | **P3** | Còn treo |
| 12 | Đồng bộ tên cấu hình trong các CSV đã commit | `runs_ate/` | **P3** | Còn treo |
| 13 | Sinh lại prediction ATE cho khớp checkpoint | `runs_ate/` | **P3** | Đã sửa cơ chế |

---

## Phần A — Bằng chứng thực nghiệm (P1)

Đây là nhóm quyết định bài có được nhận hay không. Bốn mục đầu phải xong trước khi viết paper.

### A1 · Multi-seed và khoảng tin cậy

**Vị trí**: `common/run_multiseed.py` → `DEFAULT_SEEDS`; thư mục `runs_multiseed/` chưa tồn tại

**Vì sao**: Toàn bộ kết quả đã commit chạy **một seed duy nhất** (`DEFAULT_SEEDS = [42]`). Bóc tách mức tăng Macro-F1 +8.36 của `lcf_attn_cdm_resize` so với `baseline_balanced` (đọc từ `runs_ate/eval_joint_triplet_BERT.csv`):

| lớp | n_test | baseline | SCM+CDM | Δ | đóng góp vào macro |
|---|---:|---:|---:|---:|---:|
| **EXPERIENCE_negative** | **1** | 0.00 | 100.00 | **+100.00** | **+7.69** |
| AMENITY_neutral | 3 | 66.67 | 85.71 | +19.04 | +1.46 |
| SERVICE_negative | 3 | 33.33 | 40.00 | +6.67 | +0.51 |
| EXPERIENCE_positive | 32 | 61.82 | 66.67 | +4.85 | +0.37 |
| AMENITY_positive | 63 | 76.34 | 75.38 | −0.96 | −0.07 |
| FACILITY_positive | 85 | 68.93 | 66.67 | −2.26 | −0.17 |
| BRANDING_positive | 10 | 63.64 | 57.14 | −6.50 | −0.50 |
| **FACILITY_negative** | 10 | 66.67 | 54.55 | **−12.12** | −0.93 |
| 5 lớp còn lại | — | — | — | 0.00 | 0.00 |
| | | | | **+8.36** | |

**92% mức tăng đến từ một lớp có đúng 1 mẫu test, đi từ sai thành đúng.** Trong khi đó lớp nhiều mẫu hơn đi ngược: `FACILITY_negative` (10 mẫu) giảm 12 điểm. Micro-F1 cũng giảm: 69.89 so với 70.53.

Reviewer mở bảng per-class ra là thấy ngay. Không có khoảng tin cậy thì kết quả này không đứng được.

**Cần làm**:

```bash
python run_all.py --stages multiseed --seeds 42 123 456 789 2024 --backbones bert t5
```

Báo cáo mean±std cho **từng lớp**, không chỉ cho macro tổng. Nếu `EXPERIENCE_negative` nhảy 0/100 giữa các seed — kết quả hiện tại là nhiễu, và phải đổi hướng kể chuyện. Đây là thí nghiệm trả lời câu "có paper hay không", làm trước mọi thứ khác.

---

### A2 · Macro-F1 không dùng làm metric chính được

**Vị trí**: `dataset/test.apc` — phân bố lớp; mọi bảng trong `runs_ate/`

**Vì sao**: Tập test 312 mẫu chia 13 lớp `(category, sentiment)`, trong đó **6/13 lớp có dưới 8 mẫu**:

```
BRANDING_neutral      1 mẫu      EXPERIENCE_negative   1 mẫu
AMENITY_neutral       3 mẫu      FACILITY_neutral      3 mẫu
SERVICE_negative      3 mẫu      LOYALTY_positive      7 mẫu
```

Macro-F1 cho mỗi lớp trọng số bằng nhau, nên một lớp 1 mẫu có ảnh hưởng ngang một lớp 85 mẫu. Trên cấu hình này Macro-F1 đo nhiễu lấy mẫu nhiều hơn đo mô hình — chính là hiện tượng ở A1.

**Cần làm**: đổi metric chính sang **Micro-F1 + weighted-F1**. Macro-F1 giữ lại nhưng luôn kèm khoảng tin cậy và bảng per-class đầy đủ. Cân nhắc gộp các lớp cực hiếm (ví dụ gộp toàn bộ `*_neutral`) rồi mới tính macro, và nói rõ lý do gộp trong paper.

---

### A3 · k-fold CV thay tập test cố định

**Vị trí**: `common/dataset_utils.py` → `ApcFileDataset`; chia split hiện tại là 3 file `.apc` cố định

**Vì sao**: 2448/304/312 là chia cố định. Với các lớp 1–3 mẫu, việc một mẫu rơi vào test hay train quyết định luôn Macro-F1 — kết quả phụ thuộc vào cách chia hơn là vào phương pháp.

**Cần làm**: 5-fold CV trên toàn bộ 3064 mẫu. Mỗi mẫu được vào test đúng một lần, phương sai giảm mạnh, và không còn chuyện một mẫu quyết định. Chi phí tính toán thấp hơn nhiều so với lưới 126 lượt train hiện đang dự tính — có thể đánh đổi: bỏ bớt biến thể, thêm fold.

---

### A4 · Thiếu baseline pooling để so sánh

**Vị trí**: `models/fast_lcf_bert_multitask.py` → `forward`

**Vì sao**: Nếu luận điểm là "SCM là cơ chế pooling tốt hơn", thì baseline đúng phải là **các cơ chế pooling khác**, không phải chỉ "không dùng ToMe". Hiện `baseline_balanced` là không pooling — so sánh đó không trả lời được câu "tốt hơn cái gì".

Reviewer sẽ hỏi: mean pooling thì sao? attention pooling thì sao? Nếu mean pooling cho kết quả tương đương thì đóng góp của SCM bằng không.

**Cần làm**: thêm 4 baseline pooling, dùng chung mọi hyperparameter:

| baseline | mô tả |
|---|---|
| `pool_cls` | chỉ dùng vector `[CLS]` (mặc định của BERT) |
| `pool_mean` | mean pooling có mask |
| `pool_max` | max pooling có mask |
| `pool_attn` | attention pooling học được (1 lớp) |

Đây là bổ sung làm paper mạnh lên nhiều nhất trên mỗi giờ GPU bỏ ra.

---

### A5 · Chốt động cơ của ToMe rồi sửa tài liệu cho khớp

**Vị trí**: `README.md` mục "ToMe — Token Merging"; `models/fast_lcf_bert_multitask.py` → `forward`

**Vì sao**: README hiện viết *"Gộp token dư thừa để **giảm độ phức tạp tính toán** (CVPR 2023)"* — đúng động cơ của paper gốc, nhưng số liệu trong repo nói ngược lại. Từ `runs_ate/eval_results_BERT.csv`:

| config | train (s) | infer (s) | Micro-F1 |
|---|---:|---:|---:|
| `baseline_balanced` | **257** | **10.8** | 52.86 |
| `lcf_seq_cdm_resize` | 553 | 14.9 | **53.57** |
| `attn_resize` (SCM) | 684 | 37.6 | 51.90 |

Mọi biến thể ToMe đều chậm hơn baseline. Nguyên nhân nằm trong `forward`: `self.bert(...)` chạy đủ 12 lớp trên chuỗi đầy đủ **rồi mới** gọi `self.tome.forward_with_trace(...)`. Gộp sau encoder không tiết kiệm FLOP nào của encoder, chỉ thu nhỏ đầu vào cho lớp SA và 2 head phía sau. Cả 12 cấu hình chính đều `use_pre_tome=False`.

**Cần làm**: chọn một trong hai và viết nhất quán từ abstract tới kết luận.

- **Hướng pooling** (đang theo): sửa README nói rõ ToMe dùng làm cơ chế aggregate biểu diễn, **giải thích vì sao gộp sau encoder là lựa chọn có chủ đích**, và báo cáo chi phí tăng thêm một cách trung thực như cái giá phải trả. Bắt buộc kèm A4.
- **Hướng efficiency**: chuyển sang pre-encoder merging (mục 11), đo FLOPs (mục 10), chứng minh giảm chi phí mà F1 không giảm.

Không được để hai hướng lẫn vào nhau — đó là điểm reviewer công kích đầu tiên.

---

## Phần B — Tài liệu và phạm vi dữ liệu (P2)

### B1 · Bảy chỗ mô tả dữ liệu là tiếng Việt

**Vị trí**:

| File | Dòng | Nội dung |
|---|---|---|
| `README.md` | 1 | tiêu đề "…cho Đánh giá Khách sạn **Tiếng Việt**" |
| `README.md` | 9 | "từ một **câu đánh giá khách sạn tiếng Việt**, trích xuất…" |
| `README.md` | 129 | "`dataset/` — **Dữ liệu** đánh giá khách sạn tiếng Việt" |
| `README.md` | 222–223 | ví dụ định dạng `.apc` viết bằng tiếng Việt |
| `thesis/ket_luan.tex` | 15 | "…tiếng Việt, tổ chức thành bốn stage…" |
| `thesis/ket_luan.tex` | 122 | "…đánh giá khách sạn tiếng Việt" |

**Vì sao**: Dữ liệu thật là tiếng Anh — đếm trên `dataset/train.apc`: **1 ký tự Việt có dấu / 222 599 ký tự = 0.000%**. Ví dụ thật trong tập train:

```
The atmosphere is cool and especially $T$ is friendly and fun with guests
The staff is very enthusiastic, the guides are thoughtful...
```

Backbone cũng là model tiếng Anh (`bert-base-uncased`, `t5-base`), nhất quán với dữ liệu — chỉ có phần mô tả là lệch. Dòng 9 và 129 mô tả trực tiếp *đầu vào* và *thư mục dataset*, nên không phải chuyện dịch tiêu đề.

**Cần làm**: khi chuyển báo cáo sang tiếng Anh, sửa nội dung 7 chỗ này, không dịch nguyên văn. Ví dụ `.apc` ở dòng 222 phải thay bằng mẫu tiếng Anh lấy từ chính `dataset/train.apc`.

---

### B2 · Nguồn gốc dữ liệu chưa khai báo

**Vị trí**: `README.md` mục "Bài toán & Dữ liệu"; `dataset/`

**Vì sao**: Không nói rõ dữ liệu từ đâu ra. Có hai khả năng và cách viết khác hẳn nhau:

- **Dịch máy từ tiếng Việt** → phải khai: dịch bằng công cụ gì, có hiệu đính không, đánh giá chất lượng dịch thế nào, và quan trọng nhất — **aspect term sau khi dịch còn khớp span trong câu không**. Nếu không khớp thì toàn bộ bài toán ATE bị ảnh hưởng.
- **Corpus tiếng Anh gốc** → mô tả nguồn crawl, quy trình gán nhãn, số annotator, **inter-annotator agreement** (Cohen's κ hoặc Krippendorff's α).

Reviewer sẽ hỏi. Tự khai trước thì thành limitation bình thường; bị hỏi mới nói thì thành điểm trừ nặng.

**Cần làm**: thêm một mục "Dataset construction" đầy đủ. Nếu có κ thì báo; không có thì nói rõ là không có và giải thích.

---

### B3 · Chỉ một dataset

**Vị trí**: `dataset/`

**Vì sao**: 2448 mẫu train, một domain (khách sạn), một nguồn. Không so được với SOTA và không chứng minh được phương pháp tổng quát hoá.

**Điểm thuận lợi**: dữ liệu là tiếng Anh nên **SemEval-2014 Restaurant** dùng được ngay — cùng lĩnh vực hospitality, là benchmark chuẩn của ABSA, và định dạng chuyển sang `.apc` không khó. Trước đây nếu là tiếng Việt thì không có lựa chọn này.

**Cần làm**: tối thiểu thêm SemEval-2014 Restaurant + Laptop. Có điều kiện thì thêm ASTE-Data-V2 cho bài toán triplet.

---

### B4 · Số của GAS là chép từ paper, không chạy lại

**Vị trí**: `common/run_multiseed.py` → `REF_METHODS`

**Vì sao**: Bảng `tab:paper` so sánh với `GAS 69.51` và `TOFA 62.65`, cả hai là hằng số hard-code lấy từ bài báo khác — đo trên dữ liệu khác, split khác, có thể cả metric khác. So sánh trực tiếp với kết quả của mình trên dữ liệu này là không hợp lệ.

Repo **đã có** module GAS một bước đầy đủ (`gas/`) nhưng chưa ai train: không có `checkpoints_gas/`, không có `runs_gas/eval_test.json`.

**Cần làm**:

```bash
python run_all.py --stages gas --seeds 42 123 456
```

Báo con số tự đo. Số chép từ paper nếu vẫn muốn giữ thì để bảng riêng và ghi rõ "reported on different data".

---

## Phần C — Đừng đụng vào

Những chỗ hiện **đang đúng** và rất dễ phá khi refactor. Mỗi mục: nó là gì + cái gì làm hỏng nó.

### C1 · ATE CSV khớp `test.apc` theo **chỉ số dòng**, không theo text

`experiments/eval_joint_triplet_run.py` và `run_all.py` → `driver_triplet` thay câu trong ATE CSV bằng câu gold theo đúng thứ tự dòng, vì so khớp chuỗi không đáng tin (khác hoa thường, khác dấu câu).

**Làm hỏng bằng cách**: sort lại CSV, lọc bớt dòng, hoặc đổi số mẫu của một trong hai file. Lệch một dòng là toàn bộ eval end-to-end sai mà không báo lỗi — chỉ ra F1 thấp bất thường.

### C2 · `results_raw.csv` là trạng thái resume duy nhất của multi-seed

`common/run_multiseed.py` dựng `done_keys` từ file này và kiểm tra **trước** khi train. Nhờ vậy checkpoint của `multiseed` xoá được ngay sau mỗi lát mà vẫn resume được — đây là thứ khiến chạy trên Kaggle (20 GB) khả thi, trong khi tổng checkpoint là 75,9 GB.

**Làm hỏng bằng cách**: xoá file này để "chạy lại cho sạch". Mất nó là mất toàn bộ tiến độ, phải train lại từ đầu.

### C3 · Nhãn được dựng từ hợp của cả 3 split

`common/dataset_utils.py` → `build_label_maps_from_apc(train, dev, test)`. Chỉ số lớp trong mọi `best_model.pt` đã lưu phụ thuộc vào thứ tự này, và được ghi lại trong `meta.json` cạnh checkpoint.

**Làm hỏng bằng cách**: đổi sang chỉ dựng từ train, hoặc thêm/bớt mẫu làm xuất hiện lớp mới. Checkpoint cũ sẽ load được nhưng gán nhãn lệch — sai âm thầm, không crash. `best_model.pt` và `meta.json` phải luôn đi cùng nhau.

### C4 · Dữ liệu supplement chỉ nuôi head sentiment

`ApcFileDataset` gắn cờ `is_supplement=True` cho các dòng từ `supplement/*.tsv`; head category bỏ qua chúng khi tính loss. Có chủ đích: supplement không có nhãn category đáng tin.

**Làm hỏng bằng cách**: bỏ cờ này để "dùng hết dữ liệu". Head category sẽ học trên nhãn rác.

### C5 · SCM bảo vệ token aspect và CLS/SEP

`token_merging/tome_1d.py`, chiến lược `sequential_cosine`. Đây **chính là** đóng góp của luận văn — khác biệt so với ToMe gốc.

**Làm hỏng bằng cách**: "đơn giản hoá" phần protect mask khi refactor. Mất nó thì SCM trở thành ToMe thường và bài không còn đóng góp.

---

## Phần D — Hạ tầng đo lường còn thiếu (P2–P3)

### D1 · Không đo FLOPs, tham số, throughput

**Vị trí**: toàn repo

**Vì sao**: `grep` không tìm thấy `thop`, `ptflops`, `fvcore`, `macs`, cũng không có thống kê độ dài chuỗi sau khi gộp. Chỉ có `train_time_sec` và `inference_per_sample_ms` trong `run_multiseed.py`.

Wall-clock time phụ thuộc GPU, batch size, tải máy — reviewer không chấp nhận làm bằng chứng hiệu năng. Nếu đi hướng efficiency (A5) thì đây là bắt buộc; nếu đi hướng pooling thì vẫn cần để báo cáo trung thực chi phí tăng thêm.

**Cần làm**: thêm vào bảng kết quả — FLOPs/mẫu, số tham số, throughput (mẫu/giây), và **độ dài chuỗi trung bình trước/sau khi gộp**. Cột cuối là thứ trực tiếp cho thấy SCM thực sự gộp được bao nhiêu.

### D2 · Nhóm `pretome` chưa chạy

**Vị trí**: `run_all.py` → `VARIANTS`, nhóm `pretome`

**Vì sao**: Ba biến thể `lcf_pre_bip`, `lcf_pre_seq`, `lcf_pre_scm` gộp token **trước** encoder — chỗ duy nhất tiết kiệm FLOP thật sự. Chưa có kết quả nào.

**Cần làm**: chạy nếu chọn hướng efficiency. Nếu chọn hướng pooling thì vẫn nên chạy làm ablation "gộp ở đâu thì tốt hơn" — câu hỏi này tự nhiên và reviewer sẽ nghĩ tới.

```bash
python run_all.py --stages multiseed --variants pretome --seeds 42 123 456
```

### D3 · Tên cấu hình trong CSV đã commit không khớp code

**Vị trí**: `runs_ate/eval_results_*.csv`, `runs_ate/eval_joint_triplet_*.csv`

**Vì sao**: Các file này dùng tên cũ `attn_resize`, `lcf_attn_cdm_resize`, trong khi code hiện tại gọi chiến lược đó là `sequential_cosine` (SCM). README đã giải thích việc đổi tên, nhưng dữ liệu kết quả thì chưa đổi. Người đọc đối chiếu bảng trong paper với CSV trong repo sẽ không khớp tên.

Ngoài ra `eval_joint_triplet_BERT.csv` có các dòng hậu tố `bert sup`, `compact` trộn lẫn nhiều lần chạy khác nhau, không rõ cấu hình.

**Cần làm**: sinh lại các CSV này sau khi chạy multi-seed, dùng tên thống nhất. Nếu công bố dữ liệu kèm paper thì đây là bắt buộc.

### D4 · Prediction ATE lệch pha với checkpoint

**Vị trí**: `runs_ate/test_ate_predictions.csv` (commit 17/06) vs `checkpoints/best/` (23/07)

**Vì sao**: Hai file cách nhau **868 giờ**. Prediction đã commit không phải do checkpoint đã commit sinh ra. Mọi kết quả end-to-end đã có đều dựa trên file prediction cũ này.

**Trạng thái**: cơ chế đã sửa — `run_all.py` → `stage_ate_infer` giờ so mtime và tự sinh lại khi checkpoint mới hơn. Nhưng **các CSV kết quả đã commit vẫn là số cũ**.

**Cần làm**: chạy lại `ate` + `ate_infer` + toàn bộ eval để mọi con số trong paper đến từ cùng một checkpoint.

---

## Ghi chú khi chạy lại

Sau khi chạy multi-seed, kiểm tra các bất biến sau trước khi tin kết quả:

1. **Số combo**: `runs_multiseed/results_raw.csv` phải có đúng `số_seed × số_backbone × số_biến_thể` dòng. Thiếu dòng nghĩa là có lát bị bỏ giữa chừng.
2. **Align ATE**: số dòng `runs_ate/test_ate_predictions.csv` phải bằng số mẫu trong `dataset/test.apc` (312). Lệch là eval e2e sai âm thầm — xem C1.
3. **Cùng một checkpoint**: mtime của `test_ate_predictions.csv` phải **mới hơn** mtime của checkpoint ATE. Xem D4.
4. **Per-class trước macro**: luôn in bảng per-class kèm `n_test` trước khi nhìn Macro-F1. Lớp dưới 8 mẫu phải đánh dấu rõ trong mọi bảng.
5. **Độ lệch chuẩn**: nếu std của Macro-F1 vượt quá mức chênh lệch giữa các phương pháp thì kết luận "phương pháp A hơn B" không đứng được — phải nói thẳng điều đó trong paper thay vì bỏ qua.
6. **Kiểm định**: paired t-test hoặc bootstrap CI giữa cấu hình đề xuất và baseline mạnh nhất, không phải giữa đề xuất và baseline yếu nhất.

---

## Còn treo trước khi viết paper

Ba việc phải xong, theo đúng thứ tự:

1. **A1 — multi-seed.** Trả lời câu "có tín hiệu thật hay không". Nếu `EXPERIENCE_negative` nhảy 0/100 giữa các seed thì mức tăng +8.36 hiện tại là nhiễu, và phải tìm hướng kể chuyện khác trước khi viết dòng nào.
2. **A5 — chốt động cơ.** Pooling hay efficiency. Quyết định này chi phối abstract, related work, thiết kế thí nghiệm và phần thảo luận. Không chốt trước là viết lại từ đầu.
3. **A4 — baseline pooling.** Nếu chọn hướng pooling ở bước 2 thì đây là thí nghiệm bắt buộc, không có thì luận điểm không có nội dung.

Xong ba việc này mới biết bài hướng về đâu. A2, A3 và Phần B làm song song được.

---

*Vị trí trong tài liệu này dẫn theo **tên hàm** và **tên file**, không theo số dòng, trừ phần B1 cần chỉ chính xác dòng cần sửa. Số dòng sẽ lệch sau vài commit.*
