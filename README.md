# Trích xuất Bộ ba Khía cạnh – Danh mục – Cực tính cho Đánh giá Khách sạn Tiếng Việt

**Khóa luận tốt nghiệp** · Phân tích Cảm xúc dựa trên Khía cạnh *(Aspect-Based Sentiment Analysis)*

---

## Tổng quan

Hệ thống giải quyết bài toán **Joint Triplet Extraction** trong Aspect-Based Sentiment Analysis (ABSA): từ một câu đánh giá khách sạn tiếng Việt, trích xuất toàn bộ bộ ba `(aspect_term, aspect_category, sentiment)`.

Hai kiến trúc chính được nghiên cứu và so sánh:

| Kiến trúc | Mô tả | Ưu điểm |
|---|---|---|
| **Joint** | ATE + APC huấn luyện cùng nhau, đánh giá end-to-end | Không có lỗi lan truyền giữa 2 bước |
| **Pipeline** | ATE (T5) → APC (BERT), hai bước độc lập | Linh hoạt, dễ thay thế từng thành phần |

Ngoài ra, hệ thống bao gồm:
- **GAS** (Generative Aspect Sentiment): T5 sinh bộ ba trong một bước duy nhất
- **UOS** (Unit Opinion Sentence): tách câu bằng LLM để thu hẹp ngữ cảnh cho APC
- **Web Interface**: FastAPI backend + React frontend cho demo tương tác

Toàn bộ nội dung luận văn (LaTeX) nằm ở [`thesis/`](thesis/README.md), tách biệt khỏi phần code ở đây.

---

## Chạy nhanh (Quick Start) — Inference qua Web UI

Checkpoint mặc định (`checkpoints/gas_t5_ate/best` và `runs_joint/lcf_scm_cdm_resize`) đã có sẵn trong repo, có thể chạy thẳng không cần train lại. Thứ tự khởi động: **Ollama (nếu dùng UOS) → Backend → Frontend**.

### Cách 1 — Script khởi động tất cả (Windows)

```bat
run_inference.bat
```

hoặc chạy trực tiếp bằng PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File run_inference.ps1
```

Script [`run_inference.ps1`](run_inference.ps1) tự động: kiểm tra/khởi động Ollama, kiểm tra/pull model `qwen3:8b`, cài `frontend/node_modules` nếu chưa có, rồi mở **3 cửa sổ PowerShell riêng** cho Ollama, Backend (dùng Python trong `.venv`) và Frontend. Đóng từng cửa sổ để dừng service tương ứng.

Tuỳ chọn:
```powershell
# Đổi chế độ tách câu (mặc định: uos)
powershell -ExecutionPolicy Bypass -File run_inference.ps1 -ClauseSplitMode rulebase

# Bỏ qua bước kiểm tra/khởi động Ollama (đã tự chạy sẵn, hoặc dùng none/rulebase)
powershell -ExecutionPolicy Bypass -File run_inference.ps1 -SkipOllama
```

### Cách 2 — Chạy thủ công từng bước

#### 1. Ollama — chỉ cần nếu dùng tách câu bằng LLM (`CLAUSE_SPLIT_MODE=uos`)

```bash
ollama serve
ollama pull qwen3:8b
```

Nếu không chạy Ollama, dùng `CLAUSE_SPLIT_MODE=none` hoặc `rulebase` thay thế. UOS cũng tự fallback về câu gốc nếu Ollama không kết nối được (không crash).

#### 2. Backend (FastAPI)

```bash
# Windows (Command Prompt)
set CLAUSE_SPLIT_MODE=uos
uvicorn server.app:app --host 0.0.0.0 --port 5000

# Linux / macOS
CLAUSE_SPLIT_MODE=uos uvicorn server.app:app --host 0.0.0.0 --port 5000
```

Các biến môi trường khác (`ATE_CHECKPOINT`, `APC_CHECKPOINT_DIR`, `BERT_NAME`) giữ mặc định nếu không set — xem chi tiết ở [Biến môi trường](#biến-môi-trường). Backend log dòng `[server] Loading pipeline: ATE=... APC=... clause_split_mode=...` khi khởi động thành công.

#### 3. Frontend (React + Vite)

```bash
cd frontend
npm install    # chỉ cần lần đầu
npm run dev
```

Mở `http://localhost:5173` — frontend gọi tới backend tại `http://localhost:5000`. Nhập trực tiếp một câu hoặc upload file `.txt`/`.docx` để xem kết quả. Chi tiết API xem [Web Interface](#web-interface).

---

## Cấu trúc thư mục

```
thesis_apc_baseline/
│
├── src/                            # Module ATE — T5 Aspect Term Extraction
│   ├── train.py                    # CLI huấn luyện T5 ATE
│   ├── model.py                    # T5AspectExtractor (T5ForConditionalGeneration wrapper)
│   ├── trainer.py                  # ATETrainer: AMP, early stopping, cosine scheduler
│   ├── dataset.py                  # ATEDataset + create_ate_dataloaders()
│   ├── inference.py                # predict_aspects(): sinh + chuẩn hoá Levenshtein
│   ├── metrics.py                  # Exact-match Precision / Recall / F1
│   └── normalization.py            # n-gram Levenshtein normalization
│
├── models/
│   └── fast_lcf_bert_multitask.py  # BERT đa nhiệm: LCF (CDM/CDW) + ToMe
│
├── experiments/                    # Scripts huấn luyện và đánh giá APC
│   ├── run_joint_experiments.py    # Chạy tuần tự 12 cấu hình thực nghiệm
│   ├── eval_joint_triplet.py       # Đánh giá bộ ba (aspect, category, sentiment)
│   ├── eval_joint_select.py        # Đánh giá một danh sách folder cụ thể
│   ├── run_ate_inference.py        # Infer ATE trên tập test → CSV
│   └── eval_results.py             # Đánh giá ATE F1 (standalone)
│
├── gas/                            # GAS — Generative Aspect Sentiment (1 bước)
│   ├── model.py                    # GasT5Model: T5 sinh bộ ba trực tiếp
│   └── train_gas.py, trainer.py, dataset.py, metrics.py, infer.py
│
├── uos/                            # UOS — Unit Opinion Sentence (tách câu LLM)
│   ├── segmenter.py                # OllamaUOSSegmenter (Qwen 8B via Ollama)
│   ├── prompt.py                   # Quy tắc ngôn ngữ SPLIT / DO NOT SPLIT
│   ├── parsing.py, pipeline.py, metrics.py, records.py, validation.py
│   ├── run_llm_uos_eval.py
│   └── README.md
│
├── token_merging/                  # ToMe — Token Merging
│   ├── tome_1d.py                  # bipartite / sequential_local / sequential_cosine (SCM)
│   └── __init__.py
│
├── dataset/                        # Dữ liệu đánh giá khách sạn tiếng Việt
│   ├── train.apc                   # 2 448 mẫu (9 792 dòng)
│   ├── dev.apc                     # 304 mẫu
│   ├── test.apc                    # 312 mẫu
│   ├── test_sentences_id.csv       # Gold aspect terms cho tập test
│   └── supplement/
│       ├── negative.tsv            # Dữ liệu bổ sung cân bằng nhãn âm
│       └── neutral.tsv             # Dữ liệu bổ sung cân bằng nhãn trung tính
│
├── checkpoints/
│   └── gas_t5_ate/best/            # Checkpoint T5 ATE tốt nhất (~891 MB, gitignored)
│
├── checkpoints_gas/                # Checkpoint GAS T5 (joint generation, gitignored)
│
├── runs_joint/                     # Output APC — mỗi config 1 thư mục con (gitignored)
│   └── <config_name>/
│       ├── best_model.pt           # Checkpoint tốt nhất (theo dev joint F1)
│       └── meta.json               # Labels, flags, best_dev_f1, best_epoch
│
├── runs_ate/                       # Kết quả đánh giá (CSV, tracked)
│   ├── test_ate_predictions.csv
│   ├── eval_joint_triplet_BERT.csv
│   ├── eval_joint_triplet_T5.csv
│   ├── eval_results_BERT.csv
│   └── eval_results_T5.csv
│
├── Bert/, T5/                      # Checkpoint lưu trữ các sweep cũ hơn (gitignored, rất lớn)
│
├── server/
│   └── app.py                      # FastAPI backend (REST API)
│
├── frontend/                       # React + Vite frontend
│
├── scripts/
│   ├── generate_thesis_figures.py  # Sinh hình cho luận văn → thesis/figures/
│   ├── visualize_scm.py            # Minh hoạ từng bước thuật toán SCM
│   └── visualize_scm_example.py    # Minh hoạ SCM trên câu ví dụ cụ thể
│
├── notebooks/                      # Notebook thăm dò/so sánh (không nằm trong pipeline chính)
│   ├── gas_paper.ipynb
│   └── train_ate.ipynb
│
├── docs/                           # Tài liệu tham khảo, PDF luận văn đã biên dịch
│
├── thesis/                         # Toàn bộ LaTeX của luận văn (xem thesis/README.md)
│   ├── main.tex, mo_dau.tex, co_so_ly_thuyet.tex, ...
│   ├── references.bib
│   └── figures/
│
├── common/                         # Utility dùng chung + script CLI gốc (import as common.xxx)
│   ├── pipeline_inference.py       # PipelineInference: ATE -> Clause Split -> APC
│   ├── infer_aspect_term.py        # CLI infer APC cho một aspect term cụ thể
│   ├── dataset_utils.py            # ApcFileDataset: parser .apc + supplement
│   ├── ate_dataset_utils.py        # ATEDataset: parser .apc cho T5
│   ├── clause_splitting.py         # Tách câu: none / rulebase / uos
│   ├── eval_bert_gold_aspects.py   # Đánh giá APC với gold aspect (oracle)
│   ├── run_multiseed.py            # Huấn luyện nhiều seed, tổng hợp bảng cho luận văn
│   └── run_multiseed_ate.py        # Tương tự, cho module ATE
│
├── run_all.py                      # Chạy FULL luồng 21 biến thể + sinh reports/
├── reports/                        # Báo cáo tổng hợp do run_all.py sinh ra
│   ├── REPORT.md                   # Báo cáo chính (trạng thái stage + bảng kết quả)
│   ├── manifest.csv                # Danh mục mọi artifact + trạng thái
│   └── logs/<stage>.log            # Log đầy đủ từng stage
│
├── run_inference.ps1               # Khởi động Ollama + Backend + Frontend (Windows)
├── run_inference.bat               # Wrapper double-click cho run_inference.ps1
└── requirements.txt
```

---

## Bài toán & Dữ liệu

### Bài toán

**Aspect-Based Sentiment Analysis (ABSA) — Joint Triplet Extraction**

> Cho câu đánh giá $x$, tìm tập hợp $\mathcal{T} = \{(a_i, c_i, s_i)\}$ trong đó $a_i$ là aspect term, $c_i$ là aspect category, $s_i$ là cực tính.

| Thành phần | Tập giá trị |
|---|---|
| `aspect_term` | Cụm từ xuất hiện trong câu |
| `aspect_category` | `AMENITY` · `BRANDING` · `EXPERIENCE` · `FACILITY` · `LOYALTY` · `SERVICE` |
| `sentiment` | `Positive` · `Negative` · `Neutral` |

**Tiêu chí đánh giá:** Bộ ba `(aspect_term, category, sentiment)` được tính là **TP** khi và chỉ khi **cả ba thành phần** khớp chính xác với nhãn gold.

### Định dạng file `.apc`

Mỗi mẫu gồm **4 dòng liên tiếp**, cách nhau bằng dòng trắng:

```
$T$ rất chuyên nghiệp và chu đáo từ bộ phận nhà hàng, buồng phòng đến lễ tân.
nhân viên phục vụ
SERVICE
Positive

Phòng $T$ sạch sẽ và rộng rãi.
ngủ
FACILITY
Positive
```

- **Dòng 1:** Câu gốc, `$T$` là placeholder cho aspect term
- **Dòng 2:** Aspect term (thay thế `$T$` trong câu)
- **Dòng 3:** Aspect category
- **Dòng 4:** Sentiment label

### Thống kê dữ liệu

| Tập | Số mẫu | Ghi chú |
|---|---|---|
| Train | 2 448 | Thêm `supplement/negative.tsv` + `neutral.tsv` để cân bằng nhãn |
| Dev | 304 | Dùng cho early stopping và chọn checkpoint |
| Test | 312 | Đánh giá chính thức — **không dùng trong huấn luyện** |

**Phân bố nhãn (train):** Mất cân bằng nghiêm trọng — `SERVICE_positive` chiếm tỉ lệ cao nhất, `BRANDING_neutral` và `FACILITY_neutral` rất hiếm → lý do sử dụng weighted loss và dữ liệu supplement.

---

## Kiến trúc mô hình

### 1. T5 Aspect Term Extraction (ATE)

```
Input:  "Phòng sạch sẽ, nhân viên nhiệt tình nhưng thang máy hay hỏng."
          ↓
   T5ForConditionalGeneration (t5-base)
   Beam search: beam=4, max_len=64
          ↓
Output: "(phòng); (nhân viên); (thang máy)"
          ↓
   Levenshtein normalization → khớp về văn bản gốc
          ↓
Result: ["phòng", "nhân viên", "thang máy"]
```

**Đặc điểm:**
- Sinh aspect terms dưới dạng chuỗi có cấu trúc, không cần BIO tagging
- Chuẩn hoá đầu ra bằng Levenshtein n-gram matching để xử lý lỗi sinh
- Checkpoint: `checkpoints/gas_t5_ate/best/` (~891 MB)

### 2. FastLcfBertMultiTask (APC)

```
Input: (sentence, aspect_term)
          ↓
[Pre-BERT ToMe] — tuỳ chọn: gộp token trước BERT
          ↓
BERT encoder (bert-base-uncased, 12 layers, hidden=768)
          ↓
[Post-BERT ToMe] — gộp token sau BERT (mặc định 2 bước)
          ↓
  ┌─ Local stream:  H × CDW(lcf_vec)  ← ngữ cảnh quanh aspect (SRD=5)
  └─ Global stream: H (không che)     ← ngữ cảnh toàn câu
          ↓
Linear(2H→H) → Dropout(0.1) → Self-Attention → Pooler
  ├─ Sentiment head: Linear(H→3) → Softmax   [positive / negative / neutral]
  └─ Category head:  Linear(H→6) → Softmax   [AMENITY / ... / SERVICE]
```

**LCF — Local Context Focus:**
Tập trung vào ngữ cảnh cục bộ xung quanh aspect term với hai chiến lược scoring:

| Chiến lược | Cơ chế | Đặc điểm |
|---|---|---|
| **CDM** (Context Dynamic Mask) | Mask nhị phân: token ngoài SRD=5 bị che hoàn toàn | Tập trung cứng, phù hợp nhãn thiểu số |
| **CDW** (Context Dynamic Weight) | Giảm trọng số tuyến tính theo khoảng cách đến aspect | Tập trung mềm, giữ thông tin toàn cầu |

**ToMe — Token Merging:**
Gộp token dư thừa để giảm độ phức tạp tính toán (CVPR 2023):

| Chiến lược | Mô tả |
|---|---|
| `bipartite` | Ghép cặp token theo nhóm chẵn/lẻ (phương pháp gốc) |
| `sequential_local` | Gộp hàng xóm trái–phải theo thứ tự |
| `sequential_cosine` (SCM) | Duyệt trái → phải, chọn token trái nhất chưa bảo vệ và gộp với hàng xóm có cosine similarity cao nhất trong toàn chuỗi; bảo vệ token aspect (LCF) và CLS/SEP |

> Chiến lược `sequential_cosine` trước đây được gọi là `attention_weighted`/AWM. Tên cũ gây hiểu nhầm: cài đặt thực tế không dùng attention score từ BERT (chỉ dùng vị trí + cosine similarity), nên đã đổi tên cho khớp với hành vi thật (xem `token_merging/tome_1d.py`).

Khi `tome_resize=True`: nội suy khôi phục độ dài chuỗi ban đầu sau khi gộp.

**Multi-task Loss:**
```
L_total = w_sent × L_sentiment + w_cat × L_category
```
- `L_sentiment`: CrossEntropyLoss có class weights (train.apc + supplement)
- `L_category`: CrossEntropyLoss có class weights (chỉ train.apc, bỏ qua supplement)

---

## Cài đặt

### Yêu cầu môi trường

- Python ≥ 3.9
- CUDA ≥ 11.7 (khuyến nghị cho huấn luyện; CPU inference được hỗ trợ)
- *(Tuỳ chọn)* [Ollama](https://ollama.com/) + model `qwen3:8b` cho chế độ UOS

### Cài thư viện

```bash
pip install -r requirements.txt
```

**Thư viện chính:**

| Thư viện | Phiên bản | Mục đích |
|---|---|---|
| `torch` | ≥ 2.0.0 | Deep learning framework |
| `transformers` | ≥ 4.36.0 | T5 / BERT pretrained models |
| `pyabsa` | ≥ 2.4.0, < 3 | ABSA utilities & tokenizer helpers |
| `python-Levenshtein` | ≥ 0.25.0 | Chuẩn hoá đầu ra ATE |
| `scikit-learn` | ≥ 1.2.0 | Weighted loss, classification metrics |
| `fastapi` + `uvicorn` | ≥ 0.109.0 | REST API backend |
| `python-docx` | ≥ 0.8.12 | Upload file .docx |
| `python-multipart` | ≥ 0.0.6 | Multipart form data |

---

## Chạy full luồng bằng một lệnh — `run_all.py`

`run_all.py` điều phối toàn bộ pipeline (train → inference → đánh giá → hình →
báo cáo) cho **21 biến thể** mô hình, rồi gom mọi kết quả vào `reports/`.
Script không cài đặt lại logic nào — nó gọi đúng các script đã có trong repo và
inject cấu hình cho những script vốn hard-code hằng số ở module level.

```bash
python run_all.py                              # full: 3 seeds × 2 backbone × 21 biến thể
python run_all.py --list                       # xem danh sách stage + biến thể
python run_all.py --dry-run                    # in ra lệnh sẽ chạy, không thực thi
python run_all.py --resume                     # chạy tiếp, bỏ qua combo đã xong
python run_all.py --seeds 42 --backbones bert  # bản rút gọn cho máy yếu
python run_all.py --variants resize            # chỉ 12 cấu hình resize
python run_all.py --smoke                      # CHẠY THỬ trên data tí hon, ~5-10 phút
python run_all.py --skip uos gas               # bỏ UOS (cần Ollama) và GAS
python run_all.py --stages report              # chỉ sinh lại báo cáo từ kết quả cũ
```

### Các stage (chạy theo thứ tự này)

| Stage | Việc làm | Kết quả |
|---|---|---|
| `env` | Preflight: thư viện, CUDA, dữ liệu, checkpoint sẵn có | `reports/00_environment.txt` |
| `ate` | Train T5 ATE (GAS) đa seed | `checkpoints/gas_t5_ate/seed_<N>/best`, `runs_ate/results_ate_multiseed.csv`, `results_ate_summary.txt` |
| `ate_infer` | Infer ATE trên tập test | `runs_ate/test_ate_predictions.csv` |
| `apc` | Train mọi biến thể APC (seed đầu tiên) | `runs_joint/<config>/best_model.pt` (BERT), `runs_joint_t5/…` (T5), `experiment_results_joint.{csv,txt}` |
| `multiseed` | Train đa seed + oracle/e2e eval | `runs_multiseed/thesis_tables.txt` (8 bảng luận văn), `results_raw.csv`, `results_aggregated.csv` |
| `gold` | Oracle eval với gold aspect term | `runs_bert_gold/{BERT,T5}/eval_gold_aspects.csv` |
| `triplet` | Eval bộ ba end-to-end (ATE → APC) | `runs_ate/eval_joint_triplet_{BERT,T5}.csv` |
| `gas` | Train + eval GAS một bước | `checkpoints_gas/best`, `runs_gas/eval_test.{json,csv}` |
| `results` | Eval trên `results.csv` (cần checkpoint GAS) | `runs_ate/eval_results_{BERT,T5}.csv` |
| `uos` | Tách câu bằng LLM qua Ollama | `uos/output/test/{results.jsonl,metrics.json}` |
| `figures` | Sinh hình luận văn | `thesis/figures/*.{pdf,png}` |
| `report` | Gom tất cả lại | `reports/REPORT.md`, `reports/manifest.csv` |

Mỗi stage chạy trong một tiến trình con riêng (GPU memory được giải phóng giữa
các stage) và ghi log đầy đủ vào `reports/logs/<stage>.log`. Một stage hỏng
không làm chết cả lượt chạy — dùng `--fail-fast` nếu muốn dừng ngay.

### 21 biến thể

| Nhóm | Số lượng | Nội dung |
|---|---|---|
| `resize` | 12 | Baseline, LCF-only (CDM/CDW), ToMe-only (Bip/SLM/SCM), LCF×ToMe×{CDM,CDW} — đều `tome_resize=True` |
| `compact` | 6 | LCF×{Bip,SLM,SCM}×{CDM,CDW} với `tome_resize=False` (đối chứng resize vs compact) |
| `pretome` | 3 | Gộp token **trước** BERT encoder: `lcf_pre_bip`, `lcf_pre_seq`, `lcf_pre_scm` |

Xem đầy đủ cờ của từng biến thể bằng `python run_all.py --list`.

> **Lưu ý về đặt tên:** `runs_multiseed/` dùng id ngắn (`lcf_scm_cdm`) vì các
> bảng luận văn tham chiếu theo id đó, còn `runs_joint/` giữ quy ước có hậu tố
> (`lcf_scm_cdm_resize`) cho khớp với `README` và mặc định
> `APC_CHECKPOINT_DIR` của `server/app.py`. `run_all.py` giữ cả hai id cho mỗi
> biến thể nên hai bộ kết quả luôn khớp nhau.

### Chạy thử trước khi tốn GPU — `--smoke`

```bash
python run_all.py --smoke
```

Dựng một dataset tí hon (72 train / 32 dev / 32 test) rồi chạy
**toàn bộ** đường ống với 1 epoch, 1 seed, 3 biến thể phủ đủ 3 nhánh code
(`lcf_scm_cdm` = post-ToMe resize, `lcf_scm_cdm_compact` = post-ToMe compact,
`lcf_pre_scm` = pre-ToMe). Khoảng 5–10 phút trên GPU.

Mẫu được chọn **phủ đủ 6 category × 3 sentiment**, không phải N mẫu đầu: dataset
sắp xếp theo category nên 48 mẫu đầu chỉ có `SERVICE/Positive` — một lớp duy
nhất, và `classification_report(target_names=...)` trong
`run_joint_experiments.py` sẽ ném `ValueError`. Chỉ số được giữ nguyên thứ tự
để `test_sentences_id.csv` vẫn align theo index với `test.apc`.

Mọi output đi vào `smoke_run/` — **không đè lên kết quả thật**:

```
dataset_smoke/          # dữ liệu tí hon
smoke_run/
├── checkpoints/        ├── runs_joint/        ├── runs_bert_gold/
├── checkpoints_gas/    ├── runs_joint_t5/     ├── runs_gas/
├── runs_ate/           ├── runs_multiseed/    ├── thesis_figures/
└── reports/REPORT.md + manifest.csv + logs/<stage>.log
```

Hai cờ dùng chung với chế độ thật:

```bash
python run_all.py --max-epochs 2          # giới hạn epoch cho mọi bước train
python run_all.py --out-root /tmp/thu     # đổi thư mục output gốc
python run_all.py --data-dir dataset_alt  # đổi thư mục dữ liệu
```

Stage `results` bị bỏ qua khi smoke: `experiments/eval_results.py` hard-code
`ROOT/"dataset"` ở module level (dòng 103, 213, 350) nên không chuyển sang dataset
nhỏ được.

### Chạy trên Kaggle

[`notebooks/kaggle_run_all.ipynb`](notebooks/kaggle_run_all.ipynb) — clone code từ
GitHub (clone lần đầu, `fetch --all --prune` + `reset --hard origin/<branch>` những
lần sau), cài thư viện còn thiếu, chạy smoke test, rồi chạy thật với `--resume`, và
đóng gói kết quả thành zip để tải về.

Cần bật **Accelerator = GPU** và **Internet = On** trong panel bên phải. Phiên Kaggle
tối đa 9–12 giờ nên lượt đầy đủ phải chia nhiều phiên — `--resume` lo phần nối tiếp.

### Yêu cầu tài nguyên

Lần chạy đầy đủ mặc định gồm 126 lượt train APC (21 biến thể × 3 seed × 2
backbone) ở stage `multiseed`, cộng 42 lượt ở stage `apc` và 3 lượt train T5
ATE. Trên một GPU đơn, hãy tính bằng ngày chứ không phải giờ. Dùng `--resume`
để chạy nhiều phiên, hoặc thu hẹp bằng `--seeds 42 --backbones bert --variants resize`.

---

## Huấn luyện

### Bước 1 — Huấn luyện ATE (T5)

```bash
python src/train.py \
  --data-dir dataset \
  --output-dir checkpoints/gas_t5_ate \
  --model-name t5-base \
  --epochs 20 \
  --batch-size 16 \
  --learning-rate 3e-4 \
  --max-input-length 128 \
  --max-target-length 64 \
  --seed 42
```

Checkpoint tốt nhất (theo dev F1) → `checkpoints/gas_t5_ate/best/`
Checkpoint cuối cùng → `checkpoints/gas_t5_ate/last/`

**Các tham số CLI:**

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--data-dir` | `dataset/` | Thư mục chứa `train.apc`, `dev.apc`, `test.apc` |
| `--output-dir` | `checkpoints/gas_t5_ate` | Nơi lưu checkpoint |
| `--model-name` | `t5-base` | Backbone: `t5-small` / `t5-base` / `t5-large` |
| `--batch-size` | `16` | Batch size |
| `--learning-rate` | `3e-4` | Learning rate (AdamW) |
| `--epochs` | `20` | Số epoch tối đa |
| `--max-input-length` | `128` | Độ dài input T5 tối đa |
| `--max-target-length` | `64` | Độ dài target tối đa |
| `--seed` | `42` | Random seed |
| `--num-workers` | `0` | DataLoader workers |

### Bước 2 — Huấn luyện APC đa nhiệm (BERT + LCF + ToMe)

```bash
python experiments/run_joint_experiments.py
```

Script chạy tuần tự **12 cấu hình** được định nghĩa trong biến `CONFIGS`. Kết quả từng config lưu tại `runs_joint/<config_name>/`.

**Hyperparameters cố định:**

| Biến | Giá trị | Mô tả |
|---|---|---|
| `PRETRAINED_MODEL` | `bert-base-uncased` | BERT backbone |
| `NUM_EPOCHS` | `15` | Số epoch tối đa |
| `PATIENCE` | `4` | Early stopping patience |
| `BATCH_SIZE` | `16` | Batch size |
| `LR` | `2e-5` | Learning rate |
| `MAX_SEQ_LEN` | `128` | Độ dài chuỗi BERT tối đa |
| `DROPOUT` | `0.1` | Dropout rate |
| `SRD_THRESHOLD` | `5` | Bán kính ngữ cảnh LCF (SRD) |
| `TOME_MERGE_STEPS` | `2` | Số bước gộp token (post-BERT) |
| `SEED` | `42` | Random seed |
| `USE_MIXED_PRECISION` | `True` | AMP (tự động tắt nếu không có CUDA) |

**12 cấu hình thực nghiệm:**

| Config name | LCF | Scoring | ToMe strategy | Resize |
|---|---|---|---|---|
| `baseline_balanced` | ✗ | — | — | — |
| `scm_resize` | ✗ | — | `sequential_cosine` | ✓ |
| `seq_resize` | ✗ | — | `sequential_local` | ✓ |
| `bip_resize` | ✗ | — | `bipartite` | ✓ |
| `lcf_only_cdm` | ✓ | CDM | — | — |
| `lcf_only_cdw` | ✓ | CDW | — | — |
| `lcf_scm_cdm_resize` | ✓ | CDM | `sequential_cosine` | ✓ |
| `lcf_scm_cdw_resize` | ✓ | CDW | `sequential_cosine` | ✓ |
| `lcf_bip_cdm_resize` | ✓ | CDM | `bipartite` | ✓ |
| `lcf_bip_cdw_resize` | ✓ | CDW | `bipartite` | ✓ |
| `lcf_seq_cdm_resize` | ✓ | CDM | `sequential_local` | ✓ |
| `lcf_seq_cdw_resize` | ✓ | CDW | `sequential_local` | ✓ |

---

## Đánh giá

### Quy trình đánh giá Pipeline

```bash
# Bước 1: Infer ATE trên tập test
python experiments/run_ate_inference.py
# → runs_ate/test_ate_predictions.csv

# Bước 2: Đánh giá joint triplet (BERT APC backbone)
python experiments/eval_joint_triplet_run.py --model-type bert
# → runs_ate/eval_joint_triplet_BERT.csv

# Bước 2 (T5 APC backbone)
python experiments/eval_joint_triplet.py --model-type t5
# → runs_ate/eval_joint_triplet_T5.csv

# Đánh giá ATE đơn thuần
python experiments/eval_results.py --model-type bert
python experiments/eval_results.py --model-type t5
# → runs_ate/eval_results_BERT.csv, eval_results_T5.csv

# Đánh giá một danh sách folder cụ thể (thay vì toàn bộ runs-dir)
python experiments/eval_joint_select.py --model-type bert \
  --folders "baseline_balanced" "lcf_scm_cdm_resize" "lcf_seq_cdm_resize"
```

### Metrics

| Metric | Công thức | Ý nghĩa |
|---|---|---|
| **ATE F1** | $F_1 = \frac{2PR}{P+R}$ (exact-match) | Chất lượng trích xuất aspect term |
| **Micro F1** | Tính TP/FP/FN trên toàn bộ mẫu | Hiệu suất tổng thể, ưu tiên nhãn phổ biến |
| **Macro F1** | Trung bình F1 qua các nhãn | Công bằng qua tất cả nhãn, phạt nhãn thiểu số |
| **Oracle Cat Acc** | Acc category khi aspect gold | Giới hạn trên của classifier category |
| **Oracle Sent Acc** | Acc sentiment khi aspect gold | Giới hạn trên của classifier sentiment |
| **Oracle Joint Acc** | Joint acc khi aspect gold | Giới hạn lý thuyết khi ATE hoàn hảo |

---

## Kết quả thực nghiệm

### Tổng hợp kết quả tốt nhất

| Cài đặt | Config tốt nhất | ATE F1 | Micro F1 | Macro F1 | Oracle Joint |
|---|---|---|---|---|---|
| Joint · BERT | `lcf_seq_cdm_resize` | 79.87 | **71.18** | 50.00 | 86.86 |
| Joint · BERT | `lcf_scm_cdm_resize` | 79.87 | 69.89 | **57.16** | 86.54 |
| Joint · T5 | `seq_resize` | 79.87 | **71.50** | 50.12 | **89.10** |
| Joint · T5 | `lcf_bip_cdm_resize` | 79.87 | 71.18 | **56.03** | 86.54 |
| Pipeline · BERT | `lcf_seq_cdm_resize` | 59.76 | **53.57** | 40.29 | 86.86 |
| Pipeline · T5 | `seq_resize` | 60.00 | **53.81** | 39.98 | **89.10** |

**Mô hình đề xuất:**
- **Primary:** Joint · T5 · `seq_resize` — Micro F1 = 71.50, Oracle Joint = 89.10
- **Backup:** Joint · BERT · `lcf_seq_cdm_resize` — Micro F1 = 71.18, huấn luyện nhanh hơn ~40%

### Nhận xét

**1. Joint vượt Pipeline ~18 điểm Micro F1.**
ATE trong Pipeline chỉ đạt ~60% F1 (so với 79.87% khi Joint), lỗi lan truyền là nguyên nhân chính. Oracle Joint Acc tương đương giữa hai kiến trúc xác nhận: hạn chế nằm ở bước ATE, không phải classifier.

**2. BERT và T5 cho kết quả tương đương về Micro F1 (~71%).**
T5 nhỉnh hơn về Oracle Joint Acc (89.10 vs 86.86), cho thấy tiềm năng khi ATE được cải thiện. BERT nhanh hơn đáng kể với các biến thể LCF (+40-56%).

**3. CDM tốt hơn CDW về Macro F1.**
LCF-CDM (mask nhị phân) giúp xử lý nhãn thiểu số tốt hơn CDW (weight tuyến tính), dẫn đến Macro F1 cao hơn ~7 điểm.

**4. BRANDING_neutral và FACILITY_neutral: F1 = 0.0 trên toàn bộ cấu hình.**
Phản ánh mất cân bằng nhãn nghiêm trọng trong dữ liệu — không đủ mẫu để học. Đây là vấn đề dữ liệu, không phải lỗi mô hình.

---

## Inference

### Pipeline end-to-end (ATE → APC)

```bash
python common/pipeline_inference.py \
  --ate-checkpoint checkpoints/gas_t5_ate/best \
  --apc-checkpoint-dir runs_joint/lcf_seq_cdm_resize \
  --bert-name bert-base-uncased \
  --clause-split-mode none \
  --sentence "The room was spotless, but the elevator broke down frequently. Staff were incredibly helpful."
```

**Ví dụ đầu ra:**
```json
[
  {"aspect": "room",     "sentiment": "positive", "category": "FACILITY"},
  {"aspect": "elevator", "sentiment": "negative", "category": "FACILITY"},
  {"aspect": "staff",    "sentiment": "positive", "category": "SERVICE"}
]
```

### APC cho một aspect đã biết

```bash
python common/infer_aspect_term.py \
  --apc-checkpoint-dir runs_joint/lcf_seq_cdm_resize \
  --bert-name bert-base-uncased \
  --sentence "The bedroom was clean and very spacious." \
  --aspect "bedroom"
```

### Chế độ tách câu (Clause Splitting)

> Clause splitting **chỉ áp dụng khi inference**, không áp dụng trong huấn luyện.

| Mode | Mô tả | Yêu cầu |
|---|---|---|
| `none` | Dùng toàn bộ câu gốc | — |
| `rulebase` | Tách tại `,` `;` và liên từ đối lập | — |
| `uos` | Tách bằng LLM Qwen 8B (Ollama) | `ollama serve` đang chạy |

```bash
# Khởi động Ollama (nếu dùng mode uos)
ollama serve
ollama pull qwen3:8b
```

Khi Ollama không kết nối được, UOS tự động fallback về câu gốc (không crash).

---

## Web Interface

### Backend (FastAPI)

```bash
# Linux / macOS
ATE_CHECKPOINT=checkpoints/gas_t5_ate/best \
APC_CHECKPOINT_DIR=runs_joint/lcf_scm_cdm_resize \
BERT_NAME=bert-base-uncased \
CLAUSE_SPLIT_MODE=none \
uvicorn server.app:app --host 0.0.0.0 --port 5000

# Windows (Command Prompt)
set ATE_CHECKPOINT=checkpoints/gas_t5_ate/best
set APC_CHECKPOINT_DIR=runs_joint/lcf_scm_cdm_resize
set BERT_NAME=bert-base-uncased
set CLAUSE_SPLIT_MODE=rulebase
uvicorn server.app:app --host 0.0.0.0 --port 5000
```

**API Endpoints:**

| Endpoint | Method | Body | Mô tả |
|---|---|---|---|
| `/predict` | POST | `{"text": "..."}` | Dự đoán một câu → JSON array |
| `/batch_predict` | POST | `file=@file.txt` | Upload `.txt` / `.docx` (mỗi dòng 1 câu) |

**Ví dụ curl:**
```bash
# Single sentence
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The room was clean but the service was slow."}'

# Batch file
curl -X POST http://localhost:5000/batch_predict \
  -F file=@sentences.txt
```

### Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
# Truy cập: http://localhost:5173
```

**Tính năng:**
- Nhập trực tiếp một câu → bảng kết quả (aspect, sentiment, category)
- Upload file `.txt` / `.docx` → kết quả theo từng dòng

---

## Mô-đun bổ sung

### GAS — Generative Aspect Sentiment (một bước)

T5 sinh bộ ba `(aspect, category, sentiment)` trực tiếp, không cần bước ATE riêng.

```bash
python gas/train_gas.py \
  --data-dir dataset \
  --output-dir checkpoints_gas \
  --epochs 20
```

**Định dạng đầu ra:**
```
"(nhân viên, SERVICE, positive); (thang máy, FACILITY, negative)"
```

### UOS — Unit Opinion Sentence

Phân tách câu đánh giá dài thành các đơn vị ngữ nghĩa, mỗi đơn vị chứa đúng một opinion, trước khi đưa vào APC.

```bash
python uos/run_llm_uos_eval.py
```

Chi tiết: [`uos/README.md`](uos/README.md).

### Minh hoạ thuật toán Token Merging

```bash
python scripts/visualize_scm.py            # Từng bước SCM trên chuỗi tổng quát
python scripts/visualize_scm_example.py    # SCM trên câu ví dụ cụ thể
```

Xuất hình vào `thesis/figures/`.

---

## Biến môi trường

| Biến | Mặc định | Mô tả |
|---|---|---|
| `ATE_CHECKPOINT` | `checkpoints/gas_t5_ate/best` | Đường dẫn checkpoint T5 ATE |
| `APC_CHECKPOINT_DIR` | `runs_joint/lcf_scm_cdm_resize` | Thư mục checkpoint BERT APC |
| `BERT_NAME` | `bert-base-uncased` | Tên pretrained BERT model |
| `CLAUSE_SPLIT_MODE` | `none` | Chế độ tách câu: `none` / `rulebase` / `uos` |

---

## Tài liệu khác

- [`thesis/README.md`](thesis/README.md) — hướng dẫn biên dịch luận văn LaTeX (Overleaf)
- [`uos/README.md`](uos/README.md) — chi tiết module UOS
- [`docs/`](docs/) — tài liệu tham khảo (paper gốc, mô tả phương pháp) và PDF luận văn đã biên dịch
- [`notebooks/`](notebooks/) — notebook thăm dò dữ liệu / so sánh kết quả, không thuộc pipeline huấn luyện chính

---

## Tài liệu tham khảo

- **LCF-ATEPC:** Zeng et al. (2019). *LCF: A Local Context Focus Mechanism for Aspect-Based Sentiment Classification.* Applied Sciences.
- **ToMe:** Bolya et al. (2023). *Token Merging: Your ViT But Faster.* CVPR 2023.
- **T5:** Raffel et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* JMLR.
- **GAS:** Zhang et al. (2021). *Towards Generative Aspect-Based Sentiment Analysis.* ACL-IJCNLP 2021.
- **PyABSA:** Yang et al. *PyABSA: A Modularized Framework for Reproducible Aspect-based Sentiment Analysis.*

scm cdm resize uos 