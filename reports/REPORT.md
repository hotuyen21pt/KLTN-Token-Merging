# Báo cáo thực nghiệm — ABSA Token Merging

Sinh tự động bởi `run_all.py` lúc 2026-09-21T15:12:38.

- Seeds: `[42, 123, 456]`
- Backbones: `['bert', 't5']`
- Biến thể (21): `baseline`, `lcf_only_cdm`, `lcf_only_cdw`, `bip`, `seq`, `scm`, `lcf_bip_cdm`, `lcf_bip_cdw`, `lcf_seq_cdm`, `lcf_seq_cdw`, `lcf_scm_cdm`, `lcf_scm_cdw`, `lcf_bip_cdm_compact`, `lcf_bip_cdw_compact`, `lcf_seq_cdm_compact`, `lcf_seq_cdw_compact`, `lcf_scm_cdm_compact`, `lcf_scm_cdw_compact`, `lcf_pre_bip`, `lcf_pre_seq`, `lcf_pre_scm`
- ATE F1 dùng cho header bảng: `79.87`

## 1. Trạng thái các stage

| Stage | Trạng thái | Thời gian | Ghi chú |
|---|---|---|---|
| `env` | OK | 0.1 phút |  |
| `report` | OK | 0.0 phút |  |

## 2. Danh mục kết quả (artifact)

| Nhóm | File | Trạng thái | KB | Mô tả |
|---|---|---|---|---|
| Môi trường | `reports/00_environment.txt` | OK | 4.1 | Snapshot môi trường + kế hoạch chạy |
| ATE | `runs_ate/results_ate_multiseed.csv` | THIẾU |  | P/R/F1 ATE theo từng seed |
| ATE | `runs_ate/results_ate_summary.txt` | THIẾU |  | ATE mean±std qua các seed |
| ATE | `runs_ate/seed_*/test_predictions.csv` | THIẾU |  | Prediction ATE theo seed (đầu vào e2e) |
| ATE | `runs_ate/test_ate_predictions.csv` | OK | 26.1 | Prediction ATE dùng chung cho eval e2e |
| APC | `runs_joint/experiment_results_joint.csv` | OK | 0.5 | Toàn bộ metric APC — BERT |
| APC | `runs_joint/experiment_results_joint.txt` | OK | 4.3 | Bảng tổng hợp APC — BERT |
| APC | `runs_joint_t5/experiment_results_joint.csv` | THIẾU |  | Toàn bộ metric APC — T5 |
| APC | `runs_joint_t5/experiment_results_joint.txt` | THIẾU |  | Bảng tổng hợp APC — T5 |
| Multi-seed | `runs_multiseed/thesis_tables.txt` | THIẾU |  | 8 bảng luận văn (oracle, e2e, compact, paper) |
| Multi-seed | `runs_multiseed/results_raw.csv` | THIẾU |  | Kết quả thô từng (backbone, config, seed) |
| Multi-seed | `runs_multiseed/results_aggregated.csv` | THIẾU |  | Kết quả gộp mean±std |
| Multi-seed | `runs_multiseed/results_summary.txt` | THIẾU |  | Tóm tắt lượt chạy multi-seed |
| Oracle | `runs_bert_gold/*/eval_gold_aspects.csv` | THIẾU |  | APC với gold aspect (trần trên) |
| End-to-end | `runs_ate/eval_joint_triplet_BERT.csv` | OK | 4.5 | Eval bộ ba e2e theo backbone |
| End-to-end | `runs_ate/eval_joint_triplet_T5.csv` | OK | 3.4 | Eval bộ ba e2e theo backbone |
| End-to-end | `runs_ate/eval_results_BERT.csv` | OK | 2.9 | Eval trên results.csv theo backbone |
| End-to-end | `runs_ate/eval_results_T5.csv` | OK | 2.9 | Eval trên results.csv theo backbone |
| GAS | `runs_gas/eval_test.json` | THIẾU |  | GAS một bước — metric JSON |
| GAS | `runs_gas/eval_test.csv` | THIẾU |  | GAS một bước — metric CSV |
| UOS | `uos/output/test/metrics.json` | THIẾU |  | Thống kê tách câu UOS |
| UOS | `uos/output/test/results.jsonl` | OK | 0.5 | Kết quả tách câu từng câu |
| Hình | `thesis/figures/scm_example.pdf` | OK | 53.6 | Hình luận văn (PDF) |
| Hình | `thesis/figures/scm_example.png` | OK | 229.5 | Hình luận văn (PNG) |
| Log | `reports/logs/*.log` | THIẾU |  | Log đầy đủ của từng stage |

**11/25 artifact có mặt.** Bản đầy đủ: `reports/manifest.csv`.

## 3. Bảng kết quả chính

### 3.2 Tổng hợp APC — BERT

`runs_joint/experiment_results_joint.txt`

```text

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
EXPERIMENT SUMMARY — Joint training (Sent: main+supp | Cat: main only)
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  Seed: 42  MaxEpochs: 15  Patience: 4  Batch: 16  LR: 2e-05  Device: cuda  UseSupp: True  UseWeights: True

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Configuration               LCF Strategy         Resize  Time(s)  BestEp   Sent-F1   Cat-F1  Joint-F1  SentAcc   CatAcc
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  LCF+Bip (resize)              Y bipartite           yes    720.5       6    69.19%   85.23%    66.15%    91.35%    88.46%
  Configuration               LCF  F1-positive  F1-negative   F1-neutral
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  LCF+Bip (resize)              Y       95.27%       67.86%       44.44%

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Configuration               LCF F1-AMENITY F1-BRANDIN F1-EXPERIE F1-FACILIT F1-LOYALTY F1-SERVICE
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  LCF+Bip (resize)              Y     90.00%     72.73%     73.33%     85.85%     92.31%     97.18%

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  Sent training : main .apc + supplement (negative.tsv + neutral.tsv)
  Cat  training : main .apc only — supplement masked out via is_supplement
  bipartite → ToMe CVPR 2023 | sequential_local → new neighbour merge
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

```

### 3.7 Môi trường chạy

`reports/00_environment.txt`

```text
Thời điểm     : 2026-09-21T15:10:31
Repo root     : C:/Users/User1/Desktop/Linh tin/KLTN-Token-Merging
Python        : 3.14.3  (C:/msys64/mingw64/bin/python.exe)
Platform      : Windows-11-10.0.26200-SP0

  torch         : KHÔNG CÀI ĐƯỢC (ModuleNotFoundError)
  transformers  : KHÔNG CÀI ĐƯỢC (ModuleNotFoundError)
  sklearn       : KHÔNG CÀI ĐƯỢC (ModuleNotFoundError)
  numpy         : KHÔNG CÀI ĐƯỢC (ModuleNotFoundError)
  pandas        : KHÔNG CÀI ĐƯỢC (ModuleNotFoundError)
  matplotlib    : KHÔNG CÀI ĐƯỢC (ModuleNotFoundError)
  seaborn       : KHÔNG CÀI ĐƯỢC (ModuleNotFoundError)
  Levenshtein   : KHÔNG CÀI ĐƯỢC (ModuleNotFoundError)
  fastapi       : 0.135.1
  tqdm          : KHÔNG CÀI ĐƯỢC (ModuleNotFoundError)

CUDA check lỗi: No module named 'torch'

Dữ liệu:
  [OK]   dataset/train.apc                         227.0 KB  9792 dòng
  [OK]   dataset/dev.apc                            27.2 KB  1216 dòng
  [OK]   dataset/test.apc                           27.4 KB  1248 dòng
  [OK]   dataset/test_sentences_id.csv              30.0 KB  313 dòng
  [OK]   dataset/supplement/negative.tsv            73.5 KB  664 dòng
  [OK]   dataset/supplement/neutral.tsv             10.4 KB  115 dòng
  [OK]   results.csv                                69.9 KB  613 dòng

ATE checkpoint hiện có : C:/Users/User1/Desktop/Linh tin/KLTN-Token-Merging/checkpoints/best
GAS checkpoint hiện có : KHÔNG CÓ (stage `gas` sẽ train mới)
APC checkpoint BERT  : 1 cấu hình đã có → ['lcf_scm_cdm_resize']
APC checkpoint T5    : 0 cấu hình đã có → []
Ollama (cho UOS)       : đang chạy

Kế hoạch:
  Seeds     : [42, 123, 456]
  Backbones : ['bert', 't5']
  Biến thể  : 21 (resize, compact, pretome)
    - baseline               Base                   lcf=0 cdm=0 tome=0 resize=1 pre=0 strategy=bipartite
    - lcf_only_cdm           CDM                    lcf=1 cdm=1 tome=0 resize=1 pre=0 strategy=bipartite
    - lcf_only_cdw           CDW                    lcf=1 cdm=0 tome=0 resize=1 pre=0 strategy=bipartite
    - bip                    BiToMe                 lcf=0 cdm=0 tome=1 resize=1 pre=0 strategy=bipartite
    - seq                    SLM                    lcf=0 cdm=0 tome=1 resize=1 pre=0 strategy=sequential_local
    - scm                    SCM                    lcf=0 cdm=0 tome=1 resize=1 pre=0 strategy=sequential_cosine
    - lcf_bip_cdm            BiToMe+CDM             lcf=1 cdm=1 tome=1 resize=1 pre=0 strategy=bipartite
    - lcf_bip_cdw            BiToMe+CDW             lcf=1 cdm=0 tome=1 resize=1 pre=0 strategy=bipartite
    - lcf_seq_cdm            SLM+CDM                lcf=1 cdm=1 tome=1 resize=1 pre=0 strategy=sequential_local
    - lcf_seq_cdw            SLM+CDW                lcf=1 cdm=0 tome=1 resize=1 pre=0 strategy=sequential_local
    - lcf_scm_cdm            SCM+CDM                lcf=1 cdm=1 tome=1 resize=1 pre=0 strategy=sequential_cosine
    - lcf_scm_cdw            SCM+CDW                lcf=1 cdm=0 tome=1 resize=1 pre=0 strategy=sequential_cosine
    - lcf_bip_cdm_compact    BiToMe+CDM(compact)    lcf=1 cdm=1 tome=1 resize=0 pre=0 strategy=bipartite
    - lcf_bip_cdw_compact    BiToMe+CDW(compact)    lcf=1 cdm=0 tome=1 resize=0 pre=0 strategy=bipartite
    - lcf_seq_cdm_compact    SLM+CDM(compact)       lcf=1 cdm=1 tome=1 resize=0 pre=0 strategy=sequential_local
    - lcf_seq_cdw_compact    SLM+CDW(compact)       lcf=1 cdm=0 tome=1 resize=0 pre=0 strategy=sequential_local
    - lcf_scm_cdm_compact    SCM+CDM(compact)       lcf=1 cdm=1 tome=1 resize=0 pre=0 strategy=sequential_cosine
    - lcf_scm_cdw_compact    SCM+CDW(compact)       lcf=1 cdm=0 tome=1 resize=0 pre=0 strategy=sequential_cosine
    - lcf_pre_bip            LCF+PreBip             lcf=1 cdm=1 tome=0 resize=0 pre=1 strategy=bipartite
    - lcf_pre_seq            LCF+PreSLM             lcf=1 cdm=1 tome=0 resize=0 pre=1 strategy=sequential_local
    - lcf_pre_scm            LCF+PreSCM             lcf=1 cdm=1 tome=0 resize=0 pre=1 strategy=sequential_cosine
  Tổng số lần train APC (stage multiseed): 126
```

## 4. Tái tạo

```bash
python run_all.py --list          # xem stage + biến thể
python run_all.py --dry-run       # xem lệnh sẽ chạy
python run_all.py --resume        # chạy tiếp, bỏ qua phần đã xong
python run_all.py --stages report # chỉ sinh lại báo cáo này
```

Log chi tiết từng stage: `reports/logs/<stage>.log`.

