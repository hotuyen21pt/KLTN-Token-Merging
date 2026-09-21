# Chạy hệ thống bằng Docker

Toàn bộ hệ thống (Ollama + Backend FastAPI + Frontend React) được đóng gói qua
`docker compose`. Cấu hình mặc định dùng **PyTorch CPU-only** nên chạy được trên
mọi máy, không cần GPU.

## Yêu cầu
- Docker Desktop (đã bật, có `docker compose`).
- Các checkpoint có sẵn trong repo: `checkpoints/best` và
  `runs_joint/lcf_scm_cdm_resize` (được **mount** vào container, không nằm trong image).

## Chạy

```bash
docker compose up --build
```

Lần đầu sẽ:
1. Build image backend (tải PyTorch CPU + pyabsa + transformers — vài phút).
2. Kéo image Ollama và **pull model `qwen3:8b` (~5GB)** — chỉ 1 lần, lưu vào volume.
3. Tải `bert-base-uncased` từ HuggingFace khi backend khởi động lần đầu (cache lại).

Khi xong, mở trình duyệt: **http://localhost:5173**

| Service   | URL                     | Ghi chú                                   |
|-----------|-------------------------|-------------------------------------------|
| Frontend  | http://localhost:5173   | Giao diện demo (React + Vite)             |
| Backend   | http://localhost:5000   | FastAPI (`/predict`, `/batch_predict`)    |
| Ollama    | http://localhost:11434  | LLM cho UOS (tách câu), model `qwen3:8b`  |

Chạy nền:
```bash
docker compose up --build -d
docker compose logs -f backend    # xem log backend
```

Dừng:
```bash
docker compose down          # giữ lại model & cache
docker compose down -v       # xoá luôn volume (model Ollama, cache HF)
```

## Cấu hình (biến môi trường trong `docker-compose.yml`, service `backend`)

| Biến                 | Mặc định                          | Ý nghĩa                                    |
|----------------------|-----------------------------------|--------------------------------------------|
| `ATE_CHECKPOINT`     | `checkpoints/best`                | Checkpoint T5 ATE                          |
| `APC_CHECKPOINT_DIR` | `runs_joint/lcf_scm_cdm_resize`   | Thư mục checkpoint BERT APC                |
| `BERT_NAME`          | `bert-base-uncased`               | Base model BERT                            |
| `CLAUSE_SPLIT_MODE`  | `uos`                             | `none` / `rulebase` / `uos`                |
| `OLLAMA_HOST`        | `http://ollama:11434`             | Endpoint Ollama (cho UOS)                  |

### Không muốn dùng Ollama/UOS?
Đặt `CLAUSE_SPLIT_MODE=none` cho service `backend`, và có thể bỏ qua các service
`ollama` / `ollama-pull`:

```bash
docker compose up --build backend frontend
```

UOS cũng **tự fallback** về câu gốc nếu không kết nối được Ollama (không crash).

## Ghi chú
- Backend chạy **CPU-only**: inference chậm hơn GPU nhưng ổn định cho demo.
- Model Ollama và cache HuggingFace được lưu trong Docker volume
  (`ollama-data`, `hf-cache`) nên không phải tải lại ở các lần chạy sau.
- Sửa code backend/frontend → chạy lại `docker compose up --build` để build lại.
