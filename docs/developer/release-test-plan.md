---
weight: 0
---

# Release Test Plan

Manual checklist for verifying a GuideLLM release candidate. The goal is reasonable confidence that featured paths work end-to-end and that known regression hotspots have not resurfaced.

This plan **complements** automated gates (`tox`, unit/integration/e2e). It does not replace them. Prefer short, constrained runs (`max_requests` / short `max_duration`) unless a section says otherwise.

**Out of scope:** WEKA / OTEL trace replay (still under active development).

## Prerequisites

Install a release candidate (or local checkout) with the extras you will exercise:

```bash
# From PyPI / wheel under test
pip install "guidellm[recommended,vision,audio,plot]"

# Or from a local checkout
uv sync --extra recommended --extra vision --extra audio --extra plot
```

Commands below use `guidellm …`. From a local checkout, prefer `uv run guidellm …`.

| Need                                     | Notes                                                                           |
| ---------------------------------------- | ------------------------------------------------------------------------------- |
| OpenAI-compatible HTTP server            | Most `openai_http` sections (chat, ASR, embeddings, VLM)                        |
| vLLM (or compatible) with `/v1/realtime` | Websocket audio transcription                                                   |
| Hugging Face Hub access                  | HF dataset sections (cache warmed helps)                                        |
| In-process vLLM install                  | `vllm_python` / gated `vllm_offline`                                            |
| Results workspace                        | Use a dedicated temp directory; set `GUIDELLM__DEFAULT_RESULTS_DIR` where noted |

Suggested servers / models (adjust to what you have available):

| Role         | Example                                         |
| ------------ | ----------------------------------------------- |
| Chat / text  | Any chat-completions model on `:8000`           |
| VLM          | `vllm serve Qwen/Qwen3-VL-2B-Instruct`          |
| ASR (HTTP)   | `vllm serve openai/whisper-small`               |
| Embeddings   | `vllm serve BAAI/bge-small-en-v1.5 --port 8000` |
| In-line vLLM | Small model e.g. `Qwen/Qwen3-0.6B`              |

## Baseline automated gate

Run before the manual matrix (from a clone of the release tag / candidate):

```bash
tox -e lint-check
tox -e type-check
tox -e tests
# Optional but recommended when infrastructure allows:
# docker build . -f tests/e2e/vllm-sim.Dockerfile -o type=local,dest=./
# tox -e test-e2e
```

**Pass:** all selected tox environments exit 0.

______________________________________________________________________

## Cross-cutting checks

### 1. `api_key` / `SecretStr` (regression hotspot)

`api_key` is stored as Pydantic `SecretStr`. This has broken in several forms: key not forwarded on the wire, or raw secret leaking into reports.

**Setup:** Any HTTP endpoint that records request headers (mock server, proxy, or `guidellm mock-server` plus a packet/header inspector). An authenticated backend is **not** required—only that the `Authorization: Bearer …` header is present when `api_key` is set.

**CLI path:**

```bash
export RESULTS_DIR="$(mktemp -d)"
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,api_key=sk-release-test-secret-do-not-leak \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=5 \
  --data kind=synthetic_text,prompt_tokens=64,output_tokens=16 \
  --output kind=json,path="${RESULTS_DIR}/api_key.json" \
  --output kind=csv,path="${RESULTS_DIR}/api_key.csv" \
  --output kind=html,path="${RESULTS_DIR}/api_key.html" \
  --output kind=console
```

**Env path (also verify):**

```bash
GUIDELLM__SPEC__BACKEND__API_KEY=sk-env-secret-do-not-leak \
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=5 \
  --data kind=synthetic_text,prompt_tokens=64,output_tokens=16 \
  --output kind=json,path="${RESULTS_DIR}/api_key_env.json"
```

**Pass criteria:**

- Outbound HTTP includes `Authorization: Bearer sk-release-test-secret-do-not-leak` (CLI) / env secret (env path).
- JSON / CSV / HTML / console output **do not** contain the raw secret string (value must be masked or absent).
- Benchmark completes without auth-related client crashes when the server ignores the header.

**Fail signals:** missing bearer header; `sk-…` appearing in any report artifact; serialization dumping `get_secret_value()` plaintext.

### 2. `GUIDELLM__DEFAULT_RESULTS_DIR`

```bash
export GUIDELLM__DEFAULT_RESULTS_DIR="$(mktemp -d)/guidellm-results"
mkdir -p "${GUIDELLM__DEFAULT_RESULTS_DIR}"

guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=5 \
  --data kind=synthetic_text,prompt_tokens=64,output_tokens=16 \
  --output kind=json \
  --output kind=csv \
  --output kind=html
```

**Pass criteria:** `benchmarks.json`, `benchmarks.csv`, and `benchmarks.html` are created under `$GUIDELLM__DEFAULT_RESULTS_DIR`.

**Note:** `kind=plot` defaults to `./benchmarks.png` and does **not** currently use `GUIDELLM__DEFAULT_RESULTS_DIR`. When testing plot, pass an explicit `path=`.

### 3. Output matrix (`run` and `from-file`)

Exercise `console`, `json`, `csv`, `html`, and `plot` for both entrypoints.

**A. `guidellm run`**

```bash
export OUT="$(mktemp -d)"

guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=10 \
  --data kind=synthetic_text,prompt_tokens=128,output_tokens=32 \
  --output kind=console \
  --output kind=json,path="${OUT}/run.json" \
  --output kind=csv,path="${OUT}/run.csv" \
  --output kind=html,path="${OUT}/run.html" \
  --output kind=plot,path="${OUT}/run.png",dpi=100
```

**B. `guidellm benchmark from-file`**

```bash
guidellm benchmark from-file "${OUT}/run.json" \
  --output kind=console \
  --output kind=json,path="${OUT}/from_file.json" \
  --output kind=csv,path="${OUT}/from_file.csv" \
  --output kind=html,path="${OUT}/from_file.html" \
  --output kind=plot,path="${OUT}/from_file.png",dpi=100
```

**Pass criteria:**

| Kind      | Expect                                                 |
| --------- | ------------------------------------------------------ |
| `console` | Metadata / info / stats tables print without traceback |
| `json`    | Valid JSON; contains benchmarks and metrics            |
| `csv`     | Non-empty summary rows                                 |
| `html`    | Opens in a browser; tables/charts render               |
| `plot`    | Image file written (requires `guidellm[plot]`)         |

**Fail signals:** empty files, unserializeable JSON, HTML/plot exceptions, `from-file` unable to load a report just produced by `run`.

______________________________________________________________________

## CLI surface

### 4. `guidellm benchmark from-file` (functional)

Covered in depth by the output matrix above. Additionally confirm defaults:

```bash
# Uses PATH default ./benchmarks.json if present; prefer explicit path
guidellm benchmark from-file "${OUT}/run.json"
```

**Pass:** exits 0; default outputs (`console`, `json`, `html`, `csv`) write using type defaults / `GUIDELLM__DEFAULT_RESULTS_DIR` as applicable.

### 5. `guidellm preprocess dataset`

Create a tiny JSONL input, then preprocess:

```bash
export PRE="$(mktemp -d)"
cat > "${PRE}/input.jsonl" <<'EOF'
{"prompt": "What is 2+2?", "output": "4"}
{"prompt": "Name a primary color.", "output": "Red"}
EOF

guidellm preprocess dataset \
  kind=json_file,path="${PRE}/input.jsonl" \
  "${PRE}/processed.jsonl" \
  --tokenizer kind=huggingface_auto,model=gpt2 \
  --strategy kind=ignore,prompt_tokens=64,output_tokens=16,prefix_tokens_max=8
```

**Pass criteria:** `${PRE}/processed.jsonl` exists, is non-empty, and parses as JSONL. No crash from tokenizer / strategy resolution.

See [Datasets — Preprocessing](../guides/datasets.md#preprocessing-datasets) for strategy kinds (`ignore`, `concatenate`, `pad`, `error`).

______________________________________________________________________

## Synthetic datasets

Keep constraints small. Vision extras required for image/video.

### 6. Synthetic text

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=20 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --output kind=json,path=results/synthetic_text.json
```

**Pass:** completed requests > 0; token stats present; no deserializer errors.

### 7. Synthetic image (+ text)

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=5 \
  --data kind=synthetic_text,prompt_tokens=64 \
  --data kind=synthetic_image,width=128,height=128,format=jpeg,output_tokens=32 \
  --output kind=json,path=results/synthetic_image.json
```

**Pass:** multimodal requests complete against a VLM (or mock that accepts image payloads). See [Synthetic Visual Data](../guides/multimodal/synthetic_vision.md).

### 8. Synthetic video (+ text)

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=3 \
  --data kind=synthetic_text,prompt_tokens=32 \
  --data kind=synthetic_video,width=160,height=120,frames=4,fps=1,output_tokens=16 \
  --output kind=json,path=results/synthetic_video.json
```

**Pass:** same as image, with video payloads. Longer encode time is expected; keep frame counts low for release smoke.

______________________________________________________________________

## Hugging Face datasets

Requires Hub (or cached) access and matching server modality.

### 9. HF text

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=10 \
  --data '{"kind":"huggingface","source":"openai/gsm8k","load_kwargs":{"name":"main","split":"test"}}' \
  --output kind=json,path=results/hf_text.json
```

Alternate smoke source: `garage-bAInd/Open-Platypus`.

**Pass:** dataset loads; column mapping auto-detects or completes; requests succeed.

### 10. HF audio (HTTP transcription)

```bash
# Server example: vllm serve openai/whisper-small
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,request_format=/v1/audio/transcriptions \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=5 \
  --data '{"kind":"huggingface","source":"openslr/librispeech_asr","load_kwargs":{"name":"clean","split":"test"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"audio_column":"audio"}}' \
  --output kind=json,path=results/hf_audio.json
```

**Pass:** audio metrics / completed transcriptions; requires `guidellm[audio]`. See [Audio Guide](../guides/multimodal/audio.md).

### 11. HF visual (image)

```bash
# Server example: vllm serve Qwen/Qwen3-VL-2B-Instruct
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,request_format=/v1/chat/completions \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=5 \
  --data '{"kind":"huggingface","source":"lmms-lab/MMBench_EN","load_kwargs":{"split":"test"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"image_column":"image","text_column":"question"}}' \
  --output kind=json,path=results/hf_image.json
```

**Pass:** VQA-style requests complete. See [Image Guide](../guides/multimodal/image.md).

### 12. HF visual (video)

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,request_format=/v1/chat/completions \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=3 \
  --data '{"kind":"huggingface","source":"lmms-lab/Video-MME","load_kwargs":{"split":"test"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"video_column":"url","text_column":"question"}}' \
  --output kind=json,path=results/hf_video.json
```

**Pass:** video QA requests complete (URL fetch must be reachable from the client and/or server as configured). See [Video Guide](../guides/multimodal/video.md).

______________________________________________________________________

## Backends and APIs

### 13. Websocket audio transcription

Backend: `openai_websocket` with `request_format=/v1/realtime` (vLLM-style realtime WebSocket). Requires `guidellm[audio]` and a server that speaks `/v1/realtime`.

```bash
guidellm run \
  --backend kind=openai_websocket,target=http://localhost:8000,model=openai/whisper-large-v3,request_format=/v1/realtime \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=3 \
  --data '{"kind":"huggingface","source":"openslr/librispeech_asr","load_kwargs":{"name":"clean","split":"test"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"audio_column":"audio"}}' \
  --output kind=json,path=results/ws_realtime.json
```

**Pass:** WebSocket session opens; audio streams; transcription responses / metrics recorded; process exits cleanly.

**Note:** Automated e2e coverage today uses an in-process stub (`tests/e2e/test_realtime_ws_e2e.py`). This manual item is the live-server confidence gate. User-facing docs for websocket ASR are still thin—treat command success as the primary signal.

### 14. Embeddings API

```bash
# Server example: vllm serve BAAI/bge-small-en-v1.5 --port 8000
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000/v1,model=BAAI/bge-small-en-v1.5,request_format=/v1/embeddings \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=50 \
  --data kind=synthetic_text,prompt_tokens=128 \
  --output kind=json,path=results/embeddings.json
```

Omit `output_tokens` on synthetic text. **Pass:** latency/throughput metrics present; no streaming/TTFT expectations; completed embeddings requests. See [Embeddings Guide](../guides/embeddings.md).

### 15. In-line vLLM Python (normal)

Uses `AsyncLLMEngine` in-process—no HTTP `target`. Requires a vLLM-capable environment (see [vLLM Python backend](../guides/vllm-python-backend.md)).

```bash
guidellm run \
  --backend kind=vllm_python,model=Qwen/Qwen3-0.6B \
  --data kind=synthetic_text,prompt_tokens=128,output_tokens=32 \
  --profile kind=constant,rate=2 \
  --constraint kind=max_duration,seconds=30 \
  --output kind=json,path=results/vllm_python.json
```

Optional engine knobs:

```bash
--backend '{"kind":"vllm_python","model":"Qwen/Qwen3-0.6B","vllm_config":{"gpu_memory_utilization":0.8,"max_model_len":4096}}'
```

**Pass:** engine starts; requests complete; results written. On CPU-only hosts, allow longer duration / smaller model.

### 16. In-line vLLM offline batch (`vllm_offline`) — gated

**Skip unless the release under test includes the `vllm_offline` backend** (offline micro-batching via sync `LLM.generate()`, historically developed on `feat/vllm-offline-batching-backend`).

When present:

```bash
guidellm run \
  --backend kind=vllm_offline,model=Qwen/Qwen3-0.6B,batch_size=8 \
  --data kind=synthetic_text,prompt_tokens=128,output_tokens=32 \
  --profile kind=constant,rate=2 \
  --constraint kind=max_duration,seconds=30 \
  --output kind=json,path=results/vllm_offline.json
```

**Pass:** backend registers and runs; batching completes without streaming; JSON report written. **If backend is missing:** mark **N/A** in the sign-off table—do not fail the release solely for this item.

______________________________________________________________________

## Trace replay

### 17. Optional prelude: `trace_synthetic`

Cheap scheduler smoke before Mooncake:

```bash
export TRACE="$(mktemp -d)/trace.jsonl"
cat > "${TRACE}" <<'EOF'
{"timestamp": 0.0, "input_length": 64, "output_length": 16}
{"timestamp": 0.5, "input_length": 128, "output_length": 32}
{"timestamp": 1.0, "input_length": 96, "output_length": 24}
EOF

guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=trace_synthetic,path="${TRACE}" \
  --profile kind=replay,time_scale=1.0 \
  --output kind=json,path=results/trace_synthetic.json
```

**Pass:** requests follow relative timestamps; completes without scheduler errors.

### 18. Mooncake

Create a minimal Mooncake-format trace (see [Trace File Formats](../guides/trace_replay.md)):

```bash
export MOONCAKE="$(mktemp -d)/mooncake.jsonl"
cat > "${MOONCAKE}" <<'EOF'
{"timestamp": 0.0, "input_length": 10, "output_length": 5, "hash_ids": [0]}
{"timestamp": 0.2, "input_length": 20, "output_length": 8, "hash_ids": [0, 1]}
{"timestamp": 0.4, "input_length": 15, "output_length": 6, "hash_ids": [2]}
EOF

guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=mooncake,path="${MOONCAKE}" \
  --profile kind=replay,time_scale=1.0 \
  --output kind=json,path=results/mooncake.json
```

Optional column overrides: `hash_ids_column`, `hash_id_block_size` (default `512`), plus the shared `timestamp_column` / `prompt_tokens_column` / `output_tokens_column`.

**Pass:** deserializer accepts rows; prompts generated from hash IDs; replay profile schedules by timestamp; benchmark finishes successfully.

**Not in this plan:** WEKA (and OTEL) trace formats—exclude until shipped.

______________________________________________________________________

## Sign-off

| #   | Item                                               | Status (Pass / Fail / N/A) | Tester | Notes                  |
| --- | -------------------------------------------------- | -------------------------- | ------ | ---------------------- |
| —   | Baseline tox (`lint-check`, `type-check`, `tests`) |                            |        |                        |
| —   | Optional `tox -e test-e2e`                         |                            |        |                        |
| 1   | `api_key` / `SecretStr` (wire + reports)           |                            |        |                        |
| 2   | `GUIDELLM__DEFAULT_RESULTS_DIR`                    |                            |        |                        |
| 3   | Output matrix `run` (console/json/csv/html/plot)   |                            |        |                        |
| 3b  | Output matrix `from-file`                          |                            |        |                        |
| 4   | `benchmark from-file` defaults                     |                            |        |                        |
| 5   | `preprocess dataset`                               |                            |        |                        |
| 6   | Synthetic text                                     |                            |        |                        |
| 7   | Synthetic image                                    |                            |        |                        |
| 8   | Synthetic video                                    |                            |        |                        |
| 9   | HF text                                            |                            |        |                        |
| 10  | HF audio (HTTP transcription)                      |                            |        |                        |
| 11  | HF image                                           |                            |        |                        |
| 12  | HF video                                           |                            |        |                        |
| 13  | Websocket realtime transcription                   |                            |        |                        |
| 14  | Embeddings API                                     |                            |        |                        |
| 15  | `vllm_python` (normal)                             |                            |        |                        |
| 16  | `vllm_offline` (batch)                             |                            |        | Skip if not in release |
| 17  | `trace_synthetic` (optional)                       |                            |        |                        |
| 18  | Mooncake replay                                    |                            |        |                        |

**Release candidate:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ **Date:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ **Overall verdict:** Pass / Fail

______________________________________________________________________

## Known gaps (context for testers)

- `kind=plot` is implemented but not yet documented in the outputs guide; always pass an explicit `path=` for release runs.
- Websocket realtime ASR has little user documentation; rely on this checklist and code/tests.
- WEKA / OTEL trace replay is intentionally excluded until implementation lands.
- `vllm_offline` may be absent from the candidate—record N/A rather than Fail.
- Hugging Face multimodal paths are thinly covered by automated e2e; this manual matrix is the primary confidence gate for Hub datasets.
