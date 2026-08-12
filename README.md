# NeuroForge APEX
### Adaptive Progressive EXpert ML Architecture Design Agent

A production-grade, multi-agent AI system that designs ML, Deep Learning, and Reinforcement Learning architectures from natural language problem statements. Powered by Mistral (or NVIDIA NIM) free-tier APIs.

---

## Architecture

```
User Request
    │
    ▼
┌─────────────────────────────────────────────────┐
│             APEX PIPELINE                       │
│                                                 │
│  1. Orchestrator  — parses & plans              │
│  2. Analyzer      — decomposes problem          │
│  3. Architect     — designs DNA candidates      │
│  4. Optimizer     — hyperparameter tuning plan  │
│  5. Coder         — generates Python code       │
│  6. Validator     — catches anti-patterns       │
│  7. Explainer     — writes design document      │
│                                                 │
│  Cognitive Memory Graph (shared across agents)  │
└─────────────────────────────────────────────────┘
    │
    ▼
ArchitectureDNA + Code + Design Doc + Training Guide
```

---

## Quick Start

### 1. Install
```bash
cd neuroforge
pip install -r requirements.txt
```

### 2. Configure (already set with your Mistral key)
```bash
# .env is pre-configured — no changes needed
cat .env
```

### 3. Run API Server
```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
API docs: http://localhost:8000/docs

### 4. Use CLI
```bash
# Full design
python cli.py design "Design a transformer to detect fraud in transaction sequences"

# With options
python cli.py design "Build a RL agent for stock trading" \
  --depth complete \
  --framework PyTorch \
  --compare 2 \
  --save-code model.py \
  --save-doc design.md

# Analyze dataset
python cli.py analyze ./data/train.csv --target label

# Interactive REPL
python cli.py interactive
```

### 5. Docker
```bash
docker-compose up --build
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/design` | Design architecture (full pipeline) |
| POST | `/api/v1/analyze-dataset` | Profile a local dataset |
| POST | `/api/v1/compare` | Compare candidate architectures |
| GET  | `/api/v1/session/{id}` | Get session state |
| WS   | `/api/v1/design/stream` | Streaming WebSocket |
| GET  | `/api/v1/health` | Health check |

### Example: Design Request
```bash
curl -X POST http://localhost:8000/api/v1/design \
  -H "Content-Type: application/json" \
  -d '{
    "problem_statement": "Design a vision transformer for detecting tumors in CT scans with 3000 labeled samples",
    "depth": "complete",
    "preferred_framework": "PyTorch",
    "compare_count": 2,
    "constraints": "Must run on single A100, latency under 200ms"
  }'
```

### Example: Dataset Analysis
```bash
curl -X POST http://localhost:8000/api/v1/analyze-dataset \
  -H "Content-Type: application/json" \
  -d '{"dataset_path": "./data/train.csv", "target_column": "label"}'
```

---

## Output Depth Levels

| Depth | Agents | Output |
|-------|--------|--------|
| `sketch` | Orchestrator only | Domain + family identification |
| `detailed` | + Analyzer + Architect | Full architecture DNA + training spec |
| `complete` | Full pipeline | DNA + Code + Design Doc + Training Guide |
| `research` | Full pipeline + critique loops | + Novelty analysis + publication angle |

---

## Supported Domains

- Classical ML (XGBoost, SVM, GP, ensemble)
- Computer Vision (CNN, ViT, Swin, DINO, EfficientNet)
- NLP (BERT, T5, GPT, multilingual)
- Deep Learning (hybrid, VAE, GAN, Diffusion, Mamba/SSM)
- Reinforcement Learning (DQN, PPO, SAC, MARL, DREAMER)
- LLM Fine-tuning (LoRA, QLoRA, DPO, RLHF, PEFT)
- Graph Neural Networks (GCN, GAT, GraphSAGE)
- Time Series (TCN, Temporal Transformer, N-BEATS)
- Multimodal (CLIP, Flamingo, LLaVA-style)

---

## Run Tests
```bash
python test_neuroforge.py --quick           # Single fast test (T01)
python test_neuroforge.py --tests T01 T03  # Specific tests
python test_neuroforge.py                  # All 5 tests
```

---

## File Structure
```
neuroforge/
├── main.py                  # FastAPI app entrypoint
├── cli.py                   # Rich terminal CLI
├── config.py                # Settings & provider config
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
├── test_neuroforge.py
├── .env                     # Pre-configured with your key
├── agents/
│   ├── base.py              # Abstract agent + LLM client
│   ├── orchestrator.py      # Master controller
│   ├── analyzer.py          # Problem decomposer
│   ├── architect.py         # Architecture designer
│   ├── coder.py             # Code generator
│   ├── optimizer.py         # HP optimizer
│   ├── validator.py         # Anti-pattern checker
│   └── explainer.py         # Documentation writer
├── core/
│   ├── memory.py            # Cognitive Memory Graph
│   ├── pipeline.py          # APEX Pipeline executor
│   └── prompts.py           # Expert system prompts
├── models/
│   └── schemas.py           # All Pydantic data models
├── api/
│   └── routes.py            # FastAPI route handlers
└── utils/
    └── dataset_inspector.py # Local dataset profiler
```

---

## Provider Configuration

### Mistral (current — free tier)
- Free key: https://console.mistral.ai/
- Primary model: `mistral-large-latest`
- Code model: `codestral-latest`

### NVIDIA NIM (alternative — 250 free credits)
- Free key: https://build.nvidia.com/
- Set `LLM_PROVIDER=nvidia` in `.env`
- Primary model: `meta/llama-3.1-70b-instruct`
