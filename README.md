# NeuroForge APEX

**Adaptive Progressive EXpert ML Architecture Design Agent**

NeuroForge APEX is a multi-agent system that turns a natural-language problem description into a complete machine learning, deep learning, or reinforcement learning architecture. It runs on free-tier APIs from Mistral or NVIDIA NIM.

## Architecture

```
User request
      |
      v
  APEX Pipeline
  ------------------------------------------------
  1. Orchestrator: parses the request and plans
  2. Analyzer: decomposes the problem
  3. Architect: designs candidate architectures
  4. Optimizer: builds a hyperparameter tuning plan
  5. Coder: generates Python code
  6. Validator: checks for anti-patterns
  7. Explainer: writes the design document
  ------------------------------------------------
  Cognitive Memory Graph (shared across all agents)
      |
      v
Architecture DNA + Code + Design Doc + Training Guide
```

## Quick start

### 1. Install

```bash
cd neuroforge
pip install -r requirements.txt
```

### 2. Configure

The `.env` file is already set up with your Mistral key, so no changes are needed:

```bash
cat .env
```

### 3. Run the API server

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs are available at http://localhost:8000/docs.

### 4. Use the CLI

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

# Analyze a dataset
python cli.py analyze ./data/train.csv --target label

# Interactive REPL
python cli.py interactive
```

### 5. Docker

```bash
docker-compose up --build
```

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/design` | Design an architecture using the full pipeline |
| POST | `/api/v1/analyze-dataset` | Profile a local dataset |
| POST | `/api/v1/compare` | Compare candidate architectures |
| GET | `/api/v1/session/{id}` | Get session state |
| WS | `/api/v1/design/stream` | Streaming WebSocket |
| GET | `/api/v1/health` | Health check |

### Example: design request

```bash
curl -X POST http://localhost:8000/api/v1/design \
  -H "Content-Type: application/json" \
  -d '{
    "problem_statement": "Design a vision transformer for detecting tumors in CT scans with 3000 labeled samples",
    "depth": "complete",
    "preferred_framework": "PyTorch",
    "compare_count": 2,
    "constraints": "Must run on a single A100, latency under 200ms"
  }'
```

### Example: dataset analysis

```bash
curl -X POST http://localhost:8000/api/v1/analyze-dataset \
  -H "Content-Type: application/json" \
  -d '{"dataset_path": "./data/train.csv", "target_column": "label"}'
```

## Output depth levels

| Depth | Agents involved | Output |
|-------|------------------|--------|
| `sketch` | Orchestrator only | Domain and family identification |
| `detailed` | Orchestrator, Analyzer, Architect | Full architecture DNA and training spec |
| `complete` | Full pipeline | DNA, code, design document, training guide |
| `research` | Full pipeline, with critique loops | Adds novelty analysis and a publication angle |

## Supported domains

- Classical ML: XGBoost, SVM, Gaussian processes, ensemble methods
- Computer vision: CNN, ViT, Swin, DINO, EfficientNet
- NLP: BERT, T5, GPT, multilingual models
- Deep learning: hybrid architectures, VAE, GAN, diffusion models, Mamba and other SSM variants
- Reinforcement learning: DQN, PPO, SAC, MARL, DREAMER
- LLM fine-tuning: LoRA, QLoRA, DPO, RLHF, PEFT
- Graph neural networks: GCN, GAT, GraphSAGE
- Time series: TCN, temporal transformer, N-BEATS
- Multimodal: CLIP, Flamingo, LLaVA-style models

## Running tests

```bash
python test_neuroforge.py --quick          # single fast test (T01)
python test_neuroforge.py --tests T01 T03  # specific tests
python test_neuroforge.py                  # all five tests
```

By Ajay Khadke, Royal Holloway, University of London (RA)
