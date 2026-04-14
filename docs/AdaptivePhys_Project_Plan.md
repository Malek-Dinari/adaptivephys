# AdaptivePhys — RL-Augmented rPPG Vital Signs System

## Project Plan, Tech Stack, MCP Rules & Impact Framework

---

## 1. Project Vision

**One-liner:** An open-source, Docker-containerized rPPG toolbox that uses reinforcement learning to continuously adapt vital-sign extraction models to new subjects, environments, and cameras — achieving clinically-relevant accuracy improvements without manual retraining.

**Why it matters:** Current rPPG models degrade when deployed outside their training distribution (different skin tones, lighting, cameras, motion). AdaptivePhys treats this as a continual learning problem, using RL to close the domain gap at inference time.

---

## 2. First Experiment — Proof of Concept

**Goal:** Demonstrate that RL-based online adaptation improves EfficientPhys HR estimation on an unseen dataset, compared to the frozen pretrained model.

### Setup
- **Train** EfficientPhys on UBFC-rPPG (42 subjects, controlled lab conditions)
- **Evaluate frozen model** on PURE dataset (10 subjects, different camera/lighting) → record baseline HR MAE, RMSE, Pearson r
- **Enable RL adaptation loop:** for each PURE video, run N adaptation steps (policy gradient on last 2 conv layers), then re-evaluate
- **Reward signal (self-supervised):** composite of spectral SNR (power at dominant HR frequency vs noise floor), peak sharpness (kurtosis of PSD around HR peak), and temporal consistency (smoothness of inter-beat intervals)

### Expected outcome
- Baseline cross-dataset HR MAE: ~6–8 BPM (typical for EfficientPhys on unseen data)
- Post-adaptation HR MAE: ~4–6 BPM (target 15–25% relative reduction)
- This single experiment proves the core thesis and becomes the anchor metric for everything else

### What to measure
- HR MAE, RMSE, Pearson r (before vs after adaptation, per-subject)
- Number of adaptation steps to convergence
- Inference latency overhead from RL loop (target: <5ms additional per frame)
- Signal SNR improvement (dB) pre/post adaptation

---

## 3. Tech Stack

### Core ML
- **Python 3.11+** — primary language
- **PyTorch 2.x** — model training and inference (with `torch.compile` for speed)
- **EfficientPhys** — primary lightweight rPPG model (TSCAN-lite as secondary baseline)
- **MediaPipe Face Mesh** — real-time face detection and ROI extraction (fastest option)
- **scipy.signal** — rPPG signal processing (bandpass filtering, PSD, peak detection)
- **numpy, pandas** — data handling

### Reinforcement Learning
- **Stable-Baselines3** or custom policy gradient — RL adaptation engine
- **Gymnasium** — environment wrapper for the adaptation loop
- **torch.optim** — SGD/Adam for policy gradient fine-tuning of last layers

### Infrastructure
- **Docker + Docker Compose** — containerized services (inference, RL trainer, dashboard, Redis)
- **Redis** — signal buffer and reward queue between inference and RL containers
- **FastAPI** — REST API for the inference engine
- **Streamlit** — live monitoring dashboard (HR waveform, adaptation metrics, webcam feed)
- **Prometheus + Grafana** (optional) — latency and throughput monitoring

### Testing & CI/CD
- **pytest** — unit and integration tests
- **GitHub Actions** — CI pipeline (lint, test, build Docker images)
- **pre-commit** — ruff linter, black formatter, mypy type checking
- **DVC** (Data Version Control) — dataset and model artifact tracking

### Documentation
- **MkDocs + Material theme** — project documentation site
- **Jupyter notebooks** — experiment notebooks for benchmarking results

---

## 4. Repository Structure

```
adaptivephys/
├── docker/
│   ├── Dockerfile.inference        # Lightweight inference container
│   ├── Dockerfile.trainer          # RL training container (GPU)
│   ├── Dockerfile.dashboard        # Streamlit dashboard
│   └── docker-compose.yml          # Full stack orchestration
├── src/
│   ├── models/
│   │   ├── efficientphys.py        # EfficientPhys implementation
│   │   ├── tscan_lite.py           # TS-CAN lightweight variant
│   │   └── model_registry.py       # Model zoo + loading utilities
│   ├── rl/
│   │   ├── reward.py               # Self-supervised reward functions
│   │   ├── policy_gradient.py      # Last-layer policy gradient adaptation
│   │   ├── bandit.py               # Contextual bandit for model selection
│   │   └── adaptation_loop.py      # Main RL adaptation orchestration
│   ├── signal/
│   │   ├── filters.py              # Bandpass, detrending
│   │   ├── hr_extraction.py        # PSD-based HR estimation
│   │   ├── spo2_estimation.py      # SpO2 ratio-of-ratios
│   │   └── quality.py              # SNR, peak sharpness, signal quality
│   ├── data/
│   │   ├── datasets.py             # UBFC, PURE, MMPD loaders
│   │   ├── face_detection.py       # MediaPipe ROI extraction
│   │   └── preprocessing.py        # Frame normalization, augmentation
│   ├── api/
│   │   ├── main.py                 # FastAPI inference server
│   │   └── schemas.py              # Pydantic request/response models
│   └── dashboard/
│       └── app.py                  # Streamlit live monitoring
├── configs/
│   ├── train.yaml                  # Training hyperparameters
│   ├── adapt.yaml                  # RL adaptation hyperparameters
│   └── eval.yaml                   # Evaluation configuration
├── experiments/
│   ├── 01_baseline/                # First experiment notebooks
│   ├── 02_rl_adaptation/           # RL adaptation experiments
│   └── 03_cross_dataset/           # Multi-dataset benchmarks
├── tests/
│   ├── test_models.py
│   ├── test_reward.py
│   ├── test_signal.py
│   └── test_api.py
├── scripts/
│   ├── train.py                    # Training entry point
│   ├── evaluate.py                 # Evaluation entry point
│   └── adapt.py                    # RL adaptation entry point
├── docs/                           # MkDocs documentation
├── .github/workflows/ci.yml       # GitHub Actions
├── pyproject.toml                  # Project config (ruff, mypy, pytest)
├── README.md
└── LICENSE
```

---

## 5. MCP Rules for Claude Code in Antigravity IDE

These rules govern how Claude Code (CC) operates as the AI coding agent for this project:

### Architecture Rules
- **Never modify core model architectures** without explicit approval — EfficientPhys forward pass is sacred
- **All new modules must have a corresponding test file** in `tests/` before merging
- **Type hints are mandatory** on all function signatures — mypy strict mode
- **Config-driven:** all hyperparameters go in YAML configs under `configs/`, never hardcoded
- **Separation of concerns:** inference code (`src/models/`, `src/signal/`) must have zero RL imports; RL code (`src/rl/`) imports models as black boxes

### Code Quality Rules
- **Max function length: 50 lines.** If longer, refactor into smaller functions
- **Docstrings required** on all public functions (Google style)
- **No `print()` statements** — use `logging` module with appropriate levels
- **No wildcard imports** (`from x import *`)
- **Constants in UPPER_SNAKE_CASE** at module top

### Docker Rules
- **Every container must have a health check** in docker-compose
- **Pin all dependency versions** in requirements.txt (no `>=` without upper bound)
- **Multi-stage builds** for inference container (build stage + slim runtime)
- **GPU container uses nvidia/cuda base** only for the trainer; inference runs on CPU

### Git Rules
- **Branch naming:** `feature/`, `fix/`, `experiment/`, `docs/`
- **Commit messages:** conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`)
- **No commits to main** — always PR with at least CI passing
- **Experiment notebooks** go in `experiments/` and are gitignored for large outputs

### Data Rules
- **Never commit dataset files** to git — use DVC or document download instructions
- **Model checkpoints** stored in `checkpoints/` (gitignored), tracked with DVC
- **All data paths must be configurable** via config files or environment variables

### Safety Rules
- **RL adaptation must have a rollback mechanism** — if adapted model performs worse than frozen baseline on a validation window, revert
- **Reward function changes require a before/after comparison** logged to experiments
- **Latency budget:** inference path must stay under 30ms per frame on RTX 3090; any change that exceeds this is flagged

---

## 6. Flexible Phases

### Phase 1 — Foundation (Weeks 1–3)
**Deliverables:**
- EfficientPhys running in Docker with webcam input → HR output
- UBFC-rPPG dataloader and training pipeline
- Baseline evaluation on UBFC→PURE (frozen model, no adaptation)
- Basic FastAPI endpoint: POST video frames → GET HR estimate
- pytest suite for signal processing functions

**Exit criteria:** Baseline HR MAE on PURE documented, Docker inference container runs stably

### Phase 2 — RL Adaptation Engine (Weeks 3–6)
**Deliverables:**
- Self-supervised reward function (SNR + spectral peak + temporal consistency)
- Policy gradient adaptation loop (fine-tune last 2 layers)
- Contextual bandit for model routing (if multiple models available)
- Redis integration for async reward computation
- Adaptation experiment on UBFC→PURE showing measurable improvement

**Exit criteria:** ≥15% relative HR MAE reduction on cross-dataset eval, documented in experiment notebook

### Phase 3 — Toolbox & Benchmarking (Weeks 5–8)
**Deliverables:**
- CLI tool: `adaptivephys train`, `adaptivephys eval`, `adaptivephys adapt`, `adaptivephys serve`
- Multi-dataset evaluation suite (UBFC, PURE, MMPD)
- Streamlit dashboard with live webcam, HR waveform, adaptation status, before/after toggle
- GitHub Actions CI (lint, test, Docker build)
- Benchmarking report: tables and plots comparing frozen vs adapted across datasets

**Exit criteria:** Reproducible benchmark results, CI green, dashboard functional

### Phase 4 — Demo & Impact (Weeks 7–10)
**Deliverables:**
- Polished live webcam demo (adaptation toggle, real-time metrics overlay)
- Impact metrics report with before/after tables
- Professional README with architecture diagram, quickstart, results
- LinkedIn post draft + CV project entry
- (Stretch) Conference paper draft or technical blog post

**Exit criteria:** Demo video recorded, metrics finalized, README reviewed

---

## 7. Impact Reporting Framework

### Metrics to Report (CV / LinkedIn)

**Primary — Heart Rate Estimation:**
- Cross-dataset HR MAE reduction: target **15–25%** relative improvement
- Example: "Reduced cross-dataset heart rate MAE from 7.2 to 5.4 BPM (25% improvement) using RL-based online adaptation"

**Secondary — System Performance:**
- Inference latency: target **<30ms per frame** on consumer GPU
- Adaptation convergence: **<100 samples** to reach adapted performance
- Multi-dataset generalization: consistent improvement across **3+ public datasets**

**Tertiary — Engineering Quality:**
- Docker-containerized with **<2 min cold start**
- Test coverage: **>80%** on core modules
- CI/CD pipeline with automated benchmarking

### CV Project Entry (Draft)

> **AdaptivePhys — RL-Augmented Remote Photoplethysmography**
> *Open-source, containerized toolbox for adaptive vital sign estimation from facial video*
>
> - Developed a reinforcement learning adaptation engine that reduces cross-dataset heart rate estimation error by X% (MAE: Y→Z BPM) without manual retraining
> - Implemented self-supervised reward signal (spectral SNR + temporal consistency) enabling deployment without ground-truth sensors
> - Containerized with Docker Compose (inference, RL trainer, monitoring dashboard), achieving <30ms inference latency
> - Benchmarked across 3 public datasets (UBFC-rPPG, PURE, MMPD) with reproducible evaluation suite
> - Tech: PyTorch, EfficientPhys, Stable-Baselines3, FastAPI, Docker, Redis, Streamlit

### LinkedIn Post Structure (Draft)

1. **Hook:** "What if your vital sign monitor could learn and improve just from watching you?"
2. **Problem:** rPPG models fail on unseen subjects/cameras/lighting
3. **Solution:** RL-based online adaptation — the model gets better in real time
4. **Results:** X% MAE reduction, <30ms latency, works across 3 datasets
5. **Demo:** GIF/video of live webcam with adaptation toggle
6. **Call to action:** Link to GitHub repo

---

## 8. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| RL adaptation degrades performance | Rollback mechanism: revert to frozen model if validation MAE increases |
| Self-supervised reward is noisy | Ensemble multiple reward signals; validate against ground truth on held-out set |
| Inference latency budget exceeded | Profile early and often; RL runs async in separate container |
| Dataset access issues | Start with UBFC-rPPG (freely available); PURE requires agreement |
| Scope creep | Strict phase exit criteria; first experiment is intentionally minimal |

---

## 9. Quick Start Commands (Target)

```bash
# Clone and launch
git clone https://github.com/malekdinari/adaptivephys.git
cd adaptivephys

# Start all services
docker compose up -d

# Run baseline evaluation
docker compose exec inference python scripts/evaluate.py --config configs/eval.yaml

# Enable RL adaptation
docker compose exec trainer python scripts/adapt.py --config configs/adapt.yaml

# Open dashboard
open http://localhost:8501
```

---

*Document version: 1.0 — April 2026*
*Author: Malek Dinari*
