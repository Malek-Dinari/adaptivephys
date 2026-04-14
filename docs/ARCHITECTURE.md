# VitalSense — Real-Time rPPG Vital Signs Monitoring Platform

## Architecture Document v1.0

> *"mvlek" — engineered with intention.*

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Tech Stack Justification](#2-tech-stack-justification)
3. [rPPG Pipeline Design (CORE)](#3-rppg-pipeline-design)
4. [Real-Time Considerations](#4-real-time-considerations)
5. [Implementation Roadmap](#5-implementation-roadmap)
6. [Module Contracts & API Specification](#6-module-contracts--api-specification)
7. [Project Structure](#7-project-structure)
8. [Risks & Challenges](#8-risks--challenges)
9. [Future Extensions](#9-future-extensions)

---

## 1. Architecture Overview

### 1.1 High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        BROWSER (CLIENT)                         │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │ Webcam   │───▶│ Face Detect  │───▶│ ROI Crop + Downsize   │  │
│  │ 30fps    │    │ HaarCascade  │    │ (72×72 or 36×36)      │  │
│  └──────────┘    │ (+ optional  │    └───────────┬───────────┘  │
│                  │  FaceMesh)   │                │              │
│                  └──────────────┘                │              │
│                                                  │ WebSocket    │
│                                                  │ (JPEG/raw    │
│                                                  │  base64)     │
│                                                  ▼              │
├─────────────────────────────────────────────────────────────────┤
│                        BACKEND (Python)                         │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  SESSION MANAGER                         │   │
│  │  Per-session state: frame buffer, signal history, config │   │
│  └─────────────────────────┬────────────────────────────────┘   │
│                            │                                    │
│  ┌─────────────────────────▼────────────────────────────────┐   │
│  │              rPPG PIPELINE ORCHESTRATOR                   │   │
│  │                                                          │   │
│  │  ┌────────────┐  ┌─────────────┐  ┌──────────────────┐   │   │
│  │  │ Frame      │  │ rPPG Model  │  │ Signal           │   │   │
│  │  │ Preprocess │─▶│ Inference   │─▶│ Post-Processing  │   │   │
│  │  │            │  │             │  │                   │   │   │
│  │  │ • Resize   │  │ EfficientPh │  │ • Bandpass filter │   │   │
│  │  │ • Normalize│  │ (primary)   │  │ • Detrend         │   │   │
│  │  │ • Buffer   │  │ POS         │  │ • FFT → HR        │   │   │
│  │  │ • Chunk    │  │ (fallback)  │  │ • Confidence      │   │   │
│  │  └────────────┘  └─────────────┘  └──────┬───────────┘   │   │
│  │                                          │               │   │
│  └──────────────────────────────────────────┼───────────────┘   │
│                                             │                   │
│                                    WebSocket (JSON)             │
│                                             │                   │
├─────────────────────────────────────────────┼───────────────────┤
│                        BROWSER (CLIENT)     │                   │
│                                             ▼                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                VISUALIZATION LAYER                       │   │
│  │                                                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐  │   │
│  │  │ rPPG     │  │ Filtered │  │ FFT     │  │ HR Over  │  │   │
│  │  │ Raw Sig  │  │ Signal   │  │ Spectrum│  │ Time     │  │   │
│  │  └──────────┘  └──────────┘  └─────────┘  └──────────┘  │   │
│  │                                                          │   │
│  │        Bokeh (primary)  ──▶  uPlot (if perf issues)      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Design Rationale: Hybrid Architecture

**Why not full client-side?**
EfficientPhys is a PyTorch model. Converting to ONNX.js or TF.js introduces precision drift, limited operator support, and makes debugging miserable. Since we build on rPPG-Toolbox's native PyTorch codebase, server-side inference preserves exact reproducibility with the published checkpoints.

**Why not full server-side?**
Sending full 640×480 frames at 30fps over WebSocket = ~27 MB/s raw, ~3–5 MB/s compressed. Unnecessary. The face crop is ~72×72×3 = 15 KB raw per frame. At 30fps that's ~450 KB/s — trivially handleable even over moderate connections.

**The hybrid split:**
- **Client:** Webcam capture → face detection (HaarCascade) → crop → resize → stream cropped ROIs
- **Server:** Buffer frames → run EfficientPhys/POS → signal processing → stream results back

This minimizes bandwidth, keeps the ML stack in Python where it belongs, and allows GPU/CPU flexibility on the server side.

---

## 2. Tech Stack Justification

### 2.1 Frontend

| Component | Choice | Justification |
|-----------|--------|---------------|
| Framework | **Vanilla JS + HTML/CSS** (Phase 0–1), **React** (Phase 2+) | No framework overhead initially; React when UI complexity demands it |
| Webcam | **MediaDevices API** (`getUserMedia`) | Standard, well-supported, no library needed |
| Face Detection | **OpenCV.js HaarCascade** (client-side) | Fast, lightweight, no server round-trip for detection. Ships as a single WASM module |
| Charting | **Bokeh** (via BokehJS) → **uPlot** fallback | Bokeh: rich, Python-ecosystem aligned, good streaming support. uPlot: 35KB, <1ms render at 10k points |
| Communication | **WebSocket** (native) | Bidirectional streaming. No polling overhead |
| Styling | **CSS custom properties + minimal framework** | Clean, maintainable, no Tailwind bloat for a specialized app |

**Why OpenCV.js for face detection on the client?**

MediaPipe FaceMesh is heavier (~2MB model download, more CPU). HaarCascade is a <500KB XML file, runs in ~5ms per frame on modern hardware. Since we only need a bounding box for the primary mode (not landmarks), HaarCascade is sufficient. FaceMesh is reserved for the optional advanced skin-segmentation mode.

### 2.2 Backend

| Component | Choice | Justification |
|-----------|--------|---------------|
| Framework | **FastAPI** | Native async/await, WebSocket support, Pydantic validation, auto-generated OpenAPI docs. Flask's sync model is wrong for streaming workloads |
| WebSocket | **FastAPI WebSocket** (Starlette) | First-class support, no extra dependency |
| ML Runtime | **PyTorch** (dev), **ONNX Runtime** (prod) | PyTorch for exact rPPG-Toolbox compatibility. ONNX for 2–5× speedup in production |
| Signal Processing | **SciPy** + **NumPy** | Standard, fast, well-documented. No reinventing filters |
| Session State | **In-process dict** (Phase 0–2), **Redis** (Phase 3+) | Premature Redis adds latency and ops burden. Python dict is fine for single-user |
| Task Queue | **None** (Phase 0–2), **Celery/Redis** (Phase 3+) | Not needed until multi-user |

**Why FastAPI over Flask?**

This is a streaming workload. Flask's WSGI model means you'd need `flask-socketio` + eventlet/geventwebsocket, which is a compatibility minefield. FastAPI/Starlette gives native async WebSocket handlers, proper lifecycle management, and automatic request validation. It's the right tool.

### 2.3 ML Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| Primary Model | **EfficientPhys** (PURE checkpoint) | `PURE_EfficientPhys.pth` from rPPG-Toolbox `final_model_release/` |
| Fallback Model | **POS** (Plane-Orthogonal-to-Skin) | From `unsupervised_methods/`. Zero learned parameters, CPU-only, always available |
| Input Contract | **T×C×H×W** tensor, T=chunk_length, C=3 (RGB), H=W=72 | rPPG-Toolbox standard. Normalized to [0,1] |
| Output Contract | **T-length 1D signal** (BVP waveform) | Per-frame pulse amplitude |
| Optimization | **ONNX export** (Phase 2+) | `torch.onnx.export()` → ONNX Runtime `InferenceSession` |

### 2.4 Infrastructure (Phased)

| Phase | Stack |
|-------|-------|
| Phase 0–1 | Single process, `uvicorn`, local dev |
| Phase 2 | Docker single-container (backend + static frontend) |
| Phase 3 | Docker Compose: `backend`, `frontend` (nginx), `redis`, `postgres` |
| Phase 4 | Cloud VM with GPU (or CPU+ONNX), HTTPS via Caddy/nginx |

---

## 3. rPPG Pipeline Design

This is the core of the system. Every design decision here must be defensible.

### 3.1 Pipeline Stages

```
Frame_in ──▶ [1. Validate] ──▶ [2. Preprocess] ──▶ [3. Buffer]
                                                        │
         ┌──────────────────────────────────────────────┘
         │
         ▼
    [4. Chunk Ready?]──No──▶ (wait for more frames)
         │
        Yes
         │
         ▼
    [5. Inference] ──▶ [6. Post-Process] ──▶ [7. HR Estimate] ──▶ Result_out
```

### 3.2 Stage Details

#### Stage 1: Frame Validation

```python
class FrameValidator:
    """Validates incoming ROI frames before pipeline entry.
    
    Rejects frames that would corrupt downstream processing:
    - Wrong dimensions (expected: H×W×3, uint8)
    - Excessive darkness (mean pixel < threshold → likely occluded)
    - Excessive brightness (mean pixel > threshold → likely overexposed)
    - Excessive motion blur (Laplacian variance below threshold)
    """
    
    def validate(self, frame: np.ndarray) -> FrameStatus:
        # Returns VALID, SKIP (use last good frame), or REJECT (signal quality warning)
```

**Why validate?** Bad frames propagate through the pipeline and corrupt the entire chunk's inference. It's cheaper to detect and skip/repeat than to let a dark/blurry frame introduce artifact into 6 seconds of signal.

**Failure mode:** If >30% of frames in a window are invalid → pause HR estimation, display "Signal quality too low" in UI. Do not output a garbage HR.

#### Stage 2: Preprocessing

```python
class FramePreprocessor:
    """Prepares validated ROI frames for model consumption.
    
    Pipeline: resize → color convert → normalize → (optional) stabilize
    """
    
    def __init__(self, target_size: Tuple[int,int] = (72, 72),
                 color_space: str = "RGB",
                 normalize: bool = True):
        self.target_size = target_size
        # ...
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        # 1. Resize to target_size (bilinear interpolation)
        # 2. Convert BGR→RGB if needed
        # 3. Normalize to [0, 1] float32 (divide by 255.0)
        # 4. Optional: temporal differencing (model-specific)
        return processed_frame
```

**Key design decision: Resize on client or server?**

Client-side. The client already has the face bounding box from HaarCascade. Resizing a 200×200 crop to 72×72 in JavaScript (`canvas.drawImage` with bicubic) is trivial and reduces WebSocket payload by 7.7×. Server receives 72×72×3 frames ready for buffering.

**Normalization must match rPPG-Toolbox exactly.** The EfficientPhys checkpoint was trained with specific normalization. We must replicate `data_loader/BaseLoader.py`'s preprocessing. Any deviation = silent accuracy degradation.

#### Stage 3: Temporal Buffer

```python
class TemporalBuffer:
    """Sliding window frame buffer with configurable chunk and stride.
    
    Manages the temporal dimension of the rPPG pipeline.
    Thread-safe for async frame ingestion.
    
    — dinnarus —
    """
    
    def __init__(self, chunk_length: int = 180,  # 6s @ 30fps
                 stride: int = 15,               # 0.5s shift
                 max_buffer: int = 900):          # 30s history
        self.buffer = collections.deque(maxlen=max_buffer)
        self.chunk_length = chunk_length
        self.stride = stride
        self._frames_since_last_chunk = 0
        self._lock = threading.Lock()
    
    def add_frame(self, frame: np.ndarray, timestamp: float) -> Optional[np.ndarray]:
        """Add frame. Returns chunk (T,C,H,W) when stride threshold reached."""
        with self._lock:
            self.buffer.append((frame, timestamp))
            self._frames_since_last_chunk += 1
            
            if (len(self.buffer) >= self.chunk_length and 
                self._frames_since_last_chunk >= self.stride):
                self._frames_since_last_chunk = 0
                return self._extract_chunk()
        return None
    
    def _extract_chunk(self) -> np.ndarray:
        """Extract last chunk_length frames as (T, C, H, W) tensor."""
        frames = list(self.buffer)[-self.chunk_length:]
        # Stack frames, transpose to (T, C, H, W)
        return np.stack([f for f, _ in frames]).transpose(0, 3, 1, 2)
```

**Critical parameters:**

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `chunk_length` | 180 (6s) | 180–900 | Longer = better frequency resolution, more latency |
| `stride` | 15 (0.5s) | 3–30 | Smaller = more frequent updates, more compute |
| `max_buffer` | 900 (30s) | 300–1800 | History for signal visualization |

**Stride vs update rate tradeoff:**

On RTX 3090, EfficientPhys forward pass on 180 frames ≈ 30–80ms. So stride=6 (0.2s, 5 inferences/sec) is feasible on GPU.

On CPU (e.g., deployed VM), forward pass ≈ 300–600ms. Stride=15 (0.5s, 2 inferences/sec) is realistic. Stride=6 would create a growing backlog. The system must auto-detect compute capability and adjust stride.

#### Stage 4: Chunk Readiness Check

Not a separate module — it's the return value of `TemporalBuffer.add_frame()`. When `None` is returned, the pipeline short-circuits. When a chunk is returned, inference proceeds.

**Warm-up behavior:** The first `chunk_length` frames produce no output. UI should show "Calibrating..." with a progress indicator (frames_collected / chunk_length).

#### Stage 5: Inference (Model Abstraction)

```python
class BaseRPPGModel(ABC):
    """Abstract base for all rPPG signal extractors.
    
    Defines the contract that any model (learned or unsupervised) must satisfy.
    This abstraction is what makes the pipeline model-agnostic.
    """
    
    @abstractmethod
    def predict(self, chunk: np.ndarray) -> np.ndarray:
        """
        Args:
            chunk: (T, C, H, W) float32 array, normalized [0,1]
        
        Returns:
            signal: (T,) float32 array, predicted BVP waveform
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Returns model info: name, expected input shape, compute requirements."""
        pass


class EfficientPhysModel(BaseRPPGModel):
    """EfficientPhys wrapper for rPPG-Toolbox checkpoint.
    
    Loads PURE_EfficientPhys.pth and wraps inference with
    proper input formatting and output extraction.
    """
    
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
    
    def _load_model(self, path: str) -> nn.Module:
        # Import EfficientPhys architecture from rPPG-Toolbox
        # neural_methods/model/EfficientPhys.py
        model = EfficientPhys(...)  # Match training config
        state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict)
        return model
    
    @torch.no_grad()
    def predict(self, chunk: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(chunk).float().to(self.device)
        # EfficientPhys may expect (B, T, C, H, W) — add batch dim
        tensor = tensor.unsqueeze(0)
        output = self.model(tensor)
        # Extract BVP signal from model output
        # (architecture-dependent — verify against rPPG-Toolbox trainer)
        return output.squeeze().cpu().numpy()


class POSModel(BaseRPPGModel):
    """Plane-Orthogonal-to-Skin unsupervised rPPG method.
    
    No learned parameters. Uses color channel projections
    to isolate pulse signal from skin reflectance changes.
    
    Reference: Wang et al., "Algorithmic Principles of Remote PPG" (2017)
    """
    
    def predict(self, chunk: np.ndarray) -> np.ndarray:
        # Spatial average per frame → (T, 3) color trace
        # Apply POS projection matrix
        # Returns (T,) BVP signal
        # Implementation from rPPG-Toolbox unsupervised_methods/
        pass
```

**Model switching strategy:**

```python
class ModelRegistry:
    """Registry of available rPPG models with automatic fallback.
    
    Priority: EfficientPhys (GPU) > EfficientPhys (CPU/ONNX) > POS
    Falls back transparently on model load failure or inference error.
    """
    
    def __init__(self):
        self._models: Dict[str, BaseRPPGModel] = {}
        self._active: str = None
    
    def register(self, name: str, model: BaseRPPGModel, priority: int):
        ...
    
    def get_active(self) -> BaseRPPGModel:
        ...
    
    def fallback(self, reason: str):
        """Demote current model, activate next priority."""
        ...
```

#### Stage 6: Signal Post-Processing

```python
class SignalProcessor:
    """Post-processes raw BVP waveform from rPPG model.
    
    Applies standard physiological signal processing chain:
    detrend → bandpass → (optional) smooth
    
    All filter parameters are grounded in physiology:
    - HR range: 40–200 BPM → 0.67–3.33 Hz
    - Respiratory influence: < 0.5 Hz (removed by bandpass)
    - Motion artifacts: broadband (partially removed by bandpass)
    """
    
    def __init__(self, fs: float = 30.0,
                 hr_min_bpm: float = 40.0,
                 hr_max_bpm: float = 200.0,
                 filter_order: int = 4):
        self.fs = fs
        self.low_hz = hr_min_bpm / 60.0   # 0.667 Hz
        self.high_hz = hr_max_bpm / 60.0   # 3.333 Hz
        self.filter_order = filter_order
        self._design_filter()
    
    def _design_filter(self):
        """Pre-compute Butterworth bandpass coefficients."""
        nyq = self.fs / 2.0
        self.sos = scipy.signal.butter(
            self.filter_order,
            [self.low_hz / nyq, self.high_hz / nyq],
            btype='bandpass',
            output='sos'  # SOS for numerical stability
        )
    
    def process(self, raw_signal: np.ndarray) -> ProcessedSignal:
        """Full processing chain.
        
        Returns:
            ProcessedSignal with fields:
                .raw: original signal
                .detrended: after polynomial detrend
                .filtered: after bandpass
                .spectrum: FFT magnitude
                .freqs: FFT frequency axis
        """
        # 1. Detrend (remove slow drift from illumination changes)
        detrended = self._detrend(raw_signal)
        
        # 2. Bandpass filter (isolate cardiac band)
        filtered = scipy.signal.sosfiltfilt(self.sos, detrended)
        
        # 3. FFT for frequency domain analysis
        freqs, spectrum = self._compute_fft(filtered)
        
        return ProcessedSignal(
            raw=raw_signal,
            detrended=detrended,
            filtered=filtered,
            spectrum=spectrum,
            freqs=freqs
        )
    
    def _detrend(self, signal: np.ndarray, order: int = 5) -> np.ndarray:
        """Polynomial detrending to remove slow baseline wander."""
        t = np.arange(len(signal))
        poly = np.polyfit(t, signal, order)
        trend = np.polyval(poly, t)
        return signal - trend
    
    def _compute_fft(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute single-sided FFT magnitude spectrum."""
        n = len(signal)
        # Zero-pad to next power of 2 for better frequency resolution
        n_fft = 2 ** int(np.ceil(np.log2(n)))
        
        # Apply Hanning window to reduce spectral leakage
        windowed = signal * np.hanning(n)
        
        fft_vals = np.fft.rfft(windowed, n=n_fft)
        freqs = np.fft.rfftfreq(n_fft, d=1.0/self.fs)
        magnitude = np.abs(fft_vals)
        
        return freqs, magnitude
```

#### Stage 7: Heart Rate Estimation

```python
class HREstimator:
    """Estimates heart rate from processed BVP signal.
    
    Uses FFT peak detection with confidence scoring.
    Maintains a rolling HR history for temporal smoothing.
    """
    
    def __init__(self, fs: float = 30.0,
                 hr_min_bpm: float = 40.0,
                 hr_max_bpm: float = 200.0,
                 history_size: int = 10):
        self.fs = fs
        self.hr_min = hr_min_bpm
        self.hr_max = hr_max_bpm
        self.hr_history = collections.deque(maxlen=history_size)
    
    def estimate(self, processed: ProcessedSignal) -> HRResult:
        """Estimate HR from processed signal.
        
        Returns:
            HRResult with fields:
                .hr_bpm: estimated heart rate
                .confidence: 0.0–1.0 quality score
                .method: 'fft_peak'
        """
        freqs = processed.freqs
        spectrum = processed.spectrum
        
        # Mask to cardiac frequency band
        mask = (freqs >= self.hr_min/60) & (freqs <= self.hr_max/60)
        cardiac_freqs = freqs[mask]
        cardiac_mag = spectrum[mask]
        
        if len(cardiac_mag) == 0:
            return HRResult(hr_bpm=0, confidence=0.0, method='fft_peak')
        
        # Find dominant frequency
        peak_idx = np.argmax(cardiac_mag)
        peak_freq = cardiac_freqs[peak_idx]
        hr_bpm = peak_freq * 60.0
        
        # Confidence: ratio of peak energy to total cardiac band energy
        # High confidence = sharp, dominant peak
        peak_energy = cardiac_mag[peak_idx] ** 2
        total_energy = np.sum(cardiac_mag ** 2)
        snr = peak_energy / (total_energy - peak_energy + 1e-10)
        confidence = min(1.0, snr / 5.0)  # Normalize: SNR=5 → confidence=1.0
        
        # Temporal smoothing with outlier rejection
        hr_bpm = self._smooth(hr_bpm, confidence)
        
        return HRResult(hr_bpm=hr_bpm, confidence=confidence, method='fft_peak')
    
    def _smooth(self, hr: float, confidence: float) -> float:
        """Weighted moving average with outlier gating."""
        if len(self.hr_history) > 0:
            median_hr = np.median([h for h, _ in self.hr_history])
            # Reject if >30 BPM away from recent median (likely artifact)
            if abs(hr - median_hr) > 30 and confidence < 0.5:
                hr = median_hr  # Hold previous estimate
        
        self.hr_history.append((hr, confidence))
        
        # Weighted average: recent + high-confidence estimates dominate
        weights = []
        values = []
        for i, (h, c) in enumerate(self.hr_history):
            recency_weight = (i + 1) / len(self.hr_history)
            weights.append(c * recency_weight)
            values.append(h)
        
        return np.average(values, weights=weights)
```

### 3.3 Pipeline Orchestrator

```python
class RPPGPipeline:
    """Orchestrates the full rPPG processing pipeline.
    
    Single entry point: feed frames in, get HR estimates out.
    Handles warm-up, model switching, error recovery.
    
    — mvlek —
    """
    
    def __init__(self, config: PipelineConfig):
        self.validator = FrameValidator(config.validation)
        self.preprocessor = FramePreprocessor(config.preprocessing)
        self.buffer = TemporalBuffer(
            chunk_length=config.chunk_length,
            stride=config.stride
        )
        self.model_registry = ModelRegistry()
        self.signal_processor = SignalProcessor(fs=config.fps)
        self.hr_estimator = HREstimator(fs=config.fps)
        
        # State
        self._is_warmed_up = False
        self._total_frames = 0
        self._valid_frames = 0
    
    async def process_frame(self, frame: np.ndarray, 
                            timestamp: float) -> Optional[PipelineResult]:
        """Process a single frame through the pipeline.
        
        Returns None during warm-up or when stride not yet reached.
        Returns PipelineResult when a new HR estimate is available.
        """
        self._total_frames += 1
        
        # Stage 1: Validate
        status = self.validator.validate(frame)
        if status == FrameStatus.REJECT:
            return PipelineResult.quality_warning(self._total_frames)
        if status == FrameStatus.SKIP:
            frame = self.validator.last_good_frame
        
        self._valid_frames += 1
        
        # Stage 2: Preprocess
        processed = self.preprocessor.process(frame)
        
        # Stage 3+4: Buffer and check chunk readiness
        chunk = self.buffer.add_frame(processed, timestamp)
        if chunk is None:
            # Not enough frames yet
            progress = len(self.buffer.buffer) / self.buffer.chunk_length
            return PipelineResult.warming_up(progress)
        
        self._is_warmed_up = True
        
        # Stage 5: Inference
        model = self.model_registry.get_active()
        try:
            raw_signal = model.predict(chunk)
        except Exception as e:
            self.model_registry.fallback(str(e))
            model = self.model_registry.get_active()
            raw_signal = model.predict(chunk)
        
        # Stage 6: Post-process
        processed_signal = self.signal_processor.process(raw_signal)
        
        # Stage 7: HR estimation
        hr_result = self.hr_estimator.estimate(processed_signal)
        
        return PipelineResult.success(
            hr=hr_result,
            signal=processed_signal,
            model_name=model.get_metadata().name,
            frame_count=self._total_frames
        )
```

### 3.4 Pipeline Configuration

```python
@dataclass
class PipelineConfig:
    """All tuneable pipeline parameters in one place.
    
    Defaults are sane for RTX 3090 local development.
    Production/CPU profiles override stride and model selection.
    """
    # Temporal
    fps: float = 30.0
    chunk_length: int = 180          # frames (6s @ 30fps)
    stride: int = 15                 # frames between inferences (0.5s)
    max_buffer_seconds: float = 30.0
    
    # Model
    primary_model: str = "efficientphys"
    fallback_model: str = "pos"
    device: str = "cpu"              # "cpu" | "cuda" | "cuda:0"
    checkpoint_path: str = "final_model_release/PURE_EfficientPhys.pth"
    
    # Signal processing
    hr_min_bpm: float = 40.0
    hr_max_bpm: float = 200.0
    bandpass_order: int = 4
    detrend_order: int = 5
    
    # Validation
    min_brightness: float = 30.0     # mean pixel value
    max_brightness: float = 230.0
    min_sharpness: float = 10.0      # Laplacian variance
    max_invalid_ratio: float = 0.3   # pause if >30% frames invalid
    
    # Profiles
    @classmethod
    def gpu_dev(cls) -> 'PipelineConfig':
        return cls(device="cuda", stride=6)  # 5 updates/sec
    
    @classmethod
    def cpu_prod(cls) -> 'PipelineConfig':
        return cls(device="cpu", stride=30)  # 1 update/sec
    
    @classmethod
    def onnx_prod(cls) -> 'PipelineConfig':
        return cls(device="cpu", stride=15,
                   primary_model="efficientphys_onnx")  # 2 updates/sec
```

---

## 4. Real-Time Considerations

### 4.1 Latency Budget (Per Update Cycle)

```
Target: < 200ms end-to-end on GPU, < 700ms on CPU

┌───────────────────────────────────┬────────┬────────┐
│ Stage                             │ GPU    │ CPU    │
├───────────────────────────────────┼────────┼────────┤
│ Client: capture + detect + crop   │ 10ms   │ 10ms   │
│ Client→Server: WebSocket transfer │ 1–5ms  │ 1–5ms  │
│ Server: validate + preprocess     │ 1ms    │ 1ms    │
│ Server: buffer management         │ <1ms   │ <1ms   │
│ Server: EfficientPhys inference   │ 30–80ms│ 300–600│
│ Server: signal processing + HR    │ 2–5ms  │ 2–5ms  │
│ Server→Client: WebSocket result   │ 1–2ms  │ 1–2ms  │
│ Client: chart update              │ 5–15ms │ 5–15ms │
├───────────────────────────────────┼────────┼────────┤
│ TOTAL                             │ ~60ms  │ ~450ms │
└───────────────────────────────────┴────────┴────────┘
```

### 4.2 Frame Rate vs Accuracy Tradeoff

The webcam runs at 30fps regardless of inference rate. All frames are buffered. The question is how often we run inference (stride):

| Stride | Update Rate | GPU Feasible | CPU Feasible | UX Feel |
|--------|-------------|-------------|-------------|---------|
| 3 (0.1s) | 10/sec | Yes | No | Jittery — signal oscillates too fast for user |
| 6 (0.2s) | 5/sec | Yes | No | Smooth, responsive |
| 15 (0.5s) | 2/sec | Yes | Marginal | Good balance |
| 30 (1.0s) | 1/sec | Yes | Yes | Slightly laggy but stable |

**Recommendation:** Default stride=15 (0.5s). Configurable. Auto-detect: measure first inference time, adjust stride so inference_time < 0.8 × stride_duration.

### 4.3 Backpressure Handling

If inference can't keep up with frame ingestion:

```python
class BackpressureMonitor:
    """Detects and handles inference backlog.
    
    If frames accumulate faster than processed:
    1. Increase stride (reduce inference frequency)
    2. If still backed up: switch to POS (faster)
    3. If still backed up: drop frames (last resort)
    """
```

### 4.4 Buffer Sizing

```
max_buffer_seconds = 30s
max_buffer_frames = 30 × 30fps = 900 frames
memory per frame = 72 × 72 × 3 × 4 bytes (float32) = 62,208 bytes ≈ 62 KB
total buffer memory = 900 × 62 KB ≈ 54 MB

This is negligible. No memory pressure concerns.
```

---

## 5. Implementation Roadmap

### Phase 0: Skeleton (3–5 days)

**Goal:** Webcam → face crop → display crop in browser. No ML, no backend yet.

**Deliverables:**
- `index.html` with webcam capture via `getUserMedia`
- OpenCV.js HaarCascade face detection running in browser
- Canvas overlay showing detected face bounding box
- Cropped, resized (72×72) face displayed in a debug panel
- FPS counter

**Why start here:** Validates the client-side capture pipeline independently. If face detection fails or webcam access is flaky, you find out immediately without backend complexity.

### Phase 1: Pipeline Core (7–10 days)

**Goal:** End-to-end HR estimation working. Ugly but functional.

**Deliverables:**
- FastAPI backend with WebSocket endpoint
- Client streams face crops to server over WebSocket
- `TemporalBuffer` with configurable chunk/stride
- `EfficientPhysModel` wrapper loading `PURE_EfficientPhys.pth`
- `POSModel` wrapper from rPPG-Toolbox `unsupervised_methods/`
- `SignalProcessor` (detrend + bandpass + FFT)
- `HREstimator` (FFT peak + confidence + smoothing)
- `RPPGPipeline` orchestrator
- Server streams back JSON: `{hr_bpm, confidence, signal_chunk, fft_data}`
- Basic Bokeh line chart showing filtered signal + HR value
- Single config file for all parameters

**Critical subtasks:**
1. Verify EfficientPhys input/output contract against rPPG-Toolbox trainer code
2. Confirm normalization matches training preprocessing exactly
3. Run EfficientPhys on a known UBFC-rPPG clip offline, compare HR to rPPG-Toolbox evaluation output — must match within 1 BPM

### Phase 2: UI & Visualization (5–7 days)

**Goal:** Beautiful, responsive dashboard.

**Deliverables:**
- Clean layout: webcam feed | signal plots | HR display
- Four real-time charts:
  1. Raw rPPG signal (scrolling, last 10s)
  2. Filtered signal (bandpass output)
  3. FFT spectrum (updating per inference)
  4. HR over time (last 60s, with confidence band)
- Warm-up progress indicator
- Signal quality indicator (green/yellow/red)
- Model selector (EfficientPhys / POS toggle)
- Chunk length selector (6s / 10s / 20s / 30s)
- FPS + latency display (debug panel, collapsible)
- Responsive design (works on 13" laptop and 27" monitor)
- Dark theme (appropriate for a monitoring dashboard)

**Bokeh strategy:** Use BokehJS `ColumnDataSource` with streaming mode. If frame drops or jitter observed at >2 updates/sec, swap to uPlot for the signal charts (keep Bokeh for FFT spectrum which updates less frequently).

### Phase 3: Production Hardening (5–7 days)

**Goal:** Deployable, observable, persistent.

**Deliverables:**
- Docker Compose setup:
  - `backend`: FastAPI + uvicorn + PyTorch/ONNX
  - `frontend`: nginx serving static files
  - `redis`: session cache (future multi-user)
  - `postgres`: measurement persistence
- ONNX export of EfficientPhys + ONNX Runtime integration
- Database schema:
  ```sql
  CREATE TABLE sessions (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      config JSONB,             -- pipeline config snapshot
      model_used VARCHAR(50),
      device VARCHAR(20)
  );
  
  CREATE TABLE measurements (
      id BIGSERIAL PRIMARY KEY,
      session_id UUID REFERENCES sessions(id),
      timestamp TIMESTAMPTZ NOT NULL,
      hr_bpm FLOAT,
      confidence FLOAT,
      signal_quality VARCHAR(10),  -- 'good' | 'fair' | 'poor'
      raw_signal FLOAT[],          -- optional: store chunks
      model_name VARCHAR(50)
  );
  
  CREATE INDEX idx_measurements_session ON measurements(session_id, timestamp);
  ```
- Prometheus metrics endpoint (`/metrics`):
  - `vitalsense_inference_duration_seconds` (histogram)
  - `vitalsense_fps` (gauge)
  - `vitalsense_signal_quality` (gauge)
  - `vitalsense_hr_bpm` (gauge)
  - `vitalsense_active_sessions` (gauge)
- Health check endpoint (`/health`)
- Graceful shutdown handling
- CORS configuration
- Environment-based configuration (`.env` file)

### Phase 4: Auth, Polish, Deploy (5–7 days)

**Goal:** Public-facing, authenticated, monetization-ready.

**Deliverables:**
- JWT authentication (FastAPI dependency)
- OAuth2 (Google sign-in)
- User sessions tied to auth
- Rate limiting (per-user WebSocket connections)
- HTTPS via Caddy (auto Let's Encrypt)
- Landing page
- "Premium" tier flag (database + middleware, no payment integration yet)
- README + API docs (auto-generated from FastAPI)
- GitHub Actions CI: lint + test + Docker build

---

## 6. Module Contracts & API Specification

### 6.1 WebSocket Protocol

**Client → Server (per frame):**
```json
{
    "type": "frame",
    "data": "<base64 encoded 72×72×3 JPEG>",
    "timestamp": 1712745600.123,
    "frame_idx": 42
}
```

**Server → Client (per inference):**
```json
{
    "type": "result",
    "hr_bpm": 72.3,
    "confidence": 0.87,
    "signal_quality": "good",
    "model": "efficientphys",
    "signals": {
        "raw": [0.1, 0.3, -0.2, ...],
        "filtered": [0.05, 0.15, -0.1, ...],
        "fft_freqs": [0.0, 0.167, 0.333, ...],
        "fft_magnitude": [0.01, 0.5, 2.3, ...]
    },
    "frame_count": 210,
    "inference_ms": 45.2
}
```

**Server → Client (during warm-up):**
```json
{
    "type": "warmup",
    "progress": 0.65,
    "frames_collected": 117,
    "frames_needed": 180
}
```

**Server → Client (quality warning):**
```json
{
    "type": "quality_warning",
    "message": "Low signal quality — ensure face is well-lit and stable",
    "invalid_ratio": 0.45
}
```

### 6.2 REST API Endpoints

```
GET  /health              → { "status": "ok", "model_loaded": true, "device": "cuda" }
GET  /config              → current PipelineConfig as JSON
POST /config              → update config (chunk_length, stride, model, etc.)
GET  /metrics             → Prometheus format
WS   /ws/vitals           → main WebSocket endpoint
```

### 6.3 Configuration API

```
POST /config
{
    "chunk_length": 300,        // 10s
    "stride": 15,
    "primary_model": "pos",     // switch to POS
    "hr_min_bpm": 50,
    "hr_max_bpm": 180
}
```

Pipeline hot-reloads on config change. Buffer is preserved; only processing parameters change.

---

## 7. Project Structure

```
vitalsense/
├── README.md
├── ARCHITECTURE.md                 # This document
├── docker-compose.yml
├── .env.example
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                     # FastAPI app entry point
│   ├── config.py                   # PipelineConfig + env loading
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── websocket.py            # WebSocket handler
│   │   ├── rest.py                 # Health, config, metrics endpoints
│   │   └── session.py              # Session manager
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── orchestrator.py         # RPPGPipeline
│   │   ├── validator.py            # FrameValidator
│   │   ├── preprocessor.py         # FramePreprocessor
│   │   ├── buffer.py               # TemporalBuffer
│   │   ├── signal_processor.py     # SignalProcessor
│   │   └── hr_estimator.py         # HREstimator
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseRPPGModel ABC
│   │   ├── registry.py             # ModelRegistry
│   │   ├── efficientphys.py        # EfficientPhysModel wrapper
│   │   ├── pos.py                  # POSModel wrapper
│   │   └── onnx_wrapper.py         # ONNX Runtime wrapper (Phase 3)
│   │
│   ├── db/                         # Phase 3+
│   │   ├── __init__.py
│   │   ├── models.py               # SQLAlchemy models
│   │   ├── session.py              # DB session factory
│   │   └── migrations/             # Alembic
│   │
│   └── tests/
│       ├── test_pipeline.py
│       ├── test_signal_processor.py
│       ├── test_hr_estimator.py
│       └── fixtures/
│           └── sample_chunk.npy    # Known-good test data
│
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   ├── app.js                  # Main application
│   │   ├── webcam.js               # Webcam capture + face detection
│   │   ├── websocket.js            # WebSocket client
│   │   ├── charts.js               # Bokeh/uPlot chart management
│   │   └── ui.js                   # UI state + controls
│   └── assets/
│       └── haarcascade_frontalface_default.xml
│
├── rppg_toolbox/                   # Cloned rPPG-Toolbox (git submodule or vendored)
│   ├── neural_methods/
│   │   └── model/
│   │       └── EfficientPhys.py    # Model architecture
│   ├── unsupervised_methods/
│   │   └── POS.py
│   ├── final_model_release/
│   │   └── PURE_EfficientPhys.pth
│   └── ...
│
└── scripts/
    ├── export_onnx.py              # EfficientPhys → ONNX conversion
    ├── benchmark.py                # Latency + accuracy benchmarks
    └── validate_checkpoint.py      # Verify checkpoint matches toolbox output
```

---

## 8. Risks & Challenges

### 8.1 Motion Artifacts

**Problem:** Head movement causes ROI shifts mid-chunk, introducing intensity changes unrelated to pulse.

**Mitigations:**
1. Frame validation rejects high-motion frames (Laplacian variance drop)
2. ROI stabilization: track bounding box centroid, apply exponential moving average to smooth jitter
3. Signal-level: bandpass filter removes broadband motion energy (partially)
4. Confidence scoring drops during motion → UI shows warning
5. Future: optical flow-based motion compensation

**Residual risk:** Fast lateral head turns will produce 2–5s of unreliable signal. This is inherent to all video-based rPPG. The correct response is to detect it and say "no estimate" rather than output noise.

### 8.2 Illumination Variation

**Problem:** Changing ambient light (clouds, screen brightness, indoor/outdoor transition) creates slow baseline drift + sudden step changes.

**Mitigations:**
1. Polynomial detrending (order 5) removes slow drift
2. Bandpass filter removes very low frequency illumination changes
3. POS method is inherently more robust to illumination (projection-based)
4. Frame validation detects sudden brightness jumps

**Residual risk:** Flickering fluorescent lights at 50/60 Hz can alias into the cardiac band (0.67–3.33 Hz). Very hard to separate. Anti-aliasing would require >60fps capture or explicit notch filter at mains frequency.

### 8.3 Model Generalization

**Problem:** PURE_EfficientPhys.pth was trained on PURE dataset (10 subjects, controlled lab). Real-world webcam conditions differ significantly.

**Mitigations:**
1. POS fallback doesn't suffer from training domain mismatch
2. Confidence scoring flags low-quality estimates
3. Future: fine-tune on UBFC-rPPG or larger datasets
4. Future: collect small self-supervised calibration data per user

**Residual risk:** Skin tone bias is a known issue in rPPG literature. PURE dataset is not diverse. This must be acknowledged transparently in documentation.

### 8.4 WebSocket Reliability

**Problem:** WebSocket connections drop, especially on mobile networks or when laptop sleeps.

**Mitigations:**
1. Automatic reconnection with exponential backoff
2. Pipeline state survives reconnection (buffer persists server-side for 30s)
3. Client-side frame counter detects gaps

### 8.5 EfficientPhys Input Contract Mismatch

**Problem:** The rPPG-Toolbox training pipeline applies specific preprocessing that may not be obvious from the model architecture alone. If our preprocessing doesn't match, the model outputs garbage.

**Mitigation:** Phase 1 includes a mandatory validation step — run the same UBFC-rPPG clip through both rPPG-Toolbox's evaluation pipeline and our pipeline, compare outputs. They must match within 1 BPM on the test set.

---

## 9. Future Extensions

### 9.1 Blood Pressure Estimation (High Priority)

Extend the pipeline to estimate systolic/diastolic BP from the rPPG waveform using pulse transit time (PTT) features and/or waveform morphology. Requires:
- Pulse wave analysis module (systolic peak, diastolic notch detection)
- Calibration per user (at least one reference measurement)
- Regression model (PTT → BP)

This aligns with the forearm-video BP estimation research direction.

### 9.2 Multi-Signal Extraction

Extract additional vitals from the same video:
- **Respiratory rate:** From low-frequency modulation of the rPPG signal (amplitude modulation) or chest motion
- **SpO2:** From ratio of red/infrared channel absorption (limited accuracy with RGB cameras)
- **HRV (Heart Rate Variability):** From inter-beat intervals of the BVP waveform

### 9.3 MediaPipe FaceMesh Integration (Phase 2+)

Switch from HaarCascade to FaceMesh when:
- Head pose exceeds ±30° yaw/pitch
- Multiple faces detected (need reliable tracking)
- Skin segmentation needed for better SNR

FaceMesh provides 468 landmarks → compute convex hull of forehead + cheek regions → binary mask → spatially-weighted ROI averaging.

### 9.4 Mobile Support

- Progressive Web App (PWA) with service worker
- Reduced chunk_length (90 frames / 3s) for faster warm-up
- Forced POS mode (lighter compute)
- Camera resolution cap (320×240)

### 9.5 ONNX Web Runtime

Bring inference back to the client entirely:
- Export EfficientPhys → ONNX
- Run via ONNX Runtime Web (WASM backend)
- Eliminates server dependency
- Enables fully offline operation

### 9.6 Recording & Playback

- Record raw frames + signals for offline analysis
- Export session data as CSV/HDF5
- Replay mode for debugging
- Comparison view: live vs recorded

---

## Appendix A: Key References

| Paper | Relevance |
|-------|-----------|
| Liu et al., "EfficientPhys" (CVPR 2023) | Primary model architecture |
| Liu & Poh et al., "rPPG-Toolbox" (NeurIPS 2023 D&B) | Codebase foundation, checkpoints, evaluation |
| Wang et al., "Algorithmic Principles of Remote PPG" (IEEE TBME 2017) | POS method |
| Yu et al., "TS-CAN" (NeurIPS 2020) | Alternative model, attention-based rPPG |

---

## Appendix B: Startup Checklist

```bash
# 1. Clone rPPG-Toolbox
git clone https://github.com/ubicomplab/rPPG-Toolbox.git rppg_toolbox

# 2. Verify checkpoint exists
ls rppg_toolbox/final_model_release/PURE_EfficientPhys.pth

# 3. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 4. Install dependencies
pip install fastapi uvicorn torch scipy numpy opencv-python-headless

# 5. Verify model loads
python -c "
import torch
from rppg_toolbox.neural_methods.model.EfficientPhys import EfficientPhys
model = EfficientPhys(frame_depth=10, img_size=72)
sd = torch.load('rppg_toolbox/final_model_release/PURE_EfficientPhys.pth', map_location='cpu')
model.load_state_dict(sd)
print('Model loaded successfully. Parameters:', sum(p.numel() for p in model.parameters()))
"

# 6. Start dev server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

---

*Document version: 1.0 — April 2026*
*Project codename: VitalSense*
*Author: Malek Dinari*
