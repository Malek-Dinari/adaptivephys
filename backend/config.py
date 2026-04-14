from dataclasses import dataclass
import os

@dataclass
class PipelineConfig:
    # Temporal
    fps: float = 30.0
    chunk_length: int = 180
    stride: int = 15
    max_buffer_seconds: float = 30.0
    
    # Model
    primary_model: str = "efficientphys"
    fallback_model: str = "pos"
    device: str = os.getenv("DEVICE", "cpu")
    checkpoint_path: str = "rppg_toolbox/final_model_release/PURE_EfficientPhys.pth"
    
    # Signal processing
    hr_min_bpm: float = 40.0
    hr_max_bpm: float = 200.0
    bandpass_order: int = 4
    detrend_order: int = 5
    
    # Validation
    min_brightness: float = 30.0
    max_brightness: float = 230.0
    min_sharpness: float = 10.0
    max_invalid_ratio: float = 0.3
    
    @classmethod
    def gpu_dev(cls) -> 'PipelineConfig':
        return cls(device="cuda", stride=6)
    
    @classmethod
    def cpu_prod(cls) -> 'PipelineConfig':
        return cls(device="cpu", stride=30)
    
    @classmethod
    def onnx_prod(cls) -> 'PipelineConfig':
        return cls(device="cpu", stride=15, primary_model="efficientphys_onnx")
