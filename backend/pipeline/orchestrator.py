import numpy as np
from typing import Optional
from dataclasses import dataclass
from backend.config import PipelineConfig
from backend.pipeline.validator import FrameValidator, FrameStatus
from backend.pipeline.preprocessor import FramePreprocessor
from backend.pipeline.buffer import TemporalBuffer
from backend.pipeline.signal_processor import SignalProcessor, ProcessedSignal
from backend.pipeline.hr_estimator import HREstimator, HRResult
from backend.models.registry import ModelRegistry

@dataclass
class PipelineResult:
    hr: Optional[HRResult] = None
    signal: Optional[ProcessedSignal] = None
    model_name: str = ""
    frame_count: int = 0
    status: str = "success"
    message: str = ""
    progress: float = 0.0

    @classmethod
    def warming_up(cls, progress: float):
        return cls(status="warmup", progress=progress)

    @classmethod
    def quality_warning(cls, frame_count: int):
        return cls(status="warning", message="Low signal quality", frame_count=frame_count)

    @classmethod
    def success(cls, hr, signal, model_name, frame_count):
        return cls(hr=hr, signal=signal, model_name=model_name, frame_count=frame_count, status="success")

class RPPGPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.validator = FrameValidator(config)
        self.preprocessor = FramePreprocessor((72,72))
        self.buffer = TemporalBuffer(
            chunk_length=config.chunk_length,
            stride=config.stride
        )
        self.model_registry = ModelRegistry(config)
        self.signal_processor = SignalProcessor(fs=config.fps)
        self.hr_estimator = HREstimator(fs=config.fps)
        
        self._is_warmed_up = False
        self._total_frames = 0
        self._valid_frames = 0
    
    async def process_frame(self, frame: np.ndarray, timestamp: float) -> Optional[PipelineResult]:
        self._total_frames += 1
        
        status = self.validator.validate(frame)
        if status == FrameStatus.REJECT:
            return PipelineResult.quality_warning(self._total_frames)
        if status == FrameStatus.SKIP:
            frame = self.validator.last_good_frame
            
        self._valid_frames += 1
        
        processed = self.preprocessor.process(frame)
        
        chunk = self.buffer.add_frame(processed, timestamp)
        if chunk is None:
            progress = len(self.buffer.buffer) / self.buffer.chunk_length
            return PipelineResult.warming_up(progress)
            
        self._is_warmed_up = True
        
        model = self.model_registry.get_active()
        try:
            raw_signal = model.predict(chunk)
        except Exception as e:
            self.model_registry.fallback(str(e))
            model = self.model_registry.get_active()
            raw_signal = model.predict(chunk)
            
        processed_signal = self.signal_processor.process(raw_signal)
        hr_result = self.hr_estimator.estimate(processed_signal)
        
        return PipelineResult.success(
            hr=hr_result,
            signal=processed_signal,
            model_name=model.get_metadata().name,
            frame_count=self._total_frames
        )
