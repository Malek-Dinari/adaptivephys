import numpy as np
from enum import Enum
import cv2

class FrameStatus(Enum):
    VALID = 1
    SKIP = 2
    REJECT = 3

class FrameValidator:
    def __init__(self, config=None):
        from backend.config import PipelineConfig
        self.config = config or PipelineConfig()
        self.last_good_frame = None

    def validate(self, frame: np.ndarray) -> FrameStatus:
        if frame is None or frame.size == 0:
            return FrameStatus.REJECT
        
        # Check excessive darkness or brightness
        mean_val = np.mean(frame)
        if mean_val < self.config.min_brightness or mean_val > self.config.max_brightness:
            if self.last_good_frame is not None:
                return FrameStatus.SKIP
            return FrameStatus.REJECT
            
        # Check blur using Laplacian variance
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.shape[-1] == 3 else frame
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur < self.config.min_sharpness:
            if self.last_good_frame is not None:
                return FrameStatus.SKIP
            return FrameStatus.REJECT

        self.last_good_frame = frame
        return FrameStatus.VALID
