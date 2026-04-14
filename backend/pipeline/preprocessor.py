import numpy as np
import cv2
from typing import Tuple

class FramePreprocessor:
    def __init__(self, target_size: Tuple[int,int] = (72, 72),
                 color_space: str = "RGB",
                 normalize: bool = True):
        self.target_size = target_size
        self.color_space = color_space
        self.normalize = normalize
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        # Ensure correct size
        if frame.shape[:2] != self.target_size:
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
            
        # Convert channel ordering if needed. We assume we receive RGB from browser
        # but just in case we need OpenCV BGR, handle it here.
        if self.color_space == "RGB" and len(frame.shape) == 3:
            pass # Keep RGB
            
        processed_frame = frame.astype(np.float32)
        if self.normalize:
            processed_frame = processed_frame / 255.0
            
        return processed_frame
