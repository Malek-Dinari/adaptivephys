import collections
import threading
import numpy as np
from typing import Optional

class TemporalBuffer:
    def __init__(self, chunk_length: int = 180,
                 stride: int = 15,
                 max_buffer: int = 900):
        self.buffer = collections.deque(maxlen=max_buffer)
        self.chunk_length = chunk_length
        self.stride = stride
        self._frames_since_last_chunk = 0
        self._lock = threading.Lock()
    
    def add_frame(self, frame: np.ndarray, timestamp: float) -> Optional[np.ndarray]:
        with self._lock:
            self.buffer.append((frame, timestamp))
            self._frames_since_last_chunk += 1
            
            if (len(self.buffer) >= self.chunk_length and 
                self._frames_since_last_chunk >= self.stride):
                self._frames_since_last_chunk = 0
                return self._extract_chunk()
        return None
    
    def _extract_chunk(self) -> np.ndarray:
        frames = list(self.buffer)[-self.chunk_length:]
        # Stack frames, transpose to (T, C, H, W)
        return np.stack([f for f, _ in frames]).transpose(0, 3, 1, 2)
