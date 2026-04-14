from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass

@dataclass
class ModelMetadata:
    name: str

class BaseRPPGModel(ABC):
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
        pass
