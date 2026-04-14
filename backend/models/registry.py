from typing import Dict
from backend.models.base import BaseRPPGModel
from backend.config import PipelineConfig

class ModelRegistry:
    def __init__(self, config: PipelineConfig):
        self._models: Dict[str, BaseRPPGModel] = {}
        self._active: str = None
        self.config = config
        self._initialize_models()
        self._active = config.primary_model
        if self._active not in self._models:
            self._active = config.fallback_model

    def _initialize_models(self):
        from backend.models.pos import POSModel
        self._models["pos"] = POSModel()
        
        try:
            from backend.models.efficientphys import EfficientPhysModel
            self._models["efficientphys"] = EfficientPhysModel(
                checkpoint_path=self.config.checkpoint_path,
                device=self.config.device
            )
        except Exception as e:
            print(f"Warning: Could not initialize EfficientPhys: {e}")

    def get_active(self) -> BaseRPPGModel:
        if self._active and self._active in self._models:
            return self._models[self._active]
        return self._models["pos"]

    def fallback(self, reason: str):
        print(f"Model {self._active} failed ({reason}). Falling back to POS.")
        self._active = "pos"
