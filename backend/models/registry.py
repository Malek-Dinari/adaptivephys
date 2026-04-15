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
        print("[ModelRegistry] Initializing POS model...")
        self._models["pos"] = POSModel()
        print("[ModelRegistry] ✓ POS model ready (fallback)")
        
        try:
            print(f"[ModelRegistry] Initializing EfficientPhys from: {self.config.checkpoint_path}")
            from backend.models.efficientphys import EfficientPhysModel
            self._models["efficientphys"] = EfficientPhysModel(
                checkpoint_path=self.config.checkpoint_path,
                device=self.config.device
            )
            print("[ModelRegistry] ✓ EfficientPhys model ready (primary)")
        except Exception as e:
            print(f"[ModelRegistry] ✗ Could not initialize EfficientPhys: {e}")
            print(f"[ModelRegistry] → Falling back to POS model for this session")

    def get_active(self) -> BaseRPPGModel:
        if self._active and self._active in self._models:
            print(f"[ModelRegistry] Using model: {self._active}")
            return self._models[self._active]
        print(f"[ModelRegistry] ⚠ Model '{self._active}' not available, using POS")
        return self._models["pos"]

    def fallback(self, reason: str):
        print(f"[ModelRegistry] Model {self._active} failed ({reason}). Falling back to POS.")
        self._active = "pos"
