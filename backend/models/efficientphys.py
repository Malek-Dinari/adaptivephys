import torch
import numpy as np
import os
from backend.models.base import BaseRPPGModel, ModelMetadata

class EfficientPhysModel(BaseRPPGModel):
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

    def _load_model(self, path: str):
        import sys
        
        # Add rppg_toolbox to path to import EfficientPhys properly
        # Make sure rppg_toolbox is correctly cloned and structured
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
        try:
            from rppg_toolbox.neural_methods.model.EfficientPhys import EfficientPhys
            model = EfficientPhys(frame_depth=10, img_size=72)
            if os.path.exists(path):
                state_dict = torch.load(path, map_location=self.device)
                model.load_state_dict(state_dict)
            else:
                print(f"Warning: EfficientPhys checkpoint not found at {path}")
            model = model.to(self.device)
            return model
        except Exception as e:
            print(f"Failed to load EfficientPhys model: {e}")
            raise e

    @torch.no_grad()
    def predict(self, chunk: np.ndarray) -> np.ndarray:
        # Expected shape (B, C, T, H, W) based on Toolbox EfficientPhys. Wait, let's look at rppg_toolbox
        # Architecture says T, C, H, W. We need to conform to what EfficientPhys takes.
        # rPPG-Toolbox uses (B, C, T, H, W). 
        # chunk is (T, C, H, W)
        tensor = torch.from_numpy(chunk).float()
        
        # transpose to (C, T, H, W)
        tensor = tensor.permute(1, 0, 2, 3)
        # Add B dim -> (1, C, T, H, W)
        tensor = tensor.unsqueeze(0).to(self.device)
        
        output, _, _, _ = self.model(tensor)
        
        # output is likely (B, T)
        return output.squeeze().cpu().numpy()

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(name="efficientphys")
