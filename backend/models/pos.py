import numpy as np
from backend.models.base import BaseRPPGModel, ModelMetadata

class POSModel(BaseRPPGModel):
    def predict(self, chunk: np.ndarray) -> np.ndarray:
        # chunk shape: (T, C, H, W). T=frames, C=3 (RGB), H=72, W=72
        T, C, H, W = chunk.shape
        
        # Spatial average to get color trace: (T, C)
        # Assuming channel 0=R, 1=G, 2=B based on preprocessor OpenCV conversion
        trace = np.mean(chunk, axis=(2, 3))
        
        sig = np.zeros(T)
        l = 30 # sliding window length (e.g. 1 sec at 30fps)
        if T < l:
           l = T
           
        for t in range(T - l + 1):
            # c: (l, 3)
            c = trace[t:t+l]
            # mean color: (3,)
            mean_color = np.mean(c, axis=0)
            
            # normalize by mean
            # c_norm: (l, 3)
            with np.errstate(divide='ignore', invalid='ignore'):
                c_norm = c / mean_color
                c_norm[~np.isfinite(c_norm)] = 1.0 # fallback if mean_color is 0

            # Projections
            X = c_norm[:, 1] - c_norm[:, 2]         # G - B
            Y = c_norm[:, 1] + c_norm[:, 2] - 2 * c_norm[:, 0] # G + B - 2R
            
            # projection alpha
            h_X = X - np.mean(X)
            h_Y = Y - np.mean(Y)
            
            std_X = np.std(h_X)
            std_Y = np.std(h_Y)
            if std_Y == 0:
                alpha = 1.0
            else:
                alpha = std_X / std_Y
                
            h = h_X - alpha * h_Y
            # accumulate
            sig[t:t+l] += h - np.mean(h)
            
        return sig
    
    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(name="POS")
