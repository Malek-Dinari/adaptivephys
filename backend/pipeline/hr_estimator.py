import collections
import numpy as np
from dataclasses import dataclass
from backend.pipeline.signal_processor import ProcessedSignal

@dataclass
class HRResult:
    hr_bpm: float
    confidence: float
    method: str

class HREstimator:
    def __init__(self, fs: float = 30.0,
                 hr_min_bpm: float = 40.0,
                 hr_max_bpm: float = 200.0,
                 history_size: int = 10):
        self.fs = fs
        self.hr_min = hr_min_bpm
        self.hr_max = hr_max_bpm
        self.hr_history = collections.deque(maxlen=history_size)
    
    def estimate(self, processed: ProcessedSignal) -> HRResult:
        freqs = processed.freqs
        spectrum = processed.spectrum
        
        mask = (freqs >= self.hr_min/60) & (freqs <= self.hr_max/60)
        cardiac_freqs = freqs[mask]
        cardiac_mag = spectrum[mask]
        
        if len(cardiac_mag) == 0:
            return HRResult(hr_bpm=0, confidence=0.0, method='fft_peak')
        
        peak_idx = np.argmax(cardiac_mag)
        peak_freq = cardiac_freqs[peak_idx]
        hr_bpm = peak_freq * 60.0
        
        peak_energy = cardiac_mag[peak_idx] ** 2
        total_energy = np.sum(cardiac_mag ** 2)
        snr = peak_energy / (total_energy - peak_energy + 1e-10)
        confidence = min(1.0, snr / 5.0)
        
        hr_bpm = self._smooth(hr_bpm, confidence)
        
        return HRResult(hr_bpm=hr_bpm, confidence=confidence, method='fft_peak')
    
    def _smooth(self, hr: float, confidence: float) -> float:
        if len(self.hr_history) > 0:
            median_hr = np.median([h for h, _ in self.hr_history])
            if abs(hr - median_hr) > 30 and confidence < 0.5:
                hr = median_hr
        
        self.hr_history.append((hr, confidence))
        
        weights = []
        values = []
        for i, (h, c) in enumerate(self.hr_history):
            recency_weight = (i + 1) / len(self.hr_history)
            weights.append(c * recency_weight)
            values.append(h)
        
        sum_weights = sum(weights)
        if sum_weights == 0:
            return hr
        return np.average(values, weights=weights)
