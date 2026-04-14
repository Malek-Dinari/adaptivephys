import numpy as np
import scipy.signal
from typing import Tuple
from dataclasses import dataclass

@dataclass
class ProcessedSignal:
    raw: np.ndarray
    detrended: np.ndarray
    filtered: np.ndarray
    spectrum: np.ndarray
    freqs: np.ndarray

class SignalProcessor:
    def __init__(self, fs: float = 30.0,
                 hr_min_bpm: float = 40.0,
                 hr_max_bpm: float = 200.0,
                 filter_order: int = 4):
        self.fs = fs
        self.low_hz = hr_min_bpm / 60.0
        self.high_hz = hr_max_bpm / 60.0
        self.filter_order = filter_order
        self._design_filter()
    
    def _design_filter(self):
        nyq = self.fs / 2.0
        self.sos = scipy.signal.butter(
            self.filter_order,
            [self.low_hz / nyq, self.high_hz / nyq],
            btype='bandpass',
            output='sos'
        )
    
    def process(self, raw_signal: np.ndarray) -> ProcessedSignal:
        detrended = self._detrend(raw_signal)
        filtered = scipy.signal.sosfiltfilt(self.sos, detrended)
        freqs, spectrum = self._compute_fft(filtered)
        
        return ProcessedSignal(
            raw=raw_signal,
            detrended=detrended,
            filtered=filtered,
            spectrum=spectrum,
            freqs=freqs
        )
    
    def _detrend(self, signal: np.ndarray, order: int = 5) -> np.ndarray:
        t = np.arange(len(signal))
        poly = np.polyfit(t, signal, order)
        trend = np.polyval(poly, t)
        return signal - trend
    
    def _compute_fft(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(signal)
        n_fft = 2 ** int(np.ceil(np.log2(n)))
        windowed = signal * np.hanning(n)
        fft_vals = np.fft.rfft(windowed, n=n_fft)
        freqs = np.fft.rfftfreq(n_fft, d=1.0/self.fs)
        magnitude = np.abs(fft_vals)
        return freqs, magnitude
