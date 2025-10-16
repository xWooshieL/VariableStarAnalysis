"""
Feature extraction module for variable star light curves.
Python port of feature-extraction Java module.
"""

import numpy as np
from scipy import signal, fft
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from astropy.stats import LombScargle
from typing import Dict, List, Tuple
import pandas as pd


class LightCurveFeatureExtractor:
    """
    Extracts features from variable star light curves.
    Ports functionality from ApacheFFT.java, PCA.java, LombNormalizedPeriodogram.java
    """
    
    def __init__(self):
        self.pca = None
        
    def extract_statistical_features(self, lightcurve: np.ndarray) -> Dict[str, float]:
        """
        Extract basic statistical features from light curve.
        """
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(lightcurve)
        features['std'] = np.std(lightcurve)
        features['variance'] = np.var(lightcurve)
        features['min'] = np.min(lightcurve)
        features['max'] = np.max(lightcurve)
        features['range'] = features['max'] - features['min']
        features['median'] = np.median(lightcurve)
        
        # Higher order moments
        features['skewness'] = skew(lightcurve)
        features['kurtosis'] = kurtosis(lightcurve)
        
        # Percentiles
        features['q25'] = np.percentile(lightcurve, 25)
        features['q75'] = np.percentile(lightcurve, 75)
        features['iqr'] = features['q75'] - features['q25']
        
        return features
    
    def extract_fft_features(self, lightcurve: np.ndarray, sample_rate: float = 1.0) -> Dict[str, float]:
        """
        Extract FFT-based frequency domain features.
        Ports ApacheFFT.java functionality.
        """
        # Compute FFT
        fft_values = fft.fft(lightcurve)
        freqs = fft.fftfreq(len(lightcurve), d=1.0/sample_rate)
        
        # Power spectral density
        psd = np.abs(fft_values)**2
        
        # Keep only positive frequencies
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        psd_pos = psd[pos_mask]
        
        features = {}
        
        # Dominant frequency
        max_power_idx = np.argmax(psd_pos)
        features['dominant_frequency'] = freqs_pos[max_power_idx]
        features['max_power'] = psd_pos[max_power_idx]
        
        # Power in different frequency bands
        total_power = np.sum(psd_pos)
        features['total_power'] = total_power
        
        # Spectral centroid
        features['spectral_centroid'] = np.sum(freqs_pos * psd_pos) / total_power
        
        # Spectral spread
        features['spectral_spread'] = np.sqrt(np.sum(((freqs_pos - features['spectral_centroid']) ** 2) * psd_pos) / total_power)
        
        # Spectral rolloff (95% of power)
        cumsum_power = np.cumsum(psd_pos)
        rolloff_idx = np.where(cumsum_power >= 0.95 * total_power)[0]
        if len(rolloff_idx) > 0:
            features['spectral_rolloff'] = freqs_pos[rolloff_idx[0]]
        else:
            features['spectral_rolloff'] = freqs_pos[-1]
            
        return features
    
    def lomb_scargle_periodogram(self, time: np.ndarray, lightcurve: np.ndarray, 
                                min_period: float = 0.1, max_period: float = 10.0) -> Dict[str, float]:
        """
        Compute Lomb-Scargle periodogram for unevenly sampled data.
        Ports LombNormalizedPeriodogram.java functionality.
        """
        if len(time) != len(lightcurve):
            # If time array not provided, create uniform sampling
            time = np.arange(len(lightcurve))
            
        # Frequency grid
        min_freq = 1.0 / max_period
        max_freq = 1.0 / min_period
        freqs = np.linspace(min_freq, max_freq, 1000)
        
        # Compute Lomb-Scargle periodogram
        ls = LombScargle(time, lightcurve)
        power = ls.power(freqs)
        
        features = {}
        
        # Best period
        max_power_idx = np.argmax(power)
        features['best_period'] = 1.0 / freqs[max_power_idx]
        features['best_period_power'] = power[max_power_idx]
        
        # False Alarm Probability
        features['best_period_fap'] = ls.false_alarm_probability(power[max_power_idx])
        
        # Power statistics
        features['periodogram_mean_power'] = np.mean(power)
        features['periodogram_std_power'] = np.std(power)
        features['periodogram_max_power'] = np.max(power)
        
        return features
    
    def extract_pca_features(self, lightcurves: np.ndarray, n_components: int = 5) -> np.ndarray:
        """
        Extract PCA features from multiple light curves.
        Ports PCA.java functionality.
        """
        if self.pca is None:
            self.pca = PCA(n_components=n_components)
            pca_features = self.pca.fit_transform(lightcurves)
        else:
            pca_features = self.pca.transform(lightcurves)
            
        return pca_features
    
    def extract_derivatives_features(self, lightcurve: np.ndarray) -> Dict[str, float]:
        """
        Extract features based on derivatives (velocity and acceleration).
        """
        # First derivative (velocity)
        velocity = np.diff(lightcurve)
        
        # Second derivative (acceleration)
        acceleration = np.diff(velocity)
        
        features = {}
        
        # Velocity features
        features['velocity_mean'] = np.mean(velocity)
        features['velocity_std'] = np.std(velocity)
        features['velocity_max'] = np.max(np.abs(velocity))
        
        # Acceleration features
        features['acceleration_mean'] = np.mean(acceleration)
        features['acceleration_std'] = np.std(acceleration)
        features['acceleration_max'] = np.max(np.abs(acceleration))
        
        # Zero crossings
        features['velocity_zero_crossings'] = len(np.where(np.diff(np.sign(velocity)))[0])
        features['acceleration_zero_crossings'] = len(np.where(np.diff(np.sign(acceleration)))[0])
        
        return features
    
    def extract_all_features(self, lightcurve: np.ndarray, time: np.ndarray = None) -> Dict[str, float]:
        """
        Extract all available features from a single light curve.
        """
        all_features = {}
        
        # Statistical features
        all_features.update(self.extract_statistical_features(lightcurve))
        
        # FFT features
        all_features.update(self.extract_fft_features(lightcurve))
        
        # Lomb-Scargle periodogram
        if time is not None:
            all_features.update(self.lomb_scargle_periodogram(time, lightcurve))
        
        # Derivatives features
        all_features.update(self.extract_derivatives_features(lightcurve))
        
        return all_features


def extract_features_batch(lightcurves: np.ndarray, time_arrays: List[np.ndarray] = None) -> pd.DataFrame:
    """
    Extract features from multiple light curves in batch.
    
    Args:
        lightcurves: Array of shape (n_samples, n_timesteps)
        time_arrays: Optional list of time arrays for each light curve
        
    Returns:
        DataFrame with extracted features
    """
    extractor = LightCurveFeatureExtractor()
    
    features_list = []
    for i, lc in enumerate(lightcurves):
        time = time_arrays[i] if time_arrays else None
        features = extractor.extract_all_features(lc, time)
        features_list.append(features)
    
    return pd.DataFrame(features_list)