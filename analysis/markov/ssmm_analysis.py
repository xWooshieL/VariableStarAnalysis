"""
Semi-Supervised Markov Models (SSMM) for variable star classification.
Python port of ssmm-analysis Java module.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')


class SlottedTSGenerator:
    """
    Generates slotted time series representation.
    Ports SlottedTSGenerator.java functionality.
    """
    
    def __init__(self, n_slots: int = 50, slot_method: str = 'uniform'):
        """
        Initialize slotted time series generator.
        
        Args:
            n_slots: Number of time slots
            slot_method: Method for creating slots ('uniform', 'quantile')
        """
        self.n_slots = n_slots
        self.slot_method = slot_method
        
    def generate_slots(self, lightcurve: np.ndarray, time: np.ndarray = None) -> np.ndarray:
        """
        Generate slotted representation of time series.
        
        Args:
            lightcurve: Input light curve
            time: Time array (if None, assumes uniform sampling)
            
        Returns:
            Slotted time series representation
        """
        if time is None:
            time = np.arange(len(lightcurve))
            
        if self.slot_method == 'uniform':
            # Uniform time slots
            time_bins = np.linspace(time.min(), time.max(), self.n_slots + 1)
        elif self.slot_method == 'quantile':
            # Quantile-based slots
            time_bins = np.quantile(time, np.linspace(0, 1, self.n_slots + 1))
        else:
            raise ValueError(f"Unknown slot method: {self.slot_method}")
            
        # Digitize time points into slots
        slot_indices = np.digitize(time, time_bins) - 1
        slot_indices = np.clip(slot_indices, 0, self.n_slots - 1)
        
        # Average values in each slot
        slotted_series = np.zeros(self.n_slots)
        slot_counts = np.zeros(self.n_slots)
        
        for i, slot_idx in enumerate(slot_indices):
            slotted_series[slot_idx] += lightcurve[i]
            slot_counts[slot_idx] += 1
            
        # Handle empty slots
        non_empty_slots = slot_counts > 0
        slotted_series[non_empty_slots] /= slot_counts[non_empty_slots]
        
        # Interpolate empty slots
        if not np.all(non_empty_slots):
            from scipy.interpolate import interp1d
            valid_indices = np.where(non_empty_slots)[0]
            if len(valid_indices) > 1:
                interp_func = interp1d(valid_indices, slotted_series[valid_indices], 
                                     kind='linear', fill_value='extrapolate')
                empty_indices = np.where(~non_empty_slots)[0]
                slotted_series[empty_indices] = interp_func(empty_indices)
                
        return slotted_series


class MarkovModelGenerator:
    """
    Generates Markov model features from time series.
    Ports MarkovModelGenerator.java functionality.
    """
    
    def __init__(self, n_states: int = 10, quantize_method: str = 'uniform'):
        """
        Initialize Markov model generator.
        
        Args:
            n_states: Number of states in Markov model
            quantize_method: Method for quantizing values ('uniform', 'kmeans')
        """
        self.n_states = n_states
        self.quantize_method = quantize_method
        self.quantizer = None
        
    def quantize_series(self, lightcurve: np.ndarray) -> np.ndarray:
        """
        Quantize continuous time series into discrete states.
        
        Args:
            lightcurve: Input light curve
            
        Returns:
            Quantized state sequence
        """
        if self.quantize_method == 'uniform':
            # Uniform quantization
            min_val, max_val = lightcurve.min(), lightcurve.max()
            bins = np.linspace(min_val, max_val, self.n_states + 1)
            states = np.digitize(lightcurve, bins) - 1
            states = np.clip(states, 0, self.n_states - 1)
            
        elif self.quantize_method == 'kmeans':
            # K-means quantization
            if self.quantizer is None:
                self.quantizer = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
                states = self.quantizer.fit_predict(lightcurve.reshape(-1, 1))
            else:
                states = self.quantizer.predict(lightcurve.reshape(-1, 1))
        else:
            raise ValueError(f"Unknown quantization method: {self.quantize_method}")
            
        return states
    
    def build_transition_matrix(self, states: np.ndarray) -> np.ndarray:
        """
        Build transition matrix from state sequence.
        
        Args:
            states: Sequence of discrete states
            
        Returns:
            Transition probability matrix
        """
        transition_counts = np.zeros((self.n_states, self.n_states))
        
        # Count transitions
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            transition_counts[current_state, next_state] += 1
            
        # Normalize to probabilities
        row_sums = transition_counts.sum(axis=1)
        transition_matrix = np.zeros_like(transition_counts)
        
        for i in range(self.n_states):
            if row_sums[i] > 0:
                transition_matrix[i] = transition_counts[i] / row_sums[i]
            else:
                # Uniform distribution for states with no outgoing transitions
                transition_matrix[i] = 1.0 / self.n_states
                
        return transition_matrix
    
    def extract_markov_features(self, lightcurve: np.ndarray) -> np.ndarray:
        """
        Extract Markov model features from light curve.
        
        Args:
            lightcurve: Input light curve
            
        Returns:
            Flattened transition matrix as feature vector
        """
        states = self.quantize_series(lightcurve)
        transition_matrix = self.build_transition_matrix(states)
        return transition_matrix.flatten()


class SSMMGenerator:
    """
    Semi-Supervised Markov Model generator.
    Ports SSMMGenerator.java functionality.
    """
    
    def __init__(self, n_components: int = 3, n_states: int = 10):
        """
        Initialize SSMM generator.
        
        Args:
            n_components: Number of HMM components
            n_states: Number of hidden states per component
        """
        self.n_components = n_components
        self.n_states = n_states
        self.hmm_models = {}
        
    def train_class_hmms(self, X: np.ndarray, y: np.ndarray) -> Dict[int, hmm.GaussianHMM]:
        """
        Train HMM for each class.
        
        Args:
            X: Training light curves
            y: Training labels
            
        Returns:
            Dictionary mapping class labels to trained HMMs
        """
        unique_classes = np.unique(y)
        
        for class_label in unique_classes:
            # Get samples for this class
            class_mask = (y == class_label)
            class_samples = X[class_mask]
            
            # Concatenate all samples for this class
            concatenated_data = []
            lengths = []
            
            for sample in class_samples:
                # Remove NaN values and reshape
                clean_sample = sample[~np.isnan(sample)].reshape(-1, 1)
                concatenated_data.append(clean_sample)
                lengths.append(len(clean_sample))
                
            if len(concatenated_data) > 0:
                all_data = np.vstack(concatenated_data)
                
                # Train HMM
                model = hmm.GaussianHMM(n_components=self.n_states, 
                                      covariance_type="full", 
                                      random_state=42,
                                      n_iter=100)
                try:
                    model.fit(all_data, lengths)
                    self.hmm_models[class_label] = model
                except Exception as e:
                    print(f"Warning: Failed to train HMM for class {class_label}: {e}")
                    
        return self.hmm_models
    
    def extract_ssmm_features(self, lightcurve: np.ndarray) -> np.ndarray:
        """
        Extract SSMM features using trained HMMs.
        
        Args:
            lightcurve: Input light curve
            
        Returns:
            SSMM feature vector (log-likelihoods for each class HMM)
        """
        clean_lc = lightcurve[~np.isnan(lightcurve)].reshape(-1, 1)
        
        features = []
        for class_label, model in self.hmm_models.items():
            try:
                log_likelihood = model.score(clean_lc)
                features.append(log_likelihood)
            except:
                features.append(-np.inf)
                
        return np.array(features)


class SSMMAnalysis(BaseEstimator, ClassifierMixin):
    """
    Semi-Supervised Markov Model analysis classifier.
    Ports SSMMAnalysis.java functionality.
    """
    
    def __init__(self, n_states: int = 10, use_hmm: bool = True):
        """
        Initialize SSMM classifier.
        
        Args:
            n_states: Number of states for Markov models
            use_hmm: Whether to use HMM (True) or simple Markov chains (False)
        """
        self.n_states = n_states
        self.use_hmm = use_hmm
        
        if use_hmm:
            self.feature_generator = SSMMGenerator(n_states=n_states)
        else:
            self.feature_generator = MarkovModelGenerator(n_states=n_states)
            
        self.base_classifier = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit SSMM classifier.
        
        Args:
            X: Training light curves
            y: Training labels
        """
        if self.use_hmm:
            # Train class-specific HMMs
            self.feature_generator.train_class_hmms(X, y)
        else:
            # For simple Markov chains, we'll use kNN on transition matrices
            from sklearn.neighbors import KNeighborsClassifier
            
            # Extract Markov features
            markov_features = []
            for lc in X:
                features = self.feature_generator.extract_markov_features(lc)
                markov_features.append(features)
                
            markov_features = np.array(markov_features)
            
            # Train classifier
            self.base_classifier = KNeighborsClassifier(n_neighbors=3)
            self.base_classifier.fit(markov_features, y)
            
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes using SSMM.
        
        Args:
            X: Test light curves
            
        Returns:
            Predicted class labels
        """
        if self.use_hmm:
            # Use HMM log-likelihoods for classification
            predictions = []
            
            for lc in X:
                features = self.feature_generator.extract_ssmm_features(lc)
                # Predict class with highest log-likelihood
                predicted_class = list(self.feature_generator.hmm_models.keys())[np.argmax(features)]
                predictions.append(predicted_class)
                
            return np.array(predictions)
        else:
            # Use base classifier on Markov features
            markov_features = []
            for lc in X:
                features = self.feature_generator.extract_markov_features(lc)
                markov_features.append(features)
                
            markov_features = np.array(markov_features)
            return self.base_classifier.predict(markov_features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Test light curves
            
        Returns:
            Class probabilities
        """
        if self.use_hmm:
            # Convert log-likelihoods to probabilities
            n_samples = len(X)
            n_classes = len(self.feature_generator.hmm_models)
            probabilities = np.zeros((n_samples, n_classes))
            
            class_labels = list(self.feature_generator.hmm_models.keys())
            
            for i, lc in enumerate(X):
                log_likelihoods = self.feature_generator.extract_ssmm_features(lc)
                # Convert to probabilities using softmax
                exp_scores = np.exp(log_likelihoods - np.max(log_likelihoods))
                probabilities[i] = exp_scores / np.sum(exp_scores)
                
            return probabilities
        else:
            markov_features = []
            for lc in X:
                features = self.feature_generator.extract_markov_features(lc)
                markov_features.append(features)
                
            markov_features = np.array(markov_features)
            return self.base_classifier.predict_proba(markov_features)