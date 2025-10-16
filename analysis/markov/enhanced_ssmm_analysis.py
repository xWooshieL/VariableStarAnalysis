"""
Enhanced Semi-Supervised Markov Models (SSMM) with parameter optimization.
Improved version with GMMHMM and better state quantization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')


class EnhancedSlottedTSGenerator:
    """
    Enhanced slotted time series generator with adaptive binning.
    """
    
    def __init__(self, n_slots: int = 50, slot_method: str = 'adaptive'):
        """
        Initialize enhanced slotted TS generator.
        
        Args:
            n_slots: Number of time slots
            slot_method: Slotting method ('uniform', 'quantile', 'adaptive')
        """
        self.n_slots = n_slots
        self.slot_method = slot_method
        
    def generate_slots(self, lightcurve: np.ndarray, time: np.ndarray = None) -> np.ndarray:
        """
        Generate enhanced slotted representation.
        
        Args:
            lightcurve: Input light curve
            time: Time array (optional)
            
        Returns:
            Slotted time series
        """
        if time is None:
            time = np.arange(len(lightcurve))
            
        if self.slot_method == 'uniform':
            time_bins = np.linspace(time.min(), time.max(), self.n_slots + 1)
        elif self.slot_method == 'quantile':
            time_bins = np.quantile(time, np.linspace(0, 1, self.n_slots + 1))
        elif self.slot_method == 'adaptive':
            # Adaptive binning based on data density
            sorted_time = np.sort(time)
            n_per_bin = len(time) // self.n_slots
            time_bins = [sorted_time[0]]
            
            for i in range(1, self.n_slots):
                idx = i * n_per_bin
                if idx < len(sorted_time):
                    time_bins.append(sorted_time[idx])
            time_bins.append(sorted_time[-1])
            time_bins = np.array(time_bins)
        else:
            raise ValueError(f"Unknown slot method: {self.slot_method}")
            
        # Digitize and average
        slot_indices = np.digitize(time, time_bins) - 1
        slot_indices = np.clip(slot_indices, 0, self.n_slots - 1)
        
        slotted_series = np.zeros(self.n_slots)
        slot_counts = np.zeros(self.n_slots)
        
        for i, slot_idx in enumerate(slot_indices):
            slotted_series[slot_idx] += lightcurve[i]
            slot_counts[slot_idx] += 1
        
        # Average and interpolate empty slots
        non_empty = slot_counts > 0
        slotted_series[non_empty] /= slot_counts[non_empty]
        
        if not np.all(non_empty):
            from scipy.interpolate import interp1d
            valid_indices = np.where(non_empty)[0]
            if len(valid_indices) > 1:
                interp_func = interp1d(valid_indices, slotted_series[valid_indices],
                                     kind='linear', fill_value='extrapolate')
                empty_indices = np.where(~non_empty)[0]
                slotted_series[empty_indices] = interp_func(empty_indices)
        
        return slotted_series


class EnhancedMarkovModelGenerator:
    """
    Enhanced Markov model generator with better quantization.
    """
    
    def __init__(self, n_states: int = 10, quantize_method: str = 'adaptive'):
        """
        Initialize enhanced Markov generator.
        
        Args:
            n_states: Number of states
            quantize_method: Quantization method ('uniform', 'kmeans', 'gmm', 'adaptive')
        """
        self.n_states = n_states
        self.quantize_method = quantize_method
        self.quantizer = None
        self.state_boundaries = None
        
    def quantize_series(self, lightcurve: np.ndarray) -> np.ndarray:
        """
        Enhanced quantization with multiple methods.
        
        Args:
            lightcurve: Input light curve
            
        Returns:
            Quantized state sequence
        """
        if self.quantize_method == 'uniform':
            if self.state_boundaries is None:
                min_val, max_val = lightcurve.min(), lightcurve.max()
                self.state_boundaries = np.linspace(min_val, max_val, self.n_states + 1)
            states = np.digitize(lightcurve, self.state_boundaries) - 1
            states = np.clip(states, 0, self.n_states - 1)
            
        elif self.quantize_method == 'kmeans':
            if self.quantizer is None:
                self.quantizer = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
                states = self.quantizer.fit_predict(lightcurve.reshape(-1, 1))
            else:
                states = self.quantizer.predict(lightcurve.reshape(-1, 1))
                
        elif self.quantize_method == 'gmm':
            if self.quantizer is None:
                self.quantizer = GaussianMixture(n_components=self.n_states, random_state=42)
                self.quantizer.fit(lightcurve.reshape(-1, 1))
            states = self.quantizer.predict(lightcurve.reshape(-1, 1))
            
        elif self.quantize_method == 'adaptive':
            # Adaptive quantization based on data distribution
            if self.state_boundaries is None:
                percentiles = np.linspace(0, 100, self.n_states + 1)
                self.state_boundaries = np.percentile(lightcurve, percentiles)
                # Ensure boundaries are strictly increasing
                self.state_boundaries = np.unique(self.state_boundaries)
                if len(self.state_boundaries) < self.n_states + 1:
                    # Fallback to uniform if not enough unique values
                    min_val, max_val = lightcurve.min(), lightcurve.max()
                    self.state_boundaries = np.linspace(min_val, max_val, self.n_states + 1)
            
            states = np.digitize(lightcurve, self.state_boundaries) - 1
            states = np.clip(states, 0, self.n_states - 1)
        else:
            raise ValueError(f"Unknown quantization method: {self.quantize_method}")
            
        return states
    
    def build_enhanced_transition_matrix(self, states: np.ndarray) -> np.ndarray:
        """
        Build enhanced transition matrix with smoothing.
        
        Args:
            states: State sequence
            
        Returns:
            Smoothed transition matrix
        """
        transition_counts = np.zeros((self.n_states, self.n_states))
        
        # Count transitions
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            if 0 <= current_state < self.n_states and 0 <= next_state < self.n_states:
                transition_counts[current_state, next_state] += 1
        
        # Add Laplace smoothing to avoid zero probabilities
        alpha = 0.1
        transition_counts += alpha
        
        # Normalize to probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        transition_matrix = transition_counts / row_sums
        
        return transition_matrix
    
    def extract_enhanced_markov_features(self, lightcurve: np.ndarray) -> np.ndarray:
        """
        Extract enhanced Markov features including higher-order statistics.
        
        Args:
            lightcurve: Input light curve
            
        Returns:
            Enhanced feature vector
        """
        states = self.quantize_series(lightcurve)
        transition_matrix = self.build_enhanced_transition_matrix(states)
        
        # Basic features: flattened transition matrix
        basic_features = transition_matrix.flatten()
        
        # Additional features
        additional_features = [
            # Stationary distribution (eigenvalue = 1)
            *self._compute_stationary_distribution(transition_matrix),
            # Entropy measures
            self._compute_entropy(transition_matrix),
            # Mixing measures
            self._compute_mixing_time(transition_matrix),
            # State occupancy statistics
            *self._compute_state_statistics(states)
        ]
        
        return np.concatenate([basic_features, additional_features])
    
    def _compute_stationary_distribution(self, transition_matrix: np.ndarray) -> np.ndarray:
        """Compute stationary distribution of transition matrix."""
        try:
            eigenvals, eigenvecs = np.linalg.eig(transition_matrix.T)
            stationary_idx = np.argmax(np.real(eigenvals))
            stationary = np.real(eigenvecs[:, stationary_idx])
            stationary = stationary / np.sum(stationary)
            return stationary
        except:
            return np.ones(self.n_states) / self.n_states
    
    def _compute_entropy(self, transition_matrix: np.ndarray) -> float:
        """Compute entropy of transition matrix."""
        entropy = 0.0
        for i in range(self.n_states):
            row = transition_matrix[i]
            row_entropy = -np.sum(row * np.log(row + 1e-10))
            entropy += row_entropy
        return entropy / self.n_states
    
    def _compute_mixing_time(self, transition_matrix: np.ndarray) -> float:
        """Estimate mixing time from second largest eigenvalue."""
        try:
            eigenvals = np.linalg.eigvals(transition_matrix)
            eigenvals = np.sort(np.abs(eigenvals))[::-1]
            if len(eigenvals) > 1:
                second_largest = eigenvals[1]
                if second_largest > 1e-10:
                    return -1.0 / np.log(second_largest)
        except:
            pass
        return 1.0  # Default mixing time
    
    def _compute_state_statistics(self, states: np.ndarray) -> List[float]:
        """Compute statistics about state occupancy."""
        unique_states, counts = np.unique(states, return_counts=True)
        total_states = len(states)
        
        # State diversity (number of unique states)
        n_unique = len(unique_states)
        
        # State distribution entropy
        probs = counts / total_states
        state_entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Most frequent state proportion
        max_proportion = np.max(probs) if len(probs) > 0 else 0.0
        
        return [n_unique, state_entropy, max_proportion]


class EnhancedSSMMGenerator:
    """
    Enhanced SSMM generator with GMMHMM and better initialization.
    """
    
    def __init__(self, n_components: int = 5, n_mix: int = 2, covariance_type: str = "full"):
        """
        Initialize enhanced SSMM generator.
        
        Args:
            n_components: Number of HMM states
            n_mix: Number of mixture components per state
            covariance_type: Covariance type for GMM
        """
        self.n_components = n_components
        self.n_mix = n_mix
        self.covariance_type = covariance_type
        self.hmm_models = {}
        
    def train_enhanced_hmms(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train enhanced HMM models for each class.
        
        Args:
            X: Training light curves
            y: Training labels
            
        Returns:
            Dictionary of trained HMMs
        """
        unique_classes = np.unique(y)
        
        for class_label in unique_classes:
            class_mask = (y == class_label)
            class_samples = X[class_mask]
            
            if len(class_samples) < 2:
                print(f"Warning: Too few samples for class {class_label}")
                continue
                
            # Concatenate and prepare data for HMM
            concatenated_data = []
            lengths = []
            
            for sample in class_samples:
                clean_sample = sample[~np.isnan(sample)]
                if len(clean_sample) > 5:  # Minimum length
                    concatenated_data.append(clean_sample.reshape(-1, 1))
                    lengths.append(len(clean_sample))
            
            if len(concatenated_data) == 0:
                print(f"Warning: No valid samples for class {class_label}")
                continue
                
            all_data = np.vstack(concatenated_data)
            
            # Try GMMHMM first, fallback to GaussianHMM if it fails
            try:
                model = hmm.GMMHMM(
                    n_components=self.n_components,
                    n_mix=self.n_mix,
                    covariance_type=self.covariance_type,
                    random_state=42,
                    n_iter=200,
                    tol=1e-4
                )
                
                # Better initialization
                model.fit(all_data, lengths)
                self.hmm_models[class_label] = model
                print(f"Trained GMMHMM for class {class_label}")
                
            except Exception as e1:
                print(f"GMMHMM failed for class {class_label}: {e1}")
                try:
                    # Fallback to simpler GaussianHMM
                    model = hmm.GaussianHMM(
                        n_components=max(2, self.n_components // 2),
                        covariance_type="full",
                        random_state=42,
                        n_iter=100
                    )
                    model.fit(all_data, lengths)
                    self.hmm_models[class_label] = model
                    print(f"Trained GaussianHMM for class {class_label}")
                    
                except Exception as e2:
                    print(f"All HMM training failed for class {class_label}: {e2}")
                    
        return self.hmm_models
    
    def extract_enhanced_ssmm_features(self, lightcurve: np.ndarray) -> np.ndarray:
        """
        Extract enhanced SSMM features with additional statistics.
        
        Args:
            lightcurve: Input light curve
            
        Returns:
            Enhanced SSMM feature vector
        """
        clean_lc = lightcurve[~np.isnan(lightcurve)]
        if len(clean_lc) < 5:
            # Return zero features for too short sequences
            return np.zeros(len(self.hmm_models) + 5)
            
        clean_lc = clean_lc.reshape(-1, 1)
        
        # Basic log-likelihoods
        log_likelihoods = []
        posteriors_stats = []
        
        for class_label, model in self.hmm_models.items():
            try:
                # Log-likelihood
                log_likelihood = model.score(clean_lc)
                log_likelihoods.append(log_likelihood)
                
                # Posterior probabilities statistics
                if hasattr(model, 'predict_proba'):
                    try:
                        posteriors = model.predict_proba(clean_lc)
                        # Statistics of posterior distributions
                        posteriors_stats.extend([
                            np.mean(posteriors),
                            np.std(posteriors),
                            np.max(posteriors),
                            np.min(posteriors)
                        ])
                    except:
                        posteriors_stats.extend([0, 0, 0, 0])
                else:
                    posteriors_stats.extend([0, 0, 0, 0])
                    
            except Exception as e:
                print(f"Error computing features for class {class_label}: {e}")
                log_likelihoods.append(-np.inf)
                posteriors_stats.extend([0, 0, 0, 0])
        
        # Additional sequence-level features
        sequence_features = [
            len(clean_lc),  # Sequence length
            np.std(clean_lc.flatten())  # Sequence variability
        ]
        
        all_features = log_likelihoods + posteriors_stats + sequence_features
        return np.array(all_features)


class EnhancedSSMMAnalysis(BaseEstimator, ClassifierMixin):
    """
    Enhanced SSMM analysis with parameter optimization.
    """
    
    def __init__(self, n_states: int = 10, n_components: int = 5, n_mix: int = 2,
                 quantize_method: str = 'adaptive', optimize_params: bool = True):
        """
        Initialize enhanced SSMM classifier.
        
        Args:
            n_states: Number of Markov states
            n_components: Number of HMM components
            n_mix: Number of mixture components (for GMMHMM)
            quantize_method: Quantization method
            optimize_params: Whether to optimize parameters
        """
        self.n_states = n_states
        self.n_components = n_components
        self.n_mix = n_mix
        self.quantize_method = quantize_method
        self.optimize_params = optimize_params
        
        self.hmm_generator = None
        self.markov_generator = None
        self.best_params_ = {}
        self.use_hmm = True
        
    def _optimize_parameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Optimize SSMM parameters using cross-validation.
        
        Args:
            X: Training data
            y: Labels
            
        Returns:
            Best parameters
        """
        print("Optimizing SSMM parameters...")
        
        param_grid = {
            'n_states': [5, 8, 10, 12],
            'n_components': [3, 5, 7],
            'quantize_method': ['adaptive', 'gmm', 'kmeans']
        }
        
        best_score = 0.0
        best_params = {}
        
        for n_states in param_grid['n_states']:
            for n_components in param_grid['n_components']:
                for quantize_method in param_grid['quantize_method']:
                    try:
                        # Create temporary generators
                        temp_hmm_gen = EnhancedSSMMGenerator(n_components=n_components)
                        temp_markov_gen = EnhancedMarkovModelGenerator(
                            n_states=n_states, 
                            quantize_method=quantize_method
                        )
                        
                        # Train HMM models
                        temp_hmm_gen.train_enhanced_hmms(X, y)
                        
                        if len(temp_hmm_gen.hmm_models) == 0:
                            continue
                        
                        # Extract features
                        features_list = []
                        for lc in X[:min(200, len(X))]:  # Use subset for speed
                            hmm_features = temp_hmm_gen.extract_enhanced_ssmm_features(lc)
                            markov_features = temp_markov_gen.extract_enhanced_markov_features(lc)
                            combined_features = np.concatenate([hmm_features, markov_features])
                            features_list.append(combined_features)
                        
                        if len(features_list) == 0:
                            continue
                        
                        features = np.array(features_list)
                        y_subset = y[:len(features)]
                        
                        # Quick evaluation
                        from sklearn.neighbors import KNeighborsClassifier
                        temp_classifier = KNeighborsClassifier(n_neighbors=3)
                        scores = cross_val_score(temp_classifier, features, y_subset, 
                                               cv=min(3, len(np.unique(y_subset))), 
                                               scoring='f1_weighted')
                        mean_score = np.mean(scores)
                        
                        if mean_score > best_score:
                            best_score = mean_score
                            best_params = {
                                'n_states': n_states,
                                'n_components': n_components,
                                'quantize_method': quantize_method
                            }
                        
                        print(f"  Params: states={n_states}, comp={n_components}, "
                              f"quant={quantize_method} -> F1={mean_score:.4f}")
                        
                    except Exception as e:
                        print(f"  Failed with params {n_states}, {n_components}, {quantize_method}: {e}")
                        continue
        
        if best_params:
            print(f"Best SSMM parameters: {best_params} (F1-Score: {best_score:.4f})")
        else:
            print("Parameter optimization failed, using defaults")
            best_params = {
                'n_states': self.n_states,
                'n_components': self.n_components,
                'quantize_method': self.quantize_method
            }
            
        return best_params
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit enhanced SSMM classifier.
        
        Args:
            X: Training data
            y: Labels
        """
        # Optimize parameters if enabled
        if self.optimize_params:
            self.best_params_ = self._optimize_parameters(X, y)
            self.n_states = self.best_params_['n_states']
            self.n_components = self.best_params_['n_components']
            self.quantize_method = self.best_params_['quantize_method']
        
        # Create generators with optimized parameters
        self.hmm_generator = EnhancedSSMMGenerator(n_components=self.n_components, n_mix=self.n_mix)
        self.markov_generator = EnhancedMarkovModelGenerator(
            n_states=self.n_states, 
            quantize_method=self.quantize_method
        )
        
        # Train HMM models
        self.hmm_generator.train_enhanced_hmms(X, y)
        
        # Use HMM if models were successfully trained
        self.use_hmm = len(self.hmm_generator.hmm_models) > 0
        
        if not self.use_hmm:
            print("Warning: No HMM models trained, falling back to simple Markov chains")
            from sklearn.ensemble import RandomForestClassifier
            
            # Extract Markov features only
            markov_features = []
            for lc in X:
                features = self.markov_generator.extract_enhanced_markov_features(lc)
                markov_features.append(features)
            
            markov_features = np.array(markov_features)
            self.base_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.base_classifier.fit(markov_features, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using enhanced SSMM.
        
        Args:
            X: Test data
            
        Returns:
            Predictions
        """
        if self.use_hmm:
            # Use HMM log-likelihoods
            predictions = []
            
            for lc in X:
                hmm_features = self.hmm_generator.extract_enhanced_ssmm_features(lc)
                
                # Use only log-likelihoods for prediction
                n_classes = len(self.hmm_generator.hmm_models)
                log_likelihoods = hmm_features[:n_classes]
                
                if np.all(np.isinf(log_likelihoods)):
                    # Fallback to most common class
                    predicted_class = list(self.hmm_generator.hmm_models.keys())[0]
                else:
                    class_labels = list(self.hmm_generator.hmm_models.keys())
                    best_idx = np.argmax(log_likelihoods)
                    predicted_class = class_labels[best_idx]
                
                predictions.append(predicted_class)
            
            return np.array(predictions)
        else:
            # Use base classifier with Markov features
            markov_features = []
            for lc in X:
                features = self.markov_generator.extract_enhanced_markov_features(lc)
                markov_features.append(features)
            
            markov_features = np.array(markov_features)
            return self.base_classifier.predict(markov_features)


def run_enhanced_ssmm_analysis(X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray,
                              optimize_params: bool = True) -> Dict:
    """
    Run enhanced SSMM analysis with parameter optimization.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        optimize_params: Whether to optimize parameters
        
    Returns:
        Results dictionary
    """
    # Create and train enhanced SSMM
    enhanced_ssmm = EnhancedSSMMAnalysis(optimize_params=optimize_params)
    enhanced_ssmm.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = enhanced_ssmm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Enhanced SSMM Analysis Results:")
    print(f"  Best Parameters: {enhanced_ssmm.best_params_}")
    print(f"  Using HMM: {enhanced_ssmm.use_hmm}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    return {
        'method': 'Enhanced SSMM',
        'classifier': enhanced_ssmm,
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': y_pred,
        'best_params': enhanced_ssmm.best_params_
    }