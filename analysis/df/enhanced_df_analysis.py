"""
Enhanced Derivatives Fields (DF) analysis with parameter optimization.
Improved version with grid search and better feature generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d


class EnhancedDFGenerator:
    """
    Enhanced Derivatives Fields generator with parameter optimization.
    Based on Johnston et al. (2020) with improvements.
    """
    
    def __init__(self, derivative_order: int = 1, field_size: int = 32):
        """
        Initialize enhanced DF generator.
        
        Args:
            derivative_order: Order of derivative (1 or 2)
            field_size: Size of DF grid (optimized via cross-validation)
        """
        self.derivative_order = derivative_order
        self.field_size = field_size
        
    def compute_derivatives(self, lightcurve: np.ndarray) -> np.ndarray:
        """
        Compute derivatives with noise reduction.
        
        Args:
            lightcurve: Input light curve
            
        Returns:
            Derivatives array
        """
        # Apply slight smoothing before computing derivatives
        smoothed = uniform_filter1d(lightcurve, size=3)
        
        derivatives = smoothed.copy()
        for _ in range(self.derivative_order):
            derivatives = np.diff(derivatives)
            
        return derivatives
    
    def create_phase_space(self, lightcurve: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create enhanced phase space representation.
        
        Args:
            lightcurve: Input light curve
            
        Returns:
            Tuple of (values, derivatives)
        """
        if self.derivative_order == 1:
            values = lightcurve[:-1]
            derivatives = self.compute_derivatives(lightcurve)
        elif self.derivative_order == 2:
            values = lightcurve[:-2]
            # Second derivative with better numerical stability
            second_diff = np.diff(lightcurve, n=2)
            derivatives = second_diff
        else:
            raise ValueError("Derivative order must be 1 or 2")
            
        return values, derivatives
    
    def generate_df_field(self, lightcurve: np.ndarray) -> np.ndarray:
        """
        Generate enhanced DF field with adaptive binning.
        
        Args:
            lightcurve: Input light curve
            
        Returns:
            2D derivatives field
        """
        values, derivatives = self.create_phase_space(lightcurve)
        
        # Remove outliers for better binning
        value_q1, value_q3 = np.percentile(values, [25, 75])
        deriv_q1, deriv_q3 = np.percentile(derivatives, [25, 75])
        
        value_iqr = value_q3 - value_q1
        deriv_iqr = deriv_q3 - deriv_q1
        
        # Define bounds with some margin beyond IQR
        value_bounds = [value_q1 - 1.5 * value_iqr, value_q3 + 1.5 * value_iqr]
        deriv_bounds = [deriv_q1 - 1.5 * deriv_iqr, deriv_q3 + 1.5 * deriv_iqr]
        
        # Create 2D histogram with adaptive bounds
        field, x_edges, y_edges = np.histogram2d(
            values, derivatives,
            bins=self.field_size,
            range=[value_bounds, deriv_bounds],
            density=True
        )
        
        # Apply Gaussian smoothing to reduce noise
        from scipy.ndimage import gaussian_filter
        field_smoothed = gaussian_filter(field, sigma=0.5)
        
        return field_smoothed
    
    def generate_df_features(self, lightcurve: np.ndarray) -> np.ndarray:
        """
        Generate enhanced DF features with additional statistics.
        
        Args:
            lightcurve: Input light curve
            
        Returns:
            Enhanced DF feature vector
        """
        df_field = self.generate_df_field(lightcurve)
        
        # Basic flattened field
        basic_features = df_field.flatten()
        
        # Additional statistical features from the field
        additional_features = [
            np.mean(df_field),
            np.std(df_field),
            np.max(df_field),
            np.sum(df_field > np.mean(df_field)),  # Number of above-average bins
            np.trace(df_field) if df_field.shape[0] == df_field.shape[1] else 0,
            np.sum(np.diagonal(df_field)) if df_field.shape[0] == df_field.shape[1] else 0,
            np.sum(np.diag(np.fliplr(df_field))) if df_field.shape[0] == df_field.shape[1] else 0
        ]
        
        return np.concatenate([basic_features, additional_features])


class EnhancedDFOptions:
    """
    Enhanced DF configuration with parameter optimization.
    """
    
    def __init__(self):
        self.derivative_order = 1
        self.field_size = 32
        self.normalize_lightcurves = True
        self.smooth_lightcurves = True
        self.smoothing_window = 3
        self.optimize_parameters = True
        
        # Parameter search ranges for optimization
        self.param_ranges = {
            'derivative_order': [1, 2],
            'field_size': [16, 20, 24, 28, 32, 36, 40],
            'smoothing_window': [3, 5, 7]
        }
        
    def to_dict(self) -> Dict:
        """Convert options to dictionary."""
        return {
            'derivative_order': self.derivative_order,
            'field_size': self.field_size,
            'normalize_lightcurves': self.normalize_lightcurves,
            'smooth_lightcurves': self.smooth_lightcurves,
            'smoothing_window': self.smoothing_window
        }


class EnhancedDFAnalysis(BaseEstimator, ClassifierMixin):
    """
    Enhanced DF analysis with parameter optimization and better classification.
    """
    
    def __init__(self, options: EnhancedDFOptions = None, base_classifier=None):
        """
        Initialize enhanced DF classifier.
        
        Args:
            options: DF configuration options
            base_classifier: Base classifier (default: optimized RF)
        """
        self.options = options if options else EnhancedDFOptions()
        self.base_classifier = base_classifier
        self.best_params_ = {}
        self.df_generator = None
        
    def _optimize_parameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Optimize DF parameters using cross-validation.
        
        Args:
            X: Training light curves
            y: Training labels
            
        Returns:
            Best parameters
        """
        print("Optimizing DF parameters...")
        
        best_score = 0.0
        best_params = {}
        
        # Grid search over parameter combinations
        for derivative_order in self.options.param_ranges['derivative_order']:
            for field_size in self.options.param_ranges['field_size']:
                for smoothing_window in self.options.param_ranges['smoothing_window']:
                    
                    # Create temporary generator with these parameters
                    temp_generator = EnhancedDFGenerator(
                        derivative_order=derivative_order,
                        field_size=field_size
                    )
                    
                    # Extract features
                    try:
                        features_list = []
                        for lc in X:
                            # Apply smoothing if enabled
                            if self.options.smooth_lightcurves:
                                from scipy import ndimage
                                processed_lc = ndimage.uniform_filter1d(lc, size=smoothing_window)
                            else:
                                processed_lc = lc
                                
                            # Normalize if enabled
                            if self.options.normalize_lightcurves:
                                processed_lc = (processed_lc - np.mean(processed_lc)) / (np.std(processed_lc) + 1e-8)
                            
                            features = temp_generator.generate_df_features(processed_lc)
                            features_list.append(features)
                        
                        df_features = np.array(features_list)
                        
                        # Quick cross-validation with simple classifier
                        temp_classifier = KNeighborsClassifier(n_neighbors=3)
                        scores = cross_val_score(temp_classifier, df_features, y, cv=3, scoring='f1_weighted')
                        mean_score = np.mean(scores)
                        
                        if mean_score > best_score:
                            best_score = mean_score
                            best_params = {
                                'derivative_order': derivative_order,
                                'field_size': field_size,
                                'smoothing_window': smoothing_window
                            }
                            
                        print(f"  Params: order={derivative_order}, size={field_size}, smooth={smoothing_window} -> F1={mean_score:.4f}")
                        
                    except Exception as e:
                        print(f"  Failed with params: {derivative_order}, {field_size}, {smoothing_window} - {e}")
                        continue
        
        print(f"Best DF parameters: {best_params} (F1-Score: {best_score:.4f})")
        return best_params
    
    def _preprocess_lightcurve(self, lightcurve: np.ndarray) -> np.ndarray:
        """
        Preprocess light curve with optimized parameters.
        """
        lc = lightcurve.copy()
        
        # Smooth first if enabled
        if self.options.smooth_lightcurves:
            from scipy import ndimage
            smoothing_window = self.best_params_.get('smoothing_window', self.options.smoothing_window)
            lc = ndimage.uniform_filter1d(lc, size=smoothing_window)
        
        # Then normalize
        if self.options.normalize_lightcurves:
            mean_val = np.mean(lc)
            std_val = np.std(lc)
            if std_val > 1e-8:
                lc = (lc - mean_val) / std_val
                
        return lc
    
    def extract_df_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract optimized DF features.
        
        Args:
            X: Light curves array
            
        Returns:
            DF features array
        """
        if self.df_generator is None:
            raise ValueError("Must fit classifier before extracting features")
            
        n_samples = X.shape[0]
        sample_features = self.df_generator.generate_df_features(self._preprocess_lightcurve(X[0]))
        feature_dim = len(sample_features)
        
        df_features = np.zeros((n_samples, feature_dim))
        
        for i, lightcurve in enumerate(X):
            preprocessed = self._preprocess_lightcurve(lightcurve)
            df_features[i] = self.df_generator.generate_df_features(preprocessed)
        
        return df_features
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit enhanced DF classifier with parameter optimization.
        
        Args:
            X: Training light curves
            y: Training labels
        """
        # Optimize parameters if enabled
        if self.options.optimize_parameters:
            self.best_params_ = self._optimize_parameters(X, y)
            
            # Update options with best parameters
            self.options.derivative_order = self.best_params_['derivative_order']
            self.options.field_size = self.best_params_['field_size']
            self.options.smoothing_window = self.best_params_['smoothing_window']
        
        # Create DF generator with optimized parameters
        self.df_generator = EnhancedDFGenerator(
            derivative_order=self.options.derivative_order,
            field_size=self.options.field_size
        )
        
        # Extract DF features
        df_features = self.extract_df_features(X)
        
        # Train base classifier
        if self.base_classifier is None:
            # Use optimized Random Forest as default
            self.base_classifier = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        
        self.base_classifier.fit(df_features, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using enhanced DF features.
        
        Args:
            X: Test light curves
            
        Returns:
            Predicted labels
        """
        df_features = self.extract_df_features(X)
        return self.base_classifier.predict(df_features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Test light curves
            
        Returns:
            Class probabilities
        """
        df_features = self.extract_df_features(X)
        if hasattr(self.base_classifier, 'predict_proba'):
            return self.base_classifier.predict_proba(df_features)
        else:
            raise ValueError("Base classifier does not support probability prediction")
    
    def visualize_df_field(self, lightcurve: np.ndarray, title: str = "Enhanced DF Field"):
        """
        Visualize enhanced DF field.
        
        Args:
            lightcurve: Input light curve
            title: Plot title
        """
        if self.df_generator is None:
            raise ValueError("Must fit classifier before visualization")
            
        preprocessed = self._preprocess_lightcurve(lightcurve)
        df_field = self.df_generator.generate_df_field(preprocessed)
        
        plt.figure(figsize=(12, 5))
        
        # Original light curve
        plt.subplot(1, 3, 1)
        plt.plot(lightcurve)
        plt.title("Original Light Curve")
        plt.xlabel("Time")
        plt.ylabel("Magnitude")
        
        # Preprocessed light curve
        plt.subplot(1, 3, 2)
        plt.plot(preprocessed)
        plt.title("Preprocessed Light Curve")
        plt.xlabel("Time")
        plt.ylabel("Normalized Magnitude")
        
        # DF field
        plt.subplot(1, 3, 3)
        plt.imshow(df_field.T, origin='lower', aspect='auto', cmap='viridis')
        plt.title(f"{title}\n(Order: {self.options.derivative_order}, Size: {self.options.field_size})")
        plt.xlabel("Value Bins")
        plt.ylabel("Derivative Bins")
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()


def run_enhanced_df_analysis(X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            optimize_params: bool = True) -> Dict:
    """
    Run enhanced DF analysis with parameter optimization.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        optimize_params: Whether to optimize parameters
        
    Returns:
        Results dictionary
    """
    # Configure enhanced options
    options = EnhancedDFOptions()
    options.optimize_parameters = optimize_params
    
    # Create and train enhanced DF classifier
    enhanced_df = EnhancedDFAnalysis(options=options)
    enhanced_df.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = enhanced_df.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Enhanced DF Analysis Results:")
    print(f"  Best Parameters: {enhanced_df.best_params_}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    return {
        'method': 'Enhanced Derivatives Fields',
        'classifier': enhanced_df,
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': y_pred,
        'best_params': enhanced_df.best_params_
    }