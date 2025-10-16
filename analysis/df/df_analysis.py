"""
Derivatives Fields (DF) analysis module for variable star classification.
Python port of df-analysis Java module (DFAnalysis.java, DFGenerator.java).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


class DFGenerator:
    """
    Generates derivatives fields representation from light curves.
    Ports DFGenerator.java functionality.
    """
    
    def __init__(self, derivative_order: int = 1, field_size: int = 32):
        """
        Initialize DF generator.
        
        Args:
            derivative_order: Order of derivative to compute (1 or 2)
            field_size: Size of the derivatives field grid
        """
        self.derivative_order = derivative_order
        self.field_size = field_size
        
    def compute_derivatives(self, lightcurve: np.ndarray) -> np.ndarray:
        """
        Compute derivatives of the light curve.
        
        Args:
            lightcurve: Input light curve array
            
        Returns:
            Array of derivatives
        """
        derivatives = lightcurve.copy()
        
        for _ in range(self.derivative_order):
            derivatives = np.diff(derivatives)
            
        return derivatives
    
    def create_phase_space(self, lightcurve: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create phase space representation (value vs derivative).
        
        Args:
            lightcurve: Input light curve
            
        Returns:
            Tuple of (values, derivatives) for phase space plot
        """
        if self.derivative_order == 1:
            # For first derivative: (x_i, x_{i+1} - x_i)
            values = lightcurve[:-1]
            derivatives = np.diff(lightcurve)
        elif self.derivative_order == 2:
            # For second derivative: (x_i, x_{i+2} - 2*x_{i+1} + x_i)
            values = lightcurve[:-2]
            second_diff = lightcurve[2:] - 2*lightcurve[1:-1] + lightcurve[:-2]
            derivatives = second_diff
        else:
            raise ValueError("Derivative order must be 1 or 2")
            
        return values, derivatives
    
    def generate_df_field(self, lightcurve: np.ndarray) -> np.ndarray:
        """
        Generate derivatives field from light curve.
        
        Args:
            lightcurve: Input light curve
            
        Returns:
            2D field representing derivatives distribution
        """
        values, derivatives = self.create_phase_space(lightcurve)
        
        # Create 2D histogram (derivatives field)
        field, x_edges, y_edges = np.histogram2d(
            values, derivatives, 
            bins=self.field_size,
            density=True
        )
        
        return field
    
    def generate_df_features(self, lightcurve: np.ndarray) -> np.ndarray:
        """
        Generate flattened DF features vector.
        
        Args:
            lightcurve: Input light curve
            
        Returns:
            Flattened derivatives field as feature vector
        """
        df_field = self.generate_df_field(lightcurve)
        return df_field.flatten()


class DFOptions:
    """
    Configuration options for DF analysis.
    Ports DFOptions.java functionality.
    """
    
    def __init__(self):
        self.derivative_order = 1
        self.field_size = 32
        self.normalize_lightcurves = True
        self.smooth_lightcurves = False
        self.smoothing_window = 3
        
    def to_dict(self) -> Dict:
        """Convert options to dictionary."""
        return {
            'derivative_order': self.derivative_order,
            'field_size': self.field_size,
            'normalize_lightcurves': self.normalize_lightcurves,
            'smooth_lightcurves': self.smooth_lightcurves,
            'smoothing_window': self.smoothing_window
        }


class DFAnalysis(BaseEstimator, ClassifierMixin):
    """
    Derivatives Fields analysis classifier.
    Ports DFAnalysis.java functionality.
    """
    
    def __init__(self, options: DFOptions = None, base_classifier=None):
        """
        Initialize DF analysis classifier.
        
        Args:
            options: DF configuration options
            base_classifier: Base classifier to use on DF features
        """
        self.options = options if options else DFOptions()
        self.base_classifier = base_classifier
        self.df_generator = DFGenerator(
            derivative_order=self.options.derivative_order,
            field_size=self.options.field_size
        )
        
    def _preprocess_lightcurve(self, lightcurve: np.ndarray) -> np.ndarray:
        """
        Preprocess light curve according to options.
        """
        lc = lightcurve.copy()
        
        # Normalize
        if self.options.normalize_lightcurves:
            lc = (lc - np.mean(lc)) / np.std(lc)
            
        # Smooth
        if self.options.smooth_lightcurves:
            from scipy import ndimage
            lc = ndimage.uniform_filter1d(lc, size=self.options.smoothing_window)
            
        return lc
    
    def extract_df_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract DF features from multiple light curves.
        
        Args:
            X: Array of light curves (n_samples, n_timesteps)
            
        Returns:
            DF features array (n_samples, field_size^2)
        """
        n_samples = X.shape[0]
        feature_dim = self.options.field_size ** 2
        df_features = np.zeros((n_samples, feature_dim))
        
        for i, lightcurve in enumerate(X):
            preprocessed = self._preprocess_lightcurve(lightcurve)
            df_features[i] = self.df_generator.generate_df_features(preprocessed)
            
        return df_features
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the DF classifier.
        
        Args:
            X: Training light curves
            y: Training labels
        """
        # Extract DF features
        df_features = self.extract_df_features(X)
        
        # Fit base classifier if provided
        if self.base_classifier:
            self.base_classifier.fit(df_features, y)
        else:
            # Use simple kNN as default
            from sklearn.neighbors import KNeighborsClassifier
            self.base_classifier = KNeighborsClassifier(n_neighbors=3)
            self.base_classifier.fit(df_features, y)
            
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for light curves.
        
        Args:
            X: Test light curves
            
        Returns:
            Predicted class labels
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
    
    def visualize_df_field(self, lightcurve: np.ndarray, title: str = "Derivatives Field"):
        """
        Visualize derivatives field for a single light curve.
        
        Args:
            lightcurve: Input light curve
            title: Plot title
        """
        preprocessed = self._preprocess_lightcurve(lightcurve)
        df_field = self.df_generator.generate_df_field(preprocessed)
        
        plt.figure(figsize=(10, 4))
        
        # Original light curve
        plt.subplot(1, 2, 1)
        plt.plot(lightcurve)
        plt.title("Light Curve")
        plt.xlabel("Time")
        plt.ylabel("Magnitude")
        
        # Derivatives field
        plt.subplot(1, 2, 2)
        plt.imshow(df_field.T, origin='lower', aspect='auto', cmap='viridis')
        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Derivative")
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()


def run_df_analysis(X_train: np.ndarray, y_train: np.ndarray, 
                   X_test: np.ndarray, y_test: np.ndarray,
                   options: DFOptions = None) -> Dict:
    """
    Run complete DF analysis pipeline.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        options: DF analysis options
        
    Returns:
        Dictionary with results
    """
    # Initialize classifier
    df_classifier = DFAnalysis(options=options)
    
    # Train
    df_classifier.fit(X_train, y_train)
    
    # Predict
    y_pred = df_classifier.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'classifier': df_classifier,
        'accuracy': accuracy,
        'classification_report': report,
        'predictions': y_pred
    }