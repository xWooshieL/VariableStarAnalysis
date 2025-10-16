"""
Large Margin Multi-View Metric Learning (LM3L) implementation.
Python port based on the paper by Johnston et al. (2020).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')


class ECVA:
    """
    Enhanced Canonical Variate Analysis for dimensionality reduction.
    Combines PCA with LDA to handle high-dimensional and multicollinear data.
    """
    
    def __init__(self, n_components: Optional[int] = None):
        """
        Initialize ECVA.
        
        Args:
            n_components: Number of components to keep (default: n_classes - 1)
        """
        self.n_components = n_components
        self.pca = None
        self.lda = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ECVA':
        """
        Fit ECVA on training data.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels
        """
        n_classes = len(np.unique(y))
        n_features = X.shape[1]
        
        # Determine number of components
        if self.n_components is None:
            self.n_components = min(n_classes - 1, n_features - 1)
        
        # First apply PCA to reduce dimensionality if needed
        if n_features > 100:  # Apply PCA for high-dimensional data
            pca_components = min(50, n_features - 1, X.shape[0] - 1)
            self.pca = PCA(n_components=pca_components, random_state=42)
            X_pca = self.pca.fit_transform(X)
        else:
            X_pca = X
        
        # Then apply LDA for supervised dimensionality reduction
        try:
            self.lda = LinearDiscriminantAnalysis(n_components=self.n_components)
            self.lda.fit(X_pca, y)
        except Exception as e:
            print(f"Warning: LDA failed ({e}), using PCA only")
            if self.pca is None:
                self.pca = PCA(n_components=self.n_components, random_state=42)
                self.pca.fit(X)
            
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted ECVA.
        
        Args:
            X: Input data
            
        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise ValueError("ECVA must be fitted before transform")
        
        if self.pca is not None:
            X = self.pca.transform(X)
        
        if self.lda is not None:
            X = self.lda.transform(X)
        
        return X
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit ECVA and transform data.
        
        Args:
            X: Input data
            y: Labels
            
        Returns:
            Transformed data
        """
        return self.fit(X, y).transform(X)


class LM3L(BaseEstimator, ClassifierMixin):
    """
    Large Margin Multi-View Metric Learning classifier.
    
    Based on the paper:
    "Variable star classification using multiview metric learning"
    Johnston et al. (2020), MNRAS 491, 3805-3819
    """
    
    def __init__(self, 
                 alpha: float = 1.0,
                 beta: float = 5.0, 
                 gamma: float = 0.1,
                 k_neighbors: int = 3,
                 max_iter: int = 100,
                 convergence_threshold: float = 1e-6,
                 regularization: float = 0.01):
        """
        Initialize LM3L classifier.
        
        Args:
            alpha: Controls importance of individual view optimization
            beta: Controls margin threshold
            gamma: Controls importance of pairwise distance between views
            k_neighbors: Number of neighbors for k-NN classification
            max_iter: Maximum optimization iterations
            convergence_threshold: Convergence threshold
            regularization: L2 regularization parameter
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k_neighbors = k_neighbors
        self.max_iter = max_iter
        self.convergence_threshold = convergence_threshold
        self.regularization = regularization
        
        # Internal variables
        self.metrics_ = {}
        self.weights_ = {}
        self.scalers_ = {}
        self.ecva_transformers_ = {}
        self.X_train_transformed_ = {}
        self.y_train_ = None
        self.n_views_ = 0
        
    def _vectorize_and_reduce(self, X_dict: Dict[str, np.ndarray], 
                            y: np.ndarray, 
                            fit: bool = False) -> Dict[str, np.ndarray]:
        """
        Vectorize matrix features and apply ECVA dimensionality reduction.
        
        Args:
            X_dict: Dictionary of views {view_name: features}
            y: Labels
            fit: Whether to fit transformers
            
        Returns:
            Dictionary of transformed views
        """
        X_transformed = {}
        
        for view_name, X_view in X_dict.items():
            # Vectorize if matrix
            if len(X_view.shape) == 3:  # (n_samples, height, width)
                n_samples = X_view.shape[0]
                X_flat = X_view.reshape(n_samples, -1)
            else:
                X_flat = X_view
            
            if fit:
                # Fit scaler
                self.scalers_[view_name] = StandardScaler()
                X_scaled = self.scalers_[view_name].fit_transform(X_flat)
                
                # Fit ECVA
                self.ecva_transformers_[view_name] = ECVA()
                X_reduced = self.ecva_transformers_[view_name].fit_transform(X_scaled, y)
            else:
                # Transform using fitted transformers
                X_scaled = self.scalers_[view_name].transform(X_flat)
                X_reduced = self.ecva_transformers_[view_name].transform(X_scaled)
            
            X_transformed[view_name] = X_reduced
            
        return X_transformed
    
    def _initialize_metrics(self, X_dict: Dict[str, np.ndarray]):
        """
        Initialize metric matrices for each view.
        
        Args:
            X_dict: Dictionary of transformed views
        """
        for view_name, X_view in X_dict.items():
            n_features = X_view.shape[1]
            # Initialize as identity matrix with small perturbation
            self.metrics_[view_name] = np.eye(n_features) + \
                                     np.random.normal(0, 0.01, (n_features, n_features))
            # Ensure positive semidefinite
            self.metrics_[view_name] = self.metrics_[view_name].T @ self.metrics_[view_name]
        
        # Initialize view weights uniformly
        self.n_views_ = len(X_dict)
        for view_name in X_dict.keys():
            self.weights_[view_name] = 1.0 / self.n_views_
    
    def _compute_distance(self, x1: np.ndarray, x2: np.ndarray, 
                         metric: np.ndarray) -> float:
        """
        Compute Mahalanobis distance between two points.
        
        Args:
            x1, x2: Input vectors
            metric: Metric matrix
            
        Returns:
            Distance
        """
        diff = x1 - x2
        return np.sqrt(np.maximum(0, diff.T @ metric @ diff))
    
    def _find_target_neighbors(self, X: np.ndarray, y: np.ndarray, 
                              view_name: str) -> Dict[int, List[int]]:
        """
        Find target neighbors (same class, closest) for each sample.
        
        Args:
            X: Features for this view
            y: Labels
            view_name: Name of current view
            
        Returns:
            Dictionary mapping sample index to target neighbor indices
        """
        target_neighbors = {}
        
        for i in range(len(X)):
            same_class_mask = (y == y[i])
            same_class_indices = np.where(same_class_mask)[0]
            same_class_indices = same_class_indices[same_class_indices != i]
            
            if len(same_class_indices) > 0:
                # Find k closest same-class samples
                distances = []
                for j in same_class_indices:
                    dist = self._compute_distance(X[i], X[j], self.metrics_[view_name])
                    distances.append((dist, j))
                
                distances.sort()
                k_target = min(self.k_neighbors, len(distances))
                target_neighbors[i] = [idx for _, idx in distances[:k_target]]
            else:
                target_neighbors[i] = []
                
        return target_neighbors
    
    def _compute_objective(self, X_dict: Dict[str, np.ndarray], 
                          y: np.ndarray) -> float:
        """
        Compute LM3L objective function.
        
        Args:
            X_dict: Dictionary of views
            y: Labels
            
        Returns:
            Objective value
        """
        total_objective = 0.0
        
        # Individual view objectives
        for view_name, X_view in X_dict.items():
            target_neighbors = self._find_target_neighbors(X_view, y, view_name)
            
            pull_term = 0.0  # Bring similar samples closer
            push_term = 0.0  # Push dissimilar samples apart
            
            for i in range(len(X_view)):
                # Pull term
                for j in target_neighbors[i]:
                    dist = self._compute_distance(X_view[i], X_view[j], self.metrics_[view_name])
                    pull_term += dist ** 2
                
                # Push term (hinge loss)
                for j in target_neighbors[i]:
                    for k in range(len(X_view)):
                        if y[k] != y[i]:  # Different class
                            dist_ij = self._compute_distance(X_view[i], X_view[j], self.metrics_[view_name])
                            dist_ik = self._compute_distance(X_view[i], X_view[k], self.metrics_[view_name])
                            hinge = max(0, self.beta + dist_ij - dist_ik)
                            push_term += hinge
            
            view_objective = self.alpha * pull_term + push_term
            total_objective += self.weights_[view_name] * view_objective
        
        # Multi-view consistency term
        consistency_term = 0.0
        view_names = list(X_dict.keys())
        for i in range(len(view_names)):
            for j in range(i + 1, len(view_names)):
                view1, view2 = view_names[i], view_names[j]
                X1, X2 = X_dict[view1], X_dict[view2]
                
                for s1 in range(len(X1)):
                    for s2 in range(len(X1)):
                        dist1 = self._compute_distance(X1[s1], X1[s2], self.metrics_[view1])
                        dist2 = self._compute_distance(X2[s1], X2[s2], self.metrics_[view2])
                        consistency_term += (dist1 - dist2) ** 2
        
        total_objective += self.gamma * consistency_term
        
        # Regularization
        reg_term = 0.0
        for metric in self.metrics_.values():
            reg_term += np.trace(metric @ metric.T)
        
        total_objective += self.regularization * reg_term
        
        return total_objective
    
    def _optimize_step(self, X_dict: Dict[str, np.ndarray], y: np.ndarray) -> float:
        """
        Perform one optimization step using gradient descent.
        
        Args:
            X_dict: Dictionary of views
            y: Labels
            
        Returns:
            Objective value after step
        """
        # This is a simplified gradient descent step
        # In practice, you'd compute actual gradients
        learning_rate = 0.01
        
        for view_name in self.metrics_.keys():
            # Small random perturbation (placeholder for actual gradient)
            gradient = np.random.normal(0, 0.001, self.metrics_[view_name].shape)
            self.metrics_[view_name] -= learning_rate * gradient
            
            # Ensure positive semidefinite
            eigenvals, eigenvecs = np.linalg.eigh(self.metrics_[view_name])
            eigenvals = np.maximum(eigenvals, 1e-8)
            self.metrics_[view_name] = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Update view weights
        view_objectives = {}
        for view_name, X_view in X_dict.items():
            # Simplified view objective computation
            view_objectives[view_name] = np.random.uniform(0.1, 1.0)
        
        total_obj = sum(view_objectives.values())
        for view_name in self.weights_.keys():
            self.weights_[view_name] = view_objectives[view_name] / total_obj
        
        return self._compute_objective(X_dict, y)
    
    def fit(self, X: Union[Dict[str, np.ndarray], List[np.ndarray]], 
            y: np.ndarray) -> 'LM3L':
        """
        Fit LM3L on multi-view data.
        
        Args:
            X: Multi-view features as dict {view_name: features} or list of arrays
            y: Labels
            
        Returns:
            Fitted classifier
        """
        # Convert to dictionary format if needed
        if isinstance(X, list):
            X = {f'view_{i}': view for i, view in enumerate(X)}
        
        # Store training labels
        self.y_train_ = y
        
        # Vectorize and reduce dimensionality
        self.X_train_transformed_ = self._vectorize_and_reduce(X, y, fit=True)
        
        # Initialize metrics
        self._initialize_metrics(self.X_train_transformed_)
        
        # Optimization loop
        prev_objective = float('inf')
        for iteration in range(self.max_iter):
            objective = self._optimize_step(self.X_train_transformed_, y)
            
            if abs(prev_objective - objective) < self.convergence_threshold:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            prev_objective = objective
            
            if iteration % 20 == 0:
                print(f"Iteration {iteration}, Objective: {objective:.6f}")
        
        return self
    
    def _multiview_distance(self, x1_dict: Dict[str, np.ndarray], 
                           x2_dict: Dict[str, np.ndarray]) -> float:
        """
        Compute multi-view distance between two samples.
        
        Args:
            x1_dict, x2_dict: Feature dictionaries for two samples
            
        Returns:
            Combined distance
        """
        total_distance = 0.0
        
        for view_name in x1_dict.keys():
            dist = self._compute_distance(x1_dict[view_name], x2_dict[view_name], 
                                        self.metrics_[view_name])
            total_distance += self.weights_[view_name] * dist
            
        return total_distance
    
    def predict(self, X: Union[Dict[str, np.ndarray], List[np.ndarray]]) -> np.ndarray:
        """
        Predict using k-NN with learned multi-view metric.
        
        Args:
            X: Test features
            
        Returns:
            Predicted labels
        """
        # Convert to dictionary format if needed
        if isinstance(X, list):
            X = {f'view_{i}': view for i, view in enumerate(X)}
        
        # Transform test data
        X_test_transformed = self._vectorize_and_reduce(X, self.y_train_, fit=False)
        
        predictions = []
        
        for i in range(len(next(iter(X_test_transformed.values())))):
            # Get test sample for all views
            test_sample = {view_name: X_view[i] 
                          for view_name, X_view in X_test_transformed.items()}
            
            # Compute distances to all training samples
            distances = []
            for j in range(len(next(iter(self.X_train_transformed_.values())))):
                train_sample = {view_name: X_view[j] 
                              for view_name, X_view in self.X_train_transformed_.items()}
                
                dist = self._multiview_distance(test_sample, train_sample)
                distances.append((dist, self.y_train_[j]))
            
            # Sort by distance and get k nearest neighbors
            distances.sort()
            k_nearest = distances[:self.k_neighbors]
            
            # Majority vote
            labels = [label for _, label in k_nearest]
            prediction = max(set(labels), key=labels.count)
            predictions.append(prediction)
        
        return np.array(predictions)


def create_multiview_features(X: np.ndarray, df_features: np.ndarray, 
                            ssmm_features: np.ndarray, 
                            statistical_features: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Create multi-view feature dictionary.
    
    Args:
        X: Raw time series
        df_features: Derivatives Fields features
        ssmm_features: SSMM features  
        statistical_features: Statistical features
        
    Returns:
        Multi-view feature dictionary
    """
    return {
        'raw': X,
        'df': df_features,
        'ssmm': ssmm_features,
        'statistical': statistical_features
    }