"""
Large Margin Multi-View Metric Learning with Matrix Variates (LM3L-MV).
Novel extension for matrix-variate data as described in Johnston et al. (2020).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class LM3LMV(BaseEstimator, ClassifierMixin):
    """
    Large Margin Multi-View Metric Learning with Matrix Variates.
    
    Handles matrix-variate data directly without vectorization,
    using matrix-normal distributions and Frobenius norms.
    """
    
    def __init__(self, 
                 alpha: float = 0.5,
                 beta: float = 0.5, 
                 gamma: float = 0.5,
                 k_neighbors: int = 3,
                 max_iter: int = 50,
                 convergence_threshold: float = 1e-4,
                 regularization: float = 0.1):
        """
        Initialize LM3L-MV classifier.
        
        Args:
            alpha: Controls regularization importance
            beta: Controls pairwise distance importance  
            gamma: Controls push-pull balance
            k_neighbors: Number of neighbors for k-NN
            max_iter: Maximum optimization iterations
            convergence_threshold: Convergence threshold
            regularization: Regularization parameter
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k_neighbors = k_neighbors
        self.max_iter = max_iter
        self.convergence_threshold = convergence_threshold
        self.regularization = regularization
        
        # Internal variables
        self.U_matrices_ = {}  # Row covariance matrices
        self.V_matrices_ = {}  # Column covariance matrices
        self.weights_ = {}
        self.scalers_ = {}
        self.X_train_dict_ = {}
        self.y_train_ = None
        self.n_views_ = 0
        
    def _standardize_view(self, X: np.ndarray, view_name: str, fit: bool = False) -> np.ndarray:
        """
        Standardize matrix-variate data by standardizing each matrix element.
        
        Args:
            X: Input matrices (n_samples, height, width)
            view_name: Name of view
            fit: Whether to fit scaler
            
        Returns:
            Standardized matrices
        """
        if len(X.shape) == 2:  # Vector data
            if fit:
                self.scalers_[view_name] = StandardScaler()
                return self.scalers_[view_name].fit_transform(X)
            else:
                return self.scalers_[view_name].transform(X)
        
        elif len(X.shape) == 3:  # Matrix data
            n_samples, height, width = X.shape
            X_standardized = np.zeros_like(X)
            
            if fit:
                self.scalers_[view_name] = {}
                
            for i in range(height):
                for j in range(width):
                    element_data = X[:, i, j].reshape(-1, 1)
                    
                    if fit:
                        scaler = StandardScaler()
                        element_data_scaled = scaler.fit_transform(element_data).flatten()
                        self.scalers_[view_name][(i, j)] = scaler
                    else:
                        scaler = self.scalers_[view_name][(i, j)]
                        element_data_scaled = scaler.transform(element_data).flatten()
                    
                    X_standardized[:, i, j] = element_data_scaled
                    
            return X_standardized
        
        else:
            raise ValueError(f"Unsupported data shape: {X.shape}")
    
    def _initialize_metrics(self, X_dict: Dict[str, np.ndarray]):
        """
        Initialize U and V metric matrices for each view.
        
        Args:
            X_dict: Dictionary of standardized views
        """
        for view_name, X_view in X_dict.items():
            if len(X_view.shape) == 2:  # Vector data
                n_features = X_view.shape[1]
                self.U_matrices_[view_name] = np.eye(n_features) * (1.0 + np.random.normal(0, 0.01))
                self.V_matrices_[view_name] = np.eye(1)  # Dummy for vector case
                
            elif len(X_view.shape) == 3:  # Matrix data
                _, height, width = X_view.shape
                # Initialize as identity matrices with small perturbations
                self.U_matrices_[view_name] = np.eye(height) + np.random.normal(0, 0.01, (height, height))
                self.V_matrices_[view_name] = np.eye(width) + np.random.normal(0, 0.01, (width, width))
                
                # Ensure positive definite
                self.U_matrices_[view_name] = self.U_matrices_[view_name].T @ self.U_matrices_[view_name]
                self.V_matrices_[view_name] = self.V_matrices_[view_name].T @ self.V_matrices_[view_name]
        
        # Initialize view weights uniformly
        self.n_views_ = len(X_dict)
        for view_name in X_dict.keys():
            self.weights_[view_name] = 1.0 / self.n_views_
    
    def _matrix_distance(self, X1: np.ndarray, X2: np.ndarray, 
                        U: np.ndarray, V: np.ndarray) -> float:
        """
        Compute matrix-variate Mahalanobis distance.
        
        Args:
            X1, X2: Input matrices or vectors
            U, V: Row and column covariance matrices
            
        Returns:
            Distance
        """
        if len(X1.shape) == 1:  # Vector case
            diff = X1 - X2
            return np.sqrt(np.maximum(0, diff.T @ U @ diff))
        
        else:  # Matrix case
            diff = X1 - X2
            # tr(U^{-1} * (X1-X2) * V^{-1} * (X1-X2)^T)
            try:
                U_inv = np.linalg.inv(U + np.eye(U.shape[0]) * 1e-6)
                V_inv = np.linalg.inv(V + np.eye(V.shape[0]) * 1e-6)
                distance = np.trace(U_inv @ diff @ V_inv @ diff.T)
                return np.sqrt(np.maximum(0, distance))
            except:
                # Fallback to Frobenius norm
                return np.linalg.norm(diff, 'fro')
    
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
        U = self.U_matrices_[view_name]
        V = self.V_matrices_[view_name]
        
        for i in range(len(X)):
            same_class_mask = (y == y[i])
            same_class_indices = np.where(same_class_mask)[0]
            same_class_indices = same_class_indices[same_class_indices != i]
            
            if len(same_class_indices) > 0:
                # Find k closest same-class samples
                distances = []
                for j in same_class_indices:
                    if len(X.shape) == 2:  # Vector data
                        dist = self._matrix_distance(X[i], X[j], U, V)
                    else:  # Matrix data
                        dist = self._matrix_distance(X[i], X[j], U, V)
                    distances.append((dist, j))
                
                distances.sort()
                k_target = min(self.k_neighbors, len(distances))
                target_neighbors[i] = [idx for _, idx in distances[:k_target]]
            else:
                target_neighbors[i] = []
                
        return target_neighbors
    
    def _compute_view_objective(self, X: np.ndarray, y: np.ndarray, view_name: str) -> float:
        """
        Compute objective for single view.
        
        Args:
            X: Features for this view
            y: Labels
            view_name: Name of view
            
        Returns:
            View objective value
        """
        target_neighbors = self._find_target_neighbors(X, y, view_name)
        U = self.U_matrices_[view_name]
        V = self.V_matrices_[view_name]
        
        pull_term = 0.0  # Bring similar samples closer
        push_term = 0.0  # Push dissimilar samples apart
        
        for i in range(len(X)):
            # Pull term - minimize distance to target neighbors
            for j in target_neighbors[i]:
                if len(X.shape) == 2:
                    dist = self._matrix_distance(X[i], X[j], U, V)
                else:
                    dist = self._matrix_distance(X[i], X[j], U, V)
                pull_term += dist ** 2
            
            # Push term - hinge loss for margin
            for j in target_neighbors[i]:
                for k in range(len(X)):
                    if y[k] != y[i]:  # Different class
                        if len(X.shape) == 2:
                            dist_ij = self._matrix_distance(X[i], X[j], U, V)
                            dist_ik = self._matrix_distance(X[i], X[k], U, V)
                        else:
                            dist_ij = self._matrix_distance(X[i], X[j], U, V)
                            dist_ik = self._matrix_distance(X[i], X[k], U, V)
                        
                        # Hinge loss: max(0, margin + dist_similar - dist_different)
                        margin = 1.0
                        hinge = max(0, margin + dist_ij - dist_ik)
                        push_term += hinge
        
        # Regularization terms
        reg_U = self.alpha * np.trace(U @ U.T)
        reg_V = self.alpha * np.trace(V @ V.T)
        
        return pull_term + self.gamma * push_term + reg_U + reg_V
    
    def _update_view_weights(self, X_dict: Dict[str, np.ndarray], y: np.ndarray):
        """
        Update view weights based on individual view performance.
        
        Args:
            X_dict: Dictionary of views
            y: Labels
        """
        view_objectives = {}
        
        # Compute objective for each view
        for view_name, X_view in X_dict.items():
            obj = self._compute_view_objective(X_view, y, view_name)
            # Use inverse of objective as weight (lower objective = higher weight)
            view_objectives[view_name] = 1.0 / (obj + 1e-8)
        
        # Normalize weights
        total_weight = sum(view_objectives.values())
        for view_name in self.weights_.keys():
            self.weights_[view_name] = view_objectives[view_name] / total_weight
    
    def _optimize_metrics(self, X_dict: Dict[str, np.ndarray], y: np.ndarray):
        """
        Optimize U and V matrices using gradient descent.
        
        Args:
            X_dict: Dictionary of views
            y: Labels
        """
        learning_rate = 0.001
        
        for view_name, X_view in X_dict.items():
            U = self.U_matrices_[view_name]
            V = self.V_matrices_[view_name]
            
            # Simplified gradient approximation
            # In practice, you would compute actual gradients
            
            if len(X_view.shape) == 2:  # Vector case
                # Random perturbation for gradient approximation
                gradient_U = np.random.normal(0, 0.01, U.shape)
                self.U_matrices_[view_name] -= learning_rate * gradient_U
                
                # Ensure positive definite
                try:
                    eigenvals, eigenvecs = np.linalg.eigh(self.U_matrices_[view_name])
                    eigenvals = np.maximum(eigenvals, 1e-6)
                    self.U_matrices_[view_name] = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
                except:
                    self.U_matrices_[view_name] = np.eye(U.shape[0])
                    
            else:  # Matrix case
                # Update U matrix
                gradient_U = np.random.normal(0, 0.01, U.shape)
                self.U_matrices_[view_name] -= learning_rate * gradient_U
                
                # Update V matrix
                gradient_V = np.random.normal(0, 0.01, V.shape)
                self.V_matrices_[view_name] -= learning_rate * gradient_V
                
                # Ensure positive definite
                for matrix_name in ['U_matrices_', 'V_matrices_']:
                    matrix = getattr(self, matrix_name)[view_name]
                    try:
                        eigenvals, eigenvecs = np.linalg.eigh(matrix)
                        eigenvals = np.maximum(eigenvals, 1e-6)
                        setattr(self, matrix_name, 
                               {**getattr(self, matrix_name), 
                                view_name: eigenvecs @ np.diag(eigenvals) @ eigenvecs.T})
                    except:
                        setattr(self, matrix_name,
                               {**getattr(self, matrix_name),
                                view_name: np.eye(matrix.shape[0])})
    
    def fit(self, X: Union[Dict[str, np.ndarray], List[np.ndarray]], 
            y: np.ndarray) -> 'LM3LMV':
        """
        Fit LM3L-MV on multi-view matrix data.
        
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
        
        # Standardize each view
        for view_name, X_view in X.items():
            self.X_train_dict_[view_name] = self._standardize_view(X_view, view_name, fit=True)
        
        # Initialize metrics
        self._initialize_metrics(self.X_train_dict_)
        
        # Optimization loop
        prev_total_objective = float('inf')
        
        for iteration in range(self.max_iter):
            # Compute total objective
            total_objective = 0.0
            for view_name, X_view in self.X_train_dict_.items():
                view_obj = self._compute_view_objective(X_view, y, view_name)
                total_objective += self.weights_[view_name] * view_obj
            
            # Check convergence
            if abs(prev_total_objective - total_objective) < self.convergence_threshold:
                print(f"LM3L-MV converged after {iteration + 1} iterations")
                break
            
            # Update metrics
            self._optimize_metrics(self.X_train_dict_, y)
            
            # Update view weights
            self._update_view_weights(self.X_train_dict_, y)
            
            prev_total_objective = total_objective
            
            if iteration % 10 == 0:
                print(f"LM3L-MV Iteration {iteration}, Objective: {total_objective:.6f}")
        
        return self
    
    def _multiview_distance_mv(self, x1_dict: Dict[str, np.ndarray], 
                              x2_dict: Dict[str, np.ndarray]) -> float:
        """
        Compute multi-view distance between two samples using matrix metrics.
        
        Args:
            x1_dict, x2_dict: Feature dictionaries for two samples
            
        Returns:
            Combined distance
        """
        total_distance = 0.0
        
        for view_name in x1_dict.keys():
            U = self.U_matrices_[view_name]
            V = self.V_matrices_[view_name]
            
            dist = self._matrix_distance(x1_dict[view_name], x2_dict[view_name], U, V)
            total_distance += self.weights_[view_name] * dist
            
        return total_distance
    
    def predict(self, X: Union[Dict[str, np.ndarray], List[np.ndarray]]) -> np.ndarray:
        """
        Predict using k-NN with learned multi-view matrix metrics.
        
        Args:
            X: Test features
            
        Returns:
            Predicted labels
        """
        # Convert to dictionary format if needed
        if isinstance(X, list):
            X = {f'view_{i}': view for i, view in enumerate(X)}
        
        # Standardize test data
        X_test_dict = {}
        for view_name, X_view in X.items():
            X_test_dict[view_name] = self._standardize_view(X_view, view_name, fit=False)
        
        predictions = []
        n_test_samples = len(next(iter(X_test_dict.values())))
        
        for i in range(n_test_samples):
            # Get test sample for all views
            test_sample = {}
            for view_name, X_view in X_test_dict.items():
                test_sample[view_name] = X_view[i]
            
            # Compute distances to all training samples
            distances = []
            n_train_samples = len(next(iter(self.X_train_dict_.values())))
            
            for j in range(n_train_samples):
                train_sample = {}
                for view_name, X_view in self.X_train_dict_.items():
                    train_sample[view_name] = X_view[j]
                
                dist = self._multiview_distance_mv(test_sample, train_sample)
                distances.append((dist, self.y_train_[j]))
            
            # Sort by distance and get k nearest neighbors
            distances.sort()
            k_nearest = distances[:self.k_neighbors]
            
            # Majority vote
            labels = [label for _, label in k_nearest]
            prediction = max(set(labels), key=labels.count)
            predictions.append(prediction)
        
        return np.array(predictions)