"""
RESPONSIBILITY:
1. Feature reduction: Uses PCA followed by Fisher Linear Discriminant (FLD/LDA) to reduce the 
   1536-dimensional vector to a lower-dimensional space.
2. Nearest Center Classifier: Classify based on distance to class centroids.
3. Support L1, L2, and Cosine similarity measures.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import cdist
from typing import Tuple, List, Dict, Any

class IrisMatcher:
    def __init__(self, n_components: int = 200, config: Dict[str, Any] = None):
        """
        Initialize the matcher. Supports both direct parameter passing and config dicts.
        
        Args:
            n_components (int): Target dimensionality for LDA.
            config (dict): Optional configuration dictionary from main.py.
        """
        self.n_components = n_components
        
        # Override with config values if provided by the main script
        if config and 'matching' in config:
            self.n_components = config['matching'].get('n_components', 200)
            
        self.pca = PCA(n_components=None) 
        self.lda = LinearDiscriminantAnalysis()
        self.class_centers: Dict[str, np.ndarray] = {}
        self.classes: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the dimension reduction models and compute class centroids.
        
        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target labels.
        """
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))
        
        # Step 1: PCA Reduction to prevent LDA singularity issues
        pca_comps = min(n_samples - n_classes, 500)
        if pca_comps <= 0: 
            pca_comps = None
        self.pca.n_components = pca_comps
        
        X_pca = self.pca.fit_transform(X)
        
        # Step 2: Linear Discriminant Analysis (LDA)
        lda_comps = min(n_classes - 1, self.n_components)
        self.lda.n_components = lda_comps
        X_lda = self.lda.fit_transform(X_pca, y)
        
        # Step 3: Compute Centroids for Nearest Center Classification
        self.classes = np.unique(y)
        for cls in self.classes:
            idx = np.where(y == cls)[0]
            self.class_centers[cls] = np.mean(X_lda[idx], axis=0)

    def predict(self, X: np.ndarray, metric: str = 'cosine') -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels for X based on the specified metric ('l1', 'l2', 'cosine').
        
        Returns:
            Tuple: (Predicted labels, Distances to predicted centers)
        """
        X_pca = self.pca.transform(X)
        X_lda = self.lda.transform(X_pca)
        
        centers_matrix = np.array([self.class_centers[cls] for cls in self.classes])
        
        # Calculate distances based on the chosen metric
        if metric == 'l1':
            dists = cdist(X_lda, centers_matrix, metric='cityblock')
        elif metric == 'l2':
            dists = cdist(X_lda, centers_matrix, metric='euclidean')
        else:
            # Default to Cosine Similarity as per Daugman's/Ma's logic
            dists = cdist(X_lda, centers_matrix, metric='cosine')
            
        pred_indices = np.argmin(dists, axis=1)
        pred_labels = np.array([self.classes[i] for i in pred_indices])
        pred_distances = np.min(dists, axis=1)
        
        return pred_labels, pred_distances
