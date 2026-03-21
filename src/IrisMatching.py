"""
RESPONSIBILITY:
1. Feature reduction: Uses PCA followed by Fisher Linear Discriminant (FLD/LDA) to reduce the 
   1536-dimensional vector to a lower-dimensional space (e.g., 200).
2. Nearest Center Classifier: Classify based on distance to class centroids.
3. Support L1, L2, and Cosine similarity measures.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import cdist
from typing import Tuple, List, Dict  # <--- CRITICAL: Add this line

class IrisMatcher:
    def __init__(self, n_components: int = 200):
        self.n_components = n_components
        # Intermediate reduction to avoid LDA issues with small datasets
        self.pca = PCA(n_components=None) 
        self.lda = LinearDiscriminantAnalysis()
        self.class_centers: Dict[str, np.ndarray] = {}
        self.classes: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the dimension reduction models and compute class centroids."""
        # 1. Feature Reduction
        # First use PCA to reduce to a manageable size (n_samples - n_classes)
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))
        self.pca.n_components = min(n_samples - n_classes, 500)
        
        X_pca = self.pca.fit_transform(X)
        
        # Then use LDA
        self.lda.n_components = min(n_classes - 1, self.n_components)
        X_lda = self.lda.fit_transform(X_pca, y)
        
        # 2. Compute Nearest Centers
        self.classes = np.unique(y)
        for cls in self.classes:
            idx = np.where(y == cls)[0]
            self.class_centers[cls] = np.mean(X_lda[idx], axis=0)

    def predict(self, X: np.ndarray, metric: str = 'cosine') -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels for X based on the specified metric ('l1', 'l2', 'cosine').
        Returns predicted labels and the distance to the predicted center.
        """
        X_pca = self.pca.transform(X)
        X_lda = self.lda.transform(X_pca)
        
        centers_matrix = np.array([self.class_centers[cls] for cls in self.classes])
        
        # Calculate distances
        if metric == 'l1':
            dists = cdist(X_lda, centers_matrix, metric='cityblock')
        elif metric == 'l2':
            dists = cdist(X_lda, centers_matrix, metric='euclidean')
        elif metric == 'cosine':
            dists = cdist(X_lda, centers_matrix, metric='cosine')
        else:
            raise ValueError("Metric must be 'l1', 'l2', or 'cosine'")
            
        pred_indices = np.argmin(dists, axis=1)
        pred_labels = np.array([self.classes[i] for i in pred_indices])
        pred_distances = np.min(dists, axis=1)
        
        return pred_labels, pred_distances
