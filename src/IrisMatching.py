"""
LOGIC:
1. Dimensionality Reduction: Supports both Original Space (Raw) and Reduced Space (PCA+LDA).
2. Fisher Linear Discriminant: Maximizes class separation as per Ma et al. (2003).
3. Nearest Center Classifier: Uses class centroids for robust identification.

KEY VARIABLES:
- n_components: Number of dimensions for LDA (max: classes - 1).
- use_pca: Boolean flag to toggle dimension reduction.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import cdist

class IrisMatcher:
    def __init__(self, n_components: int = 107):
        self.n_components = n_components
        self.pca = PCA(n_components=120) 
        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
        self.class_centers = {}
        self.classes = None
        self._use_pca = True

    def fit(self, X: np.ndarray, y: np.ndarray, use_pca: bool = True):
        self.classes = np.unique(y)
        self._use_pca = use_pca
        
        if self._use_pca:
            # Reduced Space: PCA followed by LDA
            X_pca = self.pca.fit_transform(X)
            X_lda = self.lda.fit_transform(X_pca, y)
            transformed_X = X_lda
        else:
            # Original Space: Use raw features
            transformed_X = X
        
        # Compute Centroids
        for cls in self.classes:
            idx = np.where(y == cls)[0]
            self.class_centers[cls] = np.mean(transformed_X[idx], axis=0)

    def predict(self, X: np.ndarray, metric: str = 'l1'):
        if self._use_pca:
            X_pca = self.pca.transform(X)
            transformed_X = self.lda.transform(X_pca)
        else:
            transformed_X = X
            
        centers_matrix = np.array([self.class_centers[cls] for cls in self.classes])
        
        # Distance calculation
        metric_map = {'l1': 'cityblock', 'l2': 'euclidean', 'cosine': 'cosine'}
        dist_metric = metric_map.get(metric.lower(), 'cityblock')
        
        dists = cdist(transformed_X, centers_matrix, metric=dist_metric)
        
        pred_indices = np.argmin(dists, axis=1)
        pred_labels = np.array([self.classes[i] for i in pred_indices])
        min_dists = np.min(dists, axis=1)
        
        return pred_labels, min_dists
