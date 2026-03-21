"""
RESPONSIBILITY:
1. Feature reduction: Uses PCA followed by Fisher Linear Discriminant (FLD/LDA).
2. Nearest Center Classifier: Classify based on distance to class centroids.
3. Support L1, L2, and Cosine similarity measures.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import cdist
from typing import Tuple, List, Dict, Any

class IrisMatcher:
    def __init__(self, config: Dict[str, Any] = None):
        self.n_components = 200
        if config and 'n_components' in config:
            self.n_components = config['n_components']
            
        self.pca = PCA(n_components=None) 
        self.lda = LinearDiscriminantAnalysis()
        self.class_centers: Dict[str, np.ndarray] = {}
        self.classes: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the dimension reduction models and compute class centroids."""
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))
        
        # PCA 降维防止 LDA 在样本过少时崩溃
        pca_comps = min(n_samples - n_classes, 500)
        if pca_comps <= 0: pca_comps = None
        self.pca.n_components = pca_comps
        
        X_pca = self.pca.fit_transform(X)
        
        lda_comps = min(n_classes - 1, self.n_components)
        self.lda.n_components = lda_comps
        X_lda = self.lda.fit_transform(X_pca, y)
        
        self.classes = np.unique(y)
        for cls in self.classes:
            idx = np.where(y == cls)[0]
            self.class_centers[cls] = np.mean(X_lda[idx], axis=0)

    def predict(self, X: np.ndarray, metric: str = 'cosine') -> Tuple[np.ndarray, np.ndarray]:
        """Predict labels and return distances."""
        X_pca = self.pca.transform(X)
        X_lda = self.lda.transform(X_pca)
        
        centers_matrix = np.array([self.class_centers[cls] for cls in self.classes])
        
        if metric == 'l1':
            dists = cdist(X_lda, centers_matrix, metric='cityblock')
        elif metric == 'l2':
            dists = cdist(X_lda, centers_matrix, metric='euclidean')
        elif metric == 'cosine':
            dists = cdist(X_lda, centers_matrix, metric='cosine')
        else:
            dists = cdist(X_lda, centers_matrix, metric='cosine')
            
        pred_indices = np.argmin(dists, axis=1)
        pred_labels = np.array([self.classes[i] for i in pred_indices])
        pred_distances = np.min(dists, axis=1)
        
        return pred_labels, pred_distances
