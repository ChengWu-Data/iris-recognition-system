"""
RESPONSIBILITY:
1. Feature reduction: Uses PCA followed by Fisher Linear Discriminant (FLD/LDA) to reduce the 
   1536-dimensional vector to a lower-dimensional space (e.g., 200)[cite: 374, 378, 577].
2. Nearest Center Classifier: Classify based on distance to class centroids[cite: 374, 384].
3. Support L1, L2, and Cosine similarity measures[cite: 390].
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import cdist

class IrisMatcher:
    def __init__(self, n_components: int = 200):
        self.n_components = n_components
        self.pca = PCA(n_components=min(300, n_components * 2)) # Intermediate reduction
        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
        self.class_centers = {}
        self.classes = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the dimension reduction models and compute class centroids."""
        # 1. Feature Reduction [cite: 374]
        X_pca = self.pca.fit_transform(X)
        X_lda = self.lda.fit_transform(X_pca, y)
        
        # 2. Compute Nearest Centers
        self.classes = np.unique(y)
        for cls in self.classes:
            idx = np.where(y == cls)[0]
            self.class_centers[cls] = np.mean(X_lda[idx], axis=0)

    def predict(self, X: np.ndarray, metric: str = 'cosine') -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels for X based on the specified metric ('l1', 'l2', 'cosine')[cite: 390].
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
