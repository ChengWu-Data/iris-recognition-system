"""
LOGIC:
1. PCA: Reduces initial 1536D features to handle redundancy.
2. FLD (LDA): Maximizes between-class scatter and minimizes within-class scatter,
   as specified in Ma et al. (2003) for optimal classification.
3. Centroid Matching: Each class (person) is represented by a 1D mean vector.

KEY VARIABLES:
- n_components: Target dimensionality for LDA (Number of classes - 1).
- metrics: L1 (Manhattan) is recommended for its outlier robustness.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import cdist

class IrisMatcher:
    def __init__(self, n_components: int = 107):
        self.pca = PCA(n_components=120) 
        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
        self.class_centers = {}
        self.classes = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes = np.unique(y)
        
        X_pca = self.pca.fit_transform(X)
        
       
        X_lda = self.lda.fit_transform(X_pca, y)
        
        for cls in self.classes:
            idx = np.where(y == cls)[0]
            self.class_centers[cls] = np.mean(X_lda[idx], axis=0)

    def predict(self, X: np.ndarray, metric: str = 'l1'):
        X_test_pca = self.pca.transform(X)
        X_test_lda = self.lda.transform(X_test_pca)
        
        centers_matrix = np.array([self.class_centers[cls] for cls in self.classes])
        
        if metric.lower() == 'l1':
            dists = cdist(X_test_lda, centers_matrix, metric='cityblock')
        elif metric.lower() == 'l2':
            dists = cdist(X_test_lda, centers_matrix, metric='euclidean')
        else:
            dists = cdist(X_test_lda, centers_matrix, metric='cosine')
            
        pred_indices = np.argmin(dists, axis=1)
        pred_labels = np.array([self.classes[i] for i in pred_indices])
        min_dists = np.min(dists, axis=1)
        
        return pred_labels, min_dists
