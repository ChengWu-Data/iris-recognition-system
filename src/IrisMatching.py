"""
LOGIC:
1. Feature Scaling: Uses StandardScaler to ensure all Gabor features contribute equally.
2. PCA: Reduces 1536D to 120D to remove noise while keeping 95%+ variance.
3. FLD (LDA): The "Supervised" step. It ignores n_components=200 and forces it 
   to the theoretical maximum (n_classes - 1) to ensure the best separation.
4. Nearest Center: Calculates distances in the new 'Fisher-space'.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

class IrisMatcher:
    def __init__(self, n_components: int = 107):
        self.n_components = n_components
        self.pca = PCA(n_components=120) 
        self.lda = LinearDiscriminantAnalysis()
        self.scaler = StandardScaler()
        self.class_centers = {}
        self.classes = None
        self._use_pca = True

    def fit(self, X: np.ndarray, y: np.ndarray, use_pca: bool = True):
        self.classes = np.unique(y)
        self._use_pca = use_pca
        num_subjects = len(self.classes)
        

        max_lda_dim = num_subjects - 1
        self.lda.n_components = min(self.n_components, max_lda_dim)
        
        X_scaled = self.scaler.fit_transform(X)
        
        if self._use_pca:
            X_pca = self.pca.fit_transform(X_scaled)
            X_lda = self.lda.fit_transform(X_pca, y)
            transformed_X = X_lda
        else:
            transformed_X = X_scaled
        
        for cls in self.classes:
            idx = np.where(y == cls)[0]
            self.class_centers[cls] = np.mean(transformed_X[idx], axis=0)

    def predict(self, X: np.ndarray, metric: str = 'l1'):
        X_scaled = self.scaler.transform(X)
        
        if self._use_pca:
            X_pca = self.pca.transform(X_scaled)
            transformed_X = self.lda.transform(X_pca)
        else:
            transformed_X = X_scaled
            
        centers_matrix = np.array([self.class_centers[cls] for cls in self.classes])
        
        dist_methods = {'l1': 'cityblock', 'l2': 'euclidean', 'cosine': 'cosine'}
        method = dist_methods.get(metric.lower(), 'cityblock')
        
        dists = cdist(transformed_X, centers_matrix, metric=method)
        
        pred_indices = np.argmin(dists, axis=1)
        pred_labels = np.array([self.classes[i] for i in pred_indices])
        min_dists = np.min(dists, axis=1)
        
        return pred_labels, min_dists
