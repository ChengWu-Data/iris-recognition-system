"""
LOGIC:
1. Feature Scaling: Standardizes features to zero mean and unit variance.
2. PCA: Reduces initial 1536D features to handle redundancy.
3. FLD (LDA): Maximizes between-class scatter. The code now automatically 
   validates n_components to avoid "ValueError: n_components cannot be larger...".
4. Centroid Matching: Each class is represented by a 1D mean vector.
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
        n_classes = len(self.classes)
        


        target_lda_dim = min(self.n_components, n_classes - 1)
        self.lda.n_components = target_lda_dim

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
        

        metric_map = {'l1': 'cityblock', 'l2': 'euclidean', 'cosine': 'cosine'}
        dist_metric = metric_map.get(metric.lower(), 'cityblock')
        
        dists = cdist(transformed_X, centers_matrix, metric=dist_metric)
        
        pred_indices = np.argmin(dists, axis=1)
        pred_labels = np.array([self.classes[i] for i in pred_indices])
        min_dists = np.min(dists, axis=1)
        
        return pred_labels, min_dists
