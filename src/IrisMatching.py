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
        self.classes = None
        self.class_templates = {}
        self._use_pca = True

    def fit(self, X: np.ndarray, y: np.ndarray, use_pca: bool = True):
        self.classes = np.unique(y)
        self._use_pca = use_pca
        num_subjects = len(self.classes)
        max_lda_dim = num_subjects - 1
        self.lda.n_components = min(self.n_components, max_lda_dim)

        # 1. Normalize
        X_scaled = self.scaler.fit_transform(X)

        # 2. PCA + LDA
        if self._use_pca:
            X_pca = self.pca.fit_transform(X_scaled)
            X_trans = self.lda.fit_transform(X_pca, y)
        else:
            X_trans = X_scaled

        # 3. Store all templates per class
        self.class_templates = {cls: [] for cls in self.classes}

        for i in range(len(X_trans)):
            cls = y[i]
            self.class_templates[cls].append(X_trans[i])

        for cls in self.classes:
            self.class_templates[cls] = np.array(self.class_templates[cls])

    def predict(self, X: np.ndarray, metric: str = 'l1'):
        X_scaled = self.scaler.transform(X)
        
        if self._use_pca:
            X_pca = self.pca.transform(X_scaled)
            transformed_X = self.lda.transform(X_pca)
        else:
            transformed_X = X_scaled
            
        dist_methods = {
            'l1': 'cityblock',
            'l2': 'euclidean',
            'cosine': 'cosine'
        }
        method = dist_methods.get(metric.lower(), 'cityblock')

        preds = []
        min_dists = []

        for x in transformed_X:
            best_cls = None
            best_dist = float('inf')

            # Compare with EACH class
            for cls in self.classes:
                templates = self.class_templates[cls]

                # MIN distance over templates
                dists = cdist([x], templates, metric=method)
                dist = np.min(dists)

                if dist < best_dist:
                    best_dist = dist
                    best_cls = cls

            preds.append(best_cls)
            min_dists.append(best_dist)

        return np.array(preds), np.array(min_dists)
