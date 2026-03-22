"""
RESPONSIBILITY:
Performs feature normalization, optional dimensionality reduction, and
nearest-center classification for iris recognition.

CURRENT DESIGN:
1. Feature Scaling:
   Standardizes training and test features using StandardScaler.

2. Reduced-Space Matching:
   Uses a two-stage reduction pipeline:
   - PCA for noise reduction and compact representation
   - LDA for supervised class separation

3. Stable PCA Setting:
   Uses a deterministic full SVD solver for reproducible PCA behavior.

4. Dimensionality Constraints:
   - PCA dimension is limited by min(n_samples, n_features)
   - LDA dimension is limited by min(requested_dim, n_classes - 1, pca_dim)

5. Classification Rule:
   Computes one class center per subject in the transformed space and
   classifies each test sample by nearest-center distance.

6. Supported Metrics:
   - L1 (cityblock)
   - L2 (euclidean)
   - cosine

NOTES:
- The current best-performing configuration uses reduced-space cosine matching.
- PCA dimensionality has been retuned for the current stronger feature pipeline.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

class IrisMatcher:
    def __init__(self, n_components: int = 107, pca_components: int = 110):
        self.n_components = n_components
        self.pca_components = pca_components

        self.scaler = StandardScaler()
        self.pca = None
        self.lda = None

        self.classes = None
        self.class_centers = None
        self.use_reduction = False

    def fit(self, X: np.ndarray, y: np.ndarray, use_pca: bool = True):
        self.classes = np.unique(y)
        self.use_reduction = use_pca

        # 1) Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # 2) Reduced space: PCA -> LDA
        if use_pca:
            # PCA dimension cannot exceed min(n_samples, n_features)
            max_pca_dim = min(X_scaled.shape[0], X_scaled.shape[1])
            pca_dim = min(self.pca_components, max_pca_dim)
            self.pca = PCA(n_components=pca_dim, svd_solver="full")
            X_pca = self.pca.fit_transform(X_scaled)

            # LDA dimension cannot exceed (#classes - 1)
            max_lda_dim = len(self.classes) - 1
            lda_dim = min(self.n_components, max_lda_dim, pca_dim)

            self.lda = LinearDiscriminantAnalysis(n_components=lda_dim)
            X_trans = self.lda.fit_transform(X_pca, y)
        else:
            self.pca = None
            self.lda = None
            X_trans = X_scaled

        # 3) Compute class centers
        self.class_centers = {}
        for cls in self.classes:
            cls_samples = X_trans[y == cls]
            self.class_centers[cls] = np.mean(cls_samples, axis=0)

    def predict(self, X: np.ndarray, metric: str = 'l2'):
        metric_map = {
            'l1': 'cityblock',
            'l2': 'euclidean',
            'cosine': 'cosine'
        }

        if metric.lower() not in metric_map:
            raise ValueError(f"Unsupported metric: {metric}")

        method = metric_map[metric.lower()]

        # 1) Apply the same transforms as training
        X_scaled = self.scaler.transform(X)

        if self.use_reduction:
            X_pca = self.pca.transform(X_scaled)
            X_trans = self.lda.transform(X_pca)
        else:
            X_trans = X_scaled

        # 2) Stack class centers
        centers = np.array([self.class_centers[cls] for cls in self.classes])

        preds = []
        min_dists = []

        # 3) Nearest center classification
        for x in X_trans:
            dists = cdist([x], centers, metric=method).flatten()
            best_idx = np.argmin(dists)
            preds.append(self.classes[best_idx])
            min_dists.append(dists[best_idx])

        return np.array(preds), np.array(min_dists)
