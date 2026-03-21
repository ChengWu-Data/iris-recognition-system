"""
RESPONSIBILITY:
1. Feature reduction using PCA as requested in Step 5 of the project.
2. Nearest Center Classifier: Represent each class by its mean vector.
3. Similarity Measures: L1, L2, and Cosine as per the assignment.
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from typing import Tuple, Dict, Any

class IrisMatcher:
    def __init__(self, n_components: int = 200):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.class_centers: Dict[str, np.ndarray] = {}
        self.classes: np.ndarray = np.array([])
        self.is_reduced = False

    def fit(self, X: np.ndarray, y: np.ndarray, use_pca: bool = True):
        """
        Step 5: 'represent each class by its mean vector'
        """
        self.classes = np.unique(y)
        self.is_reduced = use_pca
        
        if use_pca:
            # PCA reduction to 'f' dimensions (e.g., 200)
            X_fit = self.pca.fit_transform(X)
        else:
            X_fit = X

        # Compute Mean Vector (Centroid) for each of the 108 classes
        for cls in self.classes:
            idx = np.where(y == cls)[0]
            self.class_centers[cls] = np.mean(X_fit[idx], axis=0)

    def predict(self, X: np.ndarray, metric: str = 'cosine') -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify using Nearest Center Classifier.
        """
        X_test = self.pca.transform(X) if self.is_reduced else X
        centers_matrix = np.array([self.class_centers[cls] for cls in self.classes])
        
        # Calculate distances to class centers
        if metric == 'l1':
            dists = cdist(X_test, centers_matrix, metric='cityblock')
        elif metric == 'l2':
            dists = cdist(X_test, centers_matrix, metric='euclidean')
        else:
            dists = cdist(X_test, centers_matrix, metric='cosine')
            
        pred_indices = np.argmin(dists, axis=1)
        pred_labels = np.array([self.classes[i] for i in pred_indices])
        pred_distances = np.min(dists, axis=1)
        
        return pred_labels, pred_distances
