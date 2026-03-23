"""
IrisMatching.py

Purpose:
    This script performs feature transformation and iris matching. It first
    projects the extracted iris features into a more discriminative space using
    feature scaling, optional PCA, and optional Fisher Linear Discriminant (LDA).
    Then it performs classification using a nearest-center strategy with
    multiple rotation templates per class.

Overall Logic:
    1. Fit the feature transformation pipeline using the original training samples only.
    2. Transform both the original training features and the rotated template features
       into the same feature space.
    3. For each iris class, build one template center for each rotation shift.
    4. For each test sample, compute its distance to all shift templates of every class.
    5. Use the minimum template distance within each class as that class score.
    6. Predict the class with the overall minimum score.

Why this script is needed:
    Raw 1536-dimensional iris features are high-dimensional and may contain
    redundancy or noise. PCA and LDA are used to improve class separability.
    In addition, iris images may be slightly rotated, so the matcher uses
    multiple shift templates for each class to improve robustness.

Key Variables and Parameters:
    lda_components:
        Target dimension used for Fisher Linear Discriminant Analysis (LDA).
    pca_components:
        Optional PCA dimension before LDA. If None, the maximum valid PCA dimension is used.
    scaler:
        StandardScaler used to normalize feature dimensions before PCA/LDA.
    pca:
        PCA model used to reduce dimensionality before LDA.
    lda:
        LDA model used to project features into a discriminative subspace.
    use_reduction:
        Boolean flag indicating whether PCA + LDA should be used.
    classes:
        Unique iris class labels in the training set.
    template_tags:
        Unique rotation shift tags used to define the 7 templates per class.
    class_centers:
        Mean feature vector of each class using original training samples.
    class_shift_templates:
        For each class, stores the 7 template centers corresponding to different shifts.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


class IrisMatcher:
    """
    Iris matcher with optional PCA + LDA dimensionality reduction and
    multi-template nearest-center classification.

    Logic:
        1. Fit the transformation pipeline on the original training data.
        2. Project rotated template features into the same space.
        3. For each class, build one template center for each shift.
        4. Match each probe against all class templates and use the minimum
           distance within each class as the final class score.

    Why this design is used:
        The paper uses Fisher Linear Discriminant for dimensionality reduction
        and nearest-center classification. It also considers multiple rotation
        templates for each class to reduce the effect of angular misalignment.

    Important Attributes:
        classes:
            All unique training class labels.
        template_tags:
            All unique shift labels used for rotation templates.
        class_centers:
            Baseline class centers in the transformed space.
        class_shift_templates:
            A dictionary storing the 7 template vectors for each class.
    """

    def __init__(self, lda_components: int = 107, pca_components: int | None = None):
        """
        Initialize the matcher.

        Args:
            lda_components:
                Maximum number of LDA components to retain.
            pca_components:
                Optional number of PCA components before LDA. If None,
                the maximum valid PCA dimension is used.

        Key Variables:
            scaler:
                Standardizes features before PCA/LDA.
            pca:
                PCA transformer.
            lda:
                LDA transformer.
            use_reduction:
                Whether dimensionality reduction is enabled.
            classes:
                Unique class labels.
            template_tags:
                Unique shift tags for template construction.
            class_shift_templates:
                7 template centers per class, one for each shift.
        """
        self.lda_components = lda_components
        self.pca_components = pca_components

        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.lda: LinearDiscriminantAnalysis | None = None
        self.use_reduction = False

        self.classes: np.ndarray | None = None
        self.template_tags: np.ndarray | None = None
        self.class_shift_templates: dict[str, np.ndarray] | None = None
        self.class_centers: dict[str, np.ndarray] | None = None

    def _fit_reduction(self, X_train_original: np.ndarray, y_train_original: np.ndarray, use_reduction: bool) -> np.ndarray:
        """
        Fit the feature transformation pipeline.

        Logic:
            1. Standardize the training features.
            2. If dimensionality reduction is disabled, return the scaled features.
            3. If enabled, perform PCA first to address the small-sample-size issue.
            4. Then fit LDA on the PCA output using the class labels.

        Args:
            X_train_original:
                Original training feature matrix.
            y_train_original:
                Class labels for the original training set.
            use_reduction:
                Whether PCA + LDA should be used.

        Returns:
            The transformed training features.

        Why PCA is used before LDA:
            In high-dimensional small-sample problems, the within-class scatter
            matrix can become singular. PCA reduces dimensionality first so that
            LDA can be applied more stably.

        Key Variables:
            X_scaled:
                Standardized training features.
            max_pca:
                Maximum valid number of PCA components.
            pca_dim:
                Actual PCA dimension used.
            X_pca:
                PCA-transformed features.
            max_lda:
                Maximum valid number of LDA components.
            lda_dim:
                Actual LDA dimension used.
        """
        X_scaled = self.scaler.fit_transform(X_train_original)

        if not use_reduction:
            self.pca = None
            self.lda = None
            return X_scaled

        # Small-sample-size safeguard: apply PCA before LDA
        max_pca = min(X_scaled.shape[0] - 1, X_scaled.shape[1])
        if max_pca < 1:
            raise ValueError("Not enough samples to fit PCA.")

        pca_dim = max_pca if self.pca_components is None else min(self.pca_components, max_pca)
        self.pca = PCA(n_components=pca_dim, svd_solver="full", random_state=0)
        X_pca = self.pca.fit_transform(X_scaled)

        max_lda = min(len(np.unique(y_train_original)) - 1, X_pca.shape[1])
        if max_lda < 1:
            raise ValueError("Not enough classes/features to fit LDA.")

        lda_dim = min(self.lda_components, max_lda)
        self.lda = LinearDiscriminantAnalysis(n_components=lda_dim)

        return self.lda.fit_transform(X_pca, y_train_original)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input features into the matching space.

        Logic:
            1. Standardize the input feature matrix.
            2. If reduction is enabled, apply PCA followed by LDA.
            3. Otherwise, return the scaled features directly.

        Args:
            X:
                Input feature matrix.

        Returns:
            Transformed feature matrix.
        """
        X_scaled = self.scaler.transform(X)

        if self.use_reduction:
            assert self.pca is not None and self.lda is not None
            return self.lda.transform(self.pca.transform(X_scaled))

        return X_scaled

    def fit(
        self,
        X_train_original: np.ndarray,
        y_train_original: np.ndarray,
        X_template: np.ndarray,
        y_template: np.ndarray,
        template_tags: np.ndarray,
        use_reduction: bool = True,
    ):
        """
        Fit the matcher and construct class template centers.

        Logic:
            1. Store the unique class labels and shift tags.
            2. Fit the feature transformation pipeline using the original training samples only.
            3. Transform the rotated template features into the same feature space.
            4. Build a baseline class center using the transformed original training samples.
            5. For each class and each shift, compute the mean of the template features.
               This produces exactly 7 shift templates per class.

        Args:
            X_train_original:
                Original training feature matrix.
            y_train_original:
                Labels of the original training samples.
            X_template:
                Rotated template feature matrix.
            y_template:
                Class labels for the template matrix.
            template_tags:
                Shift tags associated with each template sample.
            use_reduction:
                Whether PCA + LDA should be used.

        Returns:
            The fitted matcher itself.

        Key Variables:
            classes:
                Unique class labels.
            template_tags:
                Unique shift identifiers.
            X_train_trans:
                Transformed original training samples.
            X_template_trans:
                Transformed template samples.
            class_centers:
                Baseline center of each class in transformed space.
            shift_means:
                The 7 template centers for one class.
        """
        self.use_reduction = use_reduction
        self.classes = np.unique(y_train_original)
        self.template_tags = np.unique(template_tags)

        X_train_trans = self._fit_reduction(X_train_original, y_train_original, use_reduction)
        X_template_trans = self.transform(X_template)

        # Baseline class centers in the transformed space
        self.class_centers = {}
        for cls in self.classes:
            cls_train = X_train_trans[y_train_original == cls]
            self.class_centers[cls] = np.mean(cls_train, axis=0)

        # Build exactly 7 templates per class by averaging the 3 session-1 samples
        # under each rotation shift
        self.class_shift_templates = {}
        for cls in self.classes:
            shift_means = []
            cls_mask = (y_template == cls)

            for tag in self.template_tags:
                mask = cls_mask & (template_tags == tag)
                if not np.any(mask):
                    raise ValueError(f"Missing template for class={cls}, shift={tag}")

                shift_means.append(np.mean(X_template_trans[mask], axis=0))

            self.class_shift_templates[cls] = np.vstack(shift_means)

        return self

    def _metric_name(self, metric: str) -> str:
        """
        Convert a user-facing metric name into the corresponding SciPy distance name.

        Args:
            metric:
                'l1', 'l2', or 'cosine'

        Returns:
            Distance metric name used by scipy.spatial.distance.cdist.
        """
        metric_map = {
            "l1": "cityblock",
            "l2": "euclidean",
            "cosine": "cosine",
        }
        metric_key = metric.lower()

        if metric_key not in metric_map:
            raise ValueError(f"Unsupported metric: {metric}")

        return metric_map[metric_key]

    def predict(self, X: np.ndarray, metric: str = "l2"):
        """
        Predict class labels for probe samples.

        Logic:
            1. Transform the probe features into the matching space.
            2. For each probe, compute its distance to the 7 templates of each class.
            3. Use the minimum distance within a class as that class's matching score.
            4. Predict the class with the smallest score.

        Args:
            X:
                Probe feature matrix.
            metric:
                Matching metric: 'l1', 'l2', or 'cosine'.

        Returns:
            preds:
                Predicted class labels.
            min_dists:
                The final minimum matched distance for each probe sample.

        Key Variables:
            X_trans:
                Transformed probe features.
            class_best_dists:
                Minimum template distance for each class.
            templates:
                The 7 shift templates of one class.
            d:
                Distances from the current probe to the 7 templates of one class.
        """
        metric_name = self._metric_name(metric)
        X_trans = self.transform(X)

        preds = []
        min_dists = []

        for x in X_trans:
            class_best_dists = []

            for cls in self.classes:
                templates = self.class_shift_templates[cls]
                d = cdist(x[None, :], templates, metric=metric_name).ravel()
                class_best_dists.append(np.min(d))

            class_best_dists = np.asarray(class_best_dists)
            best_idx = int(np.argmin(class_best_dists))

            preds.append(self.classes[best_idx])
            min_dists.append(class_best_dists[best_idx])

        return np.asarray(preds), np.asarray(min_dists)

    def all_match_scores(self, X: np.ndarray, metric: str = "l2"):
        """
        Compute class-wise matching scores for every probe sample.

        Logic:
            For each probe and each class, return the minimum distance between
            the probe and the 7 templates of that class.

        Args:
            X:
                Probe feature matrix.
            metric:
                Matching metric: 'l1', 'l2', or 'cosine'.

        Returns:
            A score matrix of shape (n_probes, n_classes), where:
                score[i, j] = minimum distance from probe i to class j.

        Why this method is useful:
            It provides full class-wise matching information, which can be used
            for additional analysis beyond just the final predicted label.

        Key Variables:
            X_trans:
                Transformed probe features.
            scores:
                Output score matrix.
            dmat:
                Pairwise distance matrix between all probes and the 7 templates
                of the current class.
        """
        metric_name = self._metric_name(metric)
        X_trans = self.transform(X)
        scores = np.zeros((X_trans.shape[0], len(self.classes)), dtype=np.float64)

        for j, cls in enumerate(self.classes):
            templates = self.class_shift_templates[cls]
            dmat = cdist(X_trans, templates, metric=metric_name)
            scores[:, j] = np.min(dmat, axis=1)

        return scores
