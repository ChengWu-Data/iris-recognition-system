"""
PerformanceEvaluation.py

Purpose:
    This script evaluates the iris recognition system in two ways:

    1. Identification mode:
       Measures how often the system assigns the correct identity to a probe sample.
       The evaluation metric used here is Correct Recognition Rate (CRR).

    2. Verification mode:
       Measures how well the system separates genuine comparisons from impostor
       comparisons. The evaluation output is a ROC curve in terms of False Match
       Rate (FMR) and False Non-Match Rate (FNMR).

Overall Logic:
    - For identification, compare predicted class labels with true class labels.
    - For verification, use probe-to-enrolled-class matching scores:
        * genuine comparison: probe identity matches gallery class
        * impostor comparison: probe identity does not match gallery class
    - Convert distance-based scores into similarity-style scores by negating them,
      so that larger scores indicate a more likely genuine match.
    - Compute ROC statistics and save the verification curve as an image.

Why this script is needed:
    A complete iris recognition system must be evaluated from both a classification
    perspective (identification) and a matching perspective (verification).
    This script provides both evaluation modes in a format aligned with the
    project requirement.

Key Variables and Parameters:
    y_true:
        True class labels of probe samples.
    y_pred:
        Predicted class labels returned by the matcher.
    score_matrix:
        Matrix of probe-to-class matching distances.
        score_matrix[i, j] is the distance between probe i and enrolled class j.
    probe_labels:
        True class labels of all probe samples.
    gallery_labels:
        Class labels of the enrolled gallery classes.
    comparison_labels:
        Binary labels indicating whether each probe-class comparison is genuine (1)
        or impostor (0).
    comparison_scores:
        Similarity-style comparison scores obtained by negating distances.
    fpr:
        False Positive Rate, interpreted here as False Match Rate (FMR).
    tpr:
        True Positive Rate.
    fnmr:
        False Non-Match Rate, computed as 1 - tpr.
    roc_auc:
        Area Under the ROC Curve.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def evaluate_identification_crr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Correct Recognition Rate (CRR) for identification mode.

    Logic:
        1. Convert both input label arrays to NumPy arrays.
        2. Compare the predicted labels with the true labels element-wise.
        3. Compute the proportion of correctly classified samples.
        4. Convert the proportion into a percentage.

    Args:
        y_true:
            True class labels of the probe samples.
        y_pred:
            Predicted class labels of the probe samples.

    Returns:
        CRR value in percentage form.

    Why this function is needed:
        CRR is the main identification metric required by the project. It directly
        measures how often the system assigns the correct identity to each probe image.

    Key Variables:
        y_true:
            Ground-truth identities.
        y_pred:
            Predicted identities.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred) * 100.0)


def evaluate_verification_roc_from_scores(
    score_matrix: np.ndarray,
    probe_labels: np.ndarray,
    gallery_labels: np.ndarray,
    metric: str = "cosine",
    output_dir: str = ".",
):
    """
    Generate a verification ROC curve using probe-vs-enrolled-class scores.

    Logic:
        1. Treat each probe-to-class comparison as one verification trial.
        2. Mark the comparison as genuine if the probe label matches the gallery class.
        3. Mark the comparison as impostor otherwise.
        4. Convert distance values into similarity-style scores by negating them.
        5. Compute the ROC curve.
        6. Convert TPR to FNMR.
        7. Save the ROC curve as an image file.

    Args:
        score_matrix:
            Matrix of pairwise matching distances between probes and enrolled classes.
            score_matrix[i, j] is the distance between probe i and gallery class j.
        probe_labels:
            True identity labels of probe samples.
        gallery_labels:
            Identity labels of enrolled gallery classes.
        metric:
            Name of the matching metric, used only for display and output naming.
        output_dir:
            Directory where the ROC figure will be saved.

    Returns:
        out_path:
            Full file path of the saved ROC figure.
        roc_auc:
            Area Under the ROC Curve.

    Why this function is needed:
        The project requires verification evaluation in addition to identification.
        This implementation uses probe-to-gallery-class scores, which is more
        appropriate for the project setting than directly comparing probe samples
        to one another.

    Key Variables:
        score_matrix:
            Distance matrix between probes and enrolled classes.
        comparison_labels:
            Flattened binary array where:
                1 = genuine comparison
                0 = impostor comparison
        comparison_scores:
            Flattened score array used by ROC computation.
            Distances are negated so that larger values correspond to more genuine matches.
        fpr:
            False Positive Rate, interpreted here as False Match Rate (FMR).
        tpr:
            True Positive Rate.
        fnmr:
            False Non-Match Rate, computed as 1 - tpr.
        roc_auc:
            Area under the ROC curve.
        out_path:
            Output path of the saved ROC figure.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert labels to NumPy arrays for consistent indexing
    probe_labels = np.asarray(probe_labels)
    gallery_labels = np.asarray(gallery_labels)

    # Build binary labels for all probe-vs-gallery comparisons
    # 1 = genuine comparison, 0 = impostor comparison
    comparison_labels = (probe_labels[:, None] == gallery_labels[None, :]).astype(int).ravel()

    # Convert distances into similarity-style scores:
    # smaller distance -> larger score after negation
    comparison_scores = (-np.asarray(score_matrix)).ravel()

    # Compute ROC statistics
    fpr, tpr, _ = roc_curve(comparison_labels, comparison_scores)
    fnmr = 1.0 - tpr
    roc_auc = auc(fpr, tpr)

    # Plot ROC in FMR vs FNMR form
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, fnmr, lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.xscale("log")
    plt.xlim([1e-4, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Match Rate (FMR) - Log Scale")
    plt.ylabel("False Non-Match Rate (FNMR)")
    plt.title(f"Verification Mode ROC Curve ({metric.upper()})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc="upper right")

    # Save figure
    out_path = os.path.join(output_dir, f"ROC_curve_{metric.lower()}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return out_path, roc_auc
