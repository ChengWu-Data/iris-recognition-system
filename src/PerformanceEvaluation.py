"""
RESPONSIBILITY:
1. Identification Mode: Calculate Correct Recognition Rate (CRR).
2. Verification Mode: Generate data for the Receiver Operating Characteristic (ROC) curve 
   by calculating False Match Rate (FMR) and False Non-Match Rate (FNMR) at varying thresholds.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

def evaluate_identification_crr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Correct Recognition Rate (CRR).
    Formula: (Correctly Classified Samples / Total Test Samples) * 100
    """
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    crr = (correct / total) * 100.0
    return crr

def evaluate_verification_roc(y_true: np.ndarray, y_pred: np.ndarray, distances: np.ndarray, output_dir: str):
    """
    Plots the ROC curve (FMR vs FNMR) for verification mode.
    Following Ma et al. (2003) Fig. 11 style.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels from the matcher.
        distances: Distances to the nearest class centers.
        output_dir: Directory to save the resulting figure.
    """
    # 1. Convert to binary verification problem
    # y_binary = 1 for genuine match (correctly identified), 0 for imposter
    y_binary = (y_true == y_pred).astype(int)
    
    # 2. Convert distances to similarity scores
    # Sklearn's roc_curve expects higher scores for the positive class (genuine matches)
    # We use 1 / (1 + distance) to map [0, inf) to (0, 1]
    similarity_scores = 1.0 / (1.0 + distances) 

    # 3. Calculate FPR (FMR) and TPR (1 - FNMR)
    fpr, tpr, thresholds = roc_curve(y_binary, similarity_scores)
    
    # FNMR (False Non-Match Rate) is 1 - TPR
    fnmr = 1 - tpr 
    
    # Calculate Area Under Curve (for the standard ROC, though we plot FMR vs FNMR)
    roc_auc = auc(fpr, tpr)
    
    # 4. Plotting logic (Matching Ma's paper requirements)
    plt.figure(figsize=(8, 6))
    
    # We plot FMR on X-axis and FNMR on Y-axis
    plt.plot(fpr, fnmr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    
    # Baseline for random guessing
    plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle='--')
    
    # Log scale is common for biometric ROC to see detail at low FMR
    plt.xscale('log') 
    plt.xlim([10**-4, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.xlabel('False Match Rate (FMR)')
    plt.ylabel('False Non-Match Rate (FNMR)')
    plt.title('Verification Performance (FMR vs FNMR)')
    plt.legend(loc="upper right")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # 5. Save output to the configured results/figures directory
    save_path = os.path.join(output_dir, 'ROC_curve.png')
    plt.savefig(save_path)
    plt.close()
    
    print(f">>> ROC curve (FMR vs FNMR) saved as: {save_path}")
