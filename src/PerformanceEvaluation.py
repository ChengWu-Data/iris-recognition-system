"""
RESPONSIBILITY:
1. Identification Mode: Calculate Correct Recognition Rate (CRR)[cite: 473].
2. Verification Mode: Generate data for the Receiver Operating Characteristic (ROC) curve 
   by calculating False Match Rate (FMR) and False Non-Match Rate (FNMR) at varying thresholds[cite: 474, 475].
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def evaluate_identification_crr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Correct Recognition Rate (CRR)[cite: 473].
    Formula: (Correctly Classified Samples / Total Test Samples) * 100
    """
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    crr = (correct / total) * 100.0
    return crr

def evaluate_verification_roc(y_true: np.ndarray, y_pred: np.ndarray, distances: np.ndarray):
    """
    Plots the ROC curve for verification mode[cite: 474].
    Treats matching classes as positive (distance closer to 0) and non-matching as negative.
    """
    # Convert problem to binary verification: 1 if genuine match, 0 if imposter
    y_binary = (y_true == y_pred).astype(int)
    
    # Distance is inverse to similarity, we invert it for ROC calculation
    # Sklearn expects higher scores = higher probability of positive class
    similarity_scores = 1.0 / (1.0 + distances) 

    fpr, tpr, thresholds = roc_curve(y_binary, similarity_scores)
    fnr = 1 - tpr # False Non-Match Rate (FNMR) [cite: 476]
    
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, fnr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xscale('log') # Common for FMR vs FNMR plots [cite: 606]
    plt.xlim([10**-4, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Match Rate (%)')
    plt.ylabel('False Non-Match Rate (%)')
    plt.title('Receiver Operating Characteristic (Verification Mode)')
    plt.legend(loc="lower right")
    plt.grid(True, which="both", ls="--")
    
    # Save output for reporting Table 4 / Fig 11 [cite: 32]
    plt.savefig('ROC_curve.png')
    print("ROC curve saved as 'ROC_curve.png'.")
