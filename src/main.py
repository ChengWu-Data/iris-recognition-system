"""
PROJECT DESCRIPTION:
This is the main entry point for the Iris Recognition Group Project based on the Ma et al. (2003) paper. 
It processes the CASIA Iris Image Database (version 1.0)[cite: 7], extracting features from Session 1 
for training and evaluating against Session 2 for testing.

PIPELINE REMARKS:
1. Load images from the dataset directory.
2. Preprocessing: Localization -> Normalization -> Enhancement[cite: 25, 26, 28].
3. Feature Extraction: Generate a 1536-dimensional feature vector using spatial filters[cite: 373].
4. Matching: Reduce dimensions via PCA/FLD and classify using the Nearest Center Classifier[cite: 374].
5. Evaluation: Calculate Correct Recognition Rate (CRR) for identification and False Match/Non-Match 
   Rates for the ROC curve in verification[cite: 31, 32].

MODULES CALLED:
- src.IrisLocalization
- src.IrisNormalization
- src.ImageEnhancement
- src.FeatureExtraction
- src.IrisMatching
- src.PerformanceEvaluation
"""

import os
import yaml
import numpy as np
from IrisLocalization import localize_iris
from IrisNormalization import normalize_iris
from ImageEnhancement import enhance_image
from FeatureExtraction import extract_features
from IrisMatching import IrisMatcher
from PerformanceEvaluation import evaluate_identification_crr, evaluate_verification_roc

def setup_env(config):
    """
    Creates the directory structure defined in the config.
    Ensures 'results/figures' and 'results/tables' exist.
    """
    for key in ['results', 'figures', 'tables']:
        path = config['paths'][key]
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            # Create .gitkeep to ensure empty dirs are tracked by Git
            with open(os.path.join(path, ".gitkeep"), "w") as f:
                pass

def process_pipeline(img_path, cfg):
    """End-to-end processing for a single image using config parameters."""
    # Note: These modules should be updated to accept the sub-config dictionaries
    p_param, i_param = localize_iris(img_path, cfg['localization'])
    norm_img = normalize_iris(img_path, p_param, i_param, cfg['normalization'])
    enh_img = enhance_image(norm_img)
    feat = extract_features(enh_img, cfg['features'])
    return feat

def main():
    # 1. Load Configuration
    config_path = os.path.join("configs", "default.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # 2. Setup folders (results/figures, etc.)
    setup_env(cfg)
    
    train_feats, train_labels = [], []
    test_feats, test_labels = [], []

    dataset_path = cfg['data']['dataset_path']
    print(f"--- Starting Feature Extraction from {dataset_path} ---")

    # 3. Iterate through Subjects (001, 002, ...)
    # This restores your original logic for Session 1 and Session 2
    subjects = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    for subject in subjects:
        subject_path = os.path.join(dataset_path, subject)
        
        # Session 1 for training, Session 2 for testing
        for session in [cfg['data']['train_session'], cfg['data']['test_session']]:
            session_path = os.path.join(subject_path, session)
            if not os.path.isdir(session_path):
                continue
            
            is_train = (session == cfg['data']['train_session'])
            
            for img_file in os.listdir(session_path):
                if not img_file.endswith(cfg['data']['img_ext']):
                    continue
                
                img_path = os.path.join(session_path, img_file)
                try:
                    feat = process_pipeline(img_path, cfg)
                    if is_train:
                        train_feats.append(feat)
                        train_labels.append(subject)
                    else:
                        test_feats.append(feat)
                        test_labels.append(subject)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    # 4. Convert to numpy arrays for matching
    X_train, y_train = np.array(train_feats), np.array(train_labels)
    X_test, y_test = np.array(test_feats), np.array(test_labels)

    # 5. Training & Matching
    print(f"--- Training Iris Matcher with n_components={cfg['matching']['n_components']} ---")
    matcher = IrisMatcher(n_components=cfg['matching']['n_components'])
    matcher.fit(X_train, y_train)

    # 6. Evaluation
    print("\n--- Performance Evaluation ---")
    # Identification Mode
    for metric in cfg['matching']['metrics']:
        preds, dists = matcher.predict(X_test, metric=metric)
        crr = evaluate_identification_crr(y_test, preds)
        print(f"Identification CRR ({metric.upper()}): {crr:.2f}%")

    # Verification Mode (Using the distances from the last metric in loop, usually Cosine)
    # This outputs the ROC curve to results/figures/
    evaluate_verification_roc(y_test, dists, cfg['paths']['figures'])

if __name__ == "__main__":
    main()
