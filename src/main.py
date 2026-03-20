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
import sys
import yaml
import cv2
import numpy as np

# --- PATH FIX FOR LOCAL MODULES ---
# This ensures that even if you run from the root folder, 
# Python can find IrisLocalization.py, etc., inside the src/ folder.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import local modules
try:
    from IrisLocalization import localize_iris
    from IrisNormalization import normalize_iris
    from ImageEnhancement import enhance_image
    from FeatureExtraction import extract_features
    from IrisMatching import IrisMatcher
    from PerformanceEvaluation import evaluate_identification_crr, evaluate_verification_roc
except ImportError as e:
    print(f"DEBUG ERROR: Failed to import local modules. {e}")
    print(f"Current sys.path: {sys.path}")

def setup_env(config):
    """Creates the directory structure defined in the config."""
    paths_to_create = [
        config['paths']['results'],
        config['paths']['figures'],
        config['paths']['tables']
    ]
    for path in paths_to_create:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            # Create .gitkeep to ensure empty folders are tracked by Git
            with open(os.path.join(path, ".gitkeep"), "w") as f:
                pass
    print(f">>> Environment setup complete. Results will be saved to: {config['paths']['results']}")

def process_pipeline(img_path, cfg):
    """Executes the end-to-end preprocessing and extraction for a single image."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {img_path}")

    # 1. Localization
    p_param, i_param = localize_iris(img, cfg['localization'])
    
    # 2. Normalization
    norm_img = normalize_iris(img, p_param, i_param, cfg['normalization'])
    
    # 3. Enhancement
    enh_img = enhance_image(norm_img)
    
    # 4. Feature Extraction
    feature_vector = extract_features(enh_img, cfg['features'])
    
    return feature_vector

def main():
    # 1. Load Configuration from YAML
    config_path = os.path.join("configs", "default.yaml")
    if not os.path.exists(config_path):
        # Fallback if running from within src/
        config_path = os.path.join("..", "configs", "default.yaml")
        
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load config file at {config_path}. {e}")
        return

    # 2. Initialize folders
    setup_env(cfg)
    
    train_feats, train_labels = [], []
    test_feats, test_labels = [], []

    dataset_path = cfg['data']['dataset_path']
    print(f"\n--- Starting Feature Extraction Pipeline ---")
    print(f"Dataset Path: {os.path.abspath(dataset_path)}")

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset directory not found at {dataset_path}")
        return

    # 3. Iterate through Subjects (001, 002, ...)
    subjects = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    for subject in subjects:
        subject_path = os.path.join(dataset_path, subject)
        
        # Session 1 for training ('1'), Session 2 for testing ('2')
        for session in [str(cfg['data']['train_session']), str(cfg['data']['test_session'])]:
            session_path = os.path.join(subject_path, session)
            if not os.path.isdir(session_path):
                continue
            
            is_train = (session == str(cfg['data']['train_session']))
            
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

    # 4. Convert lists to numpy arrays
    X_train = np.array(train_feats)
    y_train = np.array(train_labels)
    X_test = np.array(test_feats)
    y_test = np.array(test_labels)

    if len(X_train) == 0 or len(X_test) == 0:
        print("ERROR: No features extracted. Check dataset structure and paths.")
        return

    # 5. Training Iris Matcher
    print(f"\n--- Training Iris Matcher (LDA Reduction: {cfg['matching']['n_components']}) ---")
    matcher = IrisMatcher(n_components=cfg['matching']['n_components'])
    matcher.fit(X_train, y_train)

    # 6. Performance Evaluation
    print("\n--- Performance Evaluation Results ---")
    
    # Identification (CRR)
    last_preds, last_dists = None, None
    for metric in cfg['matching']['metrics']:
        preds, dists = matcher.predict(X_test, metric=metric)
        crr = evaluate_identification_crr(y_test, preds)
        print(f"Identification CRR ({metric.upper()}): {crr:.2f}%")
        last_preds, last_dists = preds, dists # Keep last one for ROC

    # Verification (ROC Curve)
    # Using the last distance scores (e.g., Cosine) to generate Fig 11.
    print(f"\n>>> Generating ROC Curve in {cfg['paths']['figures']}...")
    evaluate_verification_roc(y_test, last_preds, last_dists, cfg['paths']['figures'])

if __name__ == "__main__":
    main()
