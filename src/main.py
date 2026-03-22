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

"""
PROJECT DESCRIPTION:
This is the main entry point for the Iris Recognition Group Project based on the Ma et al. (2003) paper. 
It processes the CASIA Iris Image Database (version 1.0), extracting features from Session 1 
for training and evaluating against Session 2 for testing.

PIPELINE REMARKS:
1. Load images from the dataset directory.
2. Preprocessing: Localization -> Normalization -> Enhancement.
3. Feature Extraction: Generate 1536-dimensional feature vectors.
4. Matching: LDA dimension reduction and Nearest Center Classifier.
5. Evaluation: Calculate CRR for identification and ROC for verification.
"""

import os
import sys
import yaml
import cv2
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from IrisLocalization import localize_iris
from IrisNormalization import normalize_iris
from ImageEnhancement import enhance_image, is_good_quality, compute_quality
from FeatureExtraction import extract_features, extract_templates
from IrisMatching import IrisMatcher
from PerformanceEvaluation import evaluate_identification_crr, evaluate_verification_roc

def setup_env(config):
    for key in ['results', 'figures', 'tables']:
        path = config['paths'][key]
        os.makedirs(path, exist_ok=True)
    print(f">>> Environment ready. Results: {config['paths']['results']}")

def process_pipeline(img_path, cfg):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(f"Failed: {img_path}")
    p_param, i_param = localize_iris(img, cfg['localization'])
    norm_img = normalize_iris(img, p_param, i_param, cfg['normalization'])
    enh_img = enhance_image(norm_img)
    return enh_img

def main():
    root_dir = os.path.dirname(current_dir)
    config_path = os.path.join(root_dir, "configs", "default.yaml")
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    setup_env(cfg)
    train_feats, train_labels, test_feats, test_labels = [], [], [], []
    dataset_path = os.path.join(root_dir, "CASIA-IrisV1")

    subjects = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

    train_top_k = 4
    test_top_k = 1

    for subject in subjects:
        for sess in [str(cfg['data']['train_session']), str(cfg['data']['test_session'])]:
            sess_path = os.path.join(dataset_path, subject, sess)
            if not os.path.isdir(sess_path):
                continue

            candidates = []

            for img_file in os.listdir(sess_path):
                if not img_file.endswith(cfg['data']['img_ext']):
                    continue

                img_path = os.path.join(sess_path, img_file)

                try:
                    enh_img = process_pipeline(img_path, cfg)

                    total_power, ratio = compute_quality(enh_img)

                    # quality ranking score
                    quality_score = ratio + 0.15 * np.log(total_power + 1e-6)

                    candidates.append({
                        "img_file": img_file,
                        "enh_img": enh_img,
                        "total_power": total_power,
                        "ratio": ratio,
                        "quality_score": quality_score,
                    })

                except Exception as e:
                    print(f"[ERROR] {img_file}: {e}")

            # sort by quality score (descending)
            candidates.sort(key=lambda x: x["quality_score"], reverse=True)

            # choose top-k depending on session
            if sess == str(cfg['data']['train_session']):
                selected = candidates[:train_top_k]
                for item in selected:
                    feat = extract_features(item["enh_img"])
                    train_feats.append(feat)
                    train_labels.append(subject)
            else:
                selected = candidates[:test_top_k]
                for item in selected:
                    feat = extract_features(item["enh_img"])
                    test_feats.append(feat)
                    test_labels.append(subject)

            # optional debug print
            if len(candidates) > 0:
                print(
                    f"[SELECT] subject={subject}, session={sess}, "
                    f"kept={len(selected)}/{len(candidates)}, "
                    f"best_score={selected[0]['quality_score']:.4f}"
                )

    X_train, y_train = np.array(train_feats), np.array(train_labels)
    X_test, y_test = np.array(test_feats), np.array(test_labels)
    print(f"[DATA] train samples={len(train_feats)}, test samples={len(test_feats)}")

    # 2. Evaluation Table Generation (Required by Step 6)
    results_list = []
    matcher = IrisMatcher(n_components=cfg['matching']['n_components'])
    
    for space_name, use_pca in [('Original', False), ('Reduced', True)]:
        print(f"\n--- Testing in {space_name} Space ---")
        matcher.fit(X_train, y_train, use_pca=use_pca)
        for metric in cfg['matching']['metrics']:
            preds, dists = matcher.predict(X_test, metric=metric)
            crr = evaluate_identification_crr(y_test, preds)
            print(f"{metric.upper()} CRR: {crr:.2f}%")
            results_list.append({
                'Space': space_name, 
                'Dim': 1536 if not use_pca else cfg['matching']['n_components'],
                'Metric': metric.upper(), 'CRR (%)': f"{crr:.2f}%"
            })
            if space_name == 'Reduced' and metric == 'cosine': # Save ROC for the best combo
                evaluate_verification_roc(y_test, preds, dists, cfg['paths']['figures'])

    # 3. Save Table
    df = pd.DataFrame(results_list)
    df.to_csv(os.path.join(cfg['paths']['tables'], 'recognition_results.csv'), index=False)
    print(f"\n>>> Table saved to {cfg['paths']['tables']}")

if __name__ == "__main__":
    main()
