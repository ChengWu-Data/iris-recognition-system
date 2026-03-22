"""
PROJECT DESCRIPTION:
Main entry point for the current iris recognition pipeline based on the
CASIA-IrisV1 dataset.

CURRENT PIPELINE:
1. Load Dataset:
   Uses Session 1 images for training and Session 2 images for testing.

2. Preprocessing:
   Applies:
   - iris localization
   - rubber-sheet normalization
   - illumination correction and contrast enhancement

3. Quality-Ranked Sample Selection:
   For each subject/session, computes a frequency-based quality score for all
   candidate images, sorts them deterministically, and keeps:
   - top 3 images for training
   - top 2 images for testing

4. Feature Extraction:
   Extracts block-based texture features from the enhanced normalized iris image
   using the current dual-filter feature encoder.

5. Matching:
   Evaluates both:
   - Original feature space
   - Reduced feature space (PCA + LDA)
   using nearest-center classification.

6. Metrics and Outputs:
   Reports Correct Recognition Rate (CRR) under:
   - L1 distance
   - L2 distance
   - cosine distance
   and saves the ROC-related verification curve for the reduced cosine setting.

CURRENT STABLE VERSION:
- deterministic file ordering
- deterministic candidate ranking
- quality-ranked sample selection
- tuned filter pair
- tuned PCA dimensionality
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

    train_top_k = 3
    test_top_k = 2

    for subject in subjects:
        for sess in [str(cfg['data']['train_session']), str(cfg['data']['test_session'])]:
            sess_path = os.path.join(dataset_path, subject, sess)
            if not os.path.isdir(sess_path):
                continue

            candidates = []

            for img_file in sorted(os.listdir(sess_path)):
                if not img_file.endswith(cfg['data']['img_ext']):
                    continue

                img_path = os.path.join(sess_path, img_file)

                try:
                    enh_img = process_pipeline(img_path, cfg)

                    total_power, ratio = compute_quality(enh_img)

                    # quality ranking score
                    quality_score = ratio

                    candidates.append({
                        "img_file": img_file,
                        "enh_img": enh_img,
                        "total_power": total_power,
                        "ratio": ratio,
                        "quality_score": quality_score,
                    })

                except Exception as e:
                    print(f"[ERROR] {img_file}: {e}")

            candidates.sort(key=lambda x: (-x["quality_score"], x["img_file"]))

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
            #debug print
            if len(candidates) > 0:
                selected_names = [item["img_file"] for item in selected]
                print(
                    f"[SELECT] subject={subject}, session={sess}, "
                    f"kept={len(selected)}/{len(candidates)}, "
                    f"best_score={selected[0]['quality_score']:.4f}, "
                    f"selected={selected_names}"
                )

    X_train, y_train = np.array(train_feats), np.array(train_labels)
    X_test, y_test = np.array(test_feats), np.array(test_labels)
    print(f"[DATA] train samples={len(train_feats)}, test samples={len(test_feats)}")

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

    df = pd.DataFrame(results_list)
    df.to_csv(os.path.join(cfg['paths']['tables'], 'recognition_results.csv'), index=False)
    print(f"\n>>> Table saved to {cfg['paths']['tables']}")

if __name__ == "__main__":
    main()
