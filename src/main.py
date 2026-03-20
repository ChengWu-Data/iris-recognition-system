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
import cv2
import numpy as np

from IrisLocalization import localize_iris
from IrisNormalization import normalize_iris
from ImageEnhancement import enhance_image
from FeatureExtraction import extract_features
from IrisMatching import IrisMatcher
from PerformanceEvaluation import evaluate_identification_crr, evaluate_verification_roc

def process_pipeline(img_path: str) -> np.ndarray:
    """Executes the end-to-end preprocessing and extraction pipeline for a single image."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {img_path}")

    pupil_params, iris_params = localize_iris(img)
    norm_img = normalize_iris(img, pupil_params, iris_params)
    enh_img = enhance_image(norm_img)
    feature_vector = extract_features(enh_img)
    
    return feature_vector

def main():
    dataset_path = "./CASIA-IrisV1" # Adjust path as necessary
    
    train_feats, train_labels = [], []
    test_feats, test_labels = [], []

    print("--- Starting Feature Extraction Pipeline ---")
    for class_dir in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_dir)
        if not os.path.isdir(class_path):
            continue
            
        # Session 1 for training, Session 2 for testing 
        for session, is_train in [('1', True), ('2', False)]:
            session_path = os.path.join(class_path, session)
            if not os.path.isdir(session_path):
                continue
                
            for img_file in os.listdir(session_path):
                if not img_file.endswith('.bmp'): # All images are BMP [cite: 12]
                    continue
                    
                img_path = os.path.join(session_path, img_file)
                try:
                    feat = process_pipeline(img_path)
                    if is_train:
                        train_feats.append(feat)
                        train_labels.append(class_dir)
                    else:
                        test_feats.append(feat)
                        test_labels.append(class_dir)
                except Exception as e:
                    print(f"Skipping {img_path} due to error: {e}")

    X_train = np.array(train_feats)
    y_train = np.array(train_labels)
    X_test = np.array(test_feats)
    y_test = np.array(test_labels)

    print("--- Training Iris Matcher ---")
    matcher = IrisMatcher(n_components=200) # Reduced to 200 features [cite: 577]
    matcher.fit(X_train, y_train)

    print("--- Performance Evaluation ---")
    # Identification
    for metric in ['l1', 'l2', 'cosine']:
        preds, _ = matcher.predict(X_test, metric=metric)
        crr = evaluate_identification_crr(y_test, preds)
        print(f"Identification CRR ({metric.upper()}): {crr:.2f}%")

    # Verification (using Cosine as standard)
    _, dists = matcher.predict(X_test, metric='cosine')
    evaluate_verification_roc(y_test, preds, dists)

if __name__ == "__main__":
    main()
