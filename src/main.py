"""
main.py

Purpose:
    This script runs the complete iris recognition pipeline for the project.
    It loads the CASIA-IrisV1 dataset, applies preprocessing, extracts features,
    performs matching, evaluates identification performance, generates
    verification ROC curves, and saves the final results.

Overall Logic:
    1. Locate the dataset and output folders.
    2. Loop through all subjects and both acquisition sessions.
    3. For each image:
       - localize the iris,
       - normalize the iris region,
       - enhance the normalized image,
       - extract iris features.
    4. Use session 1 images as the training set.
    5. Build multiple rotated templates from each session 1 image to model
       rotational variation.
    6. Use session 2 images as the testing set.
    7. Fit the matcher in original space and reduced space.
    8. Evaluate Correct Recognition Rate (CRR) for L1, L2, and cosine distance.
    9. Generate ROC curves in the reduced space.
    10. Save the final recognition table as a CSV file.

Why this script is needed:
    The individual modules only solve one part of the problem. This script
    integrates all modules into one complete end-to-end iris recognition system
    and ensures that the experimental protocol matches the assignment setting.

Key Variables and Parameters:
    dataset_dir:
        Path to the CASIA-IrisV1 dataset.
    fig_dir:
        Directory where ROC figures are saved.
    table_dir:
        Directory where CSV result tables are saved.
    shift_cols:
        Column shifts used to generate multiple rotation templates.
    X_train_original:
        Feature matrix from original session 1 images.
    y_train_original:
        Labels corresponding to the original training samples.
    X_template:
        Feature matrix of rotated templates generated from the training images.
    y_template:
        Class labels corresponding to the template feature matrix.
    template_tags:
        Rotation tag associated with each template.
    X_test:
        Feature matrix from session 2 test images.
    y_test:
        Labels corresponding to the test images.
    matcher:
        IrisMatcher object used for training and prediction.
    results_rows:
        List of result rows that will later be written to a CSV file.
"""

from __future__ import annotations
import os
import glob
import cv2
import csv
import numpy as np

from IrisLocalization import localize_iris
from IrisNormalization import normalize_iris
from ImageEnhancement import enhance_image
from FeatureExtraction import extract_features
from IrisMatching import IrisMatcher
from PerformanceEvaluation import evaluate_identification_crr, evaluate_verification_roc_from_scores


def build_paths():
    """
    Locate the dataset directory and create output folders if needed.

    Logic:
        1. Determine the current script directory and project root.
        2. Search for the dataset in a small set of candidate locations.
        3. Create figure and table output directories if they do not already exist.

    Returns:
        dataset_dir:
            Directory containing CASIA-IrisV1.
        fig_dir:
            Directory for ROC figures.
        table_dir:
            Directory for result tables.

    Key Variables:
        current_dir:
            Directory where the current script is located.
        project_root:
            Parent directory of the script folder.
        dataset_candidates:
            Candidate locations for the CASIA-IrisV1 dataset.
        dataset_dir:
            The first valid dataset path found.
        fig_dir:
            Directory used to save figures.
        table_dir:
            Directory used to save result tables.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    dataset_candidates = [
        os.path.join(project_root, "CASIA-IrisV1"),
        os.path.join(current_dir, "CASIA-IrisV1"),
    ]

    dataset_dir = None
    for p in dataset_candidates:
        if os.path.exists(p):
            dataset_dir = p
            break

    if dataset_dir is None:
        raise FileNotFoundError(
            f"CASIA-IrisV1 not found. Put dataset under: {dataset_candidates[0]}"
        )

    fig_dir = os.path.join(project_root, "results", "figures")
    table_dir = os.path.join(project_root, "results", "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    return dataset_dir, fig_dir, table_dir


def preprocess_to_enhanced(img_path: str) -> np.ndarray:
    """
    Run the preprocessing pipeline for one iris image.

    Main Steps:
        1. Read the grayscale image from disk.
        2. Localize the pupil and outer iris boundary.
        3. Normalize the iris region into a fixed-size rectangular image.
        4. Enhance the normalized iris image.

    Args:
        img_path:
            Path to the input eye image.

    Returns:
        An enhanced normalized iris image of shape (64, 512).

    Key Variables:
        img:
            Original grayscale eye image.
        pupil_params:
            Pupil center and radius returned by localization.
        iris_params:
            Outer iris circle returned by localization.
        norm_img:
            Normalized iris image.
        enh_img:
            Enhanced iris image used for feature extraction.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")

    pupil_params, iris_params = localize_iris(img)
    norm_img = normalize_iris(
        img,
        pupil_params,
        iris_params,
        {"radial_res": 64, "angular_res": 512},
    )
    enh_img = enhance_image(norm_img)
    return enh_img


def generate_template_images(enh_img: np.ndarray, shift_cols: list[int]) -> list[tuple[int, np.ndarray]]:
    """
    Generate multiple rotated template images by cyclically shifting columns.

    Logic:
        1. Treat the normalized enhanced iris image as a wrapped angular representation.
        2. Apply cyclic column shifts to simulate different initial angular positions.
        3. Return all shifted images together with their shift tags.

    Args:
        enh_img:
            Enhanced normalized iris image of shape (64, 512).
        shift_cols:
            List of column shifts representing different template rotations.

    Returns:
        A list of tuples:
            (shift_tag, shifted_image)

    Why this function is needed:
        The iris may appear with slight angular differences between acquisitions.
        Multiple shifted templates provide a simple way to model rotational variation.

    Key Variables:
        shift_cols:
            Integer column shifts used to simulate different angles.
        shifted:
            A shifted version of the enhanced normalized image.
        templates:
            List storing all generated shifted templates.
    """
    templates = []
    for s in shift_cols:
        shifted = np.roll(enh_img, shift=s, axis=1)
        templates.append((s, shifted))
    return templates


def main():
    """
    Run the complete iris recognition experiment.

    Main Steps:
        1. Build dataset and output paths.
        2. Traverse all subjects and sessions in the dataset.
        3. For session 1:
           - extract original training features
           - generate 7 rotated templates for each training image
        4. For session 2:
           - extract test features
        5. Fit the matcher in both original and reduced feature spaces.
        6. Compute identification CRR under L1, L2, and cosine distance.
        7. Compute verification ROC curves in reduced space.
        8. Save all final results to disk.

    Notes on protocol:
        - Session 1 is used for training.
        - Session 2 is used for testing.
        This follows the assignment requirement.

    Key Variables:
        shift_cols:
            The 7 paper-inspired rotation shifts.
        subject_dirs:
            All subject folders in the dataset.
        train/test arrays:
            Store extracted features and labels for later matching.
        failed:
            List of images that could not be processed successfully.
        metrics_to_test:
            Distance metrics used for identification and verification.
        results_rows:
            Final result rows written to CSV.
    """
    dataset_dir, fig_dir, table_dir = build_paths()

    print(f"[INFO] Using dataset: {dataset_dir}")
    print(f"[INFO] Figures -> {fig_dir}")
    print(f"[INFO] Tables  -> {table_dir}")
    print("[INFO] Loading dataset and extracting features...")

    # Paper-inspired 7 rotation templates:
    # angles: -9, -6, -3, 0, 3, 6, 9 degrees
    # 512 columns correspond to 360 degrees
    # columns per degree = 512 / 360 ≈ 1.422
    shift_cols = [-13, -9, -4, 0, 4, 9, 13]

    X_train_original, y_train_original = [], []
    X_template, y_template, template_tags = [], [], []
    X_test, y_test = [], []
    failed = []

    # Traverse all subjects
    subject_dirs = sorted(
        [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    )

    for subject_id in subject_dirs:
        subject_path = os.path.join(dataset_dir, subject_id)

        # Session 1 is used for training; session 2 is used for testing
        for session in ["1", "2"]:
            session_path = os.path.join(subject_path, session)
            if not os.path.exists(session_path):
                continue

            img_files = sorted(glob.glob(os.path.join(session_path, "*.bmp")))

            for img_file in img_files:
                try:
                    enh_img = preprocess_to_enhanced(img_file)

                    if session == "1":
                        # Original training feature for fitting scaler / PCA / LDA
                        feat_original = extract_features(enh_img)
                        X_train_original.append(feat_original)
                        y_train_original.append(subject_id)

                        # Build template features under 7 shifted angles
                        for tag, shifted_img in generate_template_images(enh_img, shift_cols):
                            feat_template = extract_features(shifted_img)
                            X_template.append(feat_template)
                            y_template.append(subject_id)
                            template_tags.append(tag)

                    else:
                        # Session 2 feature used as a probe/test sample
                        feat_test = extract_features(enh_img)
                        X_test.append(feat_test)
                        y_test.append(subject_id)

                except Exception as e:
                    # Keep track of failed images instead of stopping the whole run
                    failed.append((img_file, str(e)))

    # Convert collected lists into NumPy arrays
    X_train_original = np.asarray(X_train_original, dtype=np.float32)
    y_train_original = np.asarray(y_train_original)

    X_template = np.asarray(X_template, dtype=np.float32)
    y_template = np.asarray(y_template)
    template_tags = np.asarray(template_tags)

    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test)

    # Basic dataset summary
    print(f"[DATA] Training Samples: {len(X_train_original)} | Testing Samples: {len(X_test)}")
    print(f"[DATA] X_train_original shape: {X_train_original.shape}")
    print(f"[DATA] X_template shape: {X_template.shape}")
    print(f"[DATA] X_test shape: {X_test.shape}")
    print(f"[DATA] #train classes: {len(np.unique(y_train_original))}")
    print(f"[DATA] #test classes : {len(np.unique(y_test))}")

    if failed:
        print(f"[WARN] Failed images: {len(failed)}")
        for p, msg in failed[:10]:
            print(f"   {p} -> {msg}")

    # Initialize the matcher
    matcher = IrisMatcher(
        lda_components=107,
        pca_components=120,
    )

    results_rows = []
    metrics_to_test = ["l1", "l2", "cosine"]

    # Evaluate in original space and reduced space
    for space_name, use_reduction in [("Original", False), ("Reduced", True)]:
        print(f"\n--- Testing in {space_name} Space ---")

        matcher.fit(
            X_train_original=X_train_original,
            y_train_original=y_train_original,
            X_template=X_template,
            y_template=y_template,
            template_tags=template_tags,
            use_reduction=use_reduction,
        )

        for metric in metrics_to_test:
            # Identification mode: predict the class of each probe
            preds, dists = matcher.predict(X_test, metric=metric)
            crr = evaluate_identification_crr(y_test, preds)

            print(f"[{space_name}] {metric.upper()} CRR: {crr:.2f}%")

            results_rows.append({
                "Space": space_name,
                "Metric": metric.upper(),
                "CRR (%)": round(crr, 4),
            })

            # Verification mode: generate ROC only in reduced space
            if space_name == "Reduced":
                score_matrix = matcher.all_match_scores(X_test, metric=metric)
                out_path, roc_auc = evaluate_verification_roc_from_scores(
                    score_matrix=score_matrix,
                    probe_labels=y_test,
                    gallery_labels=matcher.classes,
                    metric=metric,
                    output_dir=fig_dir,
                )
                print(f"   -> ROC saved to: {out_path} | AUC={roc_auc:.4f}")

        print("-" * 30)

    # Save final recognition table
    csv_path = os.path.join(table_dir, "recognition_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Space", "Metric", "CRR (%)"])
        writer.writeheader()
        writer.writerows(results_rows)

    print(f"\n[INFO] Recognition table saved to: {csv_path}")


if __name__ == "__main__":
    main()
