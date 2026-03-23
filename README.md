# Iris Recognition System

This repository contains our implementation of an iris recognition pipeline for the group project based on **Ma et al. (2003), *Personal Identification Based on Iris Texture Analysis***. The project focuses on the stages required by the assignment: **image preprocessing, feature extraction, iris matching, and performance evaluation** using the **CASIA-IrisV1** dataset.

The system is organized as a modular end-to-end pipeline:

1. **Iris Localization**
2. **Iris Normalization**
3. **Image Enhancement**
4. **Feature Extraction**
5. **Iris Matching**
6. **Performance Evaluation**

---

## 1. Project Objective

The goal of this project is to implement an iris recognition algorithm inspired by the design in **Ma et al. (2003)** and evaluate it on **CASIA-IrisV1** under the fixed experimental protocol required by the assignment:

* **Session 1** images are used for **training**
* **Session 2** images are used for **testing**

The system reports:

* **Correct Recognition Rate (CRR)** for identification mode
* **Verification ROC curves** for different matching metrics
* A summary table of recognition results across different feature spaces and distance measures

---

## 2. Repository Structure

```text
cw3729_j17239_yj2904_IrisRecognition/
├── CASIA-IrisV1/
├── results/
│   ├── figures/
│   │   ├── ROC_curve_l1.png
│   │   ├── ROC_curve_l2.png
│   │   └── ROC_curve_cosine.png
│   └── tables/
│       └── recognition_results.csv
├── src/
│   ├── FeatureExtraction.py
│   ├── ImageEnhancement.py
│   ├── IrisLocalization.py
│   ├── IrisMatching.py
│   ├── IrisNormalization.py
│   ├── PerformanceEvaluation.py
│   └── main.py
├── .gitignore
├── LICENSE
└── README.md
```

---

## 3. Dataset and Expected Folder Layout

This project expects the **CASIA-IrisV1** dataset to be placed at the repository root.

### Expected layout

```text
cw3729_j17239_yj2904_IrisRecognition/
├── CASIA-IrisV1/
│   ├── 001/
│   │   ├── 1/
│   │   │   ├── xxx.bmp
│   │   │   ├── ...
│   │   └── 2/
│   │       ├── xxx.bmp
│   │       ├── ...
│   ├── 002/
│   ├── ...
```

Where:

* each subject folder contains iris images from two sessions
* folder `1` is used as the training session
* folder `2` is used as the testing session

The code assumes BMP files and searches for the dataset at the repository root.

---

## 4. System Pipeline

## 4.1 Iris Localization

**File:** `src/IrisLocalization.py`

This module detects the inner boundary (**pupil**) and the outer boundary (**iris**) from the raw grayscale eye image.

### Current implementation logic

1. Apply median blur to reduce small-scale noise.
2. Estimate a coarse pupil center using horizontal and vertical projection minima.
3. Refine the pupil location inside a local `120 × 120` region using adaptive thresholding and contour analysis.
4. Use a constrained **Hough Circle Transform** to refine the pupil circle.
5. Use a second constrained **Hough Circle Transform** to detect the outer iris boundary.
6. Select the most plausible outer circle based on radius and distance from the pupil center.

### Output

The module returns:

* `pupil_params = (xp, yp, rp)`
* `iris_params = (xi, yi, ri)`

### Notes

The implementation allows the iris center to differ slightly from the pupil center, which is important because the two circles are often not perfectly concentric in real images.

---

## 4.2 Iris Normalization

**File:** `src/IrisNormalization.py`

This module transforms the annular iris region into a fixed-size rectangular image.

### Current implementation logic

1. Use the detected pupil and iris boundaries.
2. Sample points along both the pupil boundary and iris boundary for each angular position.
3. Interpolate between the two boundaries along the radial direction.
4. Use **bilinear interpolation** to sample grayscale values at non-integer pixel locations.
5. Build a normalized iris image of size:

* **radial resolution = 64**
* **angular resolution = 512**

### Output

A normalized grayscale iris image of shape:

```text
64 × 512
```

### Notes

The implementation uses a non-concentric mapping formulation rather than a simplified concentric assumption. This reduces distortion when pupil and iris centers are slightly offset.

---

## 4.3 Image Enhancement

**File:** `src/ImageEnhancement.py`

This module improves contrast and compensates for nonuniform illumination in the normalized iris image.

### Current implementation logic

1. Estimate coarse background illumination using **16 × 16 block means**.
2. Expand the coarse background map to full image size using **bicubic interpolation**.
3. Subtract the estimated background from the normalized image.
4. Rescale the corrected image into the standard grayscale range.
5. Apply **local histogram equalization** in each `32 × 32` region.

### Output

An enhanced normalized iris image.

### Notes

This implementation follows the same general idea as Ma et al. (2003): first compensate for uneven illumination, then improve local contrast before feature extraction.

---

## 4.4 Feature Extraction

**File:** `src/FeatureExtraction.py`

This module extracts a compact numerical representation of iris texture from the enhanced image.

### Current implementation logic

1. Select a fixed **48 × 512** region of interest (ROI) from the normalized/enhanced iris image.
2. Construct two **circularly symmetric spatial filters** with different scale parameters.
3. Filter the ROI using both filters.
4. Divide the filtered outputs into **8 × 8** blocks.
5. For each block, compute:

   * **Mean**
   * **Average Absolute Deviation (AAD)**
6. Concatenate all block features into a single feature vector.
7. Apply final **L2 normalization** to reduce global amplitude variation.

### Output

A **1536-dimensional feature vector**.

### Why 1536 dimensions?

* ROI size: `48 × 512`
* Block size: `8 × 8`
* Number of blocks per filtered image:

  * `(48 / 8) × (512 / 8) = 6 × 64 = 384`
* Two filtered images: `384 × 2 = 768`
* Two statistics per block: `768 × 2 = 1536`

---

## 4.5 Iris Matching

**File:** `src/IrisMatching.py`

This module performs dimensionality reduction and nearest-center classification in feature space.

### Current implementation logic

1. Standardize features using `StandardScaler`.
2. Optionally apply **PCA** before LDA to address the small-sample-size issue.
3. Apply **Linear Discriminant Analysis (LDA / FLD)** to obtain a more discriminative low-dimensional representation.
4. Build class templates from the training data using multiple **rotation-shifted templates**.
5. For each class, aggregate template samples by shift tag to obtain **7 templates per class**.
6. For each test sample, compute the distance to the 7 templates of every class.
7. Use the **minimum template distance within each class** as the class score.
8. Predict the identity using the nearest class score.

### Supported distance measures

* **L1** (`cityblock`)
* **L2** (`euclidean`)
* **Cosine distance**

### Output

For a given test set, the matcher returns:

* predicted labels
* minimum distances to the matched class templates

### Notes

The implementation supports evaluation in:

* **Original Space**: scaled features without PCA/LDA
* **Reduced Space**: PCA + LDA representation

The reduced-space matching stage follows the paper more closely by combining discriminative projection with multi-template matching for rotation robustness.

---

## 4.6 Performance Evaluation

**File:** `src/PerformanceEvaluation.py`

This module evaluates the system in both identification mode and verification mode.

### Identification mode

The function

```python
evaluate_identification_crr(y_true, y_pred)
```

computes the **Correct Recognition Rate (CRR)**:

```text
CRR = (number of correctly classified test samples / total number of test samples) × 100
```

### Verification mode

The function

```python
evaluate_verification_roc_from_scores(score_matrix, probe_labels, gallery_labels, ...)
```

builds a verification ROC curve using **probe-vs-enrolled-class scores**.

In this implementation:

* `score_matrix[i, j]` is the distance between probe `i` and enrolled class `j`
* a **genuine comparison** means the probe label matches the gallery class label
* an **impostor comparison** means they do not match

Distances are converted into similarity-style scores by negating them before ROC computation.

### Saved outputs

* `results/tables/recognition_results.csv`
* `results/figures/ROC_curve_l1.png`
* `results/figures/ROC_curve_l2.png`
* `results/figures/ROC_curve_cosine.png`

---

## 5. Main Execution Script

**File:** `src/main.py`

This is the main entry point of the entire system.

### What it does

1. Locates the dataset and result directories
2. Iterates through all subject folders and both sessions
3. Runs the full preprocessing pipeline on each image:

   * localization
   * normalization
   * enhancement
4. Extracts original training features from **Session 1**
5. Generates **7 shifted template versions** of each training sample for rotation compensation
6. Extracts test features from **Session 2**
7. Runs matching in:

   * Original Space
   * Reduced Space
8. Evaluates each setting using:

   * L1
   * L2
   * Cosine
9. Saves result tables and ROC figures

### Important protocol

This script uses the fixed assignment protocol:

* **Session 1** → training
* **Session 2** → testing

---

## 6. How to Run

### Step 1: Clone the repository

```bash
git clone <your-repo-url>
cd cw3729_j17239_yj2904_IrisRecognition
```

### Step 2: Install required packages

It is recommended to use a clean virtual environment.

```bash
pip install numpy opencv-python matplotlib scikit-learn scipy
```

### Step 3: Place the dataset

Make sure the folder structure looks like this:

```text
cw3729_j17239_yj2904_IrisRecognition/
├── CASIA-IrisV1/
├── src/
├── results/
└── README.md
```

### Step 4: Run the main script

From the project root:

```bash
python src/main.py
```

### Step 5: Check outputs

After the run finishes, inspect:

```text
results/tables/recognition_results.csv
results/figures/ROC_curve_l1.png
results/figures/ROC_curve_l2.png
results/figures/ROC_curve_cosine.png
```

---

## 7. Final Experimental Results

Using the final implementation and the required Session 1 / Session 2 protocol on **CASIA-IrisV1**, the system produced the following results:

### Identification Results

#### Original Space

* **L1 CRR:** 73.38%
* **L2 CRR:** 71.99%
* **Cosine CRR:** 73.38%

#### Reduced Space

* **L1 CRR:** 80.79%
* **L2 CRR:** 81.25%
* **Cosine CRR:** 86.11%

### Verification Results (Reduced Space)

* **L1 ROC AUC:** 0.9476
* **L2 ROC AUC:** 0.9555
* **Cosine ROC AUC:** 0.9912

These results show that the reduced-space representation consistently improves identification performance, and the cosine-based matching strategy gives the strongest final result.

---

## 8. Output Files

### 8.1 Recognition table

**Path:** `results/tables/recognition_results.csv`

This file records recognition performance for different combinations of:

* feature space
* distance metric
* CRR

Expected columns:

* `Space`
* `Metric`
* `CRR (%)`

---

### 8.2 ROC figures

**Paths:**

* `results/figures/ROC_curve_l1.png`
* `results/figures/ROC_curve_l2.png`
* `results/figures/ROC_curve_cosine.png`

These files store the verification ROC curves generated from the reduced-space matching scores.

---

## 9. File-by-File Summary

### `src/IrisLocalization.py`

Detects pupil and iris circles from the raw grayscale eye image.

### `src/IrisNormalization.py`

Maps the circular iris ring into a fixed-size rectangular representation using bilinear interpolation.

### `src/ImageEnhancement.py`

Performs illumination correction and local contrast enhancement.

### `src/FeatureExtraction.py`

Applies two circularly symmetric spatial filters and encodes block-level statistics into a 1536D feature vector.

### `src/IrisMatching.py`

Standardizes features, applies PCA/LDA when enabled, and performs nearest-template-center matching using multiple distance measures.

### `src/PerformanceEvaluation.py`

Computes CRR and generates verification ROC curves from probe-vs-gallery-class score matrices.

### `src/main.py`

Runs the full end-to-end pipeline on the dataset and saves the final outputs.

---

## 10. Current Implementation Characteristics

The current version of the project has the following characteristics:

* Modular pipeline with separate preprocessing, feature extraction, matching, and evaluation stages
* Non-concentric normalization rather than a simplified concentric model
* Fixed-size ROI-based feature extraction
* PCA + FLD matching pipeline
* Multi-template matching for rotation robustness
* Evaluation in both original feature space and reduced feature space
* Identification and verification results saved automatically

---

## 11. Limitations

This implementation works as a complete end-to-end system, but several limitations remain.

### 1. Localization sensitivity

The pupil and iris boundary estimates still depend on thresholding, contour analysis, and constrained circle detection. In difficult images, imperfect localization may still affect downstream performance.

### 2. Simplified rotation modeling

The current system models rotation by cyclic column shifts of the normalized iris image. This is practical and effective, but it is still an approximation of the full angular variation encountered in real biometric systems.

### 3. No explicit occlusion masking

Although the pipeline reduces some eyelid and eyelash interference through preprocessing and ROI selection, there is no dedicated occlusion mask for eyelids, eyelashes, or reflections.

### 4. Dataset dependence

Some parameter choices, such as radius ranges and shift settings, are tuned for CASIA-IrisV1 and may require adjustment for other datasets.

### 5. Paper alignment

The implementation is strongly inspired by Ma et al. (2003), but it is still a course project reproduction rather than a line-by-line reimplementation of every detail in the original paper.

---

## 12. Possible Improvements

Several directions could further improve the current system:

1. More robust pupil and iris localization
2. Explicit eyelid / eyelash / reflection masking
3. More exact reproduction of the rotation handling strategy in the original paper
4. More systematic tuning of PCA/LDA dimensions
5. Additional experiments on ROI placement and filter settings
6. Evaluation on additional iris datasets for generalization analysis

---

## 13. Reference

Li Ma, Tieniu Tan, Yunhong Wang, and Dexin Zhang.
**Personal Identification Based on Iris Texture Analysis.**
IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 25, No. 12, December 2003.

---

## 14. Final Note

This repository is organized as a runnable course project implementation. The emphasis is on producing a clear, modular, and reproducible iris recognition pipeline that matches the assignment structure and can be inspected, tested, and improved step by step.

---

## 15. Team Members and Contributions

**Member 1: Cheng Wu (cw3729)**  

Contributed to building the iris recognition pipeline based on Ma et al. (2003), including preprocessing, feature extraction, matching, and evaluation.

Worked on iris localization using thresholding, contour detection, and Circular Hough Transform, and implemented normalization using a non-concentric rubber sheet model.

Implemented the feature extraction process using two spatial filters and block-wise statistics (Mean and MAD), resulting in a 1536-dimensional feature vector.

Worked on the matching pipeline, including feature scaling, PCA + Fisher Linear Discriminant (FLD), and nearest-center classification with L1, L2, and cosine distance metrics.

Helped debug and improve the system by fixing the LDA dimensionality issue, adjusting ROI selection to reduce eyelash interference, and resolving feature scaling inconsistencies.

Also contributed to integrating the full pipeline in main.py, generating CRR and ROC results, and preparing the README and documentation.

In the later stage of the project, further worked on refining the implementation to better align with the paper and assignment requirements, including improving the rotation-template matching logic, adjusting the preprocessing and feature extraction details, and updating the evaluation pipeline to produce the final improved results.


---
**Member 2: Jinbo Li (jl7239)**  
### Contributions

Modified feature extraction to use the **magnitude of filter responses |F(x, y)|** instead of raw values, improving robustness to phase variations and noise.

Implemented **rotation compensation** by generating multiple templates per image using circular shifts, following the approach in Ma et al. (2003).

Added an **image quality selection** step based on frequency-domain analysis to filter out blurred, occluded, or low-quality iris images.

Improved pipeline stability with additional **error handling** to skip failed samples during processing.

Updated iris normalization to use **bilinear interpolation**, resulting in more accurate texture mapping.

---

**Member 3: Yuan Ji (yj2904)**  
### Contributions

Contributed to improving the iris recognition pipeline through systematic experimentation on preprocessing, feature extraction, matching, and evaluation under the CASIA-IrisV1 setting.

Worked on the feature extraction module by refining the dual-filter design and tuning complementary filter parameters, including frequency separation and vertical-scale separation, which substantially improved texture discrimination in the normalized iris image.

Improved the feature normalization strategy by adopting L2 normalization for the extracted feature vector, and evaluated multiple block-wise encoding variants to determine the most stable feature representation.

Refined the matching pipeline by testing PCA + Fisher Linear Discriminant (FLD) + nearest-center classification, stabilizing PCA with a deterministic solver, and tuning PCA dimensionality to better match the stronger feature representation in reduced space.

Enhanced the sample selection strategy by replacing quality filtering with quality-ranked selection, sorting images within each subject/session using frequency-domain quality measurements, and retaining the top-ranked training and testing samples.

Improved the localization and pipeline stability by fixing the overflow issue in iris boundary selection, adding deterministic ordering in data processing, and checking selected samples to ensure reproducible experimental results.

Conducted extensive parameter studies on ROI choice, block size, filter configuration, PCA dimensionality, and normalization behavior, identifying the current stable best-performing configuration with improved reduced-space cosine CRR.

Integrated and validated the full pipeline in main.py, recorded the final experimental settings and results.
