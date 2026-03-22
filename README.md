# Iris Recognition System

This repository contains our implementation of an iris recognition pipeline for the group project based on **Ma et al. (2003), _Personal Identification Based on Iris Texture Analysis_**. The project focuses on the stages required by the assignment: **image preprocessing, feature extraction, iris matching, and performance evaluation** using the **CASIA-IrisV1** dataset. 

The system follows the overall structure of a classical iris recognition pipeline:

1. **Iris Localization**
2. **Iris Normalization**
3. **Image Enhancement**
4. **Feature Extraction**
5. **Iris Matching**
6. **Performance Evaluation**

The current implementation is modular and organized as a complete runnable project rather than a single script.

---

## 1. Project Objective

The goal of this project is to implement an iris recognition algorithm inspired by the design in **Ma et al. (2003)** and evaluate it on **CASIA-IrisV1** under the fixed experimental protocol required by the assignment:

- **Session 1** images are used for **training**
- **Session 2** images are used for **testing** :contentReference[oaicite:2]{index=2}

The system reports:

- **Correct Recognition Rate (CRR)** for identification mode
- **ROC-style verification result** saved as a figure
- A summary table of recognition results across different distance measures

---

## 2. Repository Structure

```text
iris-recognition-system/
├── configs/
│   └── default.yaml
├── results/
│   ├── figures/
│   │   └── ROC_curve.png
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
````

---

## 3. Dataset and Expected Folder Layout

This project expects the **CASIA-IrisV1** dataset to be placed at the repository root.

### Expected layout

```text
iris-recognition-system/
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
* folder `1` is used as training session
* folder `2` is used as testing session

The code assumes BMP files and uses the dataset path:

```python
dataset_path = os.path.join(root_dir, "CASIA-IrisV1")
```

So the dataset should be unzipped directly inside the project root.

---

## 4. System Pipeline

### 4.1 Iris Localization

**File:** `src/IrisLocalization.py`

This module detects the inner boundary (**pupil**) and outer boundary (**iris**) from the raw grayscale eye image.

#### Current implementation logic

1. Apply median blur to reduce small-scale noise.
2. Threshold the image to isolate the dark pupil region.
3. Use contour detection and `minEnclosingCircle` to estimate pupil center and radius.
4. Use **Circular Hough Transform** to detect the outer iris boundary.
5. Select the detected outer circle closest to the pupil center as the best candidate.

#### Output

The module returns:

* `pupil_params = (xp, yp, rp)`
* `iris_params = (xi, yi, ri)`

These parameters are then used by normalization.

#### Notes

The current implementation allows the iris center to differ from the pupil center, which is important because the two circles are often not perfectly concentric in real images.

---

### 4.2 Iris Normalization

**File:** `src/IrisNormalization.py`

This module transforms the annular iris region into a fixed-size rectangular image.

#### Current implementation logic

1. Use the detected pupil and iris boundaries.
2. Sample points along the pupil boundary and iris boundary for each angle.
3. Interpolate between the two boundaries along the radial direction.
4. Build a normalized iris image of size:

* **radial resolution = 64**
* **angular resolution = 512**

#### Output

A normalized grayscale iris image of shape:

```text
64 × 512
```

#### Notes

The implementation uses a non-concentric mapping formulation rather than a simplified concentric assumption. This reduces distortion when pupil and iris centers are offset.

---

### 4.3 Image Enhancement

**File:** `src/ImageEnhancement.py`

This module improves contrast and compensates for nonuniform illumination in the normalized iris image.

#### Current implementation logic

1. Estimate coarse background illumination using **16 × 16** block means.
2. Expand the coarse background to full image size using bicubic interpolation.
3. Subtract estimated illumination from the normalized image.
4. Apply local contrast enhancement using **CLAHE**.
5. Perform image quality assessment using frequency-domain analysis:
   - Compute the **2D Fourier spectrum** of the enhanced iris ROI.
   - Divide frequency into low (F1), middle (F2), and high (F3) bands.
   - Compute the quality ratio:
     `ratio = F2 / (F1 + F3)`
   - Filter out low-quality images based on this ratio.

#### Output

An enhanced normalized iris image.

#### Notes

This implementation is consistent with the general enhancement idea in Ma et al. (2003): compensate for uneven illumination first, then improve local contrast before feature extraction. 

The added quality assessment step follows the paper’s approach of analyzing frequency distribution to distinguish clear iris images from defocused, motion-blurred, or occluded ones. A threshold-based method is used instead of the original SVM classifier for simplicity.

---

### 4.4 Feature Extraction

**File:** `src/FeatureExtraction.py`

This module extracts a compact numerical representation of iris texture from the enhanced image.

#### Current implementation logic

1. Select a region of interest (ROI) from rows **10:58** of the normalized image.

   * This avoids the most heavily occluded upper region where eyelids and eyelashes often interfere.
2. Construct two **Gabor-like symmetric spatial filters** with different scale parameters.
3. Filter the ROI using both filters.
4. Divide the filtered outputs into **8 × 8** blocks.
5. For each block, compute:

   * **Mean**
   * **Mean Absolute Deviation (MAD)**
6. Concatenate all block features into a single feature vector.

#### Rotation Compensation (Template Generation)

To handle eye rotation, multiple templates are generated:

1. Shift the normalized iris image horizontally to simulate rotation:
   ```python
   np.roll(enh_img, shift, axis=1)
2. Use angles: [-9, -6, -3, 0, 3, 6, 9] degrees.
3. Extract features for each rotated version.
4. Store all resulting feature vectors as templates.

#### Output

7 feature templates per image (with rotation compensation)
**1536-dimensional feature vector**.

#### Why 1536 dimensions?

* ROI size: `48 × 512`
* Block size: `8 × 8`
* Number of blocks per filtered image:

  * `(48 / 8) × (512 / 8) = 6 × 64 = 384`
* Two filtered images: `384 × 2 = 768` blocks
* Two values per block: `768 × 2 = 1536`

---

### 4.5 Iris Matching

**File:** `src/IrisMatching.py`

This module performs dimensionality reduction and nearest-center classification in feature space.

#### Current implementation logic

1. Standardize features using `StandardScaler`.
2. Optionally apply **PCA** to reduce the original 1536D features to 120D.
3. Apply **Linear Discriminant Analysis (LDA / FLD)** to obtain a more discriminative low-dimensional representation.
4. Compute class centers using training samples.
5. For each test sample, compute distance to every class center and assign the nearest one.

#### Supported distance measures

* **L1** (`cityblock`)
* **L2** (`euclidean`)
* **Cosine distance**

#### Output

For a given test set, the matcher returns:

* predicted labels
* minimum distances to class centers

#### Notes

The implementation uses:

* **Original Space**: scaled features without PCA/LDA
* **Reduced Space**: PCA + LDA representation

This makes it possible to compare performance before and after dimensionality reduction.

---

### 4.6 Performance Evaluation

**File:** `src/PerformanceEvaluation.py`

This module evaluates the system in both identification and verification-style settings.

#### Identification mode

The function:

```python
evaluate_identification_crr(y_true, y_pred)
```

computes the **Correct Recognition Rate (CRR)**:

```text
CRR = (number of correctly classified test samples / total number of test samples) × 100
```

#### Verification-style output

The module also generates a ROC-style figure saved as:

```text
results/figures/ROC_curve.png
```

using the distance scores produced by the matcher.

#### Saved outputs

* `results/tables/recognition_results.csv`
* `results/figures/ROC_curve.png`

---

## 5. Main Execution Script

**File:** `src/main.py`

This is the main entry point of the entire system.

### What it does

1. Loads the configuration from `configs/default.yaml`
2. Creates result directories if needed
3. Iterates through the dataset
4. Runs the full pipeline on each image:

   * localization
   * normalization
   * enhancement
   * feature extraction
5. Builds training and testing feature matrices
6. Runs recognition in:

   * Original Space
   * Reduced Space
7. Evaluates each setting using:

   * L1
   * L2
   * Cosine
8. Saves result tables and figure

### Important assumption

The script assumes the dataset directory exists at:

```text
CASIA-IrisV1/
```

inside the repository root.

---

## 6. Configuration File

**File:** `configs/default.yaml`

This file stores project-level configuration values such as:

* path settings
* normalization size
* train/test session IDs
* image extension
* matching dimensions
* enabled metrics

This design keeps the main script cleaner and makes the project easier to modify without rewriting code.

---

## 7. How to Run

### Step 1: Clone the repository

```bash
git clone <your-repo-url>
cd iris-recognition-system
```

### Step 2: Install required packages

It is recommended to use a clean virtual environment.

```bash
pip install numpy opencv-python matplotlib scikit-learn scipy pandas pyyaml
```

If you create a `requirements.txt` file, you can also run:

```bash
pip install -r requirements.txt
```

### Step 3: Place the dataset

Make sure the folder structure looks like this:

```text
iris-recognition-system/
├── CASIA-IrisV1/
├── configs/
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
results/figures/ROC_curve.png
```

---

## 8. Output Files

### 8.1 Recognition table

**Path:** `results/tables/recognition_results.csv`

This file records recognition performance for different combinations of:

* feature space
* dimensionality
* distance metric

Expected columns include:

* `Space`
* `Dim`
* `Metric`
* `CRR (%)`

---

### 8.2 Verification figure

**Path:** `results/figures/ROC_curve.png`

This file stores the ROC-style curve generated from the reduced-space cosine-based result in the current implementation.

---

## 9. File-by-File Summary

### `src/IrisLocalization.py`

Detects pupil and iris circles from the raw grayscale eye image.

### `src/IrisNormalization.py`

Maps the circular iris ring into a fixed-size rectangular representation.

### `src/ImageEnhancement.py`

Performs illumination correction and local contrast enhancement.

### `src/FeatureExtraction.py`

Applies two symmetric spatial filters and encodes block-level statistics into a 1536D vector.

### `src/IrisMatching.py`

Standardizes features, applies PCA/LDA when enabled, and performs nearest-center matching using multiple distance measures.

### `src/PerformanceEvaluation.py`

Computes CRR and generates the ROC-style figure.

### `src/main.py`

Runs the full end-to-end pipeline on the dataset and saves the final outputs.

### `configs/default.yaml`

Stores configuration values for paths, data settings, normalization, and matching.

### `results/tables/recognition_results.csv`

Stores recognition performance results.

### `results/figures/ROC_curve.png`

Stores the verification-style plot.

---

## 10. Current Implementation Characteristics

The current version of the project has the following characteristics:

* Modular pipeline with separate preprocessing, feature extraction, matching, and evaluation stages
* Non-concentric normalization rather than a simplified concentric model
* ROI-based feature extraction to reduce eyelash/eyelid interference
* PCA + FLD matching pipeline
* Evaluation in both original feature space and reduced feature space

---

## 11. Limitations

This implementation works as a complete end-to-end system, but several limitations remain.

### 1. Localization sensitivity

The pupil detection currently depends on threshold-based segmentation and circle estimation. In darker or noisier images, localization may be slightly inaccurate.

### 2. Simplified outer boundary detection

The outer iris boundary is detected using Circular Hough Transform with fixed radius constraints calibrated for CASIA-IrisV1. This may reduce generalization to other datasets.

### 3. Sampling strategy in normalization

The current normalization uses direct pixel assignment without more advanced interpolation, which may introduce small distortions.

### 4. Occlusion handling

Although ROI selection reduces some interference from eyelids and eyelashes, there is no explicit eyelid/eyelash masking module in the current system.

### 5. Verification implementation

The current project generates a ROC-style verification figure from matcher outputs, but the strict pairwise genuine/impostor protocol can still be improved further for closer alignment with a classical biometric verification setting.

---

## 12. Possible Improvements

Several directions could improve the current system:

1. More robust pupil and iris localization
2. Better interpolation during normalization
3. Explicit eyelid/eyelash/reflection masking
4. Closer reproduction of the exact filter design in Ma et al. (2003)
5. Pairwise verification protocol for more standard biometric ROC analysis
6. More detailed parameter tuning for feature extraction and dimensionality reduction

---

## 13. Reference

Li Ma, Tieniu Tan, Yunhong Wang, and Dexin Zhang.
**Personal Identification Based on Iris Texture Analysis**.
IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 25, No. 12, December 2003. 

---

## 14. Final Note

This repository is organized as a runnable course project implementation. The emphasis is on producing a clear, modular, and reproducible iris recognition pipeline that matches the assignment structure and can be inspected, tested, and improved step by step. 



## 15. Team Members and Contributions

**Member 1: Cheng Wu (cw3729)**  

Contributed to building the iris recognition pipeline based on Ma et al. (2003), including preprocessing, feature extraction, matching, and evaluation.

Worked on iris localization using thresholding, contour detection, and Circular Hough Transform, and implemented normalization using a non-concentric rubber sheet model.

Implemented the feature extraction process using two spatial filters and block-wise statistics (Mean and MAD), resulting in a 1536-dimensional feature vector.

Worked on the matching pipeline, including feature scaling, PCA + Fisher Linear Discriminant (FLD), and nearest-center classification with L1, L2, and cosine distance metrics.

Helped debug and improve the system by fixing the LDA dimensionality issue, adjusting ROI selection to reduce eyelash interference, and resolving feature scaling inconsistencies.

Also contributed to integrating the full pipeline in main.py, generating CRR and ROC results, and preparing the README and documentation.

Note:
Further improvements to performance and evaluation (e.g., parameter tuning and verification protocol refinement) will be completed collaboratively within the group.

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
Contributed to improving the iris recognition pipeline through systematic experimentation on preprocessing, feature extraction, matching, and evaluation under the CASIA-IrisV1 setting.

Worked on the feature extraction module by refining the dual-filter design and tuning complementary filter parameters, including frequency separation and vertical-scale separation, which substantially improved texture discrimination in the normalized iris image.

Improved the feature normalization strategy by adopting L2 normalization for the extracted feature vector, and evaluated multiple block-wise encoding variants to determine the most stable feature representation.

Refined the matching pipeline by testing PCA + Fisher Linear Discriminant (FLD) + nearest-center classification, stabilizing PCA with a deterministic solver, and tuning PCA dimensionality to better match the stronger feature representation in reduced space.

Enhanced the sample selection strategy by replacing quality filtering with quality-ranked selection, sorting images within each subject/session using frequency-domain quality measurements, and retaining the top-ranked training and testing samples.

Improved the localization and pipeline stability by fixing the overflow issue in iris boundary selection, adding deterministic ordering in data processing, and checking selected samples to ensure reproducible experimental results.

Conducted extensive parameter studies on ROI choice, block size, filter configuration, PCA dimensionality, and normalization behavior, identifying the current stable best-performing configuration with improved reduced-space cosine CRR.

Integrated and validated the full pipeline in main.py, recorded the final experimental settings and results.
