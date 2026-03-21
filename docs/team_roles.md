# Team Roles

## Team Members

- Member 1: [Cheng Wu / cw3729]
- Member 2: [Jinbo Li / jl7239]
- Member 3: [Yuan Ji / yj2904]

## Role Assignment

### Member 1
* **Pipeline Implementation**: Designed and developed the iris recognition system from scratch, ensuring full alignment with the Ma et al. (2003) framework.
* **Core Mathematical Modules**:
    * **Image Preprocessing**: Implemented **Hough Transform** for boundary localization and **Daugman’s Rubber Sheet Model** for polar-to-rectangular normalization.
    * **Feature Extraction**: Developed the dual-channel **Gabor spatial filter bank** and coded the block-wise statistical extraction (Mean & MAD) to generate the 1536D feature vectors.
    * **Matching Engine**: Built the dimensionality reduction engine using **PCA (Principal Component Analysis)** and **FLD (Fisher Linear Discriminant)**, along with the multi-metric Nearest Center Classifier.
* **Technical Problem Solving**: 
    * Debugged and resolved the LDA dimensionality constraint error ($n\_components < n\_classes - 1$).
    * Integrated **StandardScaler** to fix feature scaling issues between Mean and MAD values.
    * Optimized the **ROI (Region of Interest)** to 15-63 rows to effectively bypass eyelash and eyelid occlusion in the CASIA-IrisV1 dataset.
* **Testing & Infrastructure**: Developed the `main.py` execution script and the automated evaluation suite to generate CRR tables and ROC curves.

### Member 2


### Member 3

## Shared Responsibilities

## Working Rules

1. Each member should work on assigned modules first.
2. Before editing shared files, always pull the latest version.
3. All major changes must include a clear commit message.
4. Intermediate outputs should be saved for inspection.
5. Code should include comments explaining logic and key parameters.
