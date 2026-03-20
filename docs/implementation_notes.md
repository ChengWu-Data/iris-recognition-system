# Implementation Notes

## Global principle

This project should be implemented as a modular pipeline.

Each module should:
1. Have a clear input
2. Have a clear output
3. Save or return interpretable intermediate results
4. Be testable independently

## Module design

### 1. IrisLocalization
Input:
- raw eye image

Output:
- pupil center
- pupil radius
- iris outer boundary
- optional mask / visualization

Main concerns:
- sensitivity to occlusion
- robustness to low contrast
- boundary estimation consistency

### 2. IrisNormalization
Input:
- raw image
- localization parameters

Output:
- normalized iris image of fixed size

Main concerns:
- coordinate mapping correctness
- handling non-concentric boundaries
- preserving usable iris texture

### 3. ImageEnhancement
Input:
- normalized iris image

Output:
- enhanced iris image

Main concerns:
- illumination correction
- contrast improvement
- avoiding over-enhancement artifacts

### 4. FeatureExtraction
Input:
- enhanced iris image

Output:
- feature vector

Main concerns:
- parameter consistency
- block partition design
- reproducibility of extracted feature dimension

### 5. IrisMatching
Input:
- training features
- testing features
- labels

Output:
- predicted labels
- similarity scores
- matching distances

Main concerns:
- support for L1, L2, and cosine similarity
- correct dimensionality reduction logic
- reproducibility of classification pipeline

### 6. PerformanceEvaluation
Input:
- predictions
- labels
- similarity scores

Output:
- CRR
- ROC curve
- saved result tables/figures

Main concerns:
- correct separation of identification and verification modes
- correct use of train/test split
- output formatting for final submission

## Integration rule

The main script should not contain detailed algorithm logic.
It should only:
- load data
- call module functions
- save outputs
- print summary results
