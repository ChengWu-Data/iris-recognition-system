# Project Plan

## Objective

Reproduce the iris recognition pipeline required by the course project based on Ma et al. (2003), using the provided CASIA Iris dataset.

## What the project must do

According to the project description, the implementation should follow the same overall design as Ma et al. (2003), with emphasis on:

1. Image preprocessing
2. Feature extraction
3. Iris matching
4. Performance evaluation

The project must produce:
- CRR for identification mode
- ROC curve for verification mode

## Required pipeline

1. Iris Localization
2. Iris Normalization
3. Image Enhancement
4. Feature Extraction
5. Iris Matching
6. Performance Evaluation

## Fixed experiment design

- Use CASIA-IrisV1
- First session images for training
- Second session images for testing

## Priority order

### Phase 1: Infrastructure
- Create repository structure
- Create module files
- Define interfaces between modules
- Define data loading assumptions
- Set package requirements

### Phase 2: Baseline pipeline
- Run a minimal end-to-end pipeline
- Confirm that all modules connect correctly
- Save intermediate outputs for inspection

### Phase 3: Performance improvement
- Improve localization
- Improve normalization stability
- Improve enhancement quality
- Tune feature extraction parameters
- Tune matching pipeline

### Phase 4: Final outputs
- Generate CRR table
- Generate ROC curve
- Organize results
- Write final README
- Package files for submission

## Project risks

- Localization errors may affect all later stages
- Different team members may produce inconsistent code style
- Matching performance may be limited if preprocessing is unstable
- Final scripts must be runnable without manual debugging

## Deliverable checklist

- [ ] All required `.py` files created
- [ ] Main pipeline runnable
- [ ] Train/test split follows assignment
- [ ] CRR reported
- [ ] ROC reported
- [ ] README completed
- [ ] Peer evaluation form completed
- [ ] Zip file packaged without dataset
