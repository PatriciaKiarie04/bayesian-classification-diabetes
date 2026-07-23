# Bayesian Classification: Diabetes Prediction

**Course:** Bayesian Inference & Decision Theory — Semester Group Project
**Topic:** Bayesian Classification (Naive Bayes & Bayesian Logistic Regression)
**Group size:** 5

## Group Members
- [Stacy Oboko]
- [Patricia Kiarie]
- [Levin Ekuam]
- [Shammar Nawate]
- [Paul Mbuvi]

## Project Overview
We are building a Bayesian classification model to predict diabetes onset using the Pima Indians Diabetes dataset, comparing a conjugate Gaussian Naive Bayes model against a Bayesian logistic regression model, with full posterior specification, diagnostics, and uncertainty quantification.

## Dataset
**Pima Indians Diabetes Database** (UCI ML, via Kaggle)
- 768 rows, 8 features + binary outcome (`Outcome`: 1 = diabetic, 0 = non-diabetic)
- Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

## Progress So Far

### 1. Data Loading & Inspection
- Loaded dataset via Google Drive into Colab
- Inspected structure (`.info()`, `.describe()`)

### 2. Data Cleaning
Identified biologically impossible zero values (treated as missing data) in:

| Feature | % Zero Values |
|---|---|
| Glucose | 0.7% |
| BloodPressure | 4.6% |
| SkinThickness | 29.6% |
| Insulin | 48.7% |
| BMI | 1.4% |

- Replaced zeros with `NaN`, then imputed using the **median within each outcome class** (rather than a flat overall median), since diabetic and non-diabetic patients have different underlying distributions for these features
- **Known limitation:** Insulin has high missingness (48.7%) — imputation here is less reliable and is flagged as a limitation in our interpretation, or considered for exclusion as a feature

### 3. Train/Test Split
- 80/20 split, stratified on `Outcome` to preserve class balance in both sets
- `random_state=42` fixed for reproducibility across the group

### 4. Prior Specification (in progress)
- **Class prior:** Beta(1,1) (uniform) on P(Outcome=1), to be compared against a weakly informative alternative for sensitivity analysis
- **Feature likelihoods:** Normal-Inverse-Gamma (NIG) conjugate prior per feature per class, with hyperparameters centred on overall (not per-class) feature means to avoid data leakage into the prior
- Baseline `sklearn` GaussianNB fit as an uninformative-prior-limit comparison point

## Next Steps
- [ ] Implement manual conjugate NIG posterior updates for Naive Bayes (not just sklearn's MLE fit)
- [ ] Build Bayesian logistic regression model (PyMC, with MCMC)
- [ ] Run convergence diagnostics (trace plots, R-hat, ESS) for the logistic regression model
- [ ] Posterior predictive checks for both models
- [ ] Model comparison (WAIC/LOO)
- [ ] Prior sensitivity analysis
- [ ] Final results interpretation and slide deck

## Repo Structure
```
/data          → diabetes.csv
/notebooks     → analysis notebooks
/slides        → final presentation deck
README.md      → this file
```
