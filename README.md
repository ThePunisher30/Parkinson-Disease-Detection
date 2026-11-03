# Parkinson's Disease Detection

Detecting Parkinson's disease from voice recordings with classical machine learning. Seven classifiers are trained, tuned and compared on the UCI Parkinsons dataset. College coursework project.

## Dataset

[UCI Parkinsons dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data): 195 voice recordings with 22 biomedical voice measures (jitter, shimmer, HNR and others). 147 recordings are from people with Parkinson's and 48 are healthy. The notebook downloads it automatically.

## What the notebook does

1. **Preprocessing:** drops the name column, checks for duplicates and missing values (none found)
2. **EDA:** class balance, correlation heatmap, box plots per class, pair plots of the jitter and shimmer groups
3. **Balancing and scaling:** SMOTE oversampling (195 to 294 rows), then min-max scaling to [-1, 1] and an 80/20 train/test split
4. **Models:** Decision Tree, Random Forest, Logistic Regression, SVM, Naive Bayes, KNN, XGBoost, with grid-search hyperparameter tuning for the tree models, SVM and XGBoost
5. **Evaluation:** classification reports, confusion matrices and ROC curves for each model, and the trained models saved with joblib

## Results (59-row test set)

| Model | Test accuracy |
| --- | --- |
| Random Forest (tuned) | 0.97 |
| KNN (k=5) | 0.97 |
| Decision Tree (tuned) | 0.93 |
| XGBoost (tuned) | 0.92 |
| Logistic Regression | 0.83 |
| SVM (linear, untuned) | 0.81 |
| Naive Bayes | 0.76 |

## A note on these numbers

SMOTE is applied before the train/test split, so synthetic samples built from the test rows' neighbours can end up in training. With only 195 recordings, these scores are likely optimistic. A stricter version would split first and oversample only the training set. This is on my list to fix.

## Run it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost joblib requests
jupyter notebook Parkinson_Disease_Detection.ipynb
```
