# CrediSure: Intelligent Credit Scoring System

CrediSure predicts whether a loan applicant is likely to be low risk or high risk using the German Credit dataset.

This project is lightweight, reproducible, and optimized for free or limited compute environments. It also provides practical outputs like model interpretation and clear rejection reasons.

## Objective

Build a machine learning pipeline to assess creditworthiness and support decision-making with understandable model outputs.

## Dataset

- Name: German Credit Data (Statlog)
- Size: ~1000 rows
- Raw files in this project:
  - `statlog+german+credit+data/german.data`
  - `statlog+german+credit+data/german.doc`

Target mapping used in code:

- `1` (Good Credit) -> `0`
- `2` (Bad Credit) -> `1`

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Joblib

## Project Structure

```text
Credit Scoring/
|- data/
|  |- german_credit.csv
|  |- processed_data.pkl
|- models/
|  |- lr_base.pkl
|  |- rf_base.pkl
|  |- lr_optimized.pkl
|  |- credit_model.pkl
|- notebooks/
|  |- target_distribution.png
|  |- credit_amount_distribution.png
|  |- correlation_matrix.png
|  |- lr_feature_importance.png
|  |- rf_feature_importance.png
|- src/
|  |- data_prep.py
|  |- train.py
|  |- optimize.py
|- Credit_Scoring_Project_Plan_Updated.md
|- README.md
```

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Run Pipeline

From the project root, run:

```bash
python src/data_prep.py
python src/train.py
python src/optimize.py
```

## What Each Script Does

- `src/data_prep.py`
  - Loads raw dataset
  - Creates `data/german_credit.csv`
  - Runs EDA and saves plots in `notebooks/`
  - Preprocesses data and stores split/artifacts in `data/processed_data.pkl`

- `src/train.py`
  - Trains Logistic Regression and Random Forest baseline models
  - Prints Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC
  - Saves baseline models in `models/`

- `src/optimize.py`
  - Tunes Logistic Regression using `GridSearchCV`
  - `param_grid = {'C': [0.1, 1, 10]}`
  - Saves optimized model and final model (`models/credit_model.pkl`)
  - Saves top feature-importance plots

## Current Progress

Completed:

- Data preparation and preprocessing
- EDA visualizations
- Baseline model training
- Model optimization and feature interpretation

Next milestone:

- Rejection explanation and improvement advice system

## 7-Day Plan

- Day 1: Data loading, cleaning, and EDA
- Day 2: Preprocessing and train/test split
- Day 3: Baseline model training (LR + RF)
- Day 4: Evaluation and model comparison
- Day 5: Hyperparameter tuning and interpretation
- Day 6: Rejection reason and advice module
- Day 7: Final report and GitHub-ready packaging

## Planned Rejection Explanation Module

For applicants predicted as high risk:

- Identify top risk factors from model behavior
- Show rejection reasons in plain language
- Provide actionable next-step advice

Example advice mapping:

- High loan amount -> Apply for a smaller amount
- Poor credit history -> Improve repayment consistency
- Low savings -> Increase savings for the next 3-6 months
- High existing debt -> Reduce outstanding liabilities

## Deliverables

- Clean dataset and processed artifacts
- Trained baseline and optimized models
- Feature importance outputs
- Final model (`credit_model.pkl`)
- Rejection reason and advice module (next)
- Documentation and final report

## Author

JISTO-C-S
