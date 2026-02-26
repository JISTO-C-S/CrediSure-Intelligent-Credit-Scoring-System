# CrediSure - Intelligent Credit Scoring System

CrediSure predicts credit risk using the German Credit dataset and provides both:
- a command-line interactive predictor
- a Flask web application with rejection explanations and improvement simulation

## What Is Implemented

- Data preparation and preprocessing pipeline
- Model training and optimization (Logistic Regression as final model)
- ROC and interpretation utilities
- Interactive CLI predictor (`src/predict_interactive.py`)
- User-friendly rejection explanation (top risk drivers + advice)
- "What-if" improvement simulation for rejected applications
- Website UI + backend prediction API (`app.py`)

## Project Structure

```text
Credit Scoring/
|-- app.py
|-- requirements.txt
|-- data/
|   |-- processed_data.pkl
|-- models/
|   |-- credit_model.pkl
|-- src/
|   |-- data_prep.py
|   |-- train.py
|   |-- optimize.py
|   |-- explain.py
|   |-- predict_interactive.py
|-- templates/
|   |-- index.html
|-- static/
|   |-- styles.css
```

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Training Workflow

Run these scripts in order if artifacts are missing:

```powershell
python src/data_prep.py
python src/train.py
python src/optimize.py
```

This produces:
- `data/processed_data.pkl`
- `models/credit_model.pkl`

## Run CLI Predictor

```powershell
python src/predict_interactive.py
```

CLI features:
- Plain-language questions for common users
- Risk score and approve/reject decision
- Rejection reason breakdown (top contributing features)
- Actionable advice
- Optional improved-profile simulation

## Run Website

Start server:

```powershell
python app.py
```

Open in browser:

`http://127.0.0.1:5000`

Website features:
- User-friendly loan application form
- Model-based risk score prediction
- Clear rejection explanation with threshold gap
- Personalized advice based on top risk contributors
- Simulated "improved profile" result

## Notes

- Final model path: `models/credit_model.pkl`
- Preprocessing artifacts path: `data/processed_data.pkl`
- The website and CLI both use the same trained model and preprocessing artifacts.
