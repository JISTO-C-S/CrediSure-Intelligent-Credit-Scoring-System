# Credit Scoring Website

This project now includes a web interface for the credit scoring model.

## Run Locally

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Start the web app:

```powershell
python app.py
```

3. Open:

`http://127.0.0.1:5000`

## Features

- Plain-language loan application form
- Risk score prediction using `models/credit_model.pkl`
- Rejection explanation with top risk factors
- Actionable advice to improve approval chance
- Simulated improved profile result
