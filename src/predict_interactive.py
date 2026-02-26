import pandas as pd
import numpy as np
import os
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data.pkl')
CREDIT_MODEL_PATH = os.path.join(MODELS_DIR, 'credit_model.pkl')

# Predefined advice mapping based on feature names
ADVICE_DICT = {
    'credit_amount': 'Apply for a smaller loan amount.',
    'credit_history': 'Improve your credit repayment consistency.',
    'savings_status': 'Increase your savings balance over the next 3-6 months.',
    'employment_duration': 'Maintain a stable employment for a longer period.',
    'existing_credits': 'Reduce outstanding debts and liabilities.',
    'duration_months': 'Opt for a shorter loan repayment duration.',
    'checking_status': 'Improve the balance in your checking account.',
    'installment_rate': 'Ensure your installment rate is manageable relative to income.',
    'age': 'Build a longer credit history.',
    'purpose': 'Re-evaluate loan purpose or provide more collateral.',
    'property': 'Consider saving for better collateral or assets.',
    'residence_duration': 'Maintain a stable residence long-term.',
    'housing': 'A more stable housing situation could improve credit valuation.'
}
DEFAULT_ADVICE = "Improve overall financial stability and credit behavior."
FRIENDLY_NAMES = {
    'checking_status': 'Checking account status',
    'duration_months': 'Loan duration',
    'credit_history': 'Credit history',
    'purpose': 'Loan purpose',
    'credit_amount': 'Loan amount',
    'savings_status': 'Savings status',
    'employment_duration': 'Employment duration',
    'installment_rate': 'Installment burden',
    'personal_status_sex': 'Personal status',
    'other_debtors': 'Other debtors',
    'residence_duration': 'Residence duration',
    'property': 'Property/assets',
    'age': 'Age',
    'other_installment_plans': 'Other installment plans',
    'housing': 'Housing',
    'existing_credits': 'Existing credits',
    'job': 'Job level',
    'num_dependents': 'Dependents',
    'telephone': 'Telephone',
    'foreign_worker': 'Foreign worker'
}

NUMERIC_HINTS = {
    'duration_months': 'How many months do you need to repay the loan? (example: 12, 24, 36)',
    'credit_amount': 'What loan amount are you requesting? (example: 5000)',
    'installment_rate': 'Installment burden level from 1 to 4 (higher means heavier monthly burden).',
    'residence_duration': 'How many years have you lived at your current address?',
    'age': 'Enter your age in years.',
    'existing_credits': 'How many active credits/loans do you already have?',
    'num_dependents': 'How many people depend on your income?'
}

CATEGORY_DESCRIPTIONS = {
    'checking_status': {
        'A11': 'Balance is less than 0',
        'A12': 'Balance is between 0 and 200',
        'A13': 'Balance is 200 or more',
        'A14': 'No checking account'
    },
    'credit_history': {
        'A30': 'No previous credits / all paid on time',
        'A31': 'All previous credits paid back on time',
        'A32': 'Current credits paid back on time so far',
        'A33': 'Past delays in payment',
        'A34': 'Critical account / serious past credit issues'
    },
    'purpose': {
        'A40': 'Car (new)',
        'A41': 'Car (used)',
        'A410': 'Other purpose',
        'A42': 'Furniture or equipment',
        'A43': 'Radio/TV',
        'A44': 'Domestic appliances',
        'A45': 'Repairs',
        'A46': 'Education',
        'A48': 'Retraining',
        'A49': 'Business'
    },
    'savings_status': {
        'A61': 'Savings less than 100',
        'A62': 'Savings between 100 and 500',
        'A63': 'Savings between 500 and 1,000',
        'A64': 'Savings 1,000 or more',
        'A65': 'No savings / unknown'
    },
    'employment_duration': {
        'A71': 'Unemployed',
        'A72': 'Less than 1 year',
        'A73': '1 to 4 years',
        'A74': '4 to 7 years',
        'A75': '7 years or more'
    },
    'personal_status_sex': {
        'A91': 'Male, divorced/separated',
        'A92': 'Female, divorced/separated/married',
        'A93': 'Male, single',
        'A94': 'Male, married/widowed'
    },
    'other_debtors': {
        'A101': 'No other debtors/guarantors',
        'A102': 'Co-applicant',
        'A103': 'Guarantor'
    },
    'property': {
        'A121': 'Real estate',
        'A122': 'Savings/life insurance',
        'A123': 'Car or other assets',
        'A124': 'No major property'
    },
    'other_installment_plans': {
        'A141': 'Bank',
        'A142': 'Store',
        'A143': 'None'
    },
    'housing': {
        'A151': 'Rent',
        'A152': 'Own home',
        'A153': 'Living for free'
    },
    'job': {
        'A171': 'Unskilled (non-resident)',
        'A172': 'Unskilled (resident)',
        'A173': 'Skilled employee',
        'A174': 'Management/self-employed/highly skilled'
    },
    'telephone': {
        'A191': 'No registered telephone',
        'A192': 'Has registered telephone'
    },
    'foreign_worker': {
        'A201': 'Yes',
        'A202': 'No'
    }
}

def friendly_name(feature):
    return FRIENDLY_NAMES.get(feature, feature.replace('_', ' ').title())

def user_value_text(feature, user_df_scaled, artifacts):
    """Returns a readable user value (decoded for categories, raw for numerics)."""
    num_cols = set(artifacts['numerical_cols'])
    cat_cols = set(artifacts['categorical_cols'])
    val = user_df_scaled.iloc[0][feature]

    if feature in num_cols:
        scaler = artifacts['scaler']
        idx = list(artifacts['numerical_cols']).index(feature)
        raw = (float(val) * float(scaler.scale_[idx])) + float(scaler.mean_[idx])
        if feature in {'duration_months', 'existing_credits', 'num_dependents', 'age', 'installment_rate', 'residence_duration'}:
            return str(int(round(raw)))
        return f"{raw:.0f}"

    if feature in cat_cols:
        try:
            code = int(round(float(val)))
            classes = artifacts['label_encoders'][feature].classes_
            if 0 <= code < len(classes):
                return str(classes[code])
        except (ValueError, KeyError, TypeError):
            pass

    return str(val)

def suggest_improved_profile(user_df_scaled, artifacts):
    """
    Creates a simple low-risk profile suggestion by adjusting common high-risk drivers.
    This is a guidance simulation, not a guaranteed approval policy.
    """
    improved = user_df_scaled.copy()

    # Work on scaled numerical columns where we can safely reduce risk.
    num_cols = set(artifacts['numerical_cols'])

    # Shorter duration and lower amount generally reduce risk.
    if 'duration_months' in num_cols:
        improved['duration_months'] = np.minimum(improved['duration_months'].values, 0.0)
    if 'credit_amount' in num_cols:
        improved['credit_amount'] = np.minimum(improved['credit_amount'].values, 0.0)
    if 'installment_rate' in num_cols:
        improved['installment_rate'] = np.minimum(improved['installment_rate'].values, 0.0)
    if 'existing_credits' in num_cols:
        improved['existing_credits'] = np.minimum(improved['existing_credits'].values, 0.0)

    # Choose better encoded categories when known by label ordering.
    # These are heuristic choices based on dataset coding and should be treated as suggestions.
    better_values = {
        'checking_status': 2,      # A13
        'savings_status': 3,       # A64
        'employment_duration': 4,  # A75
        'housing': 1,              # A152
        'telephone': 1             # A192
    }

    for col, val in better_values.items():
        if col in improved.columns:
            improved[col] = val

    return improved

def get_user_input(artifacts):
    """Prompts the user to enter data for each required feature interactively."""
    feature_names = artifacts['feature_names']
    num_cols = artifacts['numerical_cols']
    cat_cols = artifacts['categorical_cols']
    label_encoders = artifacts['label_encoders']
    
    user_data = {}
    
    print("Please answer the following questions for your loan application.")
    print("For multiple-choice questions, enter the option number.")
    print("-" * 50)
    
    for col in feature_names:
        question = friendly_name(col)
        if col in num_cols:
            hint = NUMERIC_HINTS.get(col, "Enter a number.")
            print(f"\n{question}")
            print(f"  {hint}")
            while True:
                try:
                    val = input("  Your answer: ")
                    val_float = float(val)
                    if val_float < 0:
                        print("  Please enter a non-negative value.")
                        continue
                    user_data[col] = [val_float]
                    break
                except ValueError:
                    print("  Invalid input. Please enter a valid number.")
        elif col in cat_cols:
            le = label_encoders[col]
            classes = le.classes_
            descriptions = CATEGORY_DESCRIPTIONS.get(col, {})
            print(f"\n{question}")
            print("  Choose one option:")
            for i, cls in enumerate(classes):
                desc = descriptions.get(cls, cls)
                print(f"  {i}: {desc}")
            
            while True:
                try:
                    idx_str = input(f"  Enter choice (0-{len(classes)-1}): ")
                    idx = int(idx_str)
                    if 0 <= idx < len(classes):
                        # Store the encoded integer value directly
                        user_data[col] = [idx]
                        break
                    else:
                        print("  Choice out of range.")
                except ValueError:
                    print("  Invalid input. Please enter a whole number.")
    
    df = pd.DataFrame(user_data)
    
    # The categorical inputs are already encoded interactively.
    # We just need to scale the numerical ones.
    scaler = artifacts['scaler']
    df[num_cols] = scaler.transform(df[num_cols])
    return df

def main():
    if not os.path.exists(PROCESSED_DATA_PATH) or not os.path.exists(CREDIT_MODEL_PATH):
        print("Model or processed data not found. Please run the training pipeline first.")
        return
        
    # Load artifacts
    model = joblib.load(CREDIT_MODEL_PATH)
    artifacts = joblib.load(PROCESSED_DATA_PATH)
    
    print("\n" + "="*60)
    print(" Welcome to CrediSure Interactive Credit Scorer")
    print("="*60 + "\n")
    
    # 1. Get input
    user_df_scaled = get_user_input(artifacts)
    
    # 2. Predict
    prob = model.predict_proba(user_df_scaled)[0]
    risk_score = prob[1] * 100  # Probability of Bad Credit
    
    print("\n" + "="*60)
    print("APPLICATION RESULT:")
    print(f"Risk Probability (Risk Score): {risk_score:.1f}%")
    
    # 3. Explain
    if risk_score > 50:
        print("Loan Status: REJECTED\n")
        print(f"Reason: Your risk score is above the approval threshold (50%).")
        print(f"You are above threshold by: {risk_score - 50:.1f} percentage points.\n")

        if not hasattr(model, 'coef_'):
            print("We cannot show detailed feature contributions for this model type.")
            print(f"Advice: {DEFAULT_ADVICE}")
            print("="*60)
            return

        coefs = model.coef_[0]
        feature_impact = coefs * user_df_scaled.iloc[0].values
        
        impact_df = pd.DataFrame({
            'Feature': artifacts['feature_names'],
            'Impact': feature_impact
        })
        
        # Top 3 factors pushing risk upward.
        top_risk_factors = impact_df[impact_df['Impact'] > 0].sort_values(by='Impact', ascending=False).head(3)
        if top_risk_factors.empty:
            top_risk_factors = impact_df.sort_values(by='Impact', ascending=False).head(3)
        
        print("Main Reasons for Rejection:")
        advice_list = []
        for i, (_, row) in enumerate(top_risk_factors.iterrows(), 1):
            feature = row['Feature']
            reason = friendly_name(feature)
            user_val = user_value_text(feature, user_df_scaled, artifacts)
            print(f"{i}. {reason}: your input = {user_val} (this increased your risk).")
            
            advice = ADVICE_DICT.get(feature, DEFAULT_ADVICE)
            advice_list.append(advice)
            
        print("\nActionable Advice to Improve Credit Score:")
        for i, advice in enumerate(advice_list, 1):
            print(f"{i}. {advice}")

        print("\nWould you like to see a suggested profile that may improve approval chances?")
        choice = input("Type 'yes' to simulate improved inputs, or press Enter to skip: ").strip().lower()
        if choice in {"y", "yes"}:
            improved_df_scaled = suggest_improved_profile(user_df_scaled, artifacts)
            improved_prob = model.predict_proba(improved_df_scaled)[0]
            improved_risk_score = improved_prob[1] * 100

            print("\nSIMULATED IMPROVEMENT RESULT:")
            print(f"New Risk Probability: {improved_risk_score:.1f}%")
            if improved_risk_score <= 50:
                print("Status with suggested improvements: LIKELY APPROVED")
            else:
                print("Status with suggested improvements: STILL HIGH RISK")
                print("Try reducing loan amount/duration further and improving savings/employment stability.")
    else:
        print("Loan Status: APPROVED")
        print("Congratulations! Your application meets the credit standards.")
    print("="*60)

if __name__ == "__main__":
    main()
