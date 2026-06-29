import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
import pickle
import shap

app = Flask(__name__)

model = pickle.load(open('model.sav', 'rb'))
explainer = shap.TreeExplainer(model)

train_df = pd.read_csv('tel_churn.csv')
FEATURE_COLUMNS = [col for col in train_df.columns
                   if col not in ('Churn',) and 'Unnamed' not in col]

THRESHOLD = 0.35

FEATURE_DISPLAY = {
    'SeniorCitizen': 'Senior Citizen',
    'MonthlyCharges': 'Monthly Charges',
    'TotalCharges': 'Total Charges',
    'gender_Female': 'Gender (Female)',
    'gender_Male': 'Gender (Male)',
    'Partner_No': 'No Partner',
    'Partner_Yes': 'Has Partner',
    'Dependents_No': 'No Dependents',
    'Dependents_Yes': 'Has Dependents',
    'PhoneService_No': 'No Phone Service',
    'PhoneService_Yes': 'Has Phone Service',
    'MultipleLines_No': 'No Multiple Lines',
    'MultipleLines_No phone service': 'No Phone (Multiple Lines)',
    'MultipleLines_Yes': 'Has Multiple Lines',
    'InternetService_DSL': 'DSL Internet',
    'InternetService_Fiber optic': 'Fiber Optic Internet',
    'InternetService_No': 'No Internet Service',
    'OnlineSecurity_No': 'No Online Security',
    'OnlineSecurity_No internet service': 'No Internet (Online Security)',
    'OnlineSecurity_Yes': 'Has Online Security',
    'OnlineBackup_No': 'No Online Backup',
    'OnlineBackup_No internet service': 'No Internet (Online Backup)',
    'OnlineBackup_Yes': 'Has Online Backup',
    'DeviceProtection_No': 'No Device Protection',
    'DeviceProtection_No internet service': 'No Internet (Device Protection)',
    'DeviceProtection_Yes': 'Has Device Protection',
    'TechSupport_No': 'No Tech Support',
    'TechSupport_No internet service': 'No Internet (Tech Support)',
    'TechSupport_Yes': 'Has Tech Support',
    'StreamingTV_No': 'No Streaming TV',
    'StreamingTV_No internet service': 'No Internet (Streaming TV)',
    'StreamingTV_Yes': 'Has Streaming TV',
    'StreamingMovies_No': 'No Streaming Movies',
    'StreamingMovies_No internet service': 'No Internet (Streaming Movies)',
    'StreamingMovies_Yes': 'Has Streaming Movies',
    'Contract_Month-to-month': 'Month-to-month Contract',
    'Contract_One year': 'One-year Contract',
    'Contract_Two year': 'Two-year Contract',
    'PaperlessBilling_No': 'Paper Billing',
    'PaperlessBilling_Yes': 'Paperless Billing',
    'PaymentMethod_Bank transfer (automatic)': 'Auto Bank Transfer',
    'PaymentMethod_Credit card (automatic)': 'Auto Credit Card',
    'PaymentMethod_Electronic check': 'Electronic Check',
    'PaymentMethod_Mailed check': 'Mailed Check',
    'tenure_group_1 - 12': 'Tenure 1–12 months',
    'tenure_group_13 - 24': 'Tenure 13–24 months',
    'tenure_group_25 - 36': 'Tenure 25–36 months',
    'tenure_group_37 - 48': 'Tenure 37–48 months',
    'tenure_group_49 - 60': 'Tenure 49–60 months',
    'tenure_group_61 - 72': 'Tenure 61–72 months',
}


def display_name(col):
    return FEATURE_DISPLAY.get(col, col.replace('_', ' '))


def fmt_num(val):
    if val == int(val):
        return str(int(val))
    s = f'{val:.2f}'.rstrip('0').rstrip('.')
    return s


def fmt_pct(val):
    s = f'{val:.2f}'
    if s.endswith('.00'):
        return s[:-3]
    if s.endswith('0'):
        return s[:-1]
    return s


def encode_input(senior_citizen, monthly_charges, total_charges, gender, partner,
                 dependents, phone_service, multiple_lines, internet_service,
                 online_security, online_backup, device_protection, tech_support,
                 streaming_tv, streaming_movies, contract, paperless_billing,
                 payment_method, tenure):

    labels = ['{0} - {1}'.format(i, i + 11) for i in range(1, 72, 12)]
    tenure_group = pd.cut([int(tenure)], range(1, 80, 12),
                          right=False, labels=labels)[0]

    mappings = {
        'gender': {v: f'gender_{v}' for v in ('Female', 'Male')},
        'Partner': {v: f'Partner_{v}' for v in ('No', 'Yes')},
        'Dependents': {v: f'Dependents_{v}' for v in ('No', 'Yes')},
        'PhoneService': {v: f'PhoneService_{v}' for v in ('No', 'Yes')},
        'MultipleLines': {v: f'MultipleLines_{v}' for v in ('No', 'No phone service', 'Yes')},
        'InternetService': {v: f'InternetService_{v}' for v in ('DSL', 'Fiber optic', 'No')},
        'OnlineSecurity': {v: f'OnlineSecurity_{v}' for v in ('No', 'No internet service', 'Yes')},
        'OnlineBackup': {v: f'OnlineBackup_{v}' for v in ('No', 'No internet service', 'Yes')},
        'DeviceProtection': {v: f'DeviceProtection_{v}' for v in ('No', 'No internet service', 'Yes')},
        'TechSupport': {v: f'TechSupport_{v}' for v in ('No', 'No internet service', 'Yes')},
        'StreamingTV': {v: f'StreamingTV_{v}' for v in ('No', 'No internet service', 'Yes')},
        'StreamingMovies': {v: f'StreamingMovies_{v}' for v in ('No', 'No internet service', 'Yes')},
        'Contract': {v: f'Contract_{v}' for v in ('Month-to-month', 'One year', 'Two year')},
        'PaperlessBilling': {v: f'PaperlessBilling_{v}' for v in ('No', 'Yes')},
        'PaymentMethod': {v: f'PaymentMethod_{v}' for v in
                          ('Bank transfer (automatic)', 'Credit card (automatic)',
                           'Electronic check', 'Mailed check')},
    }

    row = {col: 0 for col in FEATURE_COLUMNS}

    row['SeniorCitizen'] = int(senior_citizen)
    row['MonthlyCharges'] = float(monthly_charges)
    row['TotalCharges'] = float(total_charges)

    raw_values = {
        'gender': gender, 'Partner': partner, 'Dependents': dependents,
        'PhoneService': phone_service, 'MultipleLines': multiple_lines,
        'InternetService': internet_service, 'OnlineSecurity': online_security,
        'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
        'TechSupport': tech_support, 'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies, 'Contract': contract,
        'PaperlessBilling': paperless_billing, 'PaymentMethod': payment_method,
    }

    for feature, raw_value in raw_values.items():
        col_name = mappings[feature].get(raw_value)
        if col_name and col_name in row:
            row[col_name] = 1

    tenure_col = f'tenure_group_{tenure_group}'
    if tenure_col in row:
        row[tenure_col] = 1

    return pd.DataFrame([row])[FEATURE_COLUMNS]


@app.route('/', methods=['GET'])
def loadPage():
    return render_template('home.html')


@app.route('/', methods=['POST'])
def predict():
    try:
        fields = [request.form[f'query{i}'] for i in range(1, 20)]
    except KeyError as e:
        return f'Missing field: {e}', 400

    try:
        monthly_charges = float(fields[1])
        total_charges = float(fields[2])
        tenure = int(fields[18])
    except (ValueError, TypeError) as e:
        return f'Invalid numeric input: {e}', 400

    fields[1] = fmt_num(monthly_charges)
    fields[2] = fmt_num(total_charges)
    fields[18] = str(tenure)

    try:
        df = encode_input(*fields)
    except Exception as e:
        return f'Encoding error: {e}', 400

    prob_churn = model.predict_proba(df)[:, 1][0]

    if prob_churn >= THRESHOLD:
        o1 = 'This customer is likely to churn'
        confidence = prob_churn
    else:
        o1 = 'This customer is likely to stay'
        confidence = 1 - prob_churn

    o2 = 'Confidence: {}%'.format(fmt_pct(confidence * 100))

    shap_values = explainer.shap_values(df)
    shap_vals = shap_values[0, :, 1] if shap_values.ndim == 3 else shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    top3_idx = np.argsort(np.abs(shap_vals))[::-1][:3]
    top3 = [(display_name(FEATURE_COLUMNS[i]), round(float(shap_vals[i]), 4)) for i in top3_idx]

    return render_template('home.html', output1=o1, output2=o2, top3=top3,
                           query1=fields[0], query2=fields[1],
                            query3=fields[2], query4=fields[3],
                            query5=fields[4], query6=fields[5],
                            query7=fields[6], query8=fields[7],
                            query9=fields[8], query10=fields[9],
                            query11=fields[10], query12=fields[11],
                            query13=fields[12], query14=fields[13],
                            query15=fields[14], query16=fields[15],
                            query17=fields[16], query18=fields[17],
                            query19=fields[18])


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
    except Exception:
        return jsonify({'error': 'Invalid JSON'}), 400

    required = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
                'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'Contract', 'PaperlessBilling',
                'PaymentMethod', 'tenure']
    for field in required:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400

    try:
        df = encode_input(
            data['SeniorCitizen'], data['MonthlyCharges'], data['TotalCharges'],
            data['gender'], data['Partner'], data['Dependents'],
            data['PhoneService'], data['MultipleLines'], data['InternetService'],
            data['OnlineSecurity'], data['OnlineBackup'], data['DeviceProtection'],
            data['TechSupport'], data['StreamingTV'], data['StreamingMovies'],
            data['Contract'], data['PaperlessBilling'], data['PaymentMethod'],
            data['tenure'],
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    prob_churn = model.predict_proba(df)[:, 1][0]
    prediction = 'churn' if prob_churn >= THRESHOLD else 'no_churn'
    confidence = float(prob_churn if prediction == 'churn' else 1 - prob_churn)

    shap_values = explainer.shap_values(df)
    shap_vals = shap_values[0, :, 1] if shap_values.ndim == 3 else shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    top3_idx = np.argsort(np.abs(shap_vals))[::-1][:3]
    top3 = [{'feature': display_name(FEATURE_COLUMNS[i]), 'shap': round(float(shap_vals[i]), 4)} for i in top3_idx]

    return jsonify({'prediction': prediction, 'confidence': confidence, 'top_factors': top3})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.errorhandler(404)
def not_found(e):
    return render_template('home.html', output1='Page not found'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('home.html', output1='Server error, please try again'), 500


if __name__ == '__main__':
    app.run(debug=True)
