# Contributing

## Setup

```bash
git clone https://github.com/kairav7220/customer-churn-predictor.git
cd customer-churn-predictor
pip install -r requirements.txt
```

## Development

- Fork the repo, create a feature branch.
- Test with `python app.py` and open `http://127.0.0.1:5000`
- Ensure model encoding changes are tested against `tel_churn.csv`

## PR Guidelines

- One feature/fix per PR.
- If changing the encoding logic, verify against the API endpoint.
