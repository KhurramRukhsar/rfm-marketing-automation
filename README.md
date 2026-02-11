# Efficient Version 2

This folder contains a fresh modular refactor of `BaKhabar_SMS_Production_13.py`.

## Run
```powershell
streamlit run app.py
```

## Bundled Files
- `Strategy.xlsx`
- `your_cleaned_product_data.csv`
- `urdu_sms_examples_by_segment.txt`
- `xgb_model.pkl`
- `user_encoder.pkl`
- `product_encoder.pkl`

## Modules
- `app.py` — Streamlit UI and orchestration
- `config.py` — configuration and logging
- `examples_loader.py` — load Urdu examples
- `recommendations.py` — recommendation engine
- `templates.py` — LLM prompt/template generation
- `db.py` — PostgreSQL and shortlink helpers
- `sms.py` — SMS sending and DB logging
- `utils.py` — shared helpers (phone cleaning, product formatting)
