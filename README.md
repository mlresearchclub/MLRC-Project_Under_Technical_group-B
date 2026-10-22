# Irrigation Event Anomaly Detection

Classifies an irrigation or fertigation event as NORMAL or ANOMALY.

## Folders
- `data/`, `outputs/`, `src/event_labeling/`: event labels and the shared training file (`outputs/training_data.csv`, with fixed `split` and `fold` columns)
- `docs/`: label rules, training data guide, and the team contract
- `models/logistic_regression/`: Logistic Regression results, report and saved models
- `models/decision_tree/`: Decision Tree results and saved models
- `app.py`: Streamlit app (uses the Decision Tree, Set A)

## Important
- Labels are rule-derived, not verified faults (see `docs/` for the rules).
- Set A includes the rule columns, so its scores mostly show the rules being recovered. Set B is the honest test.
- Report accuracy-type metrics only with precision, recall, F1 and the confusion matrix.

## Run the app
1. `pip install -r requirements.txt`
2. `streamlit run app.py`
