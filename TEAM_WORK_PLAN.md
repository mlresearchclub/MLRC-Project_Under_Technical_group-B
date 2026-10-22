# Five-person parallel work plan

## Project goal
Build a transparent prototype that classifies an **irrigation event** as `NORMAL`, `ANOMALY`, or `UNKNOWN` when evidence is insufficient. Sensor-quality flags are supporting evidence and **must not automatically become irrigation-event anomaly labels**.

## Work split (one owner per module)

### Person 1 — Target definition and event-label dataset
**Owns:** `docs/target_definition.md`, `data/event_labels_template.csv`, `src/event_labeling/`
- Define the operational criteria for `NORMAL`, `ANOMALY`, and `UNKNOWN` using documented evidence.
- Review the 789 rows in `data/final_irrigation_events.csv`; do not invent labels where the evidence is missing.
- Produce an auditable label file with `event_id`, `event_label`, `evidence`, `label_source`, `review_status`.
- Keep `UNKNOWN` rows out of supervised model training.
- Deliver: label policy, labelled/unknown counts, and examples supporting each label.
**Do not edit:** model training, Streamlit UI, sensor scripts.

### Person 2 — Event-level ML pipeline
**Owns:** `src/event_model/`, `models/` (model artifacts are local unless the team agrees to use Git LFS)
- Build event-level features from `data/final_irrigation_events.csv`.
- Train and save a scikit-learn pipeline only after Person 1 supplies defensible labels from both classes.
- Use a stratified split where feasible; avoid leakage from event IDs/timestamps and report class balance.
- Save the fitted pipeline and a feature/schema manifest. Include a clear fallback when training data are insufficient.
- Deliver: reproducible training script, prediction module with `predict_event(record)`, metrics, and tests.
**Interface contract:** prediction returns `label`, `confidence` (if supported), and `explanation`/reason; `UNKNOWN` is allowed when model/data is not reliable.

### Person 3 — Streamlit application
**Owns:** `app.py`, `app_components/`
- Build a simple UI to enter/upload an irrigation event and show `NORMAL`, `ANOMALY`, or `UNKNOWN`.
- Include the key input fields: initial/final humidity, duration, stop reason, event type, and start/end timestamps when available.
- Call the public interface from Person 2; do not duplicate model logic in the UI.
- Show a warning if the model is not trained or the model artifact is absent; never display a fake prediction.
- Add a CSV upload mode and downloadable results only after the single-event flow works.
- Deliver: local launch instructions and screenshots/demo steps.

### Person 4 — Sensor quality and feature pipeline
**Owns:** `src/01_feature_engineering.py`, `src/02_random_forest_and_rules.py`, `src/03_isolation_forest_chemistry_check.py`, `src/04_final_labeling.py`, related sensor-quality docs/outputs
- Preserve the existing sensor anomaly detection as a separate subsystem.
- Verify the `-20`, missing moisture, pH-stuck and possible-flatline handling against the current data.
- Keep the September chart/labels consistent with the documented limitation: valid moisture observations end on 6 September; 7–30 September contains only missing or `-20` readings.
- Do not relabel an irrigation event as anomalous solely because a sensor is faulty.
- Deliver: reproducible scripts, checks, updated sensor-quality report/chart, and clear separation of sensor flags from event labels.

### Person 5 — Evaluation, integration, documentation and release
**Owns:** `tests/`, `reports/`, `docs/`, `README.md`, `requirements.txt`, integration/release checklist
- Create tests for schema validation, label handling, feature preparation, model-not-available behavior, and end-to-end prediction.
- Document data limitations and distinguish sensor anomaly performance from event-classifier performance.
- Review metrics and ensure no unsupported claims (e.g. detecting subtle irrigation faults from `-20` detection metrics).
- Integrate modules through pull requests; maintain setup/run instructions and final report.
- Deliver: passing tests, updated report, reproducible setup, release checklist, and final demo steps.

## Parallel dependencies and milestones
1. **Day 1 / setup:** all five clone repo, create a branch, read README and target policy. Person 1 finalizes label criteria; Person 4 audits sensor pipeline; Person 3 builds UI layout using the interface contract; Person 5 writes tests/schema docs; Person 2 prepares feature/training skeleton without asserting model validity.
2. **Day 2 / interface freeze:** Person 1 publishes the label schema and first audited labels. Person 2 and Person 3 agree on `predict_event(record)` and model artifact path. Person 5 adds tests for this contract.
3. **Day 3 / integration:** Person 2 trains only if both classes have sufficient defensible labels. Person 3 connects the real predictor. Person 4 finishes sensor quality outputs. Person 5 reviews PRs and checks full run.
4. **Final gate:** do not call the event classifier validated unless labelled data support both classes and evaluation is credible. If not, present it as a prototype and state that more verified event labels are required.

## File ownership rules
- Avoid two people editing the same file at once. Keep each branch focused on the paths listed above.
- Shared contracts live in `docs/target_definition.md` and `docs/model_interface.md`; changes to them require team review.
- Never commit virtual environments, secrets, personal paths, or large generated artifacts without agreement.
- All labels must have an evidence/source note. Do not silently convert `UNKNOWN` to `NORMAL`.
