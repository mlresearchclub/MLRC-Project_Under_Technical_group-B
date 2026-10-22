# Event label rules and results (Module 1)

This file sits next to `docs/target_definition.md`, which is the team contract. It does not
replace it. It records the rules we ran, the counts, and one conflict with the contract that the
team has to decide (section 5).

## 1. What was labelled

All 789 events in `data/final_irrigation_events.csv`. `event_id` is the event's row number
(1 to 789) in that file. Labels are in `data/event_labels.csv`, in the format of
`data/event_labels_template.csv`: `event_id, event_label, evidence, label_source, review_status,
reviewer_notes`. A longer version with the rule name and all event columns is in
`outputs/event_labels.csv`. Re-create both with `python src/event_labeling/label_events.py`.

Sensor faults (-20, missing moisture, stuck pH) are not used. They stay a separate finding.

## 2. The rules

Checked first, UNKNOWN (the evidence is unusable):

| Rule | Condition | label_source | Events |
|---|---|---|---|
| U1 | Initial or final humidity outside 0 to 100 (for example 102, 116, 717) | documented_rule (soil moisture is a 0 to 100 scale) | 36 |
| U2 | Type is `standard_irrigation` but stop reason is `fertigation_timeout` | unresolved | 69 |

ANOMALY (an event is flagged when one rule triggers; all `data_derived_rule`):

| Rule | Condition | Basis for the limit | Events |
|---|---|---|---|
| A1 | Fertigation ran longer than 0.5 h | Q1, median and Q3 of fertigation are all 0.25 h; 0.5 h is twice that | 9 |
| A2 | Standard irrigation ran longer than 6.355 h | Upper fence: Q3 (2.80) + 1.5 x IQR (2.37) | 1 |
| A3 | Standard irrigation of 1 h or more where moisture fell by more than 10 points | Usual change is 0 to +5 points | 9 |

NORMAL: valid readings and none of A1 to A3 triggered (`data_derived_rule`). This means "no
anomaly rule triggered". It does not prove the event was healthy.

## 3. Results

| Label | Events |
|---|---|
| NORMAL | 665 |
| ANOMALY | 19 |
| UNKNOWN | 105 |

All 684 NORMAL and ANOMALY events are in `outputs/training_data.csv`. Only 19 (2.8%) are anomalies.

## 4. Documented rules we checked and did not use

The dataset README (section 5.1) gives a start and stop threshold for each sector and an
irrigation window of 06:00 to 18:00. `src/event_labeling/check_documented_rules.py` checks them:

- Window: 0 events start before 06:00 or end after 18:00.
- Start threshold, sectors 1 to 4: 0 standard irrigations start above the stop threshold.
- Start threshold, sector 5: 42 of 51 standard irrigations (82%) start above the stop threshold.
  The README describes sector 5 as a different line (blueberry, 0.4 L/min), so its thresholds do
  not describe how it is run. Flagging them would label most of the sector as anomalous.

The documented rules either never fire or fire on the normal pattern, so they give no anomaly labels.

## 5. Conflict with the team contract (needs a group decision)

`docs/target_definition.md` says: "Do not invent operational thresholds. Use only limits
documented in project materials or confirmed by a domain reviewer," and lists the allowed
label sources as `documented_rule`, `operator_confirmed` and `unresolved`.

Rules A1 to A3 use limits we worked out from the data. They do not meet that wording, so their
labels are marked `data_derived_rule`, which is not one of the allowed sources.

- Anomalies from `documented_rule` or `operator_confirmed`: **0**. Under the contract as written,
  there is nothing to train an anomaly class on, and the event classifier is "model not ready".
- The brief asks for a trained classifier with accuracy, precision, recall, F1 and a confusion
  matrix. That needs both classes.

Two further weaknesses the group should know about:

- The dataset README (5.1) says long events that ended by `fertigation_timeout` or
  `humidity_threshold` were kept as genuine. 6 of our 10 duration-based anomalies (A1, A2) ended
  that way.
- All 9 A3 anomalies, and 4 of the A1 anomalies, ended by `physical_valve_closure`. The README
  says valve states can be inferred by the platform, so these may be artefacts of how the events
  were reconstructed and not real faults.

Options for the group:

1. **Amend the contract** to add a fourth label source, `data_derived_rule`, and require the
   report to say that the labels are rule-derived and unreviewed. This lets the models be trained
   and compared. It is the only route that meets the brief's evaluation requirements.
2. **Keep the contract as written.** Report the documented-rule result (0 anomalies), present
   the sensor detector as the validated work, and mark the event classifier "model not ready".

Whichever is chosen, the report must say that the labels are not verified faults.

## 6. Other weaknesses

- **Leakage.** A1 and A2 use `duration_hours`; A3 uses `initial_humidity`, `final_humidity` and
  duration. A model that sees those columns can partly copy the rule. See Set A and Set B in
  `docs/training_data_guide.md`.
- **Thresholds are choices.** For A3, a limit of -10 points matches 11 runs of 1 h or more
  before the other rules take priority, and a limit of -5 matches 26. The final A3 count is 9.
- **Few anomalies.** 19 examples is small. The test set has 5.
- **U2 is a guess about meaning.** We do not know why 69 standard events stopped by fertigation
  timeout, so we leave them out instead of labelling them.
