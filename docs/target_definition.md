# Irrigation-event target definition (team contract)

## Prediction target
The primary target is the **irrigation event**, not an individual sensor reading.

Allowed labels:
- `NORMAL`: available event evidence is consistent with documented operating conditions and there is no known operational fault.
- `ANOMALY`: the event has a documented operational fault or violates a justified, documented operating limit.
- `UNKNOWN`: available evidence is insufficient, contradictory, or cannot support a defensible binary label.

`UNKNOWN` is not a model class by default; exclude it from supervised training and report its count. The app may return `UNKNOWN` when required fields are absent, no trained model exists, or the model cannot support a reliable decision.

## Sensor-quality flags are separate
Missing moisture, sentinel `-20`, stuck pH, or a possible flatline are sensor-quality observations. They may trigger review, but **do not automatically label the irrigation event `ANOMALY`**. A sensor fault and an irrigation-process fault are different claims.

## Label provenance
Every reviewed event label must record:
- event identifier matching the source event row;
- label (`NORMAL`, `ANOMALY`, or `UNKNOWN`);
- evidence/criterion used;
- source (`documented_rule`, `operator_confirmed`, or `unresolved`);
- reviewer and review status if available.

Do not invent operational thresholds. Use only limits documented in project materials or confirmed by a domain reviewer. If no evidence supports a label, use `UNKNOWN`.

## Current data limitation
`data/final_irrigation_events.csv` contains 789 historical event records and does not include a verified `NORMAL`/`ANOMALY` target. A classifier must not be presented as validated until defensible labels for both classes exist and evaluation on held-out labelled events is completed. The existing sensor-level detector's metrics are not evidence that event-level irrigation anomalies can be detected.
