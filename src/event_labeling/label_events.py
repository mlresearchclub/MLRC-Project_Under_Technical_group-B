"""Module 1 - rule-derived labels for the 789 historical irrigation events.

Run from the project root:   python src/event_labeling/label_events.py

Reads  : data/final_irrigation_events.csv
Writes : outputs/event_labels.csv      (every event: label, rule, evidence, label_source ...)
         data/event_labels.csv         (same labels in the team template format:
                                        event_id, event_label, evidence, label_source,
                                        review_status, reviewer_notes)
         outputs/event_label_summary.txt

event_id is the row number (1 to 789) of the event in data/final_irrigation_events.csv.

IMPORTANT: these labels are RULE-DERIVED, not verified faults. The rules and
the numbers behind them are explained in docs/target_definition.md.
Plain Python only (csv module for reading); no pandas / numpy.
"""

import csv

INPUT_FILE = "data/final_irrigation_events.csv"
OUTPUT_FILE = "outputs/event_labels.csv"
TEMPLATE_FILE = "data/event_labels.csv"
SUMMARY_FILE = "outputs/event_label_summary.txt"

# ---------------------------------------------------------------- thresholds
FERT_NORMAL_HOURS = 0.25      # fertigation timeout: 15 minutes (median, Q1 and Q3 are all 0.25)
FERT_LIMIT_HOURS = 0.5        # A1: more than twice the usual timeout
LONG_STANDARD_MIN_HOURS = 1.0  # A3: only judge moisture response on runs of an hour or more
MOISTURE_DROP_LIMIT = -10      # A3: moisture fell by more than 10 points while irrigating


def percentile(values, p):
    """Linear-interpolated percentile (p between 0 and 1) with plain logic."""
    ordered = sorted(values)
    n = len(ordered)
    position = (n - 1) * p
    low = int(position)
    high = low + 1
    if high >= n:
        return ordered[low]
    fraction = position - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def humidity_is_valid(value):
    return 0 <= value <= 100


def main():
    # ------------------------------------------------------------ read events
    rows = []
    file = open(INPUT_FILE, newline="")
    for row in csv.DictReader(file):
        rows.append(row)
    file.close()

    # A2 needs a data-driven limit: Tukey upper fence on all standard runs
    standard_durations = []
    for row in rows:
        if row["type"] == "standard_irrigation":
            standard_durations.append(float(row["duration_hours"]))
    q1 = percentile(standard_durations, 0.25)
    q3 = percentile(standard_durations, 0.75)
    long_standard_limit = q3 + 1.5 * (q3 - q1)

    # ------------------------------------------------------------ label rows
    labelled = []
    source_counts = {}
    counts = {"NORMAL": 0, "ANOMALY": 0, "UNKNOWN": 0}
    rule_counts = {}

    for index in range(len(rows)):
        row = rows[index]
        kind = row["type"]
        reason = row["stop_reason"]
        hours = float(row["duration_hours"])
        start_h = float(row["initial_humidity"])
        end_h = float(row["final_humidity"])
        change = end_h - start_h

        label = "NORMAL"
        source = "data_derived_rule"
        rule = "none triggered"
        evidence = "valid readings; no anomaly criterion triggered"

        # --- UNKNOWN first: if the evidence itself is bad, we do not judge
        if not humidity_is_valid(start_h) or not humidity_is_valid(end_h):
            label = "UNKNOWN"
            source = "documented_rule"      # soil moisture is documented as a 0-100 scale
            rule = "U1 invalid humidity"
            evidence = "humidity %s -> %s is outside 0-100 (sensor problem)" % (start_h, end_h)
        elif kind == "standard_irrigation" and reason == "fertigation_timeout":
            label = "UNKNOWN"
            source = "unresolved"
            rule = "U2 type/stop-reason conflict"
            evidence = "event type is standard_irrigation but it stopped by fertigation_timeout"

        # --- ANOMALY criteria (only if the data is usable)
        elif kind == "fertigation" and hours > FERT_LIMIT_HOURS:
            label = "ANOMALY"
            rule = "A1 long fertigation"
            evidence = "fertigation ran %.2f h; usual is %.2f h (limit %.2f h)" % (
                hours, FERT_NORMAL_HOURS, FERT_LIMIT_HOURS)
        elif kind == "standard_irrigation" and hours > long_standard_limit:
            label = "ANOMALY"
            rule = "A2 long standard irrigation"
            evidence = "ran %.2f h; limit is %.2f h (Q3 %.2f + 1.5 x IQR)" % (
                hours, long_standard_limit, q3)
        elif (kind == "standard_irrigation" and hours >= LONG_STANDARD_MIN_HOURS
              and change < MOISTURE_DROP_LIMIT):
            label = "ANOMALY"
            rule = "A3 moisture fell during irrigation"
            evidence = "ran %.2f h but moisture changed %+d points (%s -> %s)" % (
                hours, change, start_h, end_h)

        counts[label] += 1
        rule_counts[rule] = rule_counts.get(rule, 0) + 1

        if label == "NORMAL":
            event_label = "0"
        elif label == "ANOMALY":
            event_label = "1"
        else:
            event_label = ""      # UNKNOWN stays blank: never forced into a class

        new_row = {"event_id": index + 1}
        new_row.update(row)
        new_row["event_label"] = event_label
        new_row["label_name"] = label
        new_row["rule"] = rule
        new_row["evidence"] = evidence
        new_row["label_source"] = source
        new_row["review_status"] = "proposed_unreviewed"
        new_row["reviewer_notes"] = ""
        source_counts[(label, source)] = source_counts.get((label, source), 0) + 1
        labelled.append(new_row)

    # ------------------------------------------------------------ write files
    fieldnames = ["event_id"] + list(rows[0].keys()) + [
        "event_label", "label_name", "rule", "evidence",
        "label_source", "review_status", "reviewer_notes"]
    out = open(OUTPUT_FILE, "w", newline="")
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    for new_row in labelled:
        writer.writerow(new_row)
    out.close()

    # same labels in the exact format of data/event_labels_template.csv (label names, not 0/1)
    template_columns = ["event_id", "event_label", "evidence", "label_source",
                        "review_status", "reviewer_notes"]
    template_out = open(TEMPLATE_FILE, "w", newline="")
    template_writer = csv.DictWriter(template_out, fieldnames=template_columns)
    template_writer.writeheader()
    for new_row in labelled:
        template_writer.writerow({
            "event_id": new_row["event_id"],
            "event_label": new_row["label_name"],
            "evidence": new_row["evidence"],
            "label_source": new_row["label_source"],
            "review_status": new_row["review_status"],
            "reviewer_notes": new_row["reviewer_notes"]})
    template_out.close()

    lines = []
    lines.append("Events read: %d" % len(rows))
    lines.append("Standard-irrigation long-run limit (A2): %.3f h  (Q1 %.2f, Q3 %.2f)" % (
        long_standard_limit, q1, q3))
    lines.append("")
    lines.append("NORMAL  (event_label 0): %d" % counts["NORMAL"])
    lines.append("ANOMALY (event_label 1): %d" % counts["ANOMALY"])
    lines.append("UNKNOWN (blank)        : %d" % counts["UNKNOWN"])
    lines.append("")
    lines.append("By rule:")
    for rule in sorted(rule_counts):
        lines.append("  %-36s %d" % (rule, rule_counts[rule]))
    lines.append("")
    lines.append("By label and label_source:")
    for key in sorted(source_counts):
        lines.append("  %-8s %-20s %d" % (key[0], key[1], source_counts[key]))
    lines.append("")
    lines.append("Anomalies from documented_rule or operator_confirmed sources: %d" % (
        source_counts.get(("ANOMALY", "documented_rule"), 0)
        + source_counts.get(("ANOMALY", "operator_confirmed"), 0)))
    lines.append("Labels are rule-derived, not independently verified.")
    text = "\n".join(lines)
    print(text)

    summary = open(SUMMARY_FILE, "w")
    summary.write(text + "\n")
    summary.close()


main()
