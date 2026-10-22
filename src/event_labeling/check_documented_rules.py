"""Module 1 - check the events against the operating rules documented in the dataset README.

Run from the project root (after label_events.py):
    python src/event_labeling/check_documented_rules.py

README section 5.1 documents, per sector, a start threshold (irrigate below it) and a stop
threshold (stop above it), and an irrigation window of 06:00 to 18:00. This script counts how
many events break them. It writes nothing; it only prints.
"""

import csv

START = {"1": 40, "2": 40, "3": 60, "4": 80, "5": 50}
STOP = {"1": 60, "2": 60, "3": 80, "4": 90, "5": 60}

rows = []
file = open("outputs/event_labels.csv", newline="")
for row in csv.DictReader(file):
    rows.append(row)
file.close()

early = 0
late = 0
for row in rows:
    start_hour = int(row["start"][11:13])
    end_hour = int(row["end"][11:13])
    end_minute = int(row["end"][14:16])
    if start_hour < 6:
        early += 1
    if end_hour > 18 or (end_hour == 18 and end_minute > 0):
        late += 1
print("Window 06:00-18:00 -> starts before 06:00: %d, ends after 18:00: %d" % (early, late))

print("")
print("Standard irrigation, start moisture against the documented thresholds (valid humidity only)")
print("sector  events  below_start  in_band  above_stop")
table = {}
for row in rows:
    if row["type"] != "standard_irrigation":
        continue
    start_moisture = float(row["initial_humidity"])
    if start_moisture < 0 or start_moisture > 100:
        continue
    sector = row["sector"]
    counts = table.setdefault(sector, [0, 0, 0, 0])
    counts[0] += 1
    if start_moisture < START[sector]:
        counts[1] += 1
    elif start_moisture <= STOP[sector]:
        counts[2] += 1
    else:
        counts[3] += 1
for sector in sorted(table):
    c = table[sector]
    print("%-7s %-7d %-12d %-8d %d" % (sector, c[0], c[1], c[2], c[3]))
