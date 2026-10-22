"""Module 1 - build the shared, model-ready training file.

Run from the project root (after label_events.py):
    python src/event_labeling/make_training_data.py

Reads  : outputs/event_labels.csv
Writes : outputs/training_data.csv

Only NORMAL (0) and ANOMALY (1) events are kept. UNKNOWN events are left out.
Every number column is numeric, so KNN, Decision Tree and Logistic Regression can
all read the file directly. Plain Python only (csv + random from the standard library).
"""

import csv
import random

INPUT_FILE = "outputs/event_labels.csv"
WEATHER_FILE = "data/zone_1_features_final.csv"   # zone-1 table; the weather columns are identical in all zones
OUTPUT_FILE = "outputs/training_data.csv"

# Weather values are hourly amounts interpolated onto a 10-minute grid, so rain in the
# previous 24 h = (sum of the 144 bins before the start) / 6.
BINS_PER_DAY = 144
BINS_PER_HOUR = 6
# Nominal valve flow (L/min) used by the dataset itself for every event type
FLOW_L_PER_MIN = {"1": 1.1, "2": 1.1, "3": 1.1, "4": 1.1, "5": 0.4}

SEED = 42            # fixed so everyone gets the identical split
TEST_FRACTION = 0.25  # share of each class held back for the final test
N_FOLDS = 5          # cross-validation folds, made only inside the training rows

SECTORS = ["1", "2", "3", "4", "5"]
STOP_REASONS = ["fertigation_timeout", "physical_valve_closure",
                "time_restriction", "humidity_threshold"]


def load_weather():
    """Return (list of (timestamp, temp, rain), dict timestamp -> position)."""
    grid = []
    position_of = {}
    file = open(WEATHER_FILE, newline="")
    for row in csv.DictReader(file):
        position_of[row["ts"]] = len(grid)
        grid.append((row["ts"], float(row["weather_temp"]), float(row["weather_rain"])))
    file.close()
    return grid, position_of


def weather_key(start):
    """'22-02-2025 06:01' -> '2025-02-22 06:00:00' (start of its 10-minute bin)."""
    day = start[0:2]
    month = start[3:5]
    year = start[6:10]
    hour = start[11:13]
    minute = int(start[14:16]) // 10 * 10
    return "%s-%s-%s %s:%02d:00" % (year, month, day, hour, minute)


def main():
    grid, position_of = load_weather()

    # ----------------------------------------------------------- read labelled rows
    kept = []
    file = open(INPUT_FILE, newline="")
    for row in csv.DictReader(file):
        if row["event_label"] == "0" or row["event_label"] == "1":
            kept.append(row)
    file.close()

    # ----------------------------------------------------------- build features
    table = []
    for position in range(len(kept)):
        row = kept[position]
        start = row["start"]               # format dd-mm-yyyy hh:mm
        start_hour = int(start[11:13])
        start_month = int(start[3:5])

        initial = float(row["initial_humidity"])
        final = float(row["final_humidity"])

        # weather at the start of the event
        grid_position = position_of[weather_key(start)]     # KeyError means a date is not covered
        temperature = grid[grid_position][1]
        rain_total = 0.0
        for k in range(grid_position - BINS_PER_DAY, grid_position):
            rain_total += grid[k][2]
        rain_24h = rain_total / BINS_PER_HOUR

        hours = float(row["duration_hours"])
        volume = hours * 60 * FLOW_L_PER_MIN[row["sector"]]

        new = {}
        new["event_id"] = int(row["event_id"])   # row number in data/final_irrigation_events.csv
        new["start"] = start               # kept for traceability, not a feature
        for s in SECTORS:
            new["sector_" + s] = 1 if row["sector"] == s else 0
        new["is_fertigation"] = 1 if row["type"] == "fertigation" else 0
        new["start_hour"] = start_hour
        new["start_month"] = start_month
        new["initial_humidity"] = initial
        new["duration_hours"] = float(row["duration_hours"])
        new["final_humidity"] = final
        new["humidity_change"] = final - initial
        new["water_volume_l"] = round(volume, 2)
        new["temperature_c"] = round(temperature, 2)
        new["rain_prev_24h_mm"] = round(rain_24h, 2)
        for reason in STOP_REASONS:
            new["stop_" + reason] = 1 if row["stop_reason"] == reason else 0
        new["event_label"] = int(row["event_label"])
        table.append(new)

    # ----------------------------------------------------------- stratified split
    # Shuffle each class on its own so both classes appear in train and test.
    rng = random.Random(SEED)
    for label in (0, 1):
        members = []
        for new in table:
            if new["event_label"] == label:
                members.append(new)
        rng.shuffle(members)

        n_test = int(round(len(members) * TEST_FRACTION))
        train_rows = []
        for i in range(len(members)):
            if i < n_test:
                members[i]["split"] = "test"
                members[i]["fold"] = 0
            else:
                members[i]["split"] = "train"
                train_rows.append(members[i])

        # deal training rows into folds one by one (1,2,3,4,5,1,2,...)
        for i in range(len(train_rows)):
            train_rows[i]["fold"] = (i % N_FOLDS) + 1

    # ----------------------------------------------------------- write file
    columns = ["event_id", "start",
               "sector_1", "sector_2", "sector_3", "sector_4", "sector_5",
               "is_fertigation", "start_hour", "start_month",
               "initial_humidity", "duration_hours", "final_humidity", "humidity_change",
               "water_volume_l", "temperature_c", "rain_prev_24h_mm",
               "stop_fertigation_timeout", "stop_physical_valve_closure",
               "stop_time_restriction", "stop_humidity_threshold",
               "event_label", "split", "fold"]
    out = open(OUTPUT_FILE, "w", newline="")
    writer = csv.DictWriter(out, fieldnames=columns)
    writer.writeheader()
    for new in table:
        writer.writerow(new)
    out.close()

    # ----------------------------------------------------------- report counts
    counts = {}
    for new in table:
        key = (new["split"], new["event_label"])
        counts[key] = counts.get(key, 0) + 1
    print("Rows written:", len(table))
    print("train  normal(0): %d   anomaly(1): %d" % (counts.get(("train", 0), 0), counts.get(("train", 1), 0)))
    print("test   normal(0): %d   anomaly(1): %d" % (counts.get(("test", 0), 0), counts.get(("test", 1), 0)))
    print("anomalies per CV fold (train only):")
    for fold in range(1, N_FOLDS + 1):
        n = 0
        for new in table:
            if new["split"] == "train" and new["fold"] == fold and new["event_label"] == 1:
                n += 1
        print("  fold %d: %d" % (fold, n))


main()
