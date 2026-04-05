import pandas as pd
from pathlib import Path
import os
import json

# PATH CONFIG
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent.parent

RAW_DIR = ROOT_DIR / "data" / "raw_landmarks"
OUTPUT_DIR = ROOT_DIR / "data" / "processed_dataset"
CONFIG_PATH = ROOT_DIR / "config" / "gesture_labels.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# LOAD CONFIG
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

GESTURES = config["gestures"]
HANDS = config["hands"]
LABEL_MAP = config["label_map"]

VALID_CLASSES = set(LABEL_MAP.keys())


# NORMALIZE FUNCTION
def normalize_row(row):
    try:
        xs = [row[f"x{i}"] for i in range(21)]
        ys = [row[f"y{i}"] for i in range(21)]
        zs = [row[f"z{i}"] for i in range(21)]
    except KeyError:
        raise ValueError("Missing landmark columns")

    max_val = max(
        max(abs(x) for x in xs),
        max(abs(y) for y in ys),
        max(abs(z) for z in zs)
    )

    max_val = max(max(abs(x) for x in xs), max(abs(y) for y in ys))

    if max_val == 0: return row

    for i in range(21):
        row[f"x{i}"] /= max_val
        row[f"y{i}"] /= max_val
        row[f"z{i}"] /= max_val

    return row


# MAIN
def main():
    files = list(RAW_DIR.glob("*.csv"))

    if not files:
        print("No raw data found")
        return

    class_data = {}
    seen_classes = set()

    # normalize each file
    for f in files:
        class_name = f.stem

        if class_name not in VALID_CLASSES:
            continue

        try:
            df = pd.read_csv(f)
            df = df.apply(normalize_row, axis=1)
        except Exception:
            continue

        if class_name not in class_data:
            class_data[class_name] = []

        class_data[class_name].append(df)
        seen_classes.add(class_name)

    # check missing class
    missing = VALID_CLASSES - seen_classes
    if missing:
        print("Missing classes:")
        for m in sorted(missing):
            print(m)

    # merge by class
    for class_name, df_list in class_data.items():
        merged_df = pd.concat(df_list, ignore_index=True)
        save_path = OUTPUT_DIR / f"{class_name}.csv"
        merged_df.to_csv(save_path, index=False)

    print("Done")


# ENTRY
if __name__ == "__main__":
    main()