import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

# PATH CONFIG
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent.parent

INPUT_FILE = ROOT_DIR / "data" / "processed_dataset" / "dataset.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "dataset_splits"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # อ่านไฟล์ที่ถูก merge มาแล้ว
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found. Run merge_dataset.py first!")
        return

    full_df = pd.read_csv(INPUT_FILE)

    # เตรียม X และ y
    X = full_df.drop(columns=["label", "user_id"], errors='ignore')
    y = full_df["label"]

    # Split 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # บันทึกไฟล์ลงใน data/dataset_splits/
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print(f"Split Complete!")
    print(f"Train: {len(train_df)} samples")
    print(f"Test: {len(test_df)} samples")

if __name__ == "__main__":
    main()