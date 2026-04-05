import pandas as pd
from pathlib import Path
import os

# PATH CONFIG
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent.parent

INPUT_DIR = ROOT_DIR / "data" / "processed_dataset"

def main():
    # รวบรวมไฟล์ CSV ทั้งหมดจาก processed_dataset
    all_files = list(INPUT_DIR.glob("*.csv"))
    
    if not all_files:
        print("No processed data found in data/processed_dataset")
        return

    data_list = []
    for f in all_files:
        # ข้ามไฟล์ dataset.csv (ถ้ามีอยู่แล้ว)
        if f.name == "dataset.csv":
            continue
        df = pd.read_csv(f)
        data_list.append(df)

    # รวมทุกคลาสเป็น DataFrame เดียว
    full_df = pd.concat(data_list, ignore_index=True)
    
    # บันทึกไฟล์รวม
    output_path = INPUT_DIR / "dataset.csv"
    full_df.to_csv(output_path, index=False)
    
    print(f"Merge Complete!")
    print(f"Total samples: {len(full_df)}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()