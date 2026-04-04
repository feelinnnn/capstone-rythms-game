import os
import json
import zipfile
import shutil
import argparse
from pathlib import Path

# --- CONFIG PATHS ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw_landmarks"
TEMP_DIR = DATA_DIR / "temp_sync"
CONFIG_DIR = ROOT_DIR / "config"
DRIVE_CFG = CONFIG_DIR / "drive_config.json"

RAW_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# setup config
def load_drive_config():
    if not DRIVE_CFG.exists():
        default_cfg = {"drive_folder_url": "ใส่ลิงก์_Google_Drive_Folder_ตรงนี้"}
        with open(DRIVE_CFG, "w", encoding="utf-8") as f:
            json.dump(default_cfg, f, indent=4)
        return default_cfg
    
    with open(DRIVE_CFG, "r", encoding="utf-8") as f:
        return json.load(f)

# คำสั่ง PUT: บีบอัดไฟล์เตรียมอัปโหลด
def put_data(user_id):
    user_files = list(RAW_DIR.glob(f"*_{user_id}.csv"))
    
    if not user_files:
        print(f"[!] ไม่พบไฟล์ CSV ของรหัส {user_id}")
        return

    zip_filename = ROOT_DIR / f"Dataset_{user_id}.zip"
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for f in user_files:
            zipf.write(f, arcname=f.name)
            
    print(f"[SUCCESS] บีบอัด {zip_filename.name} สำเร็จ ({len(user_files)} ไฟล์) - พร้อมอัปโหลดขึ้น Drive")

# คำสั่ง GET: ดาวน์โหลดและแตกไฟล์
def get_data():
    cfg = load_drive_config()
    folder_url = cfg.get("drive_folder_url", "")
    
    if not folder_url or folder_url == "ใส่ลิงก์_Google_Drive_Folder_ตรงนี้":
        print("[!] ข้อผิดพลาด: ไม่พบลิงก์ Google Drive ใน config")
        return

    try:
        import gdown
    except ImportError:
        print("[!] กรุณาติดตั้ง gdown: pip install gdown")
        return

    print("=== กำลังดาวน์โหลดและแตกไฟล์จาก Google Drive ===")
    
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        gdown.download_folder(url=folder_url, output=str(TEMP_DIR), quiet=True, use_cookies=False)
        downloaded_zips = list(TEMP_DIR.rglob("*.zip"))
        
        if not downloaded_zips:
            print("[!] ไม่พบไฟล์ .zip ในโฟลเดอร์")
        else:
            extract_count = 0
            for zip_path in downloaded_zips:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(RAW_DIR)
                    extract_count += len(zip_ref.namelist())
            
            print(f"[SUCCESS] ดึงข้อมูลอัปเดตสำเร็จ! นำเข้าทั้งหมด {extract_count} ไฟล์")
            
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)

# ARGPARSE
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="สคริปต์ Sync ข้อมูล Dataset (Put/Get)")
    parser.add_argument("--put", action="store_true", help="บีบอัดไฟล์ CSV เดิมเตรียมอัปโหลด")
    parser.add_argument("--get", action="store_true", help="ดาวน์โหลดและแตกไฟล์จาก Google Drive")
    parser.add_argument("--user", type=str, help="รหัส User ID (ใช้คู่กับ --put)")
    
    args = parser.parse_args()

    if args.put:
        if not args.user:
            print("[!] กรุณาระบุรหัสผู้ใช้ เช่น --put --user P01")
        else:
            put_data(args.user)
    elif args.get:
        get_data()
    else:
        parser.print_help()