import pandas as pd
import joblib
import time
import os
import warnings
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ปิดการแจ้งเตือนscikit-learn
warnings.filterwarnings('ignore')

# --- CONFIG PATHS ---
ROOT_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT_DIR / "data" / "dataset_splits"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "data" / "processed_dataset"

TEST_PATH = SPLITS_DIR / "test.csv"

# รายชื่อโมเดล
MODELS_TO_COMPARE = {
    "MLP": "mlp_model.pkl",
    "SVM": "svm_model.pkl",
    "Random Forest": "rf_model.pkl",
    "XGBoost": "xgb_model.pkl"
}

def benchmark_models():
    print("=== Model Benchmark (Real-time Simulation) ===")

    # 1. Load Data safely
    if not TEST_PATH.exists():
        print(f"[ERROR] Missing test data at {TEST_PATH}")
        return

    test_df = pd.read_csv(TEST_PATH)
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1]

    # จำลองภาพ 1 เฟรมจากกล้อง สำหรับทดสอบความเร็ว
    single_frame = X_test[0].reshape(1, -1)
    results = []

    # ตรวจสอบว่ามีโฟลเดอร์ RESULTS_DIR หรือยัง ถ้ายังให้สร้างก่อน
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Benchmark Loop
    for model_name, file_name in MODELS_TO_COMPARE.items():
        model_path = MODELS_DIR / file_name

        # [SAFE] ถ้ายังไม่ได้ส่งไฟล์มา โค้ดจะข้ามไปตรวจตัวอื่นต่อโดยไม่ Error
        if not model_path.exists():
            print(f"[*] {model_name}: [SKIP] File '{file_name}' not found.")
            continue

        print(f"[*] {model_name}: Evaluating...")
        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"    -> [ERROR] Failed to load {file_name}: {e}")
            continue

        # วัดขนาดไฟล์ (MB)
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)

        # วัดความแม่นยำ (Accuracy)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100

        try:
            cm = confusion_matrix(y_test, y_pred)
            # ใช้ model.classes_ เพื่อดึงชื่อท่าทางมาเป็น Label 
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            
            # สร้างกราฟขนาด 10x8
            fig, ax = plt.subplots(figsize=(10, 8))
            disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
            plt.title(f"Confusion Matrix: {model_name}")
            plt.tight_layout()
            
            # เซฟรูปเก็บไว้ (เช่น MLP_cm.png)
            cm_filename = f"{model_name.replace(' ', '_')}_cm.png"
            cm_save_path = RESULTS_DIR / cm_filename
            plt.savefig(cm_save_path, dpi=300)
            
            # ปิดกราฟเพื่อคืนหน่วยความจำ (สำคัญมากตอนรันในลูป)
            plt.close(fig)
            print(f"    -> [SUCCESS] Saved CM image to {cm_filename}")
        except Exception as e:
            print(f"    -> [ERROR] Failed to generate Confusion Matrix: {e}")

        # ป้องกันอาการ "Cold Start" ที่เฟรมแรกๆ จะใช้เวลาประมวลผลนานกว่าปกติ
        for _ in range(50):
            model.predict(single_frame)

        # ใช้ perf_counter ที่จับเวลาได้แม่นยำระดับฮาร์ดแวร์
        iterations = 1000
        start_time = time.perf_counter()
        for _ in range(iterations):
            model.predict(single_frame)
        end_time = time.perf_counter()

        # คำนวณความเร็ว
        avg_time_ms = ((end_time - start_time) / iterations) * 1000
        estimated_fps = 1000 / avg_time_ms if avg_time_ms > 0 else float('inf')

        results.append({
            "Model": model_name,
            "Accuracy (%)": round(accuracy, 2),
            "Size (MB)": round(file_size_mb, 2),
            "Latency/Frame (ms)": round(avg_time_ms, 3),
            "Max FPS": int(estimated_fps)
        })

    # 3. Generate Report
    if not results:
        print("\n[!] No models were evaluated.")
        return

    print("\n=== Benchmark Results ===")
    results_df = pd.DataFrame(results)
    
    # จัดเรียงตาราง ให้ความเร็ว(Latency)น้อยที่สุดขึ้นก่อน
    results_df = results_df.sort_values(by="Latency/Frame (ms)", ascending=True).reset_index(drop=True)
    print(results_df.to_string(index=False))

    # Save to CSV
    save_path = RESULTS_DIR / "model_comparison.csv"
    results_df.to_csv(save_path, index=False)
    print(f"\n[SUCCESS] Results saved to: {save_path}")

if __name__ == "__main__":
    benchmark_models()