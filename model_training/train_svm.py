import pandas as pd
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIG PATHS ---
ROOT_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT_DIR / "data" / "dataset_splits"
MODELS_DIR = ROOT_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = SPLITS_DIR / "train.csv"
TEST_PATH = SPLITS_DIR / "test.csv"
MODEL_SAVE_PATH = MODELS_DIR / "svm_model.pkl"

def train_svm():
    print("=== Training SVM Model ===")
    
    # Load Data
    try:
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Missing train.csv or test.csv in {SPLITS_DIR}")
        return

    # Prepare Features (X) and Labels (y)
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1]

    print(f"[*] Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

    # Create Pipeline & Train
    print("[*] Training in progress...")
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('svm', SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            cache_size=2000,
            random_state=42
        ))
    ])
    
    model_pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"[*] Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save Model
    joblib.dump(model_pipeline, MODEL_SAVE_PATH)
    print(f"[SUCCESS] Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_svm()