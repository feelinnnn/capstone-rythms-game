import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# PATH
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent.parent

TRAIN_PATH = ROOT_DIR / "data" / "dataset_splits" / "train.csv"
TEST_PATH = ROOT_DIR / "data" / "dataset_splits" / "test.csv"
MODEL_SAVE_PATH = ROOT_DIR / "models" / "rf_model.pkl"

# LOAD DATA
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


# PREPARE DATA
X = train_df.drop(columns=["label"])
y = train_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

# split train -> train + val
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.1,
    stratify=y,
    random_state=42
)

# MODEL
model = RandomForestClassifier(
    n_estimators=500,          # เยอะ = แม่นขึ้น
    max_depth=18,              # balance ไม่ overfit
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",       
    bootstrap=True,
    oob_score=True,            # เช็ค overfit
    class_weight="balanced",   # กัน class imbalance
    n_jobs=-1,
    random_state=42
)

# TRAIN
print("Training Random Forest...")
model.fit(X_train, y_train)

# VALIDATION
val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)

print(f"\n Validation Accuracy: {val_acc:.4f}")
print(f"OOB Score: {model.oob_score_:.4f}")

# TEST EVALUATION
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n Accuracy: {acc:.4f}")

print("\n Classification Report:")
print(classification_report(y_test, y_pred))

print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# FEATURE IMPORTANCE
importance = model.feature_importances_
top_idx = np.argsort(importance)[-10:]

print("\nTop Important Features Index:")
print(top_idx)

# SAVE MODEL
MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_SAVE_PATH)