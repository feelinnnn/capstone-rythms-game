import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# PATH
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent.parent

TRAIN_PATH = ROOT_DIR / "data" / "dataset_splits" / "train.csv"
TEST_PATH = ROOT_DIR / "data" / "dataset_splits" / "test.csv"
MODEL_SAVE_PATH = ROOT_DIR / "models" / "xgb_model.pkl"

# LOAD DATA
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


# PREPARE DATA
X = train_df.drop(columns=["label"])
y = train_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)


# MODEL
model = XGBClassifier(
    n_estimators=2000,         
    max_depth=8,               
    learning_rate=0.03,        
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    num_class=len(set(y)),
    tree_method="hist",        
    n_jobs=-1,
    random_state=42,
    eval_metric="mlogloss",
    early_stopping_rounds=50, 
)


print("Training with Early Stopping...")
model.fit(
    X_train, 
    y_train, 
    eval_set=[(X_val, y_val)], 
    verbose=True  
)
# EVALUATE
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

print("\n Classification Report:")
print(classification_report(y_test, y_pred))

print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# SAVE MODEL
MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_SAVE_PATH)