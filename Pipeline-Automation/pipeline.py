import os
import glob
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, silhouette_score


DATA_DIR = "data"
PROCESSED_FILE = "processed_files.txt"
MODEL_DIR = "models"
ARCHIVE_DIR = "archive"
BEST_META = os.path.join(MODEL_DIR, "best_model_meta.json")
LOG_FILE = "pipeline.log"
RANDOM_STATE = 42
SCHEDULE_TIME = "09:00"
COMMON_TARGET_NAMES = ("target","label","y","class","outcome","response")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

#  Load CSVs 
def load_new_csvs(data_folder=DATA_DIR, processed_file=PROCESSED_FILE):
    os.makedirs(data_folder, exist_ok=True)
    csvs = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
    processed = set()
    if os.path.exists(processed_file):
        with open(processed_file, "r") as f:
            processed = set(f.read().splitlines())
    new = [p for p in csvs if p not in processed]
    if not new:
        logging.info("No new CSVs.")
        return None, []
    dfs = []
    for p in new:
        try:
            dfs.append(pd.read_csv(p))
            logging.info(f"Loaded {p}")
        except Exception as e:
            logging.exception(f"Failed to read {p}: {e}")
    if not dfs:
        return None, new
    combined = pd.concat(dfs, ignore_index=True)
    with open(processed_file, "a") as f:
        for p in new:
            f.write(p + "\n")
    return combined, new

#EDA
def light_eda(df):
    logging.info(f"Data shape: {df.shape}")
    logging.info(f"Columns: {df.columns.tolist()}")
    logging.info(f"Missing per column:\n{df.isna().sum().to_dict()}")
    dtype_info = {c: str(df[c].dtype) for c in df.columns}
    logging.info(f"Dtypes: {dtype_info}")
    if df.shape[0] > 0:
        sample_card = {c: int(df[c].nunique()) for c in df.columns}
        logging.info(f"Column cardinalities (sample): {sample_card}")

#  Detect target & problem
def detect_target_and_problem(df, explicit_target=None):
    if explicit_target and explicit_target in df.columns:
        target = explicit_target
    else:
        candidates = [c for c in df.columns if c.lower() in COMMON_TARGET_NAMES]
        target = candidates[0] if candidates else None

    if target is None:
        return None, "unsupervised"

    if pd.api.types.is_numeric_dtype(df[target]):
        nunique = df[target].nunique()
        if nunique <= 20 and nunique / max(len(df),1) < 0.05:
            return target, "classification"
        return target, "regression"
    else:
        return target, "classification"

# Preprocessing
def build_preprocessor(df, drop_cols=None):
    if drop_cols is None:
        drop_cols = []
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in numeric if c not in drop_cols]
    categorical = [c for c in categorical if c not in drop_cols]

    transformers = []
    if numeric:
        num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler())])
        transformers.append(("num", num_pipe, numeric))
    if categorical:
        cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                             ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        transformers.append(("cat", cat_pipe, categorical))

    if not transformers:
        return "passthrough", numeric, categorical

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return pre, numeric, categorical

#  Models
def get_model(problem_type):
    if problem_type == "regression":
        return RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    if problem_type == "classification":
        return RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    if problem_type == "unsupervised":
        return KMeans(n_clusters=5, random_state=RANDOM_STATE)
    raise ValueError("Unknown problem_type")

#  Train 
def train_and_eval(df, target, problem_type):
    if problem_type == "unsupervised":
        pre, _, _ = build_preprocessor(df)
        X = df.copy()
        if isinstance(pre, ColumnTransformer):
            Xp = pre.fit_transform(X)
        else:
            Xp = X.values
        model = get_model("unsupervised")
        model.fit(Xp)
        score = None
        try:
            score = float(silhouette_score(Xp, model.labels_))
        except Exception:
            score = None
        metrics = {"silhouette": score}
        return model, pre, metrics

    if target not in df.columns:
        raise ValueError("Target missing for supervised run.")

    X = df.drop(columns=[target])
    y = df[target]

    pre, num_cols, cat_cols = build_preprocessor(X)

    for c in cat_cols:
        X[c] = X[c].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    model = get_model(problem_type)
    pipeline = Pipeline([("pre", pre), ("model", model)])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    metrics = {}
    if problem_type == "regression":
        metrics["mse"] = float(mean_squared_error(y_test, preds))
        metrics["r2"] = float(r2_score(y_test, preds))
    else:
        try:
            metrics["accuracy"] = float(accuracy_score(y_test, preds))
            metrics["f1"] = float(f1_score(y_test, preds, average="weighted", zero_division=0))
        except Exception:
            metrics["note"] = "could not compute standard classification metrics"
    return pipeline, pre, metrics

#Save Best Model 
def score_for_compare(metrics):
    if metrics is None:
        return -1e9
    if "mse" in metrics:
        return -metrics["mse"]
    if "accuracy" in metrics:
        return metrics["accuracy"]
    if "f1" in metrics:
        return metrics["f1"]
    if "silhouette" in metrics and metrics["silhouette"] is not None:
        return metrics["silhouette"]
    if "r2" in metrics:
        return metrics["r2"]
    return 0

def save_if_better(model_obj, metrics, name="model.pkl"):
    model_path = os.path.join(MODEL_DIR, name)
    new_score = score_for_compare(metrics)
    current = {}
    if os.path.exists(BEST_META):
        try:
            with open(BEST_META, "r") as f:
                current = json.load(f)
        except Exception:
            current = {}
    current_score = current.get("score", -1e9)
    if new_score <= current_score:
        logging.info(f"Trained model did NOT beat current best (new_score={new_score}, best={current_score}). Not overwriting.")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ts_path = os.path.join(MODEL_DIR, f"model_{ts}.pkl")
        joblib.dump(model_obj, ts_path)
        logging.info(f"Saved timestamped model: {ts_path}")
        return False

    joblib.dump(model_obj, model_path)
    with open(BEST_META, "w") as f:
        json.dump({"model_path": model_path, "metrics": metrics, "score": new_score, "saved_at": datetime.now().isoformat()}, f, indent=2)
    logging.info(f"Saved NEW best model: {model_path} with metrics {metrics}")
    return True 

# Pipeline
def run_pipeline(explicit_target=None):
  
    logging.info("Pipeline run started ")
    try:
        df, loaded_files = load_new_csvs()
        if df is None:
   
            logging.info("Pipeline skipped: no new data.")
            return
        light_eda(df)
        target, problem = detect_target_and_problem(df, explicit_target)
        logging.info(f"Detected problem: {problem} (target: {target})")
        model_obj, pre, metrics = train_and_eval(df, target, problem)
        save_if_better(model_obj, metrics)
   
    except Exception as e:
        logging.exception(f"Pipeline failed: {e}")

#Scheduler
if __name__ == "__main__":
    import argparse
    import schedule
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--target", type=str, default=None, help="Explicit target column name")
    parser.add_argument("--time", type=str, default=SCHEDULE_TIME, help="Daily run time HH:MM")
    args = parser.parse_args()

    if args.once:
        run_pipeline(explicit_target=args.target)
    else:
        run_pipeline(explicit_target=args.target)
        schedule.every().day.at(args.time).do(run_pipeline, explicit_target=args.target)
        logging.info(f"Scheduler started. Daily run at {args.time}")
        while True:
            schedule.run_pending()
            time.sleep(30)