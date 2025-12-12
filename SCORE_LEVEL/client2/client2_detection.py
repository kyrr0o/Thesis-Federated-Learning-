# client2_detection.py
import os
import gc
import time
import json
import warnings
import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from joblib import Parallel, delayed

# server imports
import requests
from pathlib import Path


warnings.filterwarnings("ignore", category=UserWarning)

# settings
DATA_PATH   = "SCORE_LEVEL/2/BTC30Min.csv"
# RESULTS_DIR = "SCORE_LEVEL/global/results"
ANOMALY_RATE = 0.05
CHUNKS       = 6  # also used as ensemble size (one model per chunk)

# utils
def choose_n_jobs(preferred=-1, cpu_safety_frac=0.5, mem_threshold_pct=80):
    cpu_count = os.cpu_count() or 1
    cpu_load  = psutil.cpu_percent(interval=0.5)
    mem_pct   = psutil.virtual_memory().percent

    if preferred == -1:
        if cpu_load > 70 or mem_pct > mem_threshold_pct:
            n_jobs = max(1, int(cpu_count * cpu_safety_frac))
        else:
            n_jobs = -1  # use all cores
    else:
        n_jobs = min(preferred, cpu_count)
        if cpu_load > 85 or mem_pct > mem_threshold_pct:
            n_jobs = max(1, int(cpu_count * cpu_safety_frac))
    return n_jobs

def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print(f"[INFO] Dataset loaded with shape {df.shape}")
    return df

# anomaly injection
def inject_extreme_anomalies(df, rate=0.02, random_state=None):
    df = df.copy()
    if random_state is None:
        random_state = np.random.randint(0, 10000)
    np.random.seed(random_state)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'datetime']

    n_anomalies = int(len(df) * rate)
    anomaly_idx = np.random.choice(df.index, n_anomalies, replace=False)

    '''
    Three distinct anomaly types will be generated:
    1. Adaptive Feature Perturbation
    2. Correlated OHLC Distortion
    3. Volume Spike Amplification
    '''
    # 1) base adaptive perturbation per numeric feature
    for col in numeric_cols:
        std = df[col].std()
        if std > 0:
            df.loc[anomaly_idx, col] = (
                df.loc[anomaly_idx, col]
                + np.random.randn(n_anomalies) * std * np.random.uniform(1.5, 2.5)
            )

    # 2) correlated OHLC anomaly
    ohlc_cols = ['open', 'high', 'low', 'close']
    if all(c in df.columns for c in ohlc_cols):
        scale = np.random.uniform(1.5, 2.5)
        df.loc[anomaly_idx, ohlc_cols] *= scale

    # 3) volume spike
    if 'volume' in df.columns:
        df.loc[anomaly_idx, 'volume'] *= np.random.uniform(1.5, 3.0)

    # label column
    df['is_anomaly'] = 0
    df.loc[anomaly_idx, 'is_anomaly'] = 1

    print(f"[INFO] Injected {n_anomalies} anomalies in numeric cols: {numeric_cols}")
    return df

'''
actually feature selection jud ni sya ayaw lang i mind ang extract since lain na sya na
type but similar logic ra with select.
'''
def extract_features(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df.fillna(df.mean(numeric_only=True), inplace=True)
    print(f"[INFO] Using {len(numeric_cols)} numeric columns: {numeric_cols}")
    return df

# data partitioning section: 60/20/20
def split_data(df, test_size=0.2, val_size=0.25, random_state=None):
    if random_state is None:
        random_state = np.random.randint(0, 10000)

    df['is_anomaly'] = df['is_anomaly'].astype(int)

    # train+val vs test
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['is_anomaly']
    )

    # train vs val
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=train_val['is_anomaly']
    )

    train = train.reset_index(drop=True)
    val   = val.reset_index(drop=True)
    test  = test.reset_index(drop=True)

    print(f"[INFO] Data split train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"[INFO] Train anomalies: {train['is_anomaly'].sum()}, "
          f"Val anomalies: {val['is_anomaly'].sum()}, "
          f"Test anomalies: {test['is_anomaly'].sum()}")

    return train, val, test

def preprocess(train, val, test):
    scaler = StandardScaler()

    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != 'is_anomaly']

    print(f"[INFO] Preprocess using {len(feature_cols)} feature columns: {feature_cols}")

    X_train = scaler.fit_transform(train[feature_cols])
    X_val   = scaler.transform(val[feature_cols])
    X_test  = scaler.transform(test[feature_cols])

    y_train = train['is_anomaly'].astype(int).values
    y_val   = val['is_anomaly'].astype(int).values
    y_test  = test['is_anomaly'].astype(int).values

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

# dual parallel section
def train_single_chunk(subX, base_params, seed):
    params = base_params.copy()
    base_n_estimators = int(params.get("n_estimators", 100))

    # Each chunk gets a subset of trees -> memory friendly
    params["n_estimators"] = base_n_estimators  # instead of // CHUNKS
    params["max_samples"]  = base_params.get("max_samples", 0.1)  # ayaw i-cap sa 0.05
    # params["bootstrap"]    = True  # match sa old best_params [pls ignore]

    params["n_jobs"]       = choose_n_jobs(-1)  # memory-aware tree-level parallel
    params["random_state"] = seed

    model = IsolationForest(**params)
    model.fit(subX.astype(np.float32))
    return model

def train_iforest_in_chunks(X, base_params, n_chunks=CHUNKS, random_state=None):
    if random_state is None:
        random_state = np.random.randint(0, 10000)

    X_local   = X.astype(np.float32)
    chunks    = np.array_split(X_local, n_chunks)
    seeds     = [(random_state + i * 97) % 2**31 for i in range(n_chunks)]

    # chunk-level parallel (second level)
    models = Parallel(n_jobs=n_chunks)(
        delayed(train_single_chunk)(chunks[i], base_params, seeds[i])
        for i in range(n_chunks)
    )

    gc.collect()
    return models  # ensemble of chunk models

def aggregate_models_scores(models, X, agg="mean"):
    X_local = X.astype(np.float32)
    score_list = []
    for m in models:
        score_list.append(m.score_samples(X_local))
        gc.collect()
    score_matrix = np.vstack(score_list)
    if agg == "mean":
        score_agg = np.mean(score_matrix, axis=0)
    else:
        score_agg = np.median(score_matrix, axis=0)
    return score_agg

def calibrate_threshold_from_scores(scores, y_val):
    scores = np.asarray(scores)
    y_val  = np.asarray(y_val)

    p1  = np.percentile(scores, 1)
    p99 = np.percentile(scores, 99)

    if p99 <= p1:
        coarse_thresh = np.linspace(p1 - 1e-6, p1 + 1e-6, 200)
    else:
        coarse_thresh = np.linspace(p1, p99, 200)

    best_f1, best_t = -1.0, coarse_thresh[0]
    for t in coarse_thresh:
        preds = (scores < t).astype(int)
        f1    = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

def evaluate_model_from_scores(scores, y_true, threshold):
    preds = (scores < threshold).astype(int)
    p = precision_score(y_true, preds, zero_division=0)
    r = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    return p, r, f1, preds

def save_results(output_path, results_dict):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=4)

# main script
def run_client(random_state=None):
    if random_state is None:
        random_state = np.random.randint(0, 10000)

    start_total = time.time()

    # --- DATA LOADING ---
    t0 = time.time()
    df = load_data(DATA_PATH)
    print(f"[TIME] Load data: {time.time() - t0:.4f} sec")

    # --- ANOMALY INJECTION ---
    t0 = time.time()
    df = inject_extreme_anomalies(df, rate=ANOMALY_RATE, random_state=random_state)
    print(f"[TIME] Inject anomalies: {time.time() - t0:.4f} sec")

    # --- FEATURE SELECTION ---
    t0 = time.time()
    df = extract_features(df)
    print(f"[TIME] Extract features: {time.time() - t0:.4f} sec")

    # --- SPLITTING ---
    t0 = time.time()
    train_df, val_df, test_df = split_data(df, random_state=random_state)
    print(f"[TIME] Split data: {time.time() - t0:.4f} sec")
    del df; gc.collect()

    # --- PREPROCESSING ---
    t0 = time.time()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess(train_df, val_df, test_df)
    print(f"[TIME] Preprocess: {time.time() - t0:.4f} sec")
    del train_df, val_df, test_df; gc.collect()

    best_params = {
        "n_estimators": 100,
        "max_samples": 0.1,
        "max_features": 0.5,
        "contamination": 0.02,
        # "bootstrap": True
    }
    print("[INFO] Using fixed parameters:", best_params)

    # --- DUAL-LEVEL PARALLEL TRAINING (CHUNK ENSEMBLE) ---
    t0 = time.time()
    models = train_iforest_in_chunks(X_train, best_params, n_chunks=CHUNKS, random_state=random_state)
    print(f"[TIME] Dual-level training (chunks x trees): {time.time() - t0:.4f} sec")

    # --- VALIDATION SCORES & THRESHOLD CALIBRATION ---
    t0 = time.time()
    scores_val = aggregate_models_scores(models, X_val)
    threshold  = calibrate_threshold_from_scores(scores_val, y_val)
    print(f"[TIME] Threshold calibration (global val): {time.time() - t0:.4f} sec")

    # --- TEST SCORES & FINAL EVALUATION ---
    t0 = time.time()
    scores_test        = aggregate_models_scores(models, X_test)
    p, r, f1, preds    = evaluate_model_from_scores(scores_test, y_test, threshold)
    print(f"[TIME] Evaluation (test): {time.time() - t0:.4f} sec")

    print(f"[RESULTS] Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}, Threshold={threshold:.5f}")

    end_total       = time.time()
    exec_time_total = end_total - start_total
    print(f"[TIME] Total runtime: {exec_time_total:.4f} sec")

    results = {
        "precision":      float(p),
        "recall":         float(r),
        "f1":             float(f1),
        "threshold":      float(threshold),
        "val_scores":     scores_val.tolist(),   # <-- for global calibration
        "y_val":          y_val.tolist(),
        "test_scores":    scores_test.tolist(),  # <-- for global test evaluation
        "y_test":         y_test.tolist(),
        "params":         best_params,
        "exec_time_total": exec_time_total
    }

    # ---- AUTO SEND TO SERVER ----
    SERVER_URL = "http://192.168.254.158"   # <-- CHANGE to server PC IP
    ROUND_ID   = "1"
    CLIENT_ID  = "client2"

    payload = json.dumps(results).encode("utf-8")

    files = {
        "payload": ("client2_results.json", payload, "application/json")
    }
    data = {
        "round_id": ROUND_ID,
        "client_id": CLIENT_ID
    }

    resp = requests.post(
        f"{SERVER_URL}/upload_scores",
        data=data,
        files=files,
        timeout=300
    )
    resp.raise_for_status()
    print("[NETWORK] Results sent to server:", resp.json())

if __name__ == "__main__":
    run_client()
