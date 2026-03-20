# =========================================================
# central_utils.py
# =========================================================

import pandas as pd
import numpy as np
import time
import psutil
import json
import os
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterSampler
from joblib import Parallel, delayed

# =========================================================
# Utilities: resource-aware helpers
# =========================================================
def choose_n_jobs(preferred=-1, cpu_safety_frac=0.5, mem_threshold_pct=80):
    """
    Decide n_jobs for sklearn based on current system load.
     - preferred: user's preferred value (e.g. -1)
     - cpu_safety_frac: fraction of CPUs to reserve for system (0.5 -> use up to half)
     - mem_threshold_pct: if memory usage > this percent, reduce jobs aggressively
    """
    cpu_count = os.cpu_count() or 1
    cpu_load = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    mem_pct = mem.percent

    # Default: use -1 (all cores) if load is low; otherwise reduce
    if preferred == -1:
        if cpu_load > 70 or mem_pct > mem_threshold_pct:
            n_jobs = max(1, int(cpu_count * cpu_safety_frac))
        else:
            n_jobs = -1
    else:
        # honor explicit requests but cap
        n_jobs = preferred
        if n_jobs > cpu_count:
            n_jobs = cpu_count

        if cpu_load > 85 or mem_pct > mem_threshold_pct:
            n_jobs = max(1, int(cpu_count * cpu_safety_frac))

    return n_jobs

# =========================================================
# Data Loading (robust for large files)
# =========================================================
def load_data(file_path, sample_frac=1.0, max_rows=None, warn_large_gb=1.0):
    """
    Loads dataset safely with memory control.
    - sample_frac: fraction to sample after loading full (1.0 = no sampling)
    - max_rows: optionally read only first N rows
    - warn_large_gb: if file larger than this (GB), print warning and try chunk-first-row strategy
    """
    try:
        file_size_gb = os.path.getsize(file_path) / (1024 ** 3)
    except Exception:
        file_size_gb = 0.0

    try:
        if max_rows is not None:
            df = pd.read_csv(file_path, encoding='utf-8-sig', nrows=max_rows, skip_blank_lines=True)
        else:
            if file_size_gb > warn_large_gb:
                # load first chunk to avoid OOM; user should use streaming/processing for full dataset
                print(f"[WARN] File size {file_size_gb:.2f} GB — loading first chunk only. Use streaming for full processing.")
                it = pd.read_csv(file_path, encoding='utf-8-sig', skip_blank_lines=True, chunksize=200_000)
                df = next(it)
            else:
                df = pd.read_csv(file_path, encoding='utf-8-sig', skip_blank_lines=True)
        # if all columns got merged into one, force re-read with comma sep
        if len(df.columns) == 1 and ',' in df.iloc[0, 0]:
            df = pd.read_csv(file_path, sep=',', encoding='utf-8-sig', skip_blank_lines=True)
        # drop unnamed index columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)
            print(f"[INFO] Sampled {sample_frac*100:.1f}% of rows for safety.")
        print(f"[INFO] Dataset loaded successfully with shape {df.shape} and columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load data from {file_path}: {e}")
        raise

# =========================================================
# Anomaly Injection (unchanged)
# =========================================================
def inject_extreme_anomalies(df, rate=0.02, random_state=None):
    df = df.copy()
    if random_state is None:
        random_state = np.random.randint(0, 10000)
    np.random.seed(random_state)

    exclude_cols = ['Timestamp', 'Date', 'datetime']
    numeric_cols = [
        col for col in df.select_dtypes(include=['float64', 'int64']).columns if col not in exclude_cols
    ]

    if not numeric_cols:
        possible_cols = ["Open", "High", "Low", "Close", "Volume"]
        numeric_cols = [col for col in possible_cols if col in df.columns]

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    if not numeric_cols:
        raise ValueError("No numeric columns found for anomaly injection!")

    n_anomalies = int(len(df) * rate)
    if n_anomalies == 0:
        print("[WARN] rate too small, no anomalies injected.")
        df['is_anomaly'] = 0
        return df

    anomaly_indices = np.random.choice(df.index, n_anomalies, replace=False)

    for col in numeric_cols:
        std_dev = df[col].std()
        if pd.isna(std_dev) or std_dev == 0:
            continue
        amplitude = np.random.uniform(2.0, 3.0)
        df.loc[anomaly_indices, col] = df[col].mean() + np.random.randn(n_anomalies) * std_dev * amplitude

    noise = np.random.normal(0, 0.01, size=df[numeric_cols].shape)
    df[numeric_cols] = df[numeric_cols] + noise

    df['is_anomaly'] = 0
    df.loc[anomaly_indices, 'is_anomaly'] = 1
    print(f"[INFO] Injected {n_anomalies} anomalies into columns: {numeric_cols}")
    return df

def extract_features(df):
    """
    Disabled feature engineering version (baseline ablation).
    Returns the dataframe as-is, except ensures numeric conversion.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df.fillna(df.mean(numeric_only=True), inplace=True)
    print(f"[INFO] (No feature engineering applied) Using {len(df.columns)} raw columns: {df.columns.tolist()}")
    return df


# =========================================================
# Preprocessing
# =========================================================
def preprocess(train_df, val_df, test_df):
    scaler = StandardScaler()
    feature_cols = [c for c in train_df.columns if c not in ['Timestamp', 'is_anomaly']]

    X_train = scaler.fit_transform(train_df[feature_cols])
    X_val = scaler.transform(val_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    y_train = train_df['is_anomaly'].values
    y_val = val_df['is_anomaly'].values
    y_test = test_df['is_anomaly'].values

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

# =========================================================
# Random Search (unchanged)
# =========================================================
def random_search_if(X_train, X_val, y_val, n_iter=20, random_state=None):
    if random_state is None:
        random_state = np.random.randint(0, 10000)

    param_dist = {
        'n_estimators': [50, 100, 150],
        'max_samples': [0.05, 0.1, 0.2],
        'max_features': [0.5, 0.7, 0.9, 1.0],
        'contamination': [0.01, 0.02, 0.03, 0.04, 0.05, 0.08],
        'bootstrap': [True, False]
    }

    sample_limit = min(len(X_train), 200_000)
    if len(X_train) > sample_limit:
        np.random.seed(random_state)
        subset_idx = np.random.choice(len(X_train), sample_limit, replace=False)
        X_train_sub = X_train[subset_idx].astype(np.float32)
    else:
        X_train_sub = X_train.astype(np.float32)

    X_val = X_val.astype(np.float32)
    y_val = np.array(y_val).astype(int)

    sampler = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=random_state))
    best_model, best_f1, best_params = None, -1, None

    print(f"[INFO] Running random search on subset of {len(X_train_sub)} samples...")

    for i, params in enumerate(sampler):
        model = IsolationForest(**params, n_jobs=1, random_state=np.random.randint(0, 10000))
        model.fit(X_train_sub)
        preds = np.where(model.predict(X_val) == -1, 1, 0)
        if len(np.unique(y_val)) < 2:
            continue
        f1 = f1_score(y_val, preds, zero_division=0)
        print(f"[Random Search] {i + 1}/{n_iter} -> F1={f1:.4f}, Params={params}")
        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            best_model = model

    print(f"[Random Search] Best F1={best_f1:.4f} with Params={best_params}")
    return best_model, best_params

# =========================================================
# Dataset Split
# - divide data into 60-20-20 ratio
# =========================================================
def split_data(df, test_size=0.2, val_size=0.25, random_state=None):
    if random_state is None:
        random_state = np.random.randint(0, 10000)

    if 'is_anomaly' not in df.columns:
        raise ValueError("DataFrame must contain 'is_anomaly' column before splitting.")

    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True, stratify=df['is_anomaly']
    )

    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, random_state=random_state, shuffle=True, stratify=train_val_df['is_anomaly']
    )

    print(f"[INFO] Dataset split into train={len(train_df)}, val={len(val_df)}, test={len(test_df)} "
          f"(stratified by is_anomaly)")
    return train_df, val_df, test_df

# =========================================================
# Time and Memory Measurement
# =========================================================
def measure_time_memory(func, *args, **kwargs):
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 ** 2)
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    mem_after = process.memory_info().rss / (1024 ** 2)
    duration = end_time - start_time
    mem_used = mem_after - mem_before
    return result, duration, mem_used, mem_after

# =========================================================
# Evaluation Metrics
# =========================================================
def evaluate_model(model, X_test, y_test, threshold=None):
    """
    If threshold is None: use model.predict (sklearn API).
    If threshold provided: use model.score_samples and compare (scores < threshold) => anomaly=1.
    NOTE: This file assumes "lower score == more anomalous" when using score_samples,
    consistent with existing calibrate_threshold logic.
    """
    if threshold is None:
        preds = model.predict(X_test)
        preds = np.where(preds == -1, 1, 0)
    else:
        scores = model.score_samples(X_test)
        preds = (scores < threshold).astype(int)

    p = precision_score(y_test, preds, zero_division=0)
    r = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    return p, r, f1

# =========================================================
# Score Calibration (single model) — kept for backward compatibility
# =========================================================
def calibrate_threshold(model, X_val, y_val):
    """
    Calibrate threshold using model.score_samples(X_val).
    Returns best threshold (float).
    """
    scores = model.score_samples(X_val)

    coarse_thresholds = np.linspace(np.percentile(scores, 1), np.percentile(scores, 99), 200)
    best_f1, best_thresh = -1, None
    for t in coarse_thresholds:
        preds = (scores < t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    fine_min = best_thresh - abs(best_thresh) * 0.05
    fine_max = best_thresh + abs(best_thresh) * 0.05
    fine_thresholds = np.linspace(fine_min, fine_max, 200)
    for t in fine_thresholds:
        preds = (scores < t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    preds = (scores < best_thresh).astype(int)
    p = precision_score(y_val, preds, zero_division=0)
    r = recall_score(y_val, preds, zero_division=0)
    print(f"[CALIBRATION] Best threshold={best_thresh:.5f} | F1={best_f1:.4f} | P={p:.4f} | R={r:.4f}")

    best_thresh *= 1.05  
    return best_thresh

# =========================================================
# New: Calibrate from precomputed continuous scores (ensemble)
# =========================================================
def calibrate_threshold_from_scores(scores, y_val):
    scores = np.asarray(scores)
    y_val = np.asarray(y_val).astype(int)

    coarse_thresholds = np.linspace(np.percentile(scores, 1), np.percentile(scores, 99), 200)
    best_f1, best_thresh = -1, None
    for t in coarse_thresholds:
        preds = (scores < t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    fine_min = best_thresh - abs(best_thresh) * 0.05
    fine_max = best_thresh + abs(best_thresh) * 0.05
    fine_thresholds = np.linspace(fine_min, fine_max, 200)
    for t in fine_thresholds:
        preds = (scores < t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    preds = (scores < best_thresh).astype(int)
    p = precision_score(y_val, preds, zero_division=0)
    r = recall_score(y_val, preds, zero_division=0)
    print(f"[CALIBRATION] Best threshold={best_thresh:.5f} | F1={best_f1:.4f} | P={p:.4f} | R={r:.4f}")
    return best_thresh

# =========================================================
# Ensemble Voting with continuous score aggregation
# =========================================================
# def ensemble_iforest(X_train, X_test, params, n_models=3, mode="serial", agg="mean", return_scores=False):
#     """
#     Trains multiple IsolationForest models and returns:
#       - preds_final: majority-vote binary predictions (0 normal, 1 anomaly)
#     Optionally (return_scores=True) also returns aggregated continuous scores across models (score_agg).
#     Parameters:
#       - mode: "serial" => each model n_jobs=1; "parallel" => per-model n_jobs chosen by resource helper
#       - agg: "mean" or "median" for score aggregation
#       - return_scores: if True, return (preds_final, score_agg)
#     Note: Lower aggregated score -> more anomalous based on score_samples semantics used here.
#     """
#     preds_list = []
#     score_list = []

#     if mode not in ("serial", "parallel"):
#         mode = "serial"

#     # choose n_jobs per model
#     preferred = 1 if mode == "serial" else -1
#     n_jobs_value = choose_n_jobs(preferred=preferred)

#     print(f"[ENSEMBLE] Training {n_models} IsolationForest models ({mode.upper()} mode, per-model n_jobs={n_jobs_value})...")
#     for i in range(n_models):
#         seed = np.random.randint(0, 10000)
#         model = IsolationForest(**params, random_state=seed, n_jobs=n_jobs_value)
#         model.fit(X_train)

#         # predictions (per-model)
#         preds = np.where(model.predict(X_test) == -1, 1, 0)
#         preds_list.append(preds)

#         # continuous scores (score_samples)
#         scores = model.score_samples(X_test)
#         score_list.append(scores)

#         print(f"[ENSEMBLE] Model {i+1}/{n_models} done (seed={seed})")
#         # small cleanup per iteration
#         gc.collect()

#     score_matrix = np.vstack(score_list)  # shape (n_models, n_samples)
#     if agg == "median":
#         score_agg = np.median(score_matrix, axis=0)
#     else:
#         score_agg = np.mean(score_matrix, axis=0)

#     preds_final = (np.sum(preds_list, axis=0) >= (n_models // 2 + 1)).astype(int)

#     if return_scores:
#         return preds_final, score_agg
#     return preds_final

def ensemble_iforest_parallel(X_train, X_test, params, n_models=3, agg="mean", return_scores=False):
    """
    Trains multiple IsolationForest models in parallel (inter-model) while each model can use n_jobs threads (intra-model).
    Returns:
      - preds_final: majority-vote binary predictions (0 normal, 1 anomaly)
      - score_agg (if return_scores=True): aggregated continuous scores across models
    """

    # Decide per-model n_jobs (intra-model parallelism)
    n_jobs_value = choose_n_jobs(preferred=-1)
    print(f"[ENSEMBLE] Training {n_models} models with intra-model n_jobs={n_jobs_value} in parallel...")

    def train_one_model(seed):
        model = IsolationForest(**params, random_state=seed, n_jobs=n_jobs_value)
        model.fit(X_train)
        preds = np.where(model.predict(X_test) == -1, 1, 0)
        scores = model.score_samples(X_test)
        gc.collect()
        return preds, scores

    # Run models in parallel (inter-model parallelism)
    seeds = np.random.randint(0, 10000, size=n_models)
    results = Parallel(n_jobs=n_models)(
        delayed(train_one_model)(s) for s in seeds
    )

    # Unpack predictions and scores
    preds_list, score_list = zip(*results)
    preds_matrix = np.vstack(preds_list)
    score_matrix = np.vstack(score_list)

    # Aggregate scores
    score_agg = np.median(score_matrix, axis=0) if agg == "median" else np.mean(score_matrix, axis=0)

    # Majority vote for final predictions
    preds_final = (np.sum(preds_matrix, axis=0) >= (n_models // 2 + 1)).astype(int)

    if return_scores:
        return preds_final, score_agg
    return preds_final

# =========================================================
# Save Results
# =========================================================
def save_results(output_path, results_dict):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=4)

# =========================================================
# Safe chunked training & aggregation helpers
# =========================================================
def train_iforest_in_chunks(X, base_params, n_chunks=4, random_state=None):
    """
    Train several small IsolationForest models on chunks/subsamples of X to avoid OOM.
    Returns a list of trained models.
    Strategy:
      - Convert X to float32 to reduce memory.
      - Each chunk gets a reduced n_estimators (split of base), n_jobs=1 and bootstrap=False to limit memory copies.
    """

    if random_state is None:
        random_state = np.random.randint(0, 10000)

    X_local = X.astype(np.float32, copy=False)
    n_samples = X_local.shape[0]
    if n_chunks <= 0:
        n_chunks = 1
    chunk_sizes = [n_samples // n_chunks] * n_chunks
    for i in range(n_samples % n_chunks):
        chunk_sizes[i] += 1

    # Determine per-chunk n_estimators so total roughly equals base n_estimators
    base_n_estimators = int(base_params.get("n_estimators", 100))
    per_chunk_estimators = max(10, base_n_estimators // max(1, n_chunks))

    models = []
    start = 0
    print(f"[SAFE CHUNKS] Training {n_chunks} chunked IF models (per-chunk trees={per_chunk_estimators})")
    for i, size in enumerate(chunk_sizes):
        end = start + size
        subX = X_local[start:end]
        params = base_params.copy()
        params["n_estimators"] = per_chunk_estimators
        params["max_samples"] = min(params.get("max_samples", 0.1), 0.05)  # be conservative
        params["bootstrap"] = False  # avoid duplication memory
        params["n_jobs"] = 1
        params["random_state"] = (random_state + i) % 2**31

        print(f"[CHUNK {i+1}/{n_chunks}] samples {start}:{end} -> fit trees={params['n_estimators']}")
        model = IsolationForest(**params)
        model.fit(subX)
        models.append(model)

        # free local memory for subX
        del subX
        gc.collect()
        start = end

    print("[SAFE CHUNKS] Done training chunks.")
    return models


def aggregate_models_scores(models, X, agg="mean"):
    """
    Given a list of trained IsolationForest models, compute aggregated continuous scores on X.
    Returns a 1D numpy array of scores (lower -> more anomalous).
    """
    X_local = X.astype(np.float32, copy=False)
    score_list = []
    for i, m in enumerate(models):

        scores = m.score_samples(X_local)
        score_list.append(scores)
        gc.collect()

    score_matrix = np.vstack(score_list)
    if agg == "median":
        score_agg = np.median(score_matrix, axis=0)
    else:
        score_agg = np.mean(score_matrix, axis=0)
    return score_agg


def aggregate_models_preds(models, X, majority_threshold=None):
    """
    Compute majority-vote binary predictions from list of models.
    Returns binary array 0/1 where 1 = anomaly.
    majority_threshold: number of models that must vote anomaly; default: >half
    """
    preds = []
    for m in models:
        p = np.where(m.predict(X) == -1, 1, 0)
        preds.append(p)
        gc.collect()
    preds = np.vstack(preds)
    n_models = preds.shape[0]
    if majority_threshold is None:
        majority_threshold = n_models // 2 + 1
    final = (np.sum(preds, axis=0) >= majority_threshold).astype(int)
    return final

