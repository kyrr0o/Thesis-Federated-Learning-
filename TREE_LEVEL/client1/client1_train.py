# clientX_train.py
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
from joblib import Parallel, delayed, dump, load  # <-- added load

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================
# SETTINGS (ADJUST PER CLIENT)
# ==========================
CLIENT_ID       = "client1"
DATA_PATH       = "TREE_LEVEL/client1/BTC15Min.csv"

BASE_DIR        = "TREE_LEVEL"
ROUNDS_DIR      = os.path.join(BASE_DIR, CLIENT_ID)

# For feedback: match aggregator paths
GLOBAL_ROUNDS_DIR  = os.path.join(BASE_DIR, "global", "rounds")
RESULTS_DIR        = os.path.join(BASE_DIR, "global", "results")

# Shared scaler path (for consistent feature scaling across clients)
GLOBAL_SCALER_DIR  = os.path.join(BASE_DIR, "global")
GLOBAL_SCALER_PATH = os.path.join(GLOBAL_SCALER_DIR, "global_scaler.npz")
os.makedirs(GLOBAL_SCALER_DIR, exist_ok=True)

ANOMALY_RATE   = 0.05
N_MODELS       = 3          # inter-model ensemble size
CHUNKS         = 6          # intra-model chunks

# Base seed per client (change per client script if you want)
CLIENT_BASE_SEED = 12345

# IMPORTANT:
# We use joblib.Parallel at the outer level, and keep each
# IsolationForest single-threaded (n_jobs=1) to avoid nested parallelism.
BEST_PARAMS = {
    "n_estimators": 100,
    "max_samples": 0.1,
    "max_features": 0.5,
    "contamination": 0.02,
    #"bootstrap": False,
    # DO NOT set n_jobs here (we control it explicitly)
}

# ==========================
# UTILS
# ==========================
def choose_outer_jobs(preferred=-1, cpu_safety_frac=0.5, mem_threshold_pct=80):
    """
    Decide how many jobs to use at the OUTER level (joblib.Parallel).
    Inner IsolationForest models are forced to n_jobs=1 to avoid
    nested parallelism / oversubscription.
    """
    cpu_count = os.cpu_count() or 1
    cpu_load = psutil.cpu_percent(interval=0.5)
    mem_pct = psutil.virtual_memory().percent

    if preferred == -1:
        # auto
        if cpu_load > 70 or mem_pct > mem_threshold_pct:
            n_jobs = max(1, int(cpu_count * cpu_safety_frac))
        else:
            n_jobs = cpu_count
    else:
        n_jobs = min(preferred, cpu_count)
        if cpu_load > 85 or mem_pct > mem_threshold_pct:
            n_jobs = max(1, int(cpu_count * cpu_safety_frac))

    return max(1, n_jobs)


def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print(f"[INFO] Dataset loaded with shape {df.shape}")
    return df


def inject_extreme_anomalies(df, rate=0.02, rng=None):
    """
    Intraday-friendly anomaly injection:
    - adaptive per feature using its own std
    - correlated OHLC scaling
    - volume spike

    Uses a local RNG for reproducibility across rounds.
    """
    df = df.copy()
    if rng is None:
        rng = np.random.default_rng()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'datetime']

    n_anomalies = int(len(df) * rate)
    if n_anomalies <= 0:
        df['is_anomaly'] = 0
        print("[WARN] n_anomalies computed as 0, skipping injection.")
        return df

    anomaly_idx = rng.choice(df.index, n_anomalies, replace=False)

    # 1) base adaptive perturbation
    for col in numeric_cols:
        std = df[col].std()
        if std > 0:
            noise = rng.normal(size=n_anomalies)
            scale = rng.uniform(1.5, 2.5)
            df.loc[anomaly_idx, col] = (
                df.loc[anomaly_idx, col] + noise * std * scale
            )

    # 2) correlated OHLC anomaly
    ohlc_cols = ['open', 'high', 'low', 'close']
    if all(c in df.columns for c in ohlc_cols):
        scale = rng.uniform(1.5, 2.5)
        df.loc[anomaly_idx, ohlc_cols] *= scale

    # 3) volume spike
    if 'volume' in df.columns:
        vscale = rng.uniform(1.5, 3.0)
        df.loc[anomaly_idx, 'volume'] *= vscale

    df['is_anomaly'] = 0
    df.loc[anomaly_idx, 'is_anomaly'] = 1

    print(f"[INFO] Injected {n_anomalies} anomalies in numeric cols: {numeric_cols}")
    return df


def extract_features(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df.fillna(df.mean(numeric_only=True), inplace=True)
    print(f"[INFO] Using {len(numeric_cols)} numeric columns: {numeric_cols}")
    return df


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
    print(
        f"[INFO] Train anomalies: {train['is_anomaly'].sum()}, "
        f"Val anomalies: {val['is_anomaly'].sum()}, "
        f"Test anomalies: {test['is_anomaly'].sum()}"
    )

    return train, val, test


def preprocess(train, val, test, use_global_scaler=True):
    """
    Preprocess with a shared global scaler if available.
    - First run can create GLOBAL_SCALER_PATH.
    - Later runs/clients load the same mean/scale -> consistent feature space.
    """
    feature_cols = [
        c for c in train.columns
        if c not in ('is_anomaly', 'datetime')
    ]

    print(f"[INFO] Preprocess using {len(feature_cols)} feature columns: {feature_cols}")

    if use_global_scaler and os.path.exists(GLOBAL_SCALER_PATH):
        data = np.load(GLOBAL_SCALER_PATH)
        mean_ = data["mean"]
        scale_ = data["scale"]

        if mean_.shape[0] != len(feature_cols):
            print("[WARN] GLOBAL scaler dimension mismatch; refitting locally instead.")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train[feature_cols]).astype(np.float32)
            X_val   = scaler.transform(val[feature_cols]).astype(np.float32)
            X_test  = scaler.transform(test[feature_cols]).astype(np.float32)
        else:
            scaler = StandardScaler()
            scaler.mean_ = mean_
            scaler.scale_ = scale_
            scaler.var_ = scale_ ** 2
            scaler.n_features_in_ = len(feature_cols)
            print("[INFO] Loaded GLOBAL scaler parameters.")

            X_train = ((train[feature_cols].values - mean_) / scale_).astype(np.float32)
            X_val   = ((val[feature_cols].values   - mean_) / scale_).astype(np.float32)
            X_test  = ((test[feature_cols].values  - mean_) / scale_).astype(np.float32)
    else:
        # Local fit, and optionally create global scaler if allowed
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[feature_cols]).astype(np.float32)
        X_val   = scaler.transform(val[feature_cols]).astype(np.float32)
        X_test  = scaler.transform(test[feature_cols]).astype(np.float32)

        if use_global_scaler and not os.path.exists(GLOBAL_SCALER_PATH):
            np.savez(GLOBAL_SCALER_PATH, mean=scaler.mean_, scale=scaler.scale_)
            print(f"[INFO] Saved GLOBAL scaler parameters to {GLOBAL_SCALER_PATH}")
        elif not use_global_scaler:
            print("[INFO] Using client-local scaler only.")

    y_train = train['is_anomaly'].values
    y_val   = val['is_anomaly'].values
    y_test  = test['is_anomaly'].values

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


# ==========================
# DUAL-LEVEL PARALLEL TRAINING
# ==========================
def train_single_chunk(subX, base_params, seed):
    """
    Train one IsolationForest on a chunk.
    IMPORTANT: n_jobs=1 here to avoid nested parallelism,
    because joblib.Parallel handles inter-chunk parallelism.
    """
    params = base_params.copy()
    base_n_estimators = int(params.get("n_estimators", 100))

    params["n_estimators"] = max(10, base_n_estimators // CHUNKS)
    params["max_samples"]  = min(params.get("max_samples", 0.1), 0.05)
    params["bootstrap"]    = False
    params["n_jobs"]       = 1            # single-threaded per chunk
    params["random_state"] = seed

    model = IsolationForest(**params)
    model.fit(subX.astype(np.float32))
    return model


def train_iforest_in_chunks(X, base_params, n_chunks=6, random_state=None):
    """
    Level 1: intra-model parallelism.
    We split data into chunks and train smaller forests in parallel,
    using joblib for outer parallelism and n_jobs=1 inside each IF.
    """
    if random_state is None:
        random_state = np.random.randint(0, 10000)

    X_local = X.astype(np.float32)
    chunks  = np.array_split(X_local, n_chunks)
    seeds   = [(random_state + i * 97) % 2**31 for i in range(n_chunks)]

    outer_jobs = min(n_chunks, choose_outer_jobs(-1))
    print(f"[INFO] Intra-model Parallel: {n_chunks} chunks, outer_jobs={outer_jobs}")

    models_parallel = Parallel(n_jobs=outer_jobs)(
        delayed(train_single_chunk)(chunks[i], base_params, seeds[i])
        for i in range(n_chunks)
    )

    gc.collect()
    return models_parallel


def ensemble_iforest_parallel(X_train, X_test, params, n_models=3, random_state=None):
    """
    Level 2: inter-model parallelism across multiple full IF models.
    Again, outer joblib parallel, inner IF single-threaded.
    """
    if random_state is None:
        random_state = np.random.randint(0, 10000)

    rng = np.random.default_rng(random_state)
    outer_jobs = min(n_models, choose_outer_jobs(-1))
    print(f"[INFO] Inter-model Parallel: {n_models} models, outer_jobs={outer_jobs}")

    def train_one(seed):
        model = IsolationForest(
            **params,
            random_state=int(seed),
            n_jobs=1  # single-threaded inside each ensemble IF
        )
        model.fit(X_train.astype(np.float32))
        scores = model.score_samples(X_test.astype(np.float32))
        preds  = (scores < 0).astype(int)  # placeholder for local eval
        gc.collect()
        return model, scores, preds

    seeds = rng.integers(0, 2**31 - 1, size=n_models)
    results = Parallel(n_jobs=outer_jobs)(
        delayed(train_one)(s) for s in seeds
    )
    models, score_list, preds_list = zip(*results)
    score_matrix = np.vstack(score_list)
    preds_matrix = np.vstack(preds_list)
    return list(models), score_matrix, preds_matrix


def calibrate_threshold_from_scores(scores, y_val):
    scores = np.asarray(scores).ravel()
    y_val  = np.asarray(y_val)

    coarse_thresh = np.linspace(
        np.percentile(scores, 1),
        np.percentile(scores, 99),
        200
    )

    best_f1, best_t = -1, None
    for t in coarse_thresh:
        preds = (scores < t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def evaluate_model_from_scores(scores, y_true, threshold):
    scores = np.asarray(scores).ravel()
    y_true = np.asarray(y_true)
    preds  = (scores < threshold).astype(int)
    p = precision_score(y_true, preds, zero_division=0)
    r = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    return p, r, f1, preds


def sanitize_iforest_models(models, threshold_round=2):
    """
    Privacy-friendly sanitation for IsolationForest models:
    - remove impurity info
    - round split thresholds (less precise distribution leakage)
    - drop feature_importances_
    NOTE: we avoid touching node sample counts / values to keep
    scikit-learn's anomaly scoring intact.
    """
    sanitized = []
    for m in models:
        if not hasattr(m, "estimators_"):
            continue
        for est in m.estimators_:
            tree = est.tree_
            # zero out impurity array if exists
            if hasattr(tree, "impurity"):
                tree.impurity[:] = 0.0
            # round thresholds to reduce precision
            if hasattr(tree, "threshold"):
                tree.threshold[:] = np.round(tree.threshold, threshold_round)
        if hasattr(m, "feature_importances_"):
            m.feature_importances_ = None
        sanitized.append(m)
    return sanitized

# ==========================
# SERVER FEEDBACK LOADER (GLOBAL → CLIENT)
# ==========================
def load_global_feedback(round_id):
    """
    Load server-side feedback (e.g., global threshold) from previous round.
    For round 1, there is no previous global feedback, so this returns None.
    """
    if round_id <= 1:
        return None  # no previous global round

    global_results_dir = os.path.join(BASE_DIR, "global", "results")
    agg_path = os.path.join(global_results_dir, f"round_{round_id - 1}_aggregation.json")

    if not os.path.exists(agg_path):
        print(f"[WARN] No global aggregation file found for round {round_id - 1}: {agg_path}")
        return None

    with open(agg_path, "r") as f:
        agg = json.load(f)

    global_thr = agg.get("global_threshold_median", None)
    if global_thr is None:
        print(f"[WARN] Global aggregation for round {round_id - 1} has no global_threshold_median.")
        return None

    print(f"[INFO] Loaded GLOBAL threshold from round {round_id - 1}: {global_thr:.6f}")
    return float(global_thr)

# ==========================
# MAIN CLIENT TRAINING (TREE-LEVEL)
# ==========================
def run_client(round_id=1, random_state=None):
    """
    Train dual-level parallel Isolation Forest on local data,
    evaluate locally, and serialize the client forest + metadata
    for tree-level federated aggregation.
    """
    # Separate seeds for data (anomalies/split) vs models (stochastic IF)
    seed_data = CLIENT_BASE_SEED
    seed_model = CLIENT_BASE_SEED + 1000 * (round_id - 1)

    if random_state is not None:
        # If user passes explicit seed, override both
        seed_data = random_state
        seed_model = random_state + 1000

    print(
        f"[INFO] Starting client {CLIENT_ID} for round {round_id} "
        f"with data_seed={seed_data}, model_seed={seed_model}"
    )
    start_total = time.time()

    # --- LOAD SERVER FEEDBACK (GLOBAL THRESHOLD FROM PREVIOUS ROUND) ---
    global_threshold = load_global_feedback(round_id)

    # --- DATA LOADING ---
    t0 = time.time()
    df = load_data(DATA_PATH)
    print(f"[TIME] Load data: {time.time() - t0:.4f} sec")

    # --- ANOMALY INJECTION (CONSISTENT PER CLIENT) ---
    t0 = time.time()
    rng_data = np.random.default_rng(seed_data)
    df = inject_extreme_anomalies(df, rate=ANOMALY_RATE, rng=rng_data)
    print(f"[TIME] Inject anomalies: {time.time() - t0:.4f} sec")

    # --- FEATURE EXTRACTION ---
    t0 = time.time()
    df = extract_features(df)
    print(f"[TIME] Extract features: {time.time() - t0:.4f} sec")

    # --- SPLITTING (CONSISTENT PER CLIENT) ---
    t0 = time.time()
    train_df, val_df, test_df = split_data(df, random_state=seed_data)
    print(f"[TIME] Split data: {time.time() - t0:.4f} sec")
    del df
    gc.collect()

    # --- PREPROCESSING WITH GLOBAL SCALER ---
    t0 = time.time()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess(
        train_df, val_df, test_df, use_global_scaler=True
    )
    print(f"[TIME] Preprocess: {time.time() - t0:.4f} sec")
    del train_df, val_df, test_df
    gc.collect()

    print("[INFO] Using fixed parameters:", BEST_PARAMS)

    # --- INTRA-MODEL PARALLEL (CHUNKS) ---
    t0 = time.time()
    models_parallel = train_iforest_in_chunks(
        X_train, BEST_PARAMS, n_chunks=CHUNKS, random_state=seed_model
    )
    print(f"[TIME] Intra-model parallel (chunks): {time.time() - t0:.4f} sec")

    # --- INTER-MODEL PARALLEL (ENSEMBLE) ---
    t0 = time.time()
    ensemble_models, _, _ = ensemble_iforest_parallel(
        X_train, X_test, BEST_PARAMS, n_models=N_MODELS, random_state=seed_model + 999
    )
    print(f"[TIME] Inter-model parallel (ensemble): {time.time() - t0:.4f} sec")

    # --- MERGE BOTH LEVELS FOR SCORING ---
    all_models = list(models_parallel) + list(ensemble_models)
    print(f"[INFO] Total local models (chunks + ensemble): {len(all_models)}")

    # --- THRESHOLD CALIBRATION (LOCAL) ---
    t0 = time.time()
    val_scores_list = []
    for m in all_models:
        s_val = m.score_samples(X_val.astype(np.float32))
        val_scores_list.append(s_val)
    val_scores = np.mean(np.vstack(val_scores_list), axis=0)
    threshold = calibrate_threshold_from_scores(val_scores, y_val)
    print(f"[TIME] Threshold calibration: {time.time() - t0:.4f} sec")

    # --- LOCAL EVALUATION (ALL MODELS) ---
    t0 = time.time()
    test_scores_list = []
    for m in all_models:
        s_test = m.score_samples(X_test.astype(np.float32))
        test_scores_list.append(s_test)
    test_scores_agg = np.mean(np.vstack(test_scores_list), axis=0)

    p, r, f1, preds_test = evaluate_model_from_scores(test_scores_agg, y_test, threshold)
    print(f"[TIME] Evaluation: {time.time() - t0:.4f} sec")
    print(
        f"[RESULTS] Client={CLIENT_ID}, "
        f"Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}, Threshold={threshold:.5f}"
    )

    end_total = time.time()
    exec_time_total = end_total - start_total
    print(f"[TIME] Total runtime: {exec_time_total:.4f} sec")

    # --- OPTIONAL: EVALUATION USING GLOBAL THRESHOLD (SERVER FEEDBACK) ---
    if global_threshold is not None:
        t0 = time.time()
        p_g, r_g, f1_g, _ = evaluate_model_from_scores(
            test_scores_agg, y_test, global_threshold
        )
        print(f"[TIME] Global-threshold evaluation: {time.time() - t0:.4f} sec")
        print(
            f"[RESULTS-GLOBAL] Client={CLIENT_ID}, "
            f"Precision={p_g:.4f}, Recall={r_g:.4f}, F1={f1_g:.4f}, "
            f"GlobalThreshold={global_threshold:.5f}"
        )
    else:
        p_g = r_g = f1_g = -1.0

    # --- SAVE TEST DATA FOR CONSISTENT GLOBAL EVAL LATER ---
    test_round_dir = os.path.join(ROUNDS_DIR, f"round_{round_id}")
    os.makedirs(test_round_dir, exist_ok=True)
    test_data_path = os.path.join(test_round_dir, f"{CLIENT_ID}_test_data.npz")
    np.savez_compressed(test_data_path, X_test=X_test, y_test=y_test)

    # sanitize models before sending to global (both levels)
    sanitized_chunk_models    = sanitize_iforest_models(models_parallel, threshold_round=2)
    sanitized_ensemble_models = sanitize_iforest_models(ensemble_models, threshold_round=2)

    # ==========================
    # SERIALIZE FOREST + META FOR TREE-LEVEL AGGREGATION
    # ==========================
    round_dir = os.path.join(ROUNDS_DIR, f"round_{round_id}")
    os.makedirs(round_dir, exist_ok=True)

    forest_path = os.path.join(round_dir, f"{CLIENT_ID}_forest.pkl")
    meta_path   = os.path.join(round_dir, f"{CLIENT_ID}_meta.json")

    client_forest_package = {
        "client_id": CLIENT_ID,
        "round_id": round_id,
        "best_params": BEST_PARAMS,
        # chunk_models now included & used in FL tree aggregation
        "chunk_models": sanitized_chunk_models,
        "ensemble_models": sanitized_ensemble_models,
    }

    dump(client_forest_package, forest_path)

    n_trees_chunks = sum(len(m.estimators_) for m in sanitized_chunk_models if hasattr(m, "estimators_"))
    n_trees_ens    = sum(len(m.estimators_) for m in sanitized_ensemble_models if hasattr(m, "estimators_"))

    meta = {
        "client_id": CLIENT_ID,
        "round_id": round_id,
        "n_models_ensemble": len(ensemble_models),
        "n_models_chunks": len(models_parallel),
        "n_trees_chunks": int(n_trees_chunks),
        "n_trees_ensemble": int(n_trees_ens),
        "n_estimators_per_model": BEST_PARAMS["n_estimators"],
        "anomaly_rate_injection": ANOMALY_RATE,
        # local metrics (client's own threshold)
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "threshold_local": float(threshold),
        # global-feedback-based metrics (if available; else -1)
        "precision_global": float(p_g),
        "recall_global": float(r_g),
        "f1_global": float(f1_g),
        "global_threshold_used": float(global_threshold) if global_threshold is not None else None,
        # runtime + data sizes
        "exec_time_total": float(exec_time_total),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        # to allow consistent re-use if needed
        "random_state_data": int(seed_data),
        "random_state_model": int(seed_model),
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)

    print(f"[INFO] Serialized client forest saved to {forest_path}")
    print(f"[INFO] Metadata saved to {meta_path}")
    print(f"[INFO] Test data cached at {test_data_path}")

    return {
        "forest_path": forest_path,
        "meta_path": meta_path,
        "metrics": meta,
        "random_state_data": seed_data,
        "random_state_model": seed_model,
    }

# ==========================
# FEEDBACK LOOP: EVAL GLOBAL MODEL ON THIS CLIENT
# ==========================
def _load_global_forest(round_id):
    round_dir = os.path.join(GLOBAL_ROUNDS_DIR, f"round_{round_id}")
    model_path = os.path.join(round_dir, "global_forest.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Global forest not found: {model_path}")
    model = load(model_path)
    return model


def _load_global_threshold(round_id):
    stats_path = os.path.join(RESULTS_DIR, f"round_{round_id}_aggregation.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"[ERROR] Aggregation stats not found: {stats_path}")
    with open(stats_path, "r") as f:
        stats = json.load(f)
    return stats.get("global_threshold_median", None)


def _build_local_test_data(round_id):
    """
    Reuse the exact local test set (X_test, y_test) that was cached
    during run_client(...) for this round. This avoids inconsistencies
    in splits and scaling when evaluating the global model.
    """
    test_round_dir = os.path.join(ROUNDS_DIR, f"round_{round_id}")
    test_data_path = os.path.join(test_round_dir, f"{CLIENT_ID}_test_data.npz")

    if not os.path.exists(test_data_path):
        raise FileNotFoundError(
            f"[ERROR] Cached test data not found for {CLIENT_ID}, round {round_id}: {test_data_path}"
        )

    data = np.load(test_data_path)
    X_test = data["X_test"]
    y_test = data["y_test"]
    return X_test, y_test


def eval_global_on_client(round_id=1, random_state=None):
    """
    Feedback loop:
    - Load global forest and global threshold from server aggregation.
    - Evaluate the global model on this client's local test set.
    NOTE:
    random_state is kept for API compatibility but test data is now
    loaded from the cached arrays saved during run_client(...).
    """
    if random_state is None:
        random_state = 448  # unused if cached test data exists

    print(f"[INFO] Evaluating GLOBAL model on {CLIENT_ID}, round {round_id}, seed={random_state}")

    global_model = _load_global_forest(round_id)
    global_threshold = _load_global_threshold(round_id)

    if global_threshold is None:
        print("[WARN] Global threshold is None; using 0.0 as fallback.")
        global_threshold = 0.0

    X_test, y_test = _build_local_test_data(round_id=round_id)
    scores = global_model.score_samples(X_test.astype(np.float32))

    p, r, f1, preds = evaluate_model_from_scores(scores, y_test, global_threshold)

    print(
        f"[GLOBAL RESULTS] Client={CLIENT_ID}, "
        f"Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}, "
        f"GlobalThreshold={global_threshold:.6f}"
    )

    return {
        "client_id": CLIENT_ID,
        "round_id": round_id,
        "precision_global": float(p),
        "recall_global": float(r),
        "f1_global": float(f1),
        "global_threshold": float(global_threshold),
    }


if __name__ == "__main__":
    # Example run; adjust round_id as needed
    run_client(round_id=1)
    # After tree_aggregator for round 2, you can call:
    # eval_global_on_client(round_id=2)
