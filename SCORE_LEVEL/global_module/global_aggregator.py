# global_aggregator.py
import os
import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# settings
CLIENT_RESULTS_DIR = "SCORE_LEVEL/global/results"
GLOBAL_RESULTS_DIR = "SCORE_LEVEL/global/results/global"

# helper functions
def safe_load_array(obj, key):
    if key in obj:
        arr = np.array(obj[key])
        return arr
    return None

def zscore_safe(s):
    s = np.asarray(s, dtype=np.float64)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    mean = np.mean(s)
    std  = np.std(s)
    if std == 0:
        return s - mean
    return (s - mean) / std

def calibrate_threshold_from_scores(scores, y_true, n_steps=200):
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)

    if len(scores) == 0:
        return 0.0

    p1  = np.percentile(scores, 1)
    p99 = np.percentile(scores, 99)

    if p99 <= p1:
        coarse_thresh = np.linspace(p1 - 1e-6, p1 + 1e-6, n_steps)
    else:
        coarse_thresh = np.linspace(p1, p99, n_steps)

    best_f1 = -1.0
    best_t  = coarse_thresh[0]
    for t in coarse_thresh:
        preds = (scores < t).astype(int)
        f1    = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t
    return best_t

def evaluate_model_from_scores(scores, y_true, threshold):
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)
    preds  = (scores < threshold).astype(int)
    p = precision_score(y_true, preds, zero_division=0)
    r = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    return p, r, f1, preds

def save_results(output_path, results_dict):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=4)

# main aggregator
def run_global_aggregation_scores():
    client_infos = []

    # load all client JSONs
    for fname in sorted(os.listdir(CLIENT_RESULTS_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(CLIENT_RESULTS_DIR, fname)
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not read {path}: {e}")
            continue

        # new fields (preferred)
        val_scores  = safe_load_array(data, "val_scores")
        y_val       = safe_load_array(data, "y_val")
        test_scores = safe_load_array(data, "test_scores")

        # backward compatibility
        if test_scores is None:
            test_scores = safe_load_array(data, "scores")

        y_test = safe_load_array(data, "y_test")
        if y_test is None:
            y_test = safe_load_array(data, "y_true")
        if y_test is None:
            y_test = safe_load_array(data, "labels")

        exec_time = data.get("exec_time_total") or data.get("exec_time") or data.get("runtime_seconds")

        # skip clients with no scores at all
        if test_scores is None and val_scores is None:
            print(f"[WARN] {fname} has no score fields -> skipping")
            continue

        client_infos.append({
            "fname":       fname,
            "val_scores":  val_scores,
            "y_val":       y_val,
            "test_scores": test_scores,
            "y_test":      y_test,
            "exec_time":   float(exec_time) if exec_time is not None else None
        })

    if len(client_infos) == 0:
        raise RuntimeError(f"No valid client JSON files found in {CLIENT_RESULTS_DIR}")

    # prepare global validation and test sets (with per-client normalization)
    val_scores_list  = []
    val_labels_list  = []
    test_scores_list = []
    test_labels_list = []

    for c in client_infos:
        if c["val_scores"] is not None and c["y_val"] is not None:
            val_scores_list.append(zscore_safe(c["val_scores"]))
            val_labels_list.append(np.asarray(c["y_val"], dtype=int))

        if c["test_scores"] is not None and c["y_test"] is not None:
            test_scores_list.append(zscore_safe(c["test_scores"]))
            test_labels_list.append(np.asarray(c["y_test"], dtype=int))

    if len(val_scores_list) == 0:
        raise RuntimeError(
            "No client provides validation scores/labels. "
            "Make sure each client saves 'val_scores' and 'y_val'."
        )

    if len(test_scores_list) == 0:
        raise RuntimeError(
            "No client provides test scores/labels. "
            "Make sure each client saves 'test_scores' and 'y_test'."
        )

    global_val_scores  = np.concatenate(val_scores_list)
    global_val_labels  = np.concatenate(val_labels_list)
    global_test_scores = np.concatenate(test_scores_list)
    global_test_labels = np.concatenate(test_labels_list)

    # Calibrate global threshold on validation
    thresh = calibrate_threshold_from_scores(global_val_scores, global_val_labels)

    # Evaluate on independent global test
    p_global, r_global, f1_global, global_preds = evaluate_model_from_scores(
        global_test_scores, global_test_labels, thresh
    )

    # Save global results
    results = {
        "global_evaluation": {
            "precision":      float(p_global),
            "recall":         float(r_global),
            "f1":             float(f1_global),
            "threshold":      float(thresh),
            "n_val_samples":  int(len(global_val_labels)),
            "n_test_samples": int(len(global_test_labels))
        },
        "per_client_summary": [
            {
                "client_file": c["fname"],
                "has_val":     c["val_scores"] is not None and c["y_val"] is not None,
                "has_test":    c["test_scores"] is not None and c["y_test"] is not None,
                "exec_time":   c["exec_time"]
            } for c in client_infos
        ]
    }

    # include global scores/preds if not too huge
    MAX_SAVE_SAMPLES = 5_000_000
    if len(global_test_scores) <= MAX_SAVE_SAMPLES:
        results["global_test_scores"] = global_test_scores.tolist()
        results["global_test_preds"]  = global_preds.tolist()
        results["global_test_labels"] = global_test_labels.tolist()
    else:
        results["global_test_scores"] = f"too_many_samples({len(global_test_scores)})"
        results["global_test_preds"]  = f"too_many_samples({len(global_preds)})"

    save_results(os.path.join(GLOBAL_RESULTS_DIR, "global_results.json"), results)
    print(f"[GLOBAL RESULTS] Precision={p_global:.4f}, Recall={r_global:.4f}, F1={f1_global:.4f}, Threshold={thresh:.5f}")
    print(f"[INFO] Global results saved in {GLOBAL_RESULTS_DIR}")

    return results

if __name__ == "__main__":
    run_global_aggregation_scores()
