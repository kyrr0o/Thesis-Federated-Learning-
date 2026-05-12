# tree_aggregator.py
import os
import time
import json
import copy
import gc
import numpy as np
from joblib import load, dump

# ==========================
# SETTINGS
# ==========================
BASE_DIR          = "TREE_LEVEL"
# CLIENT_IDS        = ["client1", "client2", "client3"] ---> static
# Dynamic:
CLIENT_IDS = [
    d for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d))
    and d.startswith("client")
]
ROUND_ID = 1

GLOBAL_ROUNDS_DIR = os.path.join(BASE_DIR, "global", "rounds")
RESULTS_DIR       = os.path.join(BASE_DIR, "global", "results")

MAX_GLOBAL_TREES  = 1500   # cap on number of trees in global forest

os.makedirs(GLOBAL_ROUNDS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==========================
# HELPERS
# ==========================
def client_round_dir(client_id, round_id):
    """
    Directory where a given client stores its forest + meta for a round.
    Expected layout from clientX_train.py:
        TREE_LEVEL/clientX/round_{round_id}/clientX_forest.pkl
        TREE_LEVEL/clientX/round_{round_id}/clientX_meta.json
    """
    return os.path.join(BASE_DIR, client_id, f"round_{round_id}")


def load_client_package(client_id, round_id):
    """
    Load the serialized client forest package (.pkl) and metadata (.json).
    Also estimate communication cost as the forest.pkl size in KB.
    """
    rdir = client_round_dir(client_id, round_id)
    forest_path = os.path.join(rdir, f"{client_id}_forest.pkl")
    meta_path   = os.path.join(rdir, f"{client_id}_meta.json")

    if not os.path.exists(forest_path):
        raise FileNotFoundError(
            f"[ERROR] Forest file not found for {client_id}, round {round_id}: {forest_path}"
        )
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"[ERROR] Meta file not found for {client_id}, round {round_id}: {meta_path}"
        )

    t_deserialize_start = time.time()

    pkg = load(forest_path)

    deserialize_time = time.time() - t_deserialize_start

    with open(meta_path, "r") as f:
        meta = json.load(f)

    comm_cost_kb = os.path.getsize(forest_path) / 1024.0

    return pkg, meta, forest_path, comm_cost_kb, deserialize_time

def sample_trees(all_trees, max_trees, random_state=None):
    """
    Randomly sample at most max_trees from list of trees.
    """
    n = len(all_trees)
    if n <= max_trees:
        return all_trees

    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=max_trees, replace=False)
    return [all_trees[i] for i in idx]

def compute_tree_diversity(trees):
    """
    Measures structural diversity using:
    - tree depth variance
    - feature usage entropy proxy
    """

    depths = []
    feature_counts = {}

    for t in trees:
        tree = t.tree_

        depths.append(tree.max_depth if hasattr(tree, "max_depth") else tree.node_count)

        for f in tree.feature:
            if f >= 0:
                feature_counts[f] = feature_counts.get(f, 0) + 1

    depth_var = float(np.var(depths)) if len(depths) > 1 else 0.0

    total = sum(feature_counts.values()) + 1e-9
    entropy = -sum((v/total) * np.log(v/total) for v in feature_counts.values())

    return {
        "depth_variance": depth_var,
        "feature_entropy": float(entropy),
        "num_unique_features": len(feature_counts)
    }

# ==========================
# MAIN AGGREGATOR
# ==========================

def collect_global_client_evaluations(round_id):
    eval_dir = os.path.join(BASE_DIR, "global", "client_evaluations")

    if not os.path.exists(eval_dir):
        print("[WARN] No global evaluation directory found.")
        return None

    evals = []

    for file in os.listdir(eval_dir):

        if not file.endswith(".json"):
            continue

        if f"round_{round_id}" not in file:
            continue

        path = os.path.join(eval_dir, file)

        with open(path, "r") as f:
            data = json.load(f)

        evals.append(data)

    if len(evals) == 0:
        print("[WARN] No client global evaluations found.")
        return None

    avg_precision = np.mean([e["precision_global"] for e in evals])
    avg_recall = np.mean([e["recall_global"] for e in evals])
    avg_f1 = np.mean([e["f1_global"] for e in evals])

    summary = {
        "round_id": round_id,
        "n_clients": len(evals),
        "avg_precision_global": float(avg_precision),
        "avg_recall_global": float(avg_recall),
        "avg_f1_global": float(avg_f1),
        "client_results": evals
    }

    summary_path = os.path.join(
        RESULTS_DIR,
        f"round_{round_id}_global_eval_summary.json"
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"[INFO] Global evaluation summary saved to {summary_path}")

    return summary

def run_tree_aggregation(round_id=1, random_state=None):
    """
    Aggregate client Isolation Forest trees at the tree level to form
    a global forest for a given round.

    Steps:
      1) Load each client's forest package and metadata.
      2) Collect all trees from sanitized chunk + ensemble models.
      3) Sample up to MAX_GLOBAL_TREES trees.
      4) Build a global IsolationForest using a template from one client.
      5) Compute a GLOBAL THRESHOLD from client-local thresholds.
      6) Save global forest and per-round aggregation statistics.

    NOTE:
    - We aggregate sanitized chunk_models and ensemble_models.
    """
    if random_state is None:
        random_state = np.random.randint(0, 10000)

    print(f"[INFO] Starting tree-level aggregation for round {round_id} (seed={random_state})")

    t_total_start = time.time()
    t_load_start = time.time()

    client_infos = []
    all_trees = []
    template_model = None
    total_comm_kb = 0.0

    # 1) Load client packages
    for cid in CLIENT_IDS:
        try:
            pkg, meta, forest_path, comm_kb, deserialize_time = load_client_package(cid, round_id)
        except FileNotFoundError as e:
            print(f"[WARN] Skipping client {cid} for round {round_id}: {e}")
            continue

        chunk_models    = pkg.get("chunk_models", []) or []
        ensemble_models = pkg.get("ensemble_models", []) or []

        client_models = list(chunk_models) + list(ensemble_models)
        client_trees = []

        for m in client_models:
            if hasattr(m, "estimators_"):
                client_trees.extend(list(m.estimators_))

        n_client_trees = len(client_trees)
        if n_client_trees == 0:
            print(f"[WARN] No trees found for client {cid} in round {round_id}.")
            continue

        all_trees.extend(client_trees)
        total_comm_kb += comm_kb

        if template_model is None and len(client_models) > 0:
            # use the first available model (chunk or ensemble) as template for the global forest
            template_model = client_models[0]

        client_infos.append({
            "client_id": cid,
            "meta": meta,
            "n_trees": n_client_trees,
            "comm_kb": comm_kb,
            "deserialize_time": deserialize_time,
        })

        print(
            f"[INFO] Loaded {n_client_trees} trees from {cid} "
            f"(file ~{comm_kb:.2f} KB)."
        )

    t_load_end = time.time()

    # ==========================
    # GLOBAL METRICS (NOW SAFE)
    # ==========================
    avg_f1 = np.mean([ci["meta"]["f1"] for ci in client_infos])
    avg_precision = np.mean([ci["meta"]["precision"] for ci in client_infos])
    avg_recall = np.mean([ci["meta"]["recall"] for ci in client_infos])

    # ==========================
    # TREE DIVERSITY (NOW SAFE)
    # ==========================
    diversity_metrics = compute_tree_diversity(all_trees)

    if len(client_infos) == 0:
        raise RuntimeError(f"[ERROR] No valid client forests found for round {round_id}.")

    if template_model is None:
        raise RuntimeError("[ERROR] Could not identify a template IsolationForest model.")

    total_trees = len(all_trees)
    print(f"[INFO] Collected total of {total_trees} trees from {len(client_infos)} clients.")

    # 2) Sample trees if needed and build global forest
    t_merge_start = time.time()

    t_sampling_start = time.time()

    sampled_trees = sample_trees(
        all_trees,
        MAX_GLOBAL_TREES,
        random_state=random_state
    )

    sampling_time = time.time() - t_sampling_start

    n_global_trees = len(sampled_trees)

    global_model = copy.deepcopy(template_model)
    global_model.estimators_ = sampled_trees
    global_model.n_estimators = n_global_trees

    t_merge_end = time.time()

    # 3) Save global forest
    round_dir = os.path.join(GLOBAL_ROUNDS_DIR, f"round_{round_id}")
    os.makedirs(round_dir, exist_ok=True)

    global_forest_path = os.path.join(round_dir, "global_forest.pkl")
    dump(global_model, global_forest_path)
    gc.collect()

    t_merge_end = time.time()
    merge_time = t_merge_end - t_merge_start
    total_time = t_merge_end - t_total_start

    print(
        f"[INFO] Global forest for round {round_id} has {n_global_trees} trees "
        f"(from {total_trees} total)."
    )
    print(f"[TIME] Tree merge time: {merge_time:.4f} sec")
    print(f"[TIME] Total aggregation time: {total_time:.4f} sec")
    print(f"[INFO] Global forest saved to {global_forest_path}")

    # 4) Compute GLOBAL THRESHOLD from client-local thresholds (feedback)
    local_thresholds = []
    for ci in client_infos:
        t_loc = ci["meta"].get("threshold_local", None)
        if t_loc is not None:
            local_thresholds.append(float(t_loc))

    if len(local_thresholds) > 0:
        global_threshold = float(np.median(local_thresholds))
        print(f"[INFO] Computed GLOBAL threshold (median of client thresholds): {global_threshold:.6f}")
    else:
        global_threshold = None
        print("[WARN] No local thresholds found; global threshold = None")

    # 5) Save aggregation statistics
    results = {
        "round_id": round_id,
        "n_clients": len(client_infos),
        "total_client_trees": int(total_trees),
        "global_trees": int(n_global_trees),
        "max_global_trees": int(MAX_GLOBAL_TREES),
        "merge_time_seconds": float(merge_time),
        "total_time_seconds": float(total_time),
        "total_comm_kb": float(total_comm_kb),
        "tree_retention_ratio": float(n_global_trees / total_trees),
        "trees_removed": int(total_trees - n_global_trees),
        "tree_diversity": diversity_metrics,
        "overhead_breakdown": {
            "loading_time": float(t_load_end - t_load_start),
            "sampling_time": float(sampling_time),
            "merging_time": float(t_merge_end - t_merge_start),
            "total_time": float(total_time),
},
        # global threshold from client feedback
        "global_threshold_median": float(global_threshold) if global_threshold is not None else None,
        "per_client": [
            {
                "client_id": ci["client_id"],
                "n_trees": int(ci["n_trees"]),
                    "tree_contribution_ratio":
                    float(ci["n_trees"] / total_trees),
                "comm_kb": float(ci["comm_kb"]),
                "f1_local": float(ci["meta"].get("f1", -1.0)),
                "precision_local": float(ci["meta"].get("precision", -1.0)),
                "recall_local": float(ci["meta"].get("recall", -1.0)),
                "threshold_local": float(ci["meta"].get("threshold_local", -1.0)),
                "exec_time_total": float(ci["meta"].get("exec_time_total", -1.0)),
                "deserialize_time_seconds":
                    float(ci["deserialize_time"]),
            }
            for ci in client_infos
        ],
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, f"round_{round_id}_aggregation.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[INFO] Aggregation stats (incl. global threshold) saved to {results_path}")

    return {
        "global_forest_path": global_forest_path,
        "results_path": results_path,
        "summary": results,
    }


if __name__ == "__main__":
    # Example: run for round 2. You can loop this for R=3, etc.
    run_tree_aggregation(round_id=ROUND_ID)
