#FL_globalserver.py

from flask import Flask, request, jsonify, send_file
import os
import pandas as pd
import numpy as np

from SCORE_LEVEL.global_module.global_aggregator import run_global_aggregation_scores

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "..",  # gikan SCORE_LEVEL padulong thesis_code
    "Centralized Isolation Forest",
    "dataset",
    "5m_dataset.csv"
)

# normalize path
DATASET_PATH = os.path.abspath(DATA_PATH)

PARTITION_DIR = os.path.join(BASE_DIR, "partitions")
RESULTS_DIR = os.path.join(BASE_DIR, "global", "results")

EXPECTED_CLIENTS = 1   # change later to 3

received_clients = set()


def partition_dataset():

    if not os.path.exists(PARTITION_DIR):
        os.makedirs(PARTITION_DIR)

    df = pd.read_csv(DATASET_PATH)

    parts = np.array_split(df, EXPECTED_CLIENTS)
    for i, part in enumerate(parts):
        part = pd.DataFrame(part)  # force as DataFrame
        path = f"{PARTITION_DIR}/client{i+1}.csv"
        part.to_csv(path, index=False)

    print("[SERVER] Dataset partitioned")
    

@app.route("/get_partition", methods=["POST"])
def get_partition():
    data = request.get_json(silent=True)  # safe: returns None if not JSON
    if not data or "client_id" not in data:
        return jsonify({"error": "client_id missing or invalid JSON"}), 400

    client_id = data["client_id"]
    path = os.path.join(PARTITION_DIR, f"{client_id}.csv")

    if not os.path.exists(path):
        return jsonify({"error": "partition not found"}), 404

    return send_file(path, as_attachment=True)


@app.route("/upload_scores", methods=["POST"])
def upload_scores():

    client_id = request.form["client_id"]

    file = request.files["payload"]

    os.makedirs(RESULTS_DIR, exist_ok=True)

    save_path = os.path.join(RESULTS_DIR, f"{client_id}.json")

    file.save(save_path)

    received_clients.add(client_id)

    print("[SERVER] Received results from", client_id)

    if len(received_clients) == EXPECTED_CLIENTS:

        print("[SERVER] All clients finished")

        run_global_aggregation_scores()

        received_clients.clear()

    return jsonify({"status": "received"})


if __name__ == "__main__":

    partition_dataset()

    app.run(host="0.0.0.0", port=5000)