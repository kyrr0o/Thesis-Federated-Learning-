# FL_globalserver_tree.py

from flask import Flask, request, jsonify, send_file
import os
import pandas as pd
import numpy as np

from TREE_LEVEL.global_module_tree.tree_aggregator import run_tree_aggregation
from TREE_LEVEL.client1.client1_train import eval_global_on_client

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "Centralized Isolation Forest",
    "dataset",
    "5m_dataset.csv"
)

DATASET_PATH = os.path.abspath(DATA_PATH)

PARTITION_DIR = os.path.join(BASE_DIR, "partitions")

EXPECTED_CLIENTS = 1   # change later to 3
ROUND_ID = 1

received_clients = set()


def partition_dataset():

    if not os.path.exists(PARTITION_DIR):
        os.makedirs(PARTITION_DIR)

    df = pd.read_csv(DATASET_PATH)

    parts = np.array_split(df, EXPECTED_CLIENTS)

    for i, part in enumerate(parts):

        part = pd.DataFrame(part)

        path = f"{PARTITION_DIR}/client{i+1}.csv"

        part.to_csv(path, index=False)

    print("[SERVER] Dataset partitioned")


@app.route("/get_partition", methods=["POST"])
def get_partition():

    data = request.get_json(silent=True)

    if not data or "client_id" not in data:
        return jsonify({"error": "client_id missing"}), 400

    client_id = data["client_id"]

    path = os.path.join(PARTITION_DIR, f"{client_id}.csv")

    if not os.path.exists(path):
        return jsonify({"error": "partition not found"}), 404

    return send_file(path, as_attachment=True)


@app.route("/upload_forest", methods=["POST"])
def upload_forest():

    client_id = request.form["client_id"]

    forest_file = request.files["forest"]
    meta_file = request.files["meta"]

    client_dir = os.path.join(
        BASE_DIR,
        client_id,
        f"round_{ROUND_ID}"
    )

    os.makedirs(client_dir, exist_ok=True)

    forest_path = os.path.join(client_dir, f"{client_id}_forest.pkl")
    meta_path = os.path.join(client_dir, f"{client_id}_meta.json")

    forest_file.save(forest_path)
    meta_file.save(meta_path)

    print("[SERVER] Received forest from", client_id)

    received_clients.add(client_id)

    if len(received_clients) == EXPECTED_CLIENTS:

        print("[SERVER] All clients finished. Running tree aggregation...")

        run_tree_aggregation(round_id=ROUND_ID)

        print("[SERVER] Running GLOBAL evaluation for all clients...")

        for cid in list(received_clients):

            try:
                if cid == "client1":
                    from TREE_LEVEL.client1.client1_train import eval_global_on_client
                    eval_global_on_client(round_id=ROUND_ID)

                # future:
                # elif cid == "client2":
                #     from TREE_LEVEL.client2.client2_train import eval_global_on_client

            except Exception as e:
                print(f"[ERROR] Global eval failed for {cid}: {e}")

        received_clients.clear()

        print("[SERVER] Round complete ✔")

    return jsonify({"status": "received"})


if __name__ == "__main__":

    partition_dataset()

    app.run(
        host="0.0.0.0",
        port=5000,
        ssl_context=("cert.pem", "key.pem")
)