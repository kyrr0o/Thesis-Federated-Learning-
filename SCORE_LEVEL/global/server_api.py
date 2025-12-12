from fastapi import FastAPI, UploadFile, File, Form
from pathlib import Path
import subprocess
import time

app = FastAPI(title="Federated Score-Level Server")

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_CLIENTS = {"client1", "client2", "client3"}
round_state = {}

@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}

@app.post("/upload_scores")
async def upload_scores(
    round_id: str = Form(...),
    client_id: str = Form(...),
    payload: UploadFile = File(...)
):
    out_path = RESULTS_DIR / f"{client_id}_results.json"
    data = await payload.read()
    out_path.write_bytes(data)

    round_state.setdefault(round_id, set()).add(client_id)

    received = sorted(round_state[round_id])
    missing  = sorted(EXPECTED_CLIENTS - set(received))

    print(f"[SERVER] Received {client_id} ({len(data)} bytes)")

    # AUTO AGGREGATE when all clients finished
    if set(received) == EXPECTED_CLIENTS:
        print("[SERVER] All clients received. Running global aggregation...")
        subprocess.run(
            ["python", "global_aggregator.py"],
            cwd=str(BASE_DIR),
            check=False
        )

    return {
        "message": "received",
        "round_id": round_id,
        "received": received,
        "missing": missing
    }
