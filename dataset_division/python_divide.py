import pandas as pd
import numpy as np
import os

DATA_PATH = "Centralized Isolation Forest/dataset/5m_dataset.csv"
OUTPUT_DIR = "dataset_division/FL_Datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[INFO] Loading dataset...")
df = pd.read_csv(DATA_PATH)

print(f"[INFO] Dataset loaded: {df.shape}")

# Shuffle rows to avoid biased splits
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

NUM_CLIENTS = 3

subset_size = len(df) // NUM_CLIENTS

subsets = []
for i in range(NUM_CLIENTS):
    start = i * subset_size
    end = (i + 1) * subset_size if i < NUM_CLIENTS - 1 else len(df)
    subset = df.iloc[start:end]
    subsets.append(subset)

    out_path = os.path.join(OUTPUT_DIR, f"client{i+1}.csv")
    subset.to_csv(out_path, index=False)
    print(f"[INFO] Saved subset for client {i+1}: {subset.shape} → {out_path}")

print("\n[INFO] Done.")
