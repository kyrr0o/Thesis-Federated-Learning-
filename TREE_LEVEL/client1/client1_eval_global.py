# TREE_LEVEL/client1/client1_eval_global.py

from client1_train import eval_global_on_client

if __name__ == "__main__":
    # IMPORTANT:
    # Gamita ang SAME seed nga nakita nimo sa training logs:
    # [INFO] Starting client client1 for round 1 with seed=4798
    eval_global_on_client(round_id=1, random_state=4798)
