import pandas as pd

input_file = "new_dataset/BTC30Min.csv"
output_file = "new_dataset/dataset/BTC30Min.csv"

N = 72000  

df = pd.read_csv(input_file)
subset = df.sample(n=N, random_state=42) 

subset.to_csv(output_file, index=False)

print(f"Saved {N} random rows to {output_file}")
