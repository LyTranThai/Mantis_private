import pandas as pd
from datasets import load_dataset

print("Checking labels.parquet...")
try:
    df = pd.read_parquet("https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/XMAD/resolve/main/data/labels.parquet")
    print(f"Shape of labels.parquet: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("Head:\n", df.head(5))
    print("\nValue counts of labels:")
    if "label" in df.columns:
        print(df["label"].value_counts())
    elif "label_id" in df.columns:
        print(df["label_id"].value_counts())
    print("\nSample paths:")
    if "path" in df.columns:
        print(df["path"].head(5))
except Exception as e:
    print(f"Error reading labels.parquet: {e}")
    import traceback
    traceback.print_exc()
