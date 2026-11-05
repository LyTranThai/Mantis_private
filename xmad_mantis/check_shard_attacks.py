import pandas as pd
import json

df = pd.read_parquet("https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/XMAD/resolve/main/data/labels.parquet")
rows_per_shard = len(df) / 200.0

# Check shard 54 and 55 index ranges
shard_54_start = int(54 * rows_per_shard)
shard_56_end = int(56 * rows_per_shard)

sub_df = df.iloc[shard_54_start:shard_56_end]
print(f"Total rows in shards 54-55: {len(sub_df)}")
print("Sample utterance_ids:")
print(sub_df['utterance_id'].head(10).tolist())
