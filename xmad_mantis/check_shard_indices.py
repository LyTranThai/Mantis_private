import pandas as pd

df = pd.read_parquet("https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/XMAD/resolve/main/data/labels.parquet")
# 200 shards
rows_per_shard = len(df) / 200.0
print(f"Approx rows per shard: {rows_per_shard:.1f}")

# Shard for index 0 (fake)
shard_fake = int(0 // rows_per_shard)
# Shard for index 38301 (real)
shard_real = int(38301 // rows_per_shard)

print(f"Fake sample at index 0 is in shard {shard_fake:05d}: data/test-{shard_fake:05d}-of-00200.parquet")
print(f"Real sample at index 38301 is in shard {shard_real:05d}: data/test-{shard_real:05d}-of-00200.parquet")
