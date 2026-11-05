import pandas as pd

df = pd.read_parquet("https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/XMAD/resolve/main/data/labels.parquet")
rows_per_shard = len(df) / 200.0

for category in ["en__commonvoice-en__fake", "en__commonvoice-en__real"]:
    cat_df = df[df['utterance_id'].str.startswith(category)]
    start_idx = cat_df.index[0]
    end_idx = cat_df.index[-1]
    start_shard = int(start_idx // rows_per_shard)
    end_shard = int(end_idx // rows_per_shard)
    print(f"{category}: count={len(cat_df)}, indices=[{start_idx}, {end_idx}], shards=[{start_shard:05d}, {end_shard:05d}]")
