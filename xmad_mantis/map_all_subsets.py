import pandas as pd

df = pd.read_parquet("https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/XMAD/resolve/main/data/labels.parquet")
rows_per_shard = len(df) / 200.0

df['category'] = df['utterance_id'].apply(lambda x: '__'.join(x.split('__')[:3]))

summary = []
for cat, group in df.groupby('category', sort=False):
    start_idx = group.index[0]
    end_idx = group.index[-1]
    start_shard = int(start_idx // rows_per_shard)
    end_shard = int(end_idx // rows_per_shard)
    label = group['label'].iloc[0]
    summary.append({
        "category": cat,
        "label": "Real (0)" if label == 0 else "Fake (1)",
        "count": len(group),
        "start_shard": start_shard,
        "end_shard": end_shard,
        "num_shards": end_shard - start_shard + 1
    })

res_df = pd.DataFrame(summary)
print(res_df.to_string(index=False))
