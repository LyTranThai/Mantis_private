import pandas as pd

df = pd.read_parquet("https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/XMAD/resolve/main/data/labels.parquet")

real_indices = df[df['label'] == 0].index
fake_indices = df[df['label'] == 1].index

print(f"Total rows: {len(df)}")
print(f"Real start index: {real_indices[0]}, end index: {real_indices[-1]}")
print(f"Fake start index: {fake_indices[0]}, end index: {fake_indices[-1]}")

print("\nFirst 5 real utterance_ids:")
print(df.loc[real_indices[:5], 'utterance_id'].tolist())

# Check languages/corpora for real and fake
df['category'] = df['utterance_id'].apply(lambda x: '__'.join(x.split('__')[:3]))
print("\nTop categories:")
print(df['category'].value_counts().head(20))
