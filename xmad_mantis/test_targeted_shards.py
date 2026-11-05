from datasets import load_dataset, Audio

print("Loading specific shards in streaming mode...")
data_files = [
    "data/test-00000-of-00200.parquet",  # Contains fake samples
    "data/test-00021-of-00200.parquet",  # Contains real samples
]

ds = load_dataset(
    "SpeechAntiSpoofingBenchmarks/XMAD", 
    data_files={"test": data_files}, 
    split="test", 
    streaming=True
)
ds = ds.cast_column("audio", Audio(decode=False))

counts = {0: 0, 1: 0}
for i, sample in enumerate(ds):
    label = sample["label"]
    counts[label] += 1
    if counts[0] >= 5 and counts[1] >= 5:
        break

print(f"Streaming verification successful! Real(0)={counts[0]}, Fake(1)={counts[1]}")
