import json
from datasets import load_dataset, Audio

print("Scanning for bonafide (0) and spoof (1) in XMAD stream...")
ds = load_dataset("SpeechAntiSpoofingBenchmarks/XMAD", split="test", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))

counts = {0: 0, 1: 0}
samples_by_class = {0: [], 1: []}

for i, sample in enumerate(ds):
    label = sample.get("label")
    counts[label] = counts.get(label, 0) + 1
    if len(samples_by_class[label]) < 3:
        samples_by_class[label].append({
            "idx": i,
            "path": sample.get("path"),
            "label": label,
            "notes": sample.get("notes")
        })
    
    if i % 100 == 0:
        print(f"Scanned {i} samples... Counts so far: Real(0)={counts.get(0, 0)}, Spoof(1)={counts.get(1, 0)}")
        
    if counts.get(0, 0) >= 10 and counts.get(1, 0) >= 10:
        break
    if i >= 1000:
        break

print(f"\nFinal tally: Real(0)={counts.get(0,0)}, Fake(1)={counts.get(1,0)}")
print("Sample examples:", json.dumps(samples_by_class, indent=2))
