import os
import sys
import io
import json
from datasets import load_dataset, Audio
from utils_audio import parse_wav_bytes

print("Attempting to connect to SpeechAntiSpoofingBenchmarks/XMAD in streaming mode (decode=False)...")
try:
    ds = load_dataset("SpeechAntiSpoofingBenchmarks/XMAD", split="test", streaming=True)
    # Disable internal audio decoder to avoid torchcodec / soundfile requirement
    ds = ds.cast_column("audio", Audio(decode=False))
    print("Dataset stream opened successfully with decode=False!")
    
    # Grab first 2 samples
    for i, sample in enumerate(ds):
        print(f"\n--- Sample {i+1} ---")
        print("Path:", sample.get("path"))
        print("Label:", sample.get("label"))
        notes = sample.get("notes")
        print("Notes:", notes)
        
        audio_data = sample.get("audio")
        if isinstance(audio_data, dict) and "bytes" in audio_data:
            raw_bytes = audio_data["bytes"]
            print(f"Raw bytes length: {len(raw_bytes)}")
            wav_arr = parse_wav_bytes(raw_bytes)
            print(f"Decoded from wave: length={len(wav_arr)}, dtype={wav_arr.dtype}, min={wav_arr.min():.3f}, max={wav_arr.max():.3f}")
        
        if i >= 1:
            break
            
    print("\nSTREAM TEST PASSED SUCCESSFULLY!")
except Exception as e:
    print(f"Error streaming dataset: {e}")
    import traceback
    traceback.print_exc()
