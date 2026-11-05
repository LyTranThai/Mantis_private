import os
import sys
import json
import io
import wave
import numpy as np

def parse_wav_bytes(audio_bytes: bytes) -> np.ndarray:
    """Decode raw WAV bytes into a 1D float32 numpy array normalized to [-1.0, 1.0]."""
    with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_frames = wf.readframes(n_frames)
        
        if sampwidth == 2:
            dtype = np.int16
            max_val = 32768.0
        elif sampwidth == 4:
            dtype = np.int32
            max_val = 2147483648.0
        elif sampwidth == 1:
            dtype = np.uint8
            # uint8 is typically offset by 128
            arr = np.frombuffer(raw_frames, dtype=dtype).astype(np.float32)
            arr = (arr - 128.0) / 128.0
            if n_channels > 1:
                arr = arr.reshape(-1, n_channels).mean(axis=1)
            return arr
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        arr = np.frombuffer(raw_frames, dtype=dtype).astype(np.float32) / max_val
        if n_channels > 1:
            arr = arr.reshape(-1, n_channels).mean(axis=1)
        return arr

if __name__ == "__main__":
    print("utils_audio module loaded successfully.")
