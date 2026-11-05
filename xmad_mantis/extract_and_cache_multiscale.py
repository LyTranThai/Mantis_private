import os
import sys
import time
import json
import logging
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, Audio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_audio import parse_wav_bytes
from mantis.architecture import MantisV2
from mantis.trainer import MantisTrainer

def get_paths(base_store_dir: str):
    shards_dir = os.path.join(base_store_dir, "shards_multiscale")
    logs_dir = os.path.join(base_store_dir, "logs")
    os.makedirs(shards_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    log_file = os.path.join(logs_dir, "extraction_multiscale_audit.log")
    ledger_file = os.path.join(logs_dir, "shard_multiscale_ledger.json")
    return shards_dir, logs_dir, log_file, ledger_file

def setup_logger(log_file: str):
    logger = logging.getLogger("XMAD_MultiScale_Extractor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def load_ledger(ledger_file: str) -> dict:
    if os.path.exists(ledger_file):
        try:
            with open(ledger_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_ledger(ledger_file: str, ledger: dict):
    with open(ledger_file, 'w', encoding='utf-8') as f:
        json.dump(ledger, f, indent=2)

def print_status(ledger: dict, total_shards=200):
    completed = [k for k, v in ledger.items() if v.get("status") == "COMPLETED"]
    failed = [k for k, v in ledger.items() if v.get("status") == "FAILED"]
    total_samples = sum(v.get("samples_embedded", 0) for v in ledger.values() if v.get("status") == "COMPLETED")
    total_bytes = sum(v.get("file_size_bytes", 0) for v in ledger.values() if v.get("status") == "COMPLETED")
    
    print("\n" + "=" * 65)
    print("XMAD MULTI-SCALE (512+1024+2048) EMBEDDING STORE AUDIT")
    print("=" * 65)
    print(f"Total Benchmark Shards:    {total_shards}")
    print(f"Completed Successfully:    {len(completed)} / {total_shards} ({len(completed)/total_shards*100:.1f}%)")
    print(f"Total Samples Embedded:    {total_samples:,} samples")
    print(f"Total Disk Space Used:     {total_bytes / (1024*1024):.2f} MB")
    print(f"Failed / Incomplete:       {len(failed)} ({', '.join(failed) if failed else 'None'})")
    print(f"Remaining Unprocessed:     {total_shards - len(completed)}")
    print("=" * 65 + "\n")

def process_single_shard_multiscale(shard_idx: int, trainer, shards_dir: str, logger, device="cpu") -> dict:
    shard_id_str = f"{shard_idx:05d}"
    shard_file_name = f"data/test-{shard_id_str}-of-00200.parquet"
    target_parquet = os.path.join(shards_dir, f"shard_{shard_id_str}.parquet")
    tmp_parquet = os.path.join(shards_dir, f".tmp_shard_{shard_id_str}.parquet")

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        logger.info(f"--- Starting Multi-Scale Shard {shard_id_str} ({shard_file_name}) [Attempt {attempt}/{max_retries}] ---")
        t0 = time.time()

        try:
            ds = load_dataset(
                "SpeechAntiSpoofingBenchmarks/XMAD",
                data_files={"test": [shard_file_name]},
                split="test",
                streaming=True
            )
            ds = ds.cast_column("audio", Audio(decode=False))

            records = []
            x512_list = []
            x1024_list = []
            x2048_list = []

            for sample in ds:
                audio_dict = sample.get("audio")
                if not audio_dict or "bytes" not in audio_dict or audio_dict["bytes"] is None:
                    continue

                try:
                    wav_arr = parse_wav_bytes(audio_dict["bytes"])
                except Exception:
                    continue

                if len(wav_arr) < 512:
                    continue

                # Multi-Scale Resampling: 512, 1024, 2048 points
                wav_tensor = torch.tensor(wav_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                r512 = F.interpolate(wav_tensor, size=512, mode="linear", align_corners=False).squeeze(0).numpy()
                r1024 = F.interpolate(wav_tensor, size=1024, mode="linear", align_corners=False).squeeze(0).numpy()
                r2048 = F.interpolate(wav_tensor, size=2048, mode="linear", align_corners=False).squeeze(0).numpy()

                x512_list.append(r512)
                x1024_list.append(r1024)
                x2048_list.append(r2048)

                meta = json.loads(sample.get("notes", "{}"))
                records.append({
                    "utterance_id": meta.get("utterance_id", sample.get("path", "")),
                    "label": int(sample.get("label", 0)),
                    "attack": meta.get("attack", "bonafide" if sample.get("label") == 0 else "unknown"),
                    "speaker_id": meta.get("speaker_id", "unknown"),
                    "language": meta.get("language", ""),
                    "corpus": meta.get("corpus", ""),
                    "duration_sec": float(len(wav_arr) / 16000.0),
                    "original_split": meta.get("split", "test")
                })

                if len(records) % 500 == 0:
                    logger.info(f"Shard {shard_id_str}: Streamed and multi-scale interpolated {len(records)} samples...")

            if not records:
                logger.warning(f"Shard {shard_id_str} had 0 valid audio samples.")
                return {"status": "FAILED", "error": "No valid samples", "samples_embedded": 0}

            # Batch Feature Extraction across 3 scales
            X512 = np.array(x512_list, dtype=np.float32)
            X1024 = np.array(x1024_list, dtype=np.float32)
            X2048 = np.array(x2048_list, dtype=np.float32)

            batch_size = 64
            z512_parts, z1024_parts, z2048_parts = [], [], []

            for b in range(0, len(X512), batch_size):
                z512_parts.append(trainer.transform(X512[b:b+batch_size]))
                z1024_parts.append(trainer.transform(X1024[b:b+batch_size]))
                z2048_parts.append(trainer.transform(X2048[b:b+batch_size]))

            Z512 = np.concatenate(z512_parts, axis=0)     # (N, 256)
            Z1024 = np.concatenate(z1024_parts, axis=0)   # (N, 256)
            Z2048 = np.concatenate(z2048_parts, axis=0)   # (N, 256)

            # Concatenate to 768-d multi-scale embedding vector
            Z_multi = np.concatenate([Z512, Z1024, Z2048], axis=1) # (N, 768)

            df = pd.DataFrame(records)
            df["embedding"] = [v.tolist() for v in Z_multi]

            # Atomic save
            if os.path.exists(tmp_parquet):
                os.remove(tmp_parquet)
            df.to_parquet(tmp_parquet, index=False)

            verify_df = pd.read_parquet(tmp_parquet)
            if len(verify_df) != len(records):
                raise ValueError(f"Integrity check failed: expected {len(records)} rows, got {len(verify_df)}")

            if os.path.exists(target_parquet):
                os.remove(target_parquet)
            os.rename(tmp_parquet, target_parquet)

            file_size = os.path.getsize(target_parquet)
            elapsed = time.time() - t0
            logger.info(f"Committed Multi-Scale Shard {shard_id_str}: {len(df)} samples, {file_size/(1024*1024):.2f} MB in {elapsed:.1f}s.")

            return {
                "status": "COMPLETED",
                "samples_embedded": len(df),
                "file_size_bytes": file_size,
                "output_file": target_parquet,
                "elapsed_sec": elapsed,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")
            }

        except Exception as e:
            logger.error(f"Error on shard {shard_id_str} (attempt {attempt}/{max_retries}): {e}")
            if os.path.exists(tmp_parquet):
                try:
                    os.remove(tmp_parquet)
                except Exception:
                    pass
            time.sleep(2 * attempt)

    return {
        "status": "FAILED",
        "error": f"Exceeded max retries ({max_retries})",
        "samples_embedded": 0,
        "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S")
    }

def main():
    parser = argparse.ArgumentParser(description="Multi-Scale XMAD Feature Extractor (512, 1024, 2048 -> 768d)")
    parser.add_argument("--store-dir", type=str, default=r"D:\Blank\xmad_store", help="Base directory for shards")
    parser.add_argument("--start-shard", type=int, default=0, help="Start shard index")
    parser.add_argument("--end-shard", type=int, default=199, help="End shard index (inclusive)")
    parser.add_argument("--target-shards", type=str, default=None, help="Comma-separated shard list (e.g. '106,113,191,81,82')")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu or cuda)")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing shards")
    args = parser.parse_args()

    shards_dir, logs_dir, log_file, ledger_file = get_paths(args.store_dir)
    logger = setup_logger(log_file)

    logger.info("=" * 65)
    logger.info("XMAD MULTI-SCALE FEATURE EXTRACTION ENGINE")
    logger.info(f"Resolutions: [512, 1024, 2048] -> 768-d Concatenated Embedding")
    logger.info(f"Device: {args.device} | Base Directory: {args.store_dir}")
    logger.info("=" * 65)

    ledger = load_ledger(ledger_file)
    print_status(ledger)

    logger.info(f"Loading MantisV2 foundation model on {args.device}...")
    network = MantisV2(device=args.device).from_pretrained("paris-noah/MantisV2")
    trainer = MantisTrainer(device=args.device, network=network)
    logger.info("MantisV2 loaded successfully.")

    if args.target_shards:
        shard_indices = [int(s.strip()) for s in args.target_shards.split(",") if s.strip()]
    else:
        shard_indices = list(range(args.start_shard, args.end_shard + 1))

    logger.info(f"Target shards to process: {len(shard_indices)} shards.")

    for s_idx in shard_indices:
        shard_key = f"{s_idx:05d}"
        target_parquet = os.path.join(shards_dir, f"shard_{shard_key}.parquet")

        if not args.force and shard_key in ledger and ledger[shard_key].get("status") == "COMPLETED" and os.path.exists(target_parquet):
            logger.info(f"Skipping Shard {shard_key} (already completed).")
            continue

        res = process_single_shard_multiscale(s_idx, trainer, shards_dir, logger, device=args.device)
        ledger[shard_key] = res
        save_ledger(ledger_file, ledger)

    print_status(ledger)
    logger.info("Multi-Scale feature extraction run complete.")

if __name__ == "__main__":
    main()
