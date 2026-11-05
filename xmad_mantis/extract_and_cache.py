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
    shards_dir = os.path.join(base_store_dir, "shards")
    logs_dir = os.path.join(base_store_dir, "logs")
    os.makedirs(shards_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    log_file = os.path.join(logs_dir, "extraction_audit.log")
    ledger_file = os.path.join(logs_dir, "shard_ledger.json")
    return shards_dir, logs_dir, log_file, ledger_file

def setup_logger(log_file: str):
    logger = logging.getLogger("XMAD_Extractor")
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
    
    print("\n" + "=" * 60)
    print("XMAD EMBEDDING STORE STATUS & AUDIT")
    print("=" * 60)
    print(f"Total Benchmark Shards:    {total_shards}")
    print(f"Completed Successfully:    {len(completed)} / {total_shards} ({len(completed)/total_shards*100:.1f}%)")
    print(f"Total Samples Cached:      {total_samples:,} samples")
    print(f"Total Disk Space Used:     {total_bytes / (1024*1024):.2f} MB")
    print(f"Failed / Incomplete:       {len(failed)} ({', '.join(failed) if failed else 'None'})")
    print(f"Remaining Unprocessed:     {total_shards - len(completed)}")
    print("=" * 60 + "\n")

def process_single_shard(shard_idx: int, trainer, shards_dir: str, logger) -> dict:
    shard_id_str = f"{shard_idx:05d}"
    shard_file_name = f"data/test-{shard_id_str}-of-00200.parquet"
    target_parquet = os.path.join(shards_dir, f"shard_{shard_id_str}.parquet")
    tmp_parquet = os.path.join(shards_dir, f".tmp_shard_{shard_id_str}.parquet")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        logger.info(f"--- Starting Shard {shard_id_str} ({shard_file_name}) [Attempt {attempt}/{max_retries}] ---")
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
            raw_waveforms = []

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

                # Interpolate to 512 points
                wav_tensor = torch.tensor(wav_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                resized_arr = F.interpolate(wav_tensor, size=512, mode="linear", align_corners=False).squeeze(0).numpy()

                raw_waveforms.append(resized_arr)
                
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
                    logger.info(f"Shard {shard_id_str}: Streamed and preprocessed {len(records)} audio samples...")

            if not raw_waveforms:
                logger.warning(f"Shard {shard_id_str} had 0 valid audio samples.")
                return {"status": "FAILED", "error": "No valid samples", "samples_embedded": 0}

            # Batch Feature Extraction
            X_arr = np.array(raw_waveforms, dtype=np.float32)
            batch_size = 64
            embs_list = []
            for b in range(0, len(X_arr), batch_size):
                embs_list.append(trainer.transform(X_arr[b:b+batch_size]))

            Z_arr = np.concatenate(embs_list, axis=0) # (N, 256)

            # Assemble DataFrame with embeddings as float lists
            df = pd.DataFrame(records)
            df["embedding"] = [v.tolist() for v in Z_arr]

            # Atomic write: save to .tmp_ then rename
            if os.path.exists(tmp_parquet):
                os.remove(tmp_parquet)
            df.to_parquet(tmp_parquet, index=False)

            # Integrity verification
            verify_df = pd.read_parquet(tmp_parquet)
            if len(verify_df) != len(records):
                raise ValueError(f"Integrity check failed: expected {len(records)} rows, got {len(verify_df)}")

            if os.path.exists(target_parquet):
                os.remove(target_parquet)
            os.rename(tmp_parquet, target_parquet)

            file_size = os.path.getsize(target_parquet)
            elapsed = time.time() - t0
            logger.info(f"Committed Shard {shard_id_str}: {len(df)} samples, {file_size/(1024*1024):.2f} MB in {elapsed:.1f}s.")

            return {
                "status": "COMPLETED",
                "samples_embedded": len(df),
                "file_size_bytes": file_size,
                "output_file": target_parquet,
                "elapsed_sec": float(elapsed),
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")
            }

        except Exception as e:
            logger.warning(f"Attempt {attempt}/{max_retries} for Shard {shard_id_str} encountered error: {e}")
            if os.path.exists(tmp_parquet):
                try:
                    os.remove(tmp_parquet)
                except OSError:
                    pass
            if attempt < max_retries:
                time.sleep(5)
            else:
                logger.error(f"Shard {shard_id_str} FAILED after {max_retries} attempts.")
                return {
                    "status": "FAILED",
                    "error": str(e),
                    "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S")
                }

def main():
    parser = argparse.ArgumentParser(description="Resumable Sharded Mantis v2 Feature Extractor for XMAD-Bench.")
    parser.add_argument("--store-dir", type=str, default=r"D:\Blank\xmad_store", help="Base directory on D: drive.")
    parser.add_argument("--status", action="store_true", help="Print audit summary and exit.")
    parser.add_argument("--batch-shards", type=int, default=10, help="Number of uncompleted shards to process in this run.")
    parser.add_argument("--shards", type=str, default="", help="Comma-separated list of specific shard IDs (e.g. '54,55,81').")
    parser.add_argument("--redo-shard", type=int, default=-1, help="Force re-extraction of a specific shard ID.")
    parser.add_argument("--repair", action="store_true", help="Re-pull all shards marked FAILED.")
    args = parser.parse_args()

    # Fallback to local workspace if D: drive is unavailable
    store_dir = args.store_dir
    if not os.path.exists(os.path.splitdrive(store_dir)[0]):
        store_dir = r"C:\Users\Admin\Desktop\SC203\xmad_mantis\store"
        print(f"Drive {args.store_dir} not detected. Falling back to {store_dir}")

    shards_dir, logs_dir, log_file, ledger_file = get_paths(store_dir)
    logger = setup_logger(log_file)
    ledger = load_ledger(ledger_file)

    if args.status:
        print_status(ledger)
        return

    # Handle single redo
    if args.redo_shard >= 0:
        shard_key = f"{args.redo_shard:05d}"
        target_file = os.path.join(shards_dir, f"shard_{shard_key}.parquet")
        if os.path.exists(target_file):
            os.remove(target_file)
        ledger.pop(shard_key, None)
        save_ledger(ledger_file, ledger)
        target_shards = [args.redo_shard]
        logger.info(f"Targeting single redo for Shard {shard_key}.")

    # Handle repair
    elif args.repair:
        failed_shards = [int(k) for k, v in ledger.items() if v.get("status") == "FAILED"]
        if not failed_shards:
            logger.info("No failed shards found in ledger to repair.")
            return
        target_shards = failed_shards
        logger.info(f"Repairing {len(target_shards)} failed shards: {target_shards}")

    # Handle specific shards list
    elif args.shards:
        target_shards = [int(s.strip()) for s in args.shards.split(",") if s.strip()]

    # Default: Next N uncompleted shards
    else:
        completed = set(int(k) for k, v in ledger.items() if v.get("status") == "COMPLETED")
        uncompleted = [i for i in range(200) if i not in completed]
        target_shards = uncompleted[:args.batch_shards]
        logger.info(f"Selected next {len(target_shards)} uncompleted shards: {target_shards}")

    if not target_shards:
        logger.info("All target shards are already completed! Run --status to inspect.")
        return

    # Initialize MantisV2 model
    logger.info("Loading MantisV2 foundation model...")
    network = MantisV2(device="cpu").from_pretrained("paris-noah/MantisV2")
    trainer = MantisTrainer(device="cpu", network=network)
    logger.info("MantisV2 loaded and ready for batch extraction.")

    # Execution loop
    for idx in target_shards:
        res = process_single_shard(idx, trainer, shards_dir, logger)
        ledger[f"{idx:05d}"] = res
        save_ledger(ledger_file, ledger) # Commit ledger immediately after every shard

    print_status(ledger)

if __name__ == "__main__":
    main()
