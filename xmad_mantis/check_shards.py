from huggingface_hub import HfApi

api = HfApi()
files = api.list_repo_files("SpeechAntiSpoofingBenchmarks/XMAD", repo_type="dataset")
parquet_files = [f for f in files if f.endswith(".parquet")]
print(f"Total parquet files: {len(parquet_files)}")
print("First 10 parquets:", parquet_files[:10])
print("Middle 5 parquets:", parquet_files[95:100])
print("Last 5 parquets:", parquet_files[-5:])
