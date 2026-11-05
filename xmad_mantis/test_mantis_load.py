import sys
import torch
print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

from mantis.architecture import MantisV2
from mantis.trainer import MantisTrainer

device = "cpu"
print(f"Loading MantisV2 on {device}...")
network = MantisV2(device=device).from_pretrained("paris-noah/MantisV2")
print("MantisV2 network loaded successfully!")
trainer = MantisTrainer(device=device, network=network)
print("MantisTrainer initialized successfully!")

# Test dummy input
import numpy as np
dummy_X = np.random.randn(2, 1, 512).astype(np.float32)
embs = trainer.transform(dummy_X)
print(f"Embedding shape: {embs.shape}")
print("ALL MANTIS TESTS PASSED!")
