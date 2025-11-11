import transformers
import sys
import torch

print(f"--- Version Check ---")
print(f"Python Executable: {sys.executable}")
print(f"Transformers version: {transformers.__version__}")
print(f"PyTorch version: {torch.__version__}")