import os
import torch

PROJECT_ROOT = r"D:\VAE_Music_Project"
assert os.path.exists(PROJECT_ROOT), "Project root not found!"

print("Python OK")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU (this is fine for our plan).")

# Small tensor test
x = torch.randn(2, 3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
print("Tensor moved to:", device)
print("Smoke test passed.")
