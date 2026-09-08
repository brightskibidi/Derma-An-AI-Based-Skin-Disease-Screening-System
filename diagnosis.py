import sys
import traceback
import zipfile
import os

import torch
import fastai
import torchvision
import PIL


print("=" * 60)
print("ENVIRONMENT")
print("=" * 60)

print("Python      :", sys.version)
print("PyTorch     :", torch.__version__)
print("FastAI      :", fastai.__version__)
print("torchvision :", torchvision.__version__)
print("Pillow      :", PIL.__version__)

print("\n" + "=" * 60)
print("CHECKPOINT")
print("=" * 60)

path = "Skin_disease.pkl"

print("Exists:", os.path.exists(path))
print("Size :", os.path.getsize(path), "bytes")

with zipfile.ZipFile(path, "r") as z:
    files = z.namelist()

    print("\nNumber of files:", len(files))

    print("\nFirst 30 files:")
    for f in files[:30]:
        print(" ", f)

    print("\nContains data.pkl:",
          any(f.endswith("data.pkl") for f in files))

    storage_files = [
        f for f in files
        if "/data/" in f or f.startswith("data/")
    ]

    print("Possible storage files:", len(storage_files))


print("\n" + "=" * 60)
print("FASTAI LOAD TEST")
print("=" * 60)

try:
    from fastai.learner import load_learner

    model = load_learner(path, cpu=True)

    print("SUCCESS!")
    print(type(model))

except Exception as e:
    print("\nFAILED:")
    print(type(e).__name__, ":", str(e))

    print("\nFULL TRACEBACK:")
    traceback.print_exc()
