"""Downloads model weights from Hugging Face into ./models"""
import os, shutil
from huggingface_hub import hf_hub_download

REPO = "ParthMalik6/rf-cnn"
FILES = ["best_model.pth", "rf_gradient.joblib",
         "rf_fft.joblib", "rf_combined.joblib"]

os.makedirs("models", exist_ok=True)
for f in FILES:
    dest = os.path.join("models", f)
    if os.path.exists(dest):
        print(f"skip {f} (already present)")
        continue
    print(f"downloading {f} ...")
    shutil.copy(hf_hub_download(repo_id=REPO, filename=f), dest)
print("\nall models in ./models")