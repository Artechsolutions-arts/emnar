from huggingface_hub import snapshot_download
import os

model_id = "rednote-hilab/dots.ocr"
local_dir = r"d:\AI_ML\onyx\backend\weights\DotsOCR"

print(f"Downloading model {model_id} to {local_dir}...")
if not os.path.exists(local_dir):
    os.makedirs(local_dir, exist_ok=True)

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print("Download complete!")
