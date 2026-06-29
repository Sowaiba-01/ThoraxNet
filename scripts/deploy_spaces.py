"""Push latest code to HuggingFace Spaces for deployment."""
import os
from huggingface_hub import HfApi

api = HfApi()
repo_id = os.environ.get("HF_SPACES_REPO", "YOUR_USERNAME/chestai")
token   = os.environ["HF_TOKEN"]

api.upload_folder(
    folder_path=".",
    repo_id=repo_id,
    repo_type="space",
    token=token,
    ignore_patterns=["*.pyc", "__pycache__", ".git", "frontend/node_modules", "notebooks/"],
)
print(f"Deployed to https://huggingface.co/spaces/{repo_id}")
