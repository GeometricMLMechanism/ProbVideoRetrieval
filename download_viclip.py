#!/usr/bin/env python
"""Download ViCLIP model from Hugging Face"""

from huggingface_hub import hf_hub_download
import os

print("Attempting to download ViCLIP model...")
print("If this fails with authentication error, you need to:")
print("1. Visit https://huggingface.co/OpenGVLab/ViCLIP")
print("2. Accept the model terms/license")
print("3. Run: huggingface-cli login")
print("4. Enter your HuggingFace token\n")

try:
    # Try to download the model
    model_path = hf_hub_download(
        repo_id="OpenGVLab/ViCLIP",
        filename="ViClip-InternVid-10M-FLT.pth",
        local_dir="tool_models/viCLIP",
        local_dir_use_symlinks=False
    )
    print(f"\n✅ Model downloaded successfully to: {model_path}")
except Exception as e:
    print(f"\n❌ Download failed: {e}")
    print("\nPlease authenticate with Hugging Face:")
    print("   huggingface-cli login")
    print("\nThen request access at:")
    print("   https://huggingface.co/OpenGVLab/ViCLIP")
