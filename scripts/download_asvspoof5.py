#!/usr/bin/env python3
"""Download ASVspoof 5 dataset subsets (shards) from HuggingFace.

We only download:
1. Protocols (metadata)
2. One shard of Training data (flac_T_aa.tar) ~7GB
3. One shard of Dev data (flac_D_aa.tar) ~6GB

Total download: ~14GB.
Auto-deletes tar files after extraction to save space.
"""
import os
import sys
import shutil
import tarfile
from pathlib import Path
from huggingface_hub import hf_hub_download

# Constants
REPO_ID = "jungjee/asvspoof5"
DATA_ROOT = Path("data/ASVspoof5")

FILES_TO_DOWNLOAD = [
    "ASVspoof5_protocols.tar",
    "flac_T_aa.tar",
    "flac_D_aa.tar",
]

def check_space(required_gb=20):
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024**3)
    print(f"Disk free: {free_gb:.2f} GB")
    return free_gb >= required_gb

def main():
    if not check_space():
        # Non-interactive mode for automation
        print("WARNING: Low disk space (<20GB).")
        # if input("Continue? (y/n) ").lower() != 'y': return

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    for f in FILES_TO_DOWNLOAD:
        print(f"Downloading {f}...")
        try:
            # Download to cache then symlink/copy
            # or force local_dir to bypass cache duplication if space is tight
            path = hf_hub_download(
                repo_id=REPO_ID, 
                filename=f, 
                repo_type="dataset", 
                local_dir=DATA_ROOT, 
                local_dir_use_symlinks=False
            )
            
            print(f"Extracting {path}...")
            try:
                with tarfile.open(path) as tar:
                    tar.extractall(path=DATA_ROOT)
                print(f"Extracted. Deleting {path} to save space...")
                os.remove(path) 
            except Exception as e:
                print(f"Error extracting {f}: {e}")
                
        except Exception as e:
            print(f"Error downloading {f}: {e}")

    print("\nASVspoof 5 subset download complete.")

if __name__ == "__main__":
    main()
