from huggingface_hub import list_datasets, list_repo_files, hf_hub_download
import shutil
from pathlib import Path

def search_and_download():
    print("Searching HF datasets for 'asvspoof 2019'...")
    try:
        datasets = list(list_datasets(search="asvspoof 2019", limit=5))
        for d in datasets:
            print(f"Found dataset: {d.id}")
            try:
                files = list_repo_files(repo_id=d.id, repo_type="dataset")
                print(f"  Example files in {d.id} ({len(files)} total): {files[:5]}")
                
                # Check for protocols
                target_file = "ASVspoof2019.LA.asv.dev.gi.trl.txt"
                # Some repos might have it under a subdirectory
                matching = [f for f in files if target_file in f]
                
                if matching:
                    print(f"  FOUND PROTOCOL FILE: {matching[0]}")
                    print(f"  Downloading {matching[0]} from {d.id}...")
                    
                    # Download to data/LA/ASVspoof2019_LA_asv_protocols/
                    dest_dir = Path("data/LA/ASVspoof2019_LA_asv_protocols")
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    local_path = hf_hub_download(repo_id=d.id, filename=matching[0], repo_type="dataset")
                    shutil.copy(local_path, dest_dir / Path(matching[0]).name)
                    print(f"  Saved to {dest_dir / Path(matching[0]).name}")
                    
                    # Also try to get enrollment files?
                    # *asv.dev*.trn.txt
                    trn_files = [f for f in files if "trn.txt" in f and "dev" in f]
                    for trn in trn_files:
                         print(f"  Downloading {trn}...")
                         lp = hf_hub_download(repo_id=d.id, filename=trn, repo_type="dataset")
                         shutil.copy(lp, dest_dir / Path(trn).name)
                         
                    return True # Success
            except Exception as e:
                print(f"  Error accessing {d.id}: {e}")
                
    except Exception as e:
        print(f"Search failed: {e}")
    return False

if __name__ == "__main__":
    found = search_and_download()
    if not found:
        print("Could not find protocols on HF automatically.")
