import os
import shutil
import time
import kagglehub

def download():
    print("Downloading dataset netslabdemo/netslab-5g-oran-idd via kagglehub...")
    max_retries = 10
    path = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Download attempt {attempt}/{max_retries}...")
            path = kagglehub.dataset_download("netslabdemo/netslab-5g-oran-idd")
            print("Path to dataset files:", path)
            break
        except Exception as e:
            print(f"Attempt {attempt} failed with error: {e}")
            if attempt == max_retries:
                raise
            print("Retrying in 5 seconds (kagglehub will resume from existing downloaded bytes)...")
            time.sleep(5)

    if not path or not os.path.exists(path):
        raise RuntimeError("Dataset path not found after download.")

    dest_dirs = ["rawdata", "data/raw"]
    for dest in dest_dirs:
        os.makedirs(dest, exist_ok=True)
        for item in os.listdir(path):
            s = os.path.join(path, item)
            d = os.path.join(dest, item)
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        print(f"Copied dataset files to {dest}/")

if __name__ == "__main__":
    download()
