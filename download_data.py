import os
import shutil
import kagglehub

def download():
    print("Downloading dataset netslabdemo/netslab-5g-oran-idd via kagglehub...")
    path = kagglehub.dataset_download("netslabdemo/netslab-5g-oran-idd")
    print("Path to dataset files:", path)

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
