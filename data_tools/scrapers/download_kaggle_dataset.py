"""
Standalone script: downloads all files from one of your own Kaggle
datasets to a local folder. Not connected to any other project code.

Requires the Kaggle API to be set up (same as when you pushed this
dataset originally):
    pip install kaggle
    # API token placed at ~/.kaggle/kaggle.json (or ~/.kaggle/access_token
    # for the newer token flow) -- see earlier setup steps if not done yet.

Usage:
    python download_kaggle_dataset.py --dataset dorast/real-old-photos-vae1 --out-dir ./downloaded_photos
"""

import argparse
import os
import zipfile


def main():
    parser = argparse.ArgumentParser(description="Download a Kaggle dataset's files to a local folder.")
    parser.add_argument("--dataset", type=str, required=True,
                         help="dataset slug in the form 'username/dataset-name', e.g. 'dorast/real-old-photos-vae1'")
    parser.add_argument("--out-dir", type=str, default="./downloaded_photos")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading dataset '{args.dataset}' to {args.out_dir} ...")
    api.dataset_download_files(args.dataset, path=args.out_dir, unzip=False, quiet=False)

    # Kaggle downloads as a single zip named after the dataset slug's second half
    zip_candidates = [f for f in os.listdir(args.out_dir) if f.endswith(".zip")]
    if zip_candidates:
        zip_path = os.path.join(args.out_dir, zip_candidates[0])
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(args.out_dir)
        os.remove(zip_path)
        print("Extraction complete, zip removed.")

    n_files = sum(len(files) for _, _, files in os.walk(args.out_dir))
    print(f"Done. {n_files} files now in {args.out_dir}")


if __name__ == "__main__":
    main()