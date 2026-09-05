"""
Standalone script: downloads files from a Google Drive folder or a single
shared file/zip, with no login required. Not connected to any other
project code.

REQUIRES the folder/file to be shared as "Anyone with the link" (Viewer is
enough) -- private files still need Google's OAuth flow, which this script
does not do.

Requires:
    pip install gdown

Usage:
    # A shared FOLDER (downloads every file inside it)
    python download_gdrive_dataset.py --folder-url "https://drive.google.com/drive/folders/XXXXXXXXXXXX" --out-dir ./downloaded_photos

    # A single shared FILE (e.g. a zip of your dataset)
    python download_gdrive_dataset.py --file-url "https://drive.google.com/file/d/XXXXXXXXXXXX/view" --out-dir ./downloaded_photos
"""

import argparse
import os
import re
import zipfile

import gdown


def extract_file_id(url_or_id: str) -> str:
    """
    Pulls the file ID out of any common Google Drive share-link format,
    or passes through a bare ID unchanged. Done manually with regex
    instead of relying on gdown's `fuzzy=True` option, since that
    parameter doesn't exist in every gdown version (this is exactly what
    broke the first version of this script).

    Handles:
        https://drive.google.com/file/d/FILE_ID/view?usp=sharing
        https://drive.google.com/open?id=FILE_ID
        https://drive.google.com/uc?id=FILE_ID
        FILE_ID   (bare ID, passed through unchanged)
    """
    match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url_or_id)
    if match:
        return match.group(1)

    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url_or_id)
    if match:
        return match.group(1)

    if re.fullmatch(r"[a-zA-Z0-9_-]+", url_or_id):
        return url_or_id

    raise ValueError(f"Could not extract a Google Drive file ID from: {url_or_id!r}")


def main():
    parser = argparse.ArgumentParser(description="Download files from a public Google Drive link.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--folder-url", type=str, help="URL of a shared Google Drive FOLDER")
    group.add_argument("--file-url", type=str, help="URL of a single shared Google Drive FILE")
    parser.add_argument("--out-dir", type=str, default="./downloaded_photos")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.folder_url:
        print(f"Downloading folder contents to {args.out_dir} ...")
        downloaded = gdown.download_folder(url=args.folder_url, output=args.out_dir, quiet=False, use_cookies=False)
        print(f"Done. {len(downloaded)} files downloaded.")

    else:
        file_id = extract_file_id(args.file_url)
        print(f"Extracted file ID: {file_id}")
        print(f"Downloading file to {args.out_dir} ...")
        output_path = gdown.download(id=file_id, output=os.path.join(args.out_dir, ""),
                                      quiet=False, use_cookies=False)

        if output_path and output_path.lower().endswith(".zip"):
            print(f"Extracting {output_path} ...")
            with zipfile.ZipFile(output_path, "r") as zf:
                zf.extractall(args.out_dir)
            os.remove(output_path)
            print("Extraction complete, zip removed.")

    n_files = sum(len(files) for _, _, files in os.walk(args.out_dir))
    print(f"Total files now in {args.out_dir}: {n_files}")


if __name__ == "__main__":
    main()
