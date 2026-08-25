"""
Scrape real old/historical photographs from the Digital Public Library of
America (DPLA) API. DPLA aggregates hundreds of US libraries, archives, and
museums (including LOC, NYPL, Smithsonian, and many university collections)
behind one API, so this is a good way to get subject variety beyond what a
single archive offers.

REQUIRES A FREE API KEY (one-time step, takes a minute):
    curl -X POST "https://api.dp.la/v2/api_key/YOUR_EMAIL@example.com"
Your key will be emailed to you (a 32-character string). Pass it with --api-key.

Docs: https://pro.dp.la/developers/api-codex

Important caveat vs. the LOC scraper: DPLA's `object` field is a THUMBNAIL
URL, not always a full-resolution image -- size varies a lot by contributing
institution. This script's built-in size validation (--min-size) will
naturally discard thumbnails that are too small, but expect a higher skip
rate here than with LOC's own API.

Also note: DPLA's `page` parameter has a hard maximum of 100 (with up to 500
results per page), so a single query is capped at ~50,000 items -- plenty
for this use case, but worth knowing if you're wondering why very large
--pages values get rejected by the API.

Usage:
    python dpla_scraper.py --api-key YOUR_KEY --query "street scene" --pages 15 --page-size 100 --out-dir ./real_old_photos
    python dpla_scraper.py --api-key YOUR_KEY --query "county fair" --start-page 5 --pages 10

Output:
    <out-dir>/images/*.jpg          -- downloaded photos
    <out-dir>/manifest.csv          -- id, title, date, provider, source url, local filename
"""

import argparse
import csv
import io
import os
import time
from urllib.parse import urlencode

import requests
from PIL import Image, UnidentifiedImageError

BASE_URL = "https://api.dp.la/v2/items"
USER_AGENT = "thesis-research-scraper/1.0 (educational use)"


def format_duration(seconds):
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def build_search_url(api_key, query, page, page_size, date_start=None, date_end=None):
    params = {
        "q": query,
        "sourceResource.type": "image",
        "page": page,
        "page_size": page_size,
        "api_key": api_key,
    }
    if date_start:
        params["sourceResource.date.after"] = date_start
    if date_end:
        params["sourceResource.date.before"] = date_end
    return f"{BASE_URL}?{urlencode(params)}"


def pick_image_url(doc):
    """DPLA gives a single thumbnail URL in `object`. Some providers also
    expose a (sometimes larger) view via `hasView`; we try that as a
    secondary option since it occasionally points to a bigger image."""
    obj = doc.get("object")
    if obj:
        return obj
    has_view = doc.get("hasView")
    if isinstance(has_view, dict):
        return has_view.get("@id")
    if isinstance(has_view, list) and has_view:
        return has_view[0].get("@id")
    return None


def fetch_page(session, url, retries=3, backoff=5):
    for attempt in range(retries):
        try:
            resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            if resp.status_code == 429:
                wait = backoff * (attempt + 1)
                print(f"  Rate limited (429). Waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 401 or resp.status_code == 403:
                print("  Authentication failed -- check that --api-key is correct.")
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"  Request failed ({e}), retrying ({attempt + 1}/{retries})...")
            time.sleep(backoff)
    print(f"  Giving up on {url} after {retries} retries.")
    return None


def download_image(session, url, out_path, min_size=254, retries=3):
    """Downloads to memory first, validates content-type/decodability/size,
    and only writes to disk if all checks pass. Returns (success, reason)."""
    for attempt in range(retries):
        try:
            resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if not content_type.startswith("image"):
                return False, f"not an image (Content-Type: {content_type or 'unknown'})"

            content = resp.content
            if not content:
                return False, "empty response body"

            try:
                with Image.open(io.BytesIO(content)) as img:
                    img.verify()
                with Image.open(io.BytesIO(content)) as img:
                    width, height = img.size
            except (UnidentifiedImageError, OSError) as e:
                return False, f"invalid/corrupt image ({e})"

            if width <= min_size or height <= min_size:
                return False, f"too small ({width}x{height}, need > {min_size}x{min_size})"

            with open(out_path, "wb") as f:
                f.write(content)
            return True, "ok"

        except requests.RequestException as e:
            print(f"    Download failed ({e}), retrying ({attempt + 1}/{retries})...")
            time.sleep(3)

    return False, "download failed after retries"


def main():
    parser = argparse.ArgumentParser(description="Scrape real old photos from the DPLA API.")
    parser.add_argument("--api-key", type=str, required=True,
                         help="your 32-character DPLA API key (request via POST to "
                              "https://api.dp.la/v2/api_key/YOUR_EMAIL)")
    parser.add_argument("--query", type=str, default="street scene",
                         help="search term, e.g. 'county fair', 'farmhouse', 'railroad depot'")
    parser.add_argument("--start-page", type=int, default=1,
                         help="page to start from (DPLA caps page at 100; use this to continue a previous run)")
    parser.add_argument("--pages", type=int, default=10, help="number of result pages to fetch, starting from --start-page")
    parser.add_argument("--page-size", type=int, default=100,
                         help="results per page (DPLA allows up to 500)")
    parser.add_argument("--min-size", type=int, default=254,
                         help="minimum width AND height in pixels; smaller images are discarded")
    parser.add_argument("--date-start", type=str, default=None, help="e.g. 1900-01-01")
    parser.add_argument("--date-end", type=str, default=None, help="e.g. 1970-01-01")
    parser.add_argument("--out-dir", type=str, default="./real_old_photos")
    parser.add_argument("--sleep", type=float, default=2.0, help="seconds to sleep between API requests")
    parser.add_argument("--download-sleep", type=float, default=0.5, help="seconds to sleep between image downloads")
    args = parser.parse_args()

    start_time = time.time()

    if args.start_page + args.pages - 1 > 100:
        print(f"Warning: DPLA's page parameter has a hard max of 100. "
              f"Your range (--start-page {args.start_page} + --pages {args.pages}) exceeds that; "
              f"pages beyond 100 will fail.")

    images_dir = os.path.join(args.out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.csv")

    session = requests.Session()
    downloaded = 0
    skipped = 0

    write_header = not os.path.exists(manifest_path)
    manifest_file = open(manifest_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(manifest_file)
    if write_header:
        writer.writerow(["id", "title", "date", "provider", "source_page_url", "image_url", "local_filename"])

    end_page = min(args.start_page + args.pages - 1, 100)
    for page in range(args.start_page, end_page + 1):
        url = build_search_url(args.api_key, args.query, page, args.page_size, args.date_start, args.date_end)
        print(f"Fetching page {page}/{end_page}: {url.split('&api_key=')[0]}&api_key=***")
        data = fetch_page(session, url)
        time.sleep(args.sleep)

        if data is None:
            print("  No data returned, stopping.")
            break

        docs = data.get("docs", [])
        if not docs:
            print("  No results on this page, stopping (may have reached the end).")
            break

        for doc in docs:
            item_id = doc.get("id", "unknown")
            source = doc.get("sourceResource", {}) or {}
            title = (source.get("title") or "untitled")
            if isinstance(title, list):
                title = title[0] if title else "untitled"
            date_field = source.get("date") or {}
            date = date_field.get("displayDate", "") if isinstance(date_field, dict) else str(date_field)
            provider = (doc.get("provider") or {}).get("name", "")
            image_url = pick_image_url(doc)
            source_page = doc.get("isShownAt", "")

            if not image_url:
                skipped += 1
                continue

            local_filename = f"{item_id}.jpg"
            local_path = os.path.join(images_dir, local_filename)

            if os.path.exists(local_path):
                skipped += 1
                continue

            ok, reason = download_image(session, image_url, local_path, min_size=args.min_size)
            if ok:
                writer.writerow([item_id, title, date, provider, source_page, image_url, local_filename])
                manifest_file.flush()
                downloaded += 1
                if downloaded % 25 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Downloaded {downloaded} images so far... (elapsed: {format_duration(elapsed)})")
            else:
                skipped += 1
                print(f"    Skipped {item_id}: {reason}")

            time.sleep(args.download_sleep)

        elapsed = time.time() - start_time
        print(f"  Page {page} done. Total so far: {downloaded} downloaded, {skipped} skipped. "
              f"Elapsed: {format_duration(elapsed)}")

        total_count = data.get("count", 0)
        items_seen = page * args.page_size
        if items_seen >= total_count:
            print("  Reached last page of results.")
            break

    manifest_file.close()
    total_elapsed = time.time() - start_time
    rate = downloaded / total_elapsed * 60 if total_elapsed > 0 else 0
    print(f"\nDone. Downloaded {downloaded} images, skipped {skipped}.")
    print(f"Total runtime: {format_duration(total_elapsed)} ({rate:.1f} images/min)")
    print(f"Images saved to: {images_dir}")
    print(f"Manifest (for citation/provenance) saved to: {manifest_path}")


if __name__ == "__main__":
    main()
