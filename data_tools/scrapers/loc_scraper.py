"""
Scrape real old/historical photographs from the Library of Congress's public
loc.gov JSON API, for use as unpaired training data for VAE1 in the
"Bringing Old Photos Back to Life" pipeline.

No API key required (the loc.gov API is public), but it IS rate-limited --
this script sleeps between requests to stay well under the limit.

Docs: https://www.loc.gov/apis/json-and-yaml/

Usage:
    Single category (original behavior, unchanged):
        python loc_scraper.py --query "portrait" --pages 20 --per-page 100 --out-dir ./real_old_photos

    Multiple categories in one run, split by weight (new):
        python loc_scraper.py \
            --categories "portrait photograph,street scene,farm landscape,railroad,parade" \
            --weights "0.2,0.2,0.2,0.2,0.2" \
            --total-images 2000 \
            --out-dir ./real_old_photos

        --weights is optional -- omit it for an equal split across categories:
        python loc_scraper.py --categories "street scene,farm landscape,railroad" --total-images 1500

Output:
    <out-dir>/images/*.jpg          -- downloaded photos
    <out-dir>/manifest.csv          -- id, title, date, category, source url, local filename
                                        (keep this for citation/provenance in your thesis)
"""

import argparse
import csv
import io
import os
import time
import sys
from urllib.parse import urlencode

import requests
from PIL import Image, UnidentifiedImageError

BASE_URL = "https://www.loc.gov/photos/"
USER_AGENT = "thesis-research-scraper/1.0 (educational use; contact via loc.gov if issue)"


def build_search_url(query, page, per_page, date_start=None, date_end=None):
    params = {
        "q": query,
        "fo": "json",
        "c": per_page,
        "sp": page,
        # restrict to items that are actually digitized/online so downloads succeed
        "fa": "online-format:image",
    }
    if date_start and date_end:
        params["dates"] = f"{date_start}/{date_end}"
    return f"{BASE_URL}?{urlencode(params)}"


def pick_image_url(item, max_pref_index=-2):
    """
    LOC's `image_url` field is a list of URLs at increasing resolution, with
    the largest/highest-res last. Downloading the very largest can mean huge
    TIFFs, so by default we take the second-to-last (max_pref_index=-2) which
    is usually a large-but-reasonable JPEG. Falls back to the last available
    entry if the list is shorter than expected.
    """
    urls = item.get("image_url")
    if not urls:
        return None
    if len(urls) == 1:
        return urls[0]
    try:
        return urls[max_pref_index]
    except IndexError:
        return urls[-1]


def fetch_page(session, url, retries=3, backoff=5):
    for attempt in range(retries):
        try:
            resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            if resp.status_code == 429:
                wait = backoff * (attempt + 1)
                print(f"  Rate limited (429). Waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"  Request failed ({e}), retrying ({attempt + 1}/{retries})...")
            time.sleep(backoff)
    print(f"  Giving up on {url} after {retries} retries.")
    return None


def download_image(session, url, out_path, min_size=254, retries=3):
    """
    Downloads to memory first, validates it, and only then writes to disk.
    Returns (success: bool, reason: str) so callers can log why something
    was skipped.
    """
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

            # Validate it's a real, decodable image (catches truncated/corrupt downloads)
            try:
                with Image.open(io.BytesIO(content)) as img:
                    img.verify()
                # verify() invalidates the file pointer -- reopen to read dimensions
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


def format_duration(seconds):
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def run_category(session, query, target_count, images_dir, writer, manifest_file, args, start_time,
                  overall_downloaded, overall_skipped, max_pages=None):
    """
    Scrapes a single category/query until either `target_count` images from
    THIS category have been downloaded, or there are no more result pages.
    Returns (category_downloaded, category_skipped, overall_downloaded, overall_skipped).

    max_pages caps how many pages we'll try before giving up on a category
    even if the target wasn't hit (protects against burning huge amounts of
    time on a query that mostly returns skippable/too-small/non-image results).
    """
    if max_pages is None:
        # generous cap: enough pages to plausibly reach target even with a
        # high skip rate, but not unbounded
        max_pages = max(10, (target_count // args.per_page + 1) * 4)

    category_downloaded = 0
    category_skipped = 0

    for page in range(1, max_pages + 1):
        if category_downloaded >= target_count:
            break

        url = build_search_url(query, page, args.per_page, args.date_start, args.date_end)
        print(f"  [{query}] Fetching page {page}: {url}")
        data = fetch_page(session, url)
        time.sleep(args.sleep)

        if not data:
            print(f"  [{query}] No data returned, stopping this category.")
            break

        results = data.get("results", [])
        if not results:
            print(f"  [{query}] No results on this page, stopping this category.")
            break

        for item in results:
            if category_downloaded >= target_count:
                break

            item_id = item.get("id", "unknown").rstrip("/").split("/")[-1]
            title = (item.get("title") or "untitled").strip()
            date = item.get("date", "")
            image_url = pick_image_url(item)
            source_page = item.get("id", "")

            if not image_url:
                category_skipped += 1
                overall_skipped += 1
                continue

            local_filename = f"{item_id}.jpg"
            local_path = os.path.join(images_dir, local_filename)

            if os.path.exists(local_path):
                category_skipped += 1
                overall_skipped += 1
                continue

            ok, reason = download_image(session, image_url, local_path, min_size=args.min_size)
            if ok:
                writer.writerow([item_id, title, date, query, source_page, image_url, local_filename])
                manifest_file.flush()
                category_downloaded += 1
                overall_downloaded += 1
                if overall_downloaded % 25 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Downloaded {overall_downloaded} images total so far... "
                          f"(elapsed: {format_duration(elapsed)})")
            else:
                category_skipped += 1
                overall_skipped += 1
                print(f"    [{query}] Skipped {item_id}: {reason}")

            time.sleep(args.download_sleep)

        pagination = data.get("pagination", {})
        if not pagination.get("next"):
            print(f"  [{query}] Reached last page of results.")
            break

    print(f"  [{query}] done: {category_downloaded}/{target_count} target images "
          f"({category_skipped} skipped)")
    return category_downloaded, category_skipped, overall_downloaded, overall_skipped


def compute_category_targets(categories, weights, total_images):
    """
    Splits total_images across categories according to weights (normalized
    automatically, so they don't need to sum to exactly 1.0). Rounding is
    corrected on the last category so the targets sum exactly to total_images.
    """
    weight_sum = sum(weights)
    normalized = [w / weight_sum for w in weights]
    targets = [round(total_images * w) for w in normalized[:-1]]
    targets.append(total_images - sum(targets))  # last one absorbs rounding error
    return dict(zip(categories, targets))


def main():
    parser = argparse.ArgumentParser(description="Scrape real old photos from the Library of Congress API.")
    parser.add_argument("--query", type=str, default="portrait photograph",
                         help="single search term (ignored if --categories is set), e.g. 'family photograph'")
    parser.add_argument("--categories", type=str, default=None,
                         help="comma-separated list of search terms to split scraping across in one run, "
                              "e.g. 'portrait,street scene,farm landscape,railroad,parade'. "
                              "Overrides --query when set.")
    parser.add_argument("--weights", type=str, default=None,
                         help="comma-separated weights matching --categories, e.g. '0.3,0.2,0.2,0.15,0.15'. "
                              "Don't need to sum to 1 (auto-normalized). Omit for an equal split.")
    parser.add_argument("--total-images", type=int, default=1000,
                         help="total images to collect across all categories (only used with --categories)")
    parser.add_argument("--start-page", type=int, default=1,
                         help="page number to start from (use this to continue a previous run, "
                              "e.g. --start-page 21 to continue after a run that covered pages 1-20)")
    parser.add_argument("--pages", type=int, default=10, help="number of result pages to fetch, starting from --start-page")
    parser.add_argument("--per-page", type=int, default=100, choices=[25, 50, 100, 150],
                         help="results per page (LOC API allows 25/50/100/150)")
    parser.add_argument("--min-size", type=int, default=254,
                         help="minimum width AND height in pixels; images not strictly larger than this in both "
                              "dimensions are discarded")
    parser.add_argument("--date-start", type=str, default=None, help="e.g. 1850")
    parser.add_argument("--date-end", type=str, default=None, help="e.g. 1970")
    parser.add_argument("--out-dir", type=str, default="./real_old_photos")
    parser.add_argument("--sleep", type=float, default=3.0,
                         help="seconds to sleep between API requests (be polite -- LOC enforces rate limits)")
    parser.add_argument("--download-sleep", type=float, default=0.5,
                         help="seconds to sleep between image downloads")
    args = parser.parse_args()

    start_time = time.time()

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
        writer.writerow(["id", "title", "date", "category", "source_page_url", "image_url", "local_filename"])
    elif args.categories:
        # Warn if we're appending category-tagged rows to a manifest that
        # predates the category column -- the CSV will have mixed row shapes.
        with open(manifest_path, "r", encoding="utf-8") as f:
            existing_header = f.readline().strip()
        if "category" not in existing_header:
            print(f"Warning: {manifest_path} already exists without a 'category' column. "
                  f"New rows will have an extra column, producing a mixed-format CSV. "
                  f"Consider using a fresh --out-dir if you want a clean manifest.")

    downloaded = 0
    skipped = 0

    if args.categories:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
        if args.weights:
            weights = [float(w.strip()) for w in args.weights.split(",")]
            if len(weights) != len(categories):
                raise ValueError(f"--weights has {len(weights)} values but --categories has "
                                  f"{len(categories)} -- they must match.")
        else:
            weights = [1.0] * len(categories)  # equal split

        targets = compute_category_targets(categories, weights, args.total_images)
        print("Category targets for this run:")
        for cat, target in targets.items():
            print(f"  {cat!r}: {target} images")
        print()

        for cat, target in targets.items():
            if target <= 0:
                continue
            cat_downloaded, cat_skipped, downloaded, skipped = run_category(
                session, cat, target, images_dir, writer, manifest_file, args,
                start_time, downloaded, skipped,
            )
            elapsed = time.time() - start_time
            print(f"Running total: {downloaded} downloaded, {skipped} skipped. "
                  f"Elapsed: {format_duration(elapsed)}\n")

    else:
        # Original single-query behavior, preserved exactly, just with a
        # 'category' column added to the manifest for consistency.
        end_page = args.start_page + args.pages - 1
        for page in range(args.start_page, end_page + 1):
            url = build_search_url(args.query, page, args.per_page, args.date_start, args.date_end)
            print(f"Fetching page {page}/{end_page}: {url}")
            data = fetch_page(session, url)
            time.sleep(args.sleep)

            if not data:
                print("  No data returned, stopping.")
                break

            results = data.get("results", [])
            if not results:
                print("  No results on this page, stopping (may have reached the end).")
                break

            for item in results:
                item_id = item.get("id", "unknown").rstrip("/").split("/")[-1]
                title = (item.get("title") or "untitled").strip()
                date = item.get("date", "")
                image_url = pick_image_url(item)
                source_page = item.get("id", "")

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
                    writer.writerow([item_id, title, date, args.query, source_page, image_url, local_filename])
                    manifest_file.flush()
                    downloaded += 1
                    if downloaded % 25 == 0:
                        elapsed = time.time() - start_time
                        print(f"  Downloaded {downloaded} images so far... (elapsed: {format_duration(elapsed)})")
                else:
                    skipped += 1
                    print(f"    Skipped {item_id}: {reason}")

                time.sleep(args.download_sleep)

            pagination = data.get("pagination", {})
            elapsed = time.time() - start_time
            print(f"  Page {page} done. Total so far: {downloaded} downloaded, {skipped} skipped. "
                  f"Elapsed: {format_duration(elapsed)}")
            if not pagination.get("next"):
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
