from __future__ import annotations

import argparse
import csv
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable
from urllib.parse import urlparse

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
TRAINING_DIR = Path(__file__).resolve().parent
DEFAULT_CSVS = [
    TRAINING_DIR / "data" / "non-annotation-dataset" / "clean" / "train_clean.csv",
    TRAINING_DIR / "data" / "non-annotation-dataset" / "clean" / "val_clean.csv",
    TRAINING_DIR / "data" / "non-annotation-dataset" / "clean" / "test_clean.csv",
]
DEFAULT_OUT_DIR = TRAINING_DIR / "data" / "non-annotation-dataset" / "images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download meme images referenced by the training CSV splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csvs",
        nargs="+",
        default=[str(path) for path in DEFAULT_CSVS],
        help="CSV files to scan for meme_post_id/image_url pairs.",
    )
    parser.add_argument(
        "--out_dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory where downloaded images will be stored.",
    )
    parser.add_argument(
        "--fail_log",
        default="",
        help="Optional CSV log path for failures. Defaults to <out_dir>/download_failures.csv.",
    )
    parser.add_argument("--timeout", type=int, default=20, help="Per-request timeout in seconds.")
    parser.add_argument("--max_retries", type=int, default=5, help="Max retries per image.")
    parser.add_argument(
        "--retry_sleep",
        type=float,
        default=2.0,
        help="Base sleep time in seconds between retries.",
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=0.2,
        help="Sleep time in seconds after each processed image.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=100,
        help="Print progress after this many processed images.",
    )
    return parser.parse_args()


def read_rows(csv_paths: Iterable[Path]) -> Dict[str, str]:
    rows: Dict[str, str] = {}
    for csv_path in csv_paths:
        with open(csv_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                meme_id = row.get("meme_post_id", "").strip()
                image_url = row.get("image_url", "").strip()
                if meme_id and image_url and meme_id not in rows:
                    rows[meme_id] = image_url
    return rows


def infer_extension(url: str) -> str:
    ext = Path(urlparse(url).path).suffix.lower()
    return ext if ext in VALID_EXTS else ".jpg"


def image_exists(out_dir: Path, meme_id: str) -> bool:
    return any((out_dir / f"{meme_id}{ext}").exists() for ext in VALID_EXTS)


def main() -> None:
    args = parse_args()
    csv_paths = [Path(path).expanduser().resolve() for path in args.csvs]
    out_dir = Path(args.out_dir).expanduser().resolve()
    fail_log = (
        Path(args.fail_log).expanduser().resolve()
        if args.fail_log
        else out_dir / "download_failures.csv"
    )

    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    fail_log.parent.mkdir(parents=True, exist_ok=True)

    rows = read_rows(csv_paths)
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]

    downloaded = 0
    skipped = 0
    failed = 0

    with open(fail_log, "w", newline="", encoding="utf-8") as log_handle:
        writer = csv.writer(log_handle)
        writer.writerow(["meme_post_id", "url", "status", "reason"])

        for index, (meme_id, url) in enumerate(rows.items(), start=1):
            if image_exists(out_dir, meme_id):
                skipped += 1
                continue

            dst = out_dir / f"{meme_id}{infer_extension(url)}"
            success = False

            for attempt in range(args.max_retries):
                try:
                    with opener.open(url, timeout=args.timeout) as response, open(dst, "wb") as out_handle:
                        out_handle.write(response.read())
                    downloaded += 1
                    success = True
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code == 404:
                        writer.writerow([meme_id, url, 404, "not_found"])
                        failed += 1
                        break
                    if exc.code == 429 and attempt < args.max_retries - 1:
                        time.sleep(args.retry_sleep * (attempt + 1) * 2)
                        continue
                    writer.writerow([meme_id, url, exc.code, "http_error"])
                    failed += 1
                    break
                except Exception as exc:
                    if attempt < args.max_retries - 1:
                        time.sleep(args.retry_sleep * (attempt + 1))
                        continue
                    writer.writerow([meme_id, url, "error", str(exc)])
                    failed += 1

            if index % args.progress_every == 0:
                print(
                    f"processed={index} downloaded={downloaded} "
                    f"skipped={skipped} failed={failed}"
                )
            if not success and dst.exists():
                dst.unlink(missing_ok=True)
            time.sleep(args.throttle)

    print(
        f"done: downloaded={downloaded} skipped={skipped} failed={failed} "
        f"fail_log={fail_log}"
    )


if __name__ == "__main__":
    main()
