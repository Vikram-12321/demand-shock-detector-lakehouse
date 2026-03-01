"""
download_m5.py — Download M5 Forecasting dataset from Kaggle.

Usage:
    # With Kaggle credentials configured (~/.kaggle/kaggle.json):
    python scripts/download_m5.py

    # With a locally provided zip file:
    python scripts/download_m5.py --zip-path /path/to/m5-forecasting-accuracy.zip

The raw files are extracted to data/raw/m5/ (gitignored).
"""

import argparse
import os
import zipfile
from pathlib import Path


RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "m5"


def download_via_kaggle(output_dir: Path) -> None:
    """Download M5 dataset using the Kaggle CLI / Python API."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise ImportError(
            "kaggle package not installed. Run: pip install kaggle\n"
            "Also ensure ~/.kaggle/kaggle.json exists with your API credentials."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading M5 dataset to {output_dir} …")
    os.system(
        f"kaggle competitions download -c m5-forecasting-accuracy -p {output_dir}"
    )
    _unzip_all(output_dir)


def extract_from_local_zip(zip_path: Path, output_dir: Path) -> None:
    """Extract a locally provided M5 zip archive."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} → {output_dir} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    _unzip_all(output_dir)
    print("Done.")


def _unzip_all(directory: Path) -> None:
    """Recursively extract any nested zip files found in directory."""
    for zf_path in directory.glob("*.zip"):
        print(f"  Extracting nested zip: {zf_path.name}")
        with zipfile.ZipFile(zf_path, "r") as zf:
            zf.extractall(directory)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download M5 Forecasting dataset")
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Path to a locally downloaded M5 zip archive (skips Kaggle download).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Directory to extract files into. Default: {RAW_DIR}",
    )
    args = parser.parse_args()

    if args.zip_path:
        extract_from_local_zip(args.zip_path, args.output_dir)
    else:
        download_via_kaggle(args.output_dir)


if __name__ == "__main__":
    main()
