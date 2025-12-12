
"""
Data loader utilities for Kaggle climate dataset.

Features:
- Download using KaggleHub (adapter or direct) when available.
- Reliable fallback to Kaggle CLI if KaggleHub API changes or isn't present.
- Organizes files under `data/raw/` and unzips archives.
- Loads the first CSV found or a specific file path.

Usage:
    from src.data_loader import download_kaggle_dataset, load_weather_data

    # Download (auto-select best method)
    download_kaggle_dataset(
        dataset_name="guillemservera/global-daily-climate-data",
        output_dir="data/raw",
    )

    # Load (first CSV in data/raw)
    df = load_weather_data()
    print(df.head())

    # Or load a specific file
    df = load_weather_data(file_path="data/raw/global_daily_climate_data.csv")
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helpers: credentials & environment checks
# -----------------------------------------------------------------------------
def ensure_kaggle_credentials() -> bool:
    """
    Ensure Kaggle credentials exist at ~/.kaggle/kaggle.json with permission 600.

    Returns:
        bool: True if credentials appear valid, else False.
    """
    cred_path = Path.home() / ".kaggle" / "kaggle.json"
    if not cred_path.exists():
        logger.warning("Kaggle credentials not found at %s", cred_path)
        return False

    try:
        mode = cred_path.stat().st_mode & 0o777
        if mode != 0o600:
            logger.info("Fixing permissions on %s (current: %o)", cred_path, mode)
            os.chmod(cred_path, 0o600)
    except Exception as e:
        logger.warning("Could not adjust permissions on %s: %s", cred_path, e)

    return True


def _kaggle_cli_available() -> bool:
    """Return True if Kaggle CLI is available in PATH."""
    from shutil import which
    return which("kaggle") is not None


def _kagglehub_has(symbol: str) -> bool:
    """
    Return True if kagglehub is importable and has the given attribute (symbol).
    """
    try:
        import kagglehub  # noqa: F401
        return hasattr(sys.modules["kagglehub"], symbol)
    except Exception:
        return False


def _run(cmd: str) -> str:
    """
    Run a shell command and return its combined stdout/stderr as text.
    Raises subprocess.CalledProcessError on non-zero exit.
    """
    logger.debug("Running: %s", cmd)
    res = subprocess.run(
        shlex.split(cmd),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return res.stdout


# -----------------------------------------------------------------------------
# KaggleHub download & adapter loaders
# -----------------------------------------------------------------------------
def _download_with_kagglehub(dataset_name: str, output_dir: Path) -> Path:
    """
    Attempt to download using kagglehub.dataset_download(dataset_name).
    Copies results under output_dir and returns that directory path.

    Raises:
        RuntimeError if kagglehub is missing or the symbol is not available.
    """
    if not _kagglehub_has("dataset_download"):
        raise RuntimeError("kagglehub.dataset_download() is not available in this environment.")

    import kagglehub  # imported only if needed

    logger.info("Using KaggleHub: dataset_download(%s)", dataset_name)
    # Returns a local path (cache dir) where files are stored.
    cache_path_str = kagglehub.dataset_download(dataset_name)
    cache_path = Path(cache_path_str)

    output_dir.mkdir(parents=True, exist_ok=True)
    target_dir = output_dir / cache_path.name

    # Copy/cache contents under output_dir for project consistency.
    if cache_path.is_dir():
        shutil.copytree(cache_path, target_dir, dirs_exist_ok=True)
        logger.info("Copied dataset directory to: %s", target_dir)
        return target_dir
    else:
        # Single file
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cache_path, target_dir / cache_path.name)
        logger.info("Copied dataset file to: %s", target_dir)
        return target_dir


def _load_with_kagglehub_adapter(
    dataset_name: str,
    file_path_in_bundle: str,
    pandas_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Load a file directly into pandas using KaggleHub adapter.

    Requires: pip install "kagglehub[pandas-datasets]"

    Args:
        dataset_name: e.g., "owner/dataset-slug"
        file_path_in_bundle: relative path of file inside the dataset archive/bundle.
        pandas_kwargs: kwargs forwarded to pandas reader internally.

    Returns:
        pd.DataFrame

    Raises:
        RuntimeError if adapter API is not available.
    """
    try:
        from kagglehub import KaggleDatasetAdapter  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "KaggleHub adapter API not available. Install with: "
            'pip install "kagglehub[pandas-datasets]==0.2.5"'
        ) from e

    import kagglehub

    logger.info("Using KaggleHub adapter to load %s :: %s", dataset_name, file_path_in_bundle)
    kwargs = {}
    if pandas_kwargs:
        kwargs["pandas_kwargs"] = pandas_kwargs

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        dataset_name,
        file_path_in_bundle,
        **kwargs,
    )
    return df


# -----------------------------------------------------------------------------
# Kaggle CLI download (reliable fallback)
# -----------------------------------------------------------------------------
def _download_with_kaggle_cli(dataset_name: str, output_dir: Path) -> Path:
    """
    Download a dataset using Kaggle CLI, unzip any archives, return the output directory.

    Requires:
        - Kaggle CLI installed (`pip install kaggle`)
        - Credentials at ~/.kaggle/kaggle.json (chmod 600)

    Raises:
        RuntimeError on failure.
    """
    if not _kaggle_cli_available():
        raise RuntimeError("Kaggle CLI not found. Install with: pip install kaggle")

    if not ensure_kaggle_credentials():
        logger.warning("Kaggle credentials missing or not correctly configured.")

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = f"kaggle datasets download -d {dataset_name} -p {shlex.quote(str(output_dir))} --unzip"
    logger.info("Downloading with Kaggle CLI: %s", dataset_name)

    try:
        out = _run(cmd)
        logger.debug("Kaggle CLI output:\n%s", out)
    except subprocess.CalledProcessError as e:
        logger.error("Kaggle CLI download failed:\n%s", e.stdout)
        raise RuntimeError("Kaggle CLI download failed") from e

    # If CLI didn't unzip (older behavior), unzip any .zip files present
    for z in output_dir.glob("*.zip"):
        logger.info("Unzipping %s ...", z)
        with zipfile.ZipFile(z, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        z.unlink(missing_ok=True)

    logger.info("Files in %s:", output_dir)
    for f in sorted(output_dir.glob("*")):
        logger.info(" - %s", f.name)

    return output_dir


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def download_kaggle_dataset(
    dataset_name: str = "guillemservera/global-daily-climate-data",
    output_dir: str = "data/raw",
    prefer: str = "auto",
) -> Path:
    """
    Download the Kaggle dataset into `output_dir`.

    Args:
        dataset_name: Kaggle dataset slug (e.g., "owner/dataset-slug").
        output_dir: Where to place the files. Default "data/raw".
        prefer: "auto" (default), "kagglehub", or "cli".

    Returns:
        Path: The directory containing the downloaded files.

    Strategy:
        - prefer="kagglehub": try kagglehub.dataset_download(), else raise.
        - prefer="cli": use Kaggle CLI.
        - prefer="auto": try kagglehub.dataset_download() if available; else use CLI.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading dataset: %s -> %s (prefer=%s)", dataset_name, out_dir, prefer)

    if prefer == "kagglehub":
        return _download_with_kagglehub(dataset_name, out_dir)
    elif prefer == "cli":
        return _download_with_kaggle_cli(dataset_name, out_dir)
    elif prefer == "auto":
        if _kagglehub_has("dataset_download"):
            try:
                return _download_with_kagglehub(dataset_name, out_dir)
            except Exception as e:
                logger.warning("kagglehub download failed; falling back to CLI: %s", e)
                return _download_with_kaggle_cli(dataset_name, out_dir)
        else:
            logger.info("kagglehub.dataset_download not available; using CLI.")
            return _download_with_kaggle_cli(dataset_name, out_dir)
    else:
        raise ValueError("prefer must be one of: 'auto', 'kagglehub', 'cli'")


def load_weather_data(
    file_path: str | Path | None = None,
    search_dir: str = "data/raw",
    pandas_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Load weather CSV (or parquet) data.

    Args:
        file_path: If provided, load that file (CSV or parquet).
        search_dir: If file_path is None, search this directory for the first CSV (or parquet).
        pandas_kwargs: Optional dict passed to pandas reader (e.g., dtype, parse_dates).

    Returns:
        pd.DataFrame

    Raises:
        FileNotFoundError if no suitable file is found.
    """
    if file_path is None:
        raw_dir = Path(search_dir).resolve()
        # Prefer CSV; fallback to parquet
        csv_files = sorted(raw_dir.glob("*.csv"))
        if csv_files:
            file_path = csv_files[0]
        else:
            pq_files = sorted(raw_dir.glob("*.parquet"))
            if pq_files:
                file_path = pq_files[0]
            else:
                raise FileNotFoundError(f"No CSV or parquet files found in {raw_dir}")

    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info("Loading data from: %s", file_path)

    kwargs = pandas_kwargs or {}

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path, **kwargs)
    elif file_path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    logger.info("Data shape: %s; Columns: %s", df.shape, list(df.columns))
    return df


def load_with_adapter_if_possible(
    dataset_name: str = "guillemservera/global-daily-climate-data",
    file_path_in_bundle: Optional[str] = None,
    pandas_kwargs: Optional[dict] = None,
    fallback_download_dir: str = "data/raw",
) -> pd.DataFrame:
    """
    Try KaggleHub adapter to load directly into pandas; if adapter isn't available
    or file_path_in_bundle is not provided, fall back to downloading + pandas.

    Args:
        dataset_name: Kaggle dataset slug.
        file_path_in_bundle: Relative file path inside the dataset bundle (e.g., 'data.csv').
        pandas_kwargs: Forwarded to pandas reader.
        fallback_download_dir: Where to place files if falling back.

    Returns:
        pd.DataFrame
    """
    if file_path_in_bundle and _kagglehub_has("load_dataset"):
        try:
            return _load_with_kagglehub_adapter(
                dataset_name=dataset_name,
                file_path_in_bundle=file_path_in_bundle,
                pandas_kwargs=pandas_kwargs,
            )
        except Exception as e:
            logger.warning("Adapter load failed; falling back to download: %s", e)

    # Fallback: download and then load
    download_kaggle_dataset(dataset_name=dataset_name, output_dir=fallback_download_dir, prefer="auto")
    return load_weather_data(search_dir=fallback_download_dir, pandas_kwargs=pandas_kwargs)


# -----------------------------------------------------------------------------
# Script entry point (quick test)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Script entry point (quick test)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
       # Quick test flow:
    DATASET = "guillemservera/global-daily-climate-data"
    OUTDIR = "data/raw"

    try:
        # Download (auto method selection)
        dest = download_kaggle_dataset(dataset_name=DATASET, output_dir=OUTDIR, prefer="auto")
        logger.info("Downloaded to: %s", dest)

        # Load first CSV found
        df = load_weather_data(search_dir=OUTDIR)
        print(df.head())

    except Exception as exc:
        logger.error("Test run failed: %s", exc)

