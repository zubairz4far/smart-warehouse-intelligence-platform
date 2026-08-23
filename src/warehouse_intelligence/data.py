from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

UCI_DATASET_ID = 352
UCI_DOI = "10.24432/C5BW33"
UCI_DATASET_URL = "https://archive.ics.uci.edu/dataset/352/online-retail"
ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip"
ARCHIVE_FILENAME = "online_retail.zip"
WORKBOOK_FILENAME = "Online Retail.xlsx"
EXPECTED_ROWS = 541_909
REQUIRED_COLUMNS = {
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
}
SERVICE_STOCK_CODES = {
    "BANK CHARGES",
    "C2",
    "CRUK",
    "D",
    "DOT",
    "M",
    "PADS",
    "POST",
}


@dataclass(frozen=True)
class RetailDataset:
    frame: pd.DataFrame
    archive_path: Path
    workbook_path: Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "warehouse-intelligence-v0.1"})
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    temporary.replace(destination)


def ensure_retail_file(data_dir: str | Path, *, download: bool = False) -> tuple[Path, Path]:
    root = Path(data_dir)
    archive_path = root / ARCHIVE_FILENAME
    workbook_path = root / WORKBOOK_FILENAME
    if workbook_path.exists():
        return archive_path, workbook_path
    if not archive_path.exists():
        if not download:
            message = (
                f"Missing {WORKBOOK_FILENAME}. Re-run with --download or place the UCI file "
                f"in {root}."
            )
            raise FileNotFoundError(message)
        _download(ARCHIVE_URL, archive_path)
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        matches = [name for name in archive.namelist() if Path(name).name == WORKBOOK_FILENAME]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {WORKBOOK_FILENAME} in {archive_path}")
        with archive.open(matches[0]) as source, workbook_path.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
    return archive_path, workbook_path


def load_online_retail(
    data_dir: str | Path,
    *,
    download: bool = False,
    validate_rows: bool = True,
) -> RetailDataset:
    archive_path, workbook_path = ensure_retail_file(data_dir, download=download)
    frame = pd.read_excel(workbook_path, engine="openpyxl")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Online Retail workbook is missing columns: {sorted(missing)}")
    if validate_rows and len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(frame)}")
    return RetailDataset(frame=frame, archive_path=archive_path, workbook_path=workbook_path)


def clean_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep physical, positive shipments and derive warehouse-relevant fields."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cleaned = frame.copy()
    cleaned["InvoiceNo"] = cleaned["InvoiceNo"].astype(str).str.strip()
    cleaned["StockCode"] = cleaned["StockCode"].astype(str).str.strip().str.upper()
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"], errors="coerce")
    cleaned["Quantity"] = pd.to_numeric(cleaned["Quantity"], errors="coerce")
    cleaned["UnitPrice"] = pd.to_numeric(cleaned["UnitPrice"], errors="coerce")

    cancellation = cleaned["InvoiceNo"].str.upper().str.startswith("C")
    physical_item = ~cleaned["StockCode"].isin(SERVICE_STOCK_CODES)
    valid = (
        ~cancellation
        & physical_item
        & cleaned["InvoiceDate"].notna()
        & cleaned["StockCode"].ne("")
        & cleaned["Quantity"].gt(0)
        & cleaned["UnitPrice"].gt(0)
    )
    cleaned = cleaned.loc[valid].copy()
    cleaned["line_value"] = cleaned["Quantity"] * cleaned["UnitPrice"]
    cleaned["week_start"] = cleaned["InvoiceDate"].dt.to_period("W-SUN").dt.start_time
    return cleaned.reset_index(drop=True)
