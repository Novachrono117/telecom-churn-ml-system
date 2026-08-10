"""Data acquisition, loading and understanding for the churn dataset."""

from churn.data.loader import (
    EXPECTED_RAW_SHA256,
    RAW_FILENAME,
    RawDatasetMismatchError,
    load_raw_text,
    load_raw_typed,
    raw_csv_path,
    verify_raw_dataset,
)
from churn.data.provenance import FileProvenance, describe_file

__all__ = [
    "EXPECTED_RAW_SHA256",
    "RAW_FILENAME",
    "FileProvenance",
    "RawDatasetMismatchError",
    "describe_file",
    "load_raw_text",
    "load_raw_typed",
    "raw_csv_path",
    "verify_raw_dataset",
]
