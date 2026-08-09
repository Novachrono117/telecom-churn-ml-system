"""File provenance: identity and integrity of the raw dataset.

The hash lets any future reader confirm they are analyzing the very same bytes
that produced the recorded results.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

_CHUNK_SIZE = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class FileProvenance:
    """Identity of a file on disk."""

    filename: str
    size_bytes: int
    sha256: str


def compute_sha256(path: Path, chunk_size: int = _CHUNK_SIZE) -> str:
    """Return the SHA-256 of a file, read in chunks.

    Args:
        path: File to hash.
        chunk_size: Read buffer size in bytes.

    Returns:
        The lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(path: Path) -> FileProvenance:
    """Collect name, size and SHA-256 of a file.

    Args:
        path: File to describe.

    Returns:
        The corresponding :class:`FileProvenance`.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Cannot describe missing file: {path}")
    return FileProvenance(
        filename=path.name,
        size_bytes=path.stat().st_size,
        sha256=compute_sha256(path),
    )
