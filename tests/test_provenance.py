"""Tests for file provenance."""

from __future__ import annotations

from pathlib import Path

import pytest

from churn.data.provenance import compute_sha256, describe_file

#: Published SHA-256 of the three bytes "abc".
SHA256_OF_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sha256_matches_known_vector(tmp_path: Path) -> None:
    path = tmp_path / "abc.bin"
    path.write_bytes(b"abc")

    assert compute_sha256(path) == SHA256_OF_ABC


def test_sha256_is_independent_of_chunk_size(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"x" * 5000)

    assert compute_sha256(path, chunk_size=7) == compute_sha256(path, chunk_size=4096)


def test_describe_file_reports_name_size_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "abc.bin"
    path.write_bytes(b"abc")

    provenance = describe_file(path)

    assert provenance.filename == "abc.bin"
    assert provenance.size_bytes == 3
    assert provenance.sha256 == SHA256_OF_ABC


def test_describe_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        describe_file(tmp_path / "absent.bin")
