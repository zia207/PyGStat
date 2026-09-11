#!/usr/bin/env python3
"""
Read Fortran files with Python.

Two use cases:
1. Read Fortran source code (.f, .f90, .f95) as text.
2. Read Fortran unformatted (binary) data files — Fortran writes each record
   as: [4-byte record length][data][4-byte record length].
"""

import struct
from pathlib import Path
from typing import BinaryIO, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# 1. Read Fortran source code (plain text)
# -----------------------------------------------------------------------------

def read_fortran_source(filepath: str) -> str:
    """Read a Fortran source file (.f, .f90, .for, etc.) as text."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    return path.read_text(encoding="utf-8", errors="replace")


# -----------------------------------------------------------------------------
# 2. Read Fortran unformatted (binary) sequential files
# -----------------------------------------------------------------------------

def read_fortran_record(f: BinaryIO, dtype: str = "f4") -> Optional[np.ndarray]:
    """
    Read one record from a Fortran unformatted sequential file.

    Fortran format: [4-byte int: record length in bytes] [data] [same 4-byte int]

    Parameters
    ----------
    f : file-like (binary)
        Open file in 'rb' mode.
    dtype : str or numpy dtype
        Data type of the record, e.g. 'f4' (float32), 'i4' (int32), 'f8'.

    Returns
    -------
    np.ndarray or None
        Data array for the record, or None at end of file.
    """
    # Read leading record length (4 bytes, big-endian by Fortran default; use 'i' for native)
    lead = f.read(4)
    if len(lead) < 4:
        return None
    rec_len = struct.unpack("i", lead)[0]
    nbytes = rec_len
    # Number of elements
    dtype = np.dtype(dtype)
    n = nbytes // dtype.itemsize
    data = np.frombuffer(f.read(nbytes), dtype=dtype, count=n)
    # Trailing record length
    trail = f.read(4)
    if len(trail) < 4:
        raise ValueError("Unexpected end of file: missing trailing record marker")
    return data


def read_fortran_unformatted(
    filepath: str,
    dtype: str = "f4",
    max_records: Optional[int] = None,
) -> list:
    """
    Read all records from a Fortran unformatted sequential file.

    Parameters
    ----------
    filepath : str
        Path to the binary file.
    dtype : str
        NumPy dtype for each record (e.g. 'f4', 'i4', 'f8').
    max_records : int, optional
        If set, stop after this many records.

    Returns
    -------
    list of np.ndarray
        One array per record.
    """
    records = []
    with open(filepath, "rb") as f:
        while True:
            rec = read_fortran_record(f, dtype=dtype)
            if rec is None:
                break
            records.append(rec)
            if max_records is not None and len(records) >= max_records:
                break
    return records


# -----------------------------------------------------------------------------
# 3. Byte order: big-endian vs little-endian
# -----------------------------------------------------------------------------

def read_fortran_record_endian(
    f: BinaryIO,
    dtype: str = "f4",
    endian: str = ">",
) -> Optional[np.ndarray]:
    """
    Read one record with explicit byte order.

    endian: '>' big-endian (typical for Fortran), '<' little-endian.
    """
    lead = f.read(4)
    if len(lead) < 4:
        return None
    rec_len = struct.unpack(f"{endian}i", lead)[0]
    dtype = np.dtype(dtype)
    n = rec_len // dtype.itemsize
    # NumPy dtype with byte order, e.g. '>f4'
    dtype_ordered = f"{endian}{dtype.str.lstrip('<>|=')}"
    data = np.frombuffer(f.read(rec_len), dtype=dtype_ordered, count=n)
    f.read(4)  # trailing length
    return data


# -----------------------------------------------------------------------------
# Example / test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python read_fortran.py <file>           # Fortran source → print text")
        print("  python read_fortran.py <file> --binary  # Fortran unformatted → print record shapes")
        sys.exit(0)

    path = sys.argv[1]
    binary = "--binary" in sys.argv

    if binary:
        print(f"Reading Fortran unformatted binary: {path}")
        records = read_fortran_unformatted(path, dtype="f4")
        for i, rec in enumerate(records):
            print(f"  Record {i}: shape={rec.shape}, dtype={rec.dtype}")
            if rec.size <= 20:
                print(f"    values = {rec}")
    else:
        print(f"Reading Fortran source: {path}")
        text = read_fortran_source(path)
        print(text[:2000])
        if len(text) > 2000:
            print("... [truncated]")
