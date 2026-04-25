# -*- coding: utf-8 -*-
"""
PyABSA discover datasets via findfile keywords: folder path must match ``train`` + ``APC``
(and ``test`` + ``APC`` for test). Raw SemEval-style ``*.xml.seg`` names are not detected.

This script copies your originals into detectable ``*.apc`` names (same content).

Run once from repo root:
  python thesis_apc_baseline/dataset/ensure_pyabsa_names.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent

# (source glob or exact name relative to dataset folder, target name)
JOBS = [
    ("Restaurants_Train.xml.seg", "train.APC.restaurants.apc"),
    ("Restaurants_Test_Gold.xml.seg", "test.APC.restaurants.apc"),
]


def main():
    for src_name, dst_name in JOBS:
        dst = _HERE / dst_name
        if dst.exists():
            print("skip (exists):", dst.name)
            continue
        src = _HERE / src_name
        if not src.exists():
            print("missing source:", src, file=sys.stderr)
            sys.exit(1)
        shutil.copyfile(src, dst)
        print("created:", dst.name, "<-", src.name)


if __name__ == "__main__":
    main()
