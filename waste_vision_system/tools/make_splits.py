"""
Assign labelled images to train / val / test by (camera, day), never by frame.
Frames from one camera on one day are near-duplicates; letting them straddle
train and test inflates every metric.

Writes datasets/bins/{train,val,test}.txt, plus a single-class copy for
detector training at datasets/bins_det/ (images hard-linked, labels rewritten
to class 0, data.yaml generated).

    python tools/make_splits.py
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINS = ROOT / "datasets" / "bins"
DET = ROOT / "datasets" / "bins_det"
SPLITS = (("train", 70), ("val", 85), ("test", 100))  # cumulative percent


def split_of(camera_id: str, captured_at: str) -> str:
    key = f"{camera_id}|{captured_at[:10]}"  # one group per camera per day
    bucket = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return next(name for name, upto in SPLITS if bucket < upto)


def main() -> None:
    with (BINS / "meta.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    files = {name: [] for name, _ in SPLITS}
    groups = {name: set() for name, _ in SPLITS}
    for r in rows:
        name = split_of(r["camera_id"], r["captured_at"])
        files[name].append(Path(r["image"]).name)
        groups[name].add((r["camera_id"], r["captured_at"][:10]))

    for sub in ("images", "labels"):
        (DET / sub).mkdir(parents=True, exist_ok=True)
    for name, names in files.items():
        listing = "".join(f"./images/{n}\n" for n in sorted(names))
        (BINS / f"{name}.txt").write_text(listing)
        (DET / f"{name}.txt").write_text(listing)
        for n in names:
            dst = DET / "images" / n
            if not dst.exists():
                try:
                    os.link(BINS / "images" / n, dst)  # no second copy on disk
                except OSError:
                    shutil.copy2(BINS / "images" / n, dst)
            label = BINS / "labels" / f"{Path(n).stem}.txt"
            lines = label.read_text().splitlines() if label.exists() else []
            (DET / "labels" / label.name).write_text(
                "".join("0 " + ln.split(maxsplit=1)[1] + "\n" for ln in lines if ln.strip()))
        print(f"{name:5}: {len(names):4} images from {len(groups[name])} camera-days")

    (DET / "data.yaml").write_text(
        f'path: "{DET.as_posix()}"\ntrain: train.txt\nval: val.txt\ntest: test.txt\n'
        "names:\n  0: trash can\n")


if __name__ == "__main__":
    main()
