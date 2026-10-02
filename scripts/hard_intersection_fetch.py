"""Fetch only pinned evaluation inputs, with hashes, resume and a disk reserve.

The dataset is CC BY 4.0, Dynamic Map Platform Co., Ltd. (2026).
Images, reconstruction and simulator assets are deliberately not in this manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

REPO = "dynamic-maps/hard-intersection-multimodal-sample"
REVISION = "e8e8d2a5d49a8b9cb63b5b5ecef9b260ff48f39c"
RAW = "pointcloud/jp_tokyo_takanawadai.las"
LABELLED = "annotation/semantic_pointcloud/jp_tokyo_takanawadai_class.las"
SMALL = ["README.md", "maps/lanelet2/jp_tokyo_takanawadai.osm",
         "annotation/semantic_images/jp_tokyo_takanawadai_images_annotations.json",
         "calibration/cameras.txt", "calibration/images.txt"] + [
    f"trajectory/{name}_260217.txt" for name in
    ["26047_Record004", "26047_Record050", "26047a_Record004", "26047a_Record084"]]


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def verify(path: Path, item: dict) -> str:
    if path.stat().st_size != item["size"]:
        raise ValueError(f"size mismatch: {item['path']}")
    actual = digest(path)
    expected = item.get("lfs", {}).get("oid")
    if expected:
        valid = actual == expected
    else:
        sha = hashlib.sha1(f"blob {item['size']}\0".encode())
        with path.open("rb") as f:
            for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
                sha.update(block)
        valid = sha.hexdigest() == item["oid"]
    if not valid:
        raise ValueError(f"pinned source hash mismatch: {item['path']}; file retained for inspection")
    return actual


def download(root: Path, item: dict) -> dict:
    name, size = item["path"], item["size"]
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return {"path": name, "bytes": size, "sha256": verify(target, item)}
    part = target.with_name(target.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    if start > size:
        raise ValueError(f"oversized partial input: {part}")
    if shutil.disk_usage(root).free < size - start + 6 * 1024**3:
        raise ValueError("download would leave less than 6 GiB free")
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}"
    with part.open("ab") as f:
        while start < size:
            end = min(size, start + 16 * 1024 * 1024) - 1
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=90) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                    raise ValueError(f"server did not return the requested bounded range for {name}")
                data = response.read(end - start + 2)
            if len(data) != end - start + 1:
                raise ValueError(f"short or oversized range for {name}")
            f.write(data)
            start = end + 1
            if start == size or start % (128 * 1024 * 1024) == 0:
                print(f"{name}: {start:,}/{size:,} bytes", flush=True)
    actual = verify(part, item)
    part.replace(target)
    return {"path": name, "bytes": size, "sha256": actual}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--clouds", action="store_true", help="Also fetch the two distinct ~1.21 GB LAS files")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    api = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?recursive=true&limit=1000"
    with urllib.request.urlopen(api, timeout=30) as response:
        if response.headers.get("Link"):
            raise ValueError("unexpected paginated manifest; inspect before downloading")
        entries = {x["path"]: x for x in json.load(response) if x["type"] == "file"}
    selected = SMALL + ([RAW, LABELLED] if args.clouds else [])
    outputs = [download(args.directory, entries[name]) for name in selected]
    manifest = {"repository": REPO, "revision": REVISION, "license": "CC-BY-4.0",
                "attribution": "Hard Intersection Multimodal Samples, Dynamic Map Platform Co., Ltd., 2026",
                "source_url": f"https://huggingface.co/datasets/{REPO}", "files": outputs,
                "raw_cloud_and_labels_are_distinct": True}
    (args.directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {sum(x['bytes'] for x in outputs):,} bytes; no images or 3DGS fetched.")


if __name__ == "__main__":
    main()
