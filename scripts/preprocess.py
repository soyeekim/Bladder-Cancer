#!/usr/bin/env python3
"""Build nnU-Net-ready label volumes from AVIEW masks on the dcm2niix image grid.

Per series (see SEGMENTATION_PIPELINE.md 0-6절 / 5단계):
  label = OR( mask/lesionAnnot3D-*.nii.gz )      # several lesion files may exist
        -> binarize to {0,1}
        -> flip the j (row) axis                   # AVIEW row order is the reverse of dcm2niix's
        -> written with the *image*'s NIfTI header  # identical dim / pixdim / affine

Every series is verified before writing:
  * dim of mask == dim of image
  * mask affine with the j-flip applied == image affine (max |diff| < 0.05 mm)
Series failing either check are recorded in the manifest and no label is written.

Single series:
    python3 scripts/preprocess.py <mask_dir> <image.nii.gz> <out_label.nii.gz>
Batch (uses data/patient_series_index.json for series -> mri_type):
    python3 scripts/preprocess.py --batch [--mri-type T2_AX ...] [--patients ID ...]
    -> data/processed/Bladder_cancer/<patient>/labels/<seriesNumber>.nii.gz
    -> reports/label_manifest.csv

No third-party dependencies (the analysis machine has no pip/nibabel).
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import struct
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BINARIZE = bytes([0] + [1] * 255)  # translate table: any non-zero byte -> 1


def read_header(path: str) -> dict:
    """Read NIfTI-1 header fields we care about (single-file .nii.gz)."""
    with gzip.open(path, "rb") as f:
        raw = f.read(352)
    dim = struct.unpack_from("<8h", raw, 40)
    srow = [list(struct.unpack_from("<4f", raw, 280 + 16 * i)) for i in range(3)]
    return dict(header=bytearray(raw), nx=dim[1], ny=dim[2], nz=dim[3], nt=dim[4] if dim[0] >= 4 else 1,
                datatype=struct.unpack_from("<h", raw, 70)[0],
                vox_offset=int(struct.unpack_from("<f", raw, 108)[0]), srow=srow)


def read_uint8_mask(path: str) -> tuple[dict, bytes]:
    """Read an AVIEW mask (datatype uint8) and return (header, binarized bytes)."""
    with gzip.open(path, "rb") as f:
        raw = f.read()
    h = _header_from_bytes(raw)
    if h["datatype"] != 2:
        raise ValueError(f"{path}: expected uint8 mask (datatype 2), got {h['datatype']}")
    n = h["nx"] * h["ny"] * h["nz"]
    return h, raw[h["vox_offset"] : h["vox_offset"] + n].translate(_BINARIZE)


def _header_from_bytes(raw: bytes) -> dict:
    dim = struct.unpack_from("<8h", raw, 40)
    srow = [list(struct.unpack_from("<4f", raw, 280 + 16 * i)) for i in range(3)]
    return dict(header=bytearray(raw[:352]), nx=dim[1], ny=dim[2], nz=dim[3], nt=dim[4] if dim[0] >= 4 else 1,
                datatype=struct.unpack_from("<h", raw, 70)[0],
                vox_offset=int(struct.unpack_from("<f", raw, 108)[0]), srow=srow)


def or_merge(masks: list[bytes]) -> bytes:
    acc = int.from_bytes(masks[0], "big")
    for m in masks[1:]:
        acc |= int.from_bytes(m, "big")
    return acc.to_bytes(len(masks[0]), "big")


def flip_j(data: bytes, nx: int, ny: int, nz: int) -> bytes:
    """Reverse the row (j) order of every slice."""
    sl = nx * ny
    out = bytearray(len(data))
    for z in range(nz):
        base = z * sl
        for y in range(ny):
            out[base + (ny - 1 - y) * nx : base + (ny - y) * nx] = data[base + y * nx : base + (y + 1) * nx]
    return bytes(out)


def flipped_affine(srow: list[list[float]], ny: int) -> list[list[float]]:
    """Affine of a volume after j -> ny-1-j reindexing."""
    return [[r[0], -r[1], r[2], r[3] + (ny - 1) * r[1]] for r in srow]


def write_label(image_header: bytearray, nx: int, ny: int, nz: int, voxels: bytes, out_path: str) -> None:
    hdr = bytearray(image_header)
    struct.pack_into("<8h", hdr, 40, 3, nx, ny, nz, 1, 1, 1, 1)
    struct.pack_into("<hh", hdr, 70, 2, 8)        # uint8
    struct.pack_into("<f", hdr, 108, 352.0)        # vox_offset
    struct.pack_into("<ff", hdr, 112, 1.0, 0.0)    # scl_slope / scl_inter
    struct.pack_into("<ff", hdr, 124, 1.0, 0.0)    # cal_max / cal_min
    hdr[148:228] = b"AVIEW lesion mask, OR-merged, j-flipped to dcm2niix grid".ljust(80, b"\x00")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with gzip.open(out_path, "wb") as f:
        f.write(bytes(hdr) + bytes(4) + voxels)


def build_label(mask_dir: str, image_path: str, out_path: str | None, affine_tol: float = 0.05) -> dict:
    """Verify + build one label. Returns a result dict (status, counts, diffs)."""
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "lesionAnnot3D-*.nii.gz")))
    res = dict(n_mask_files=len(mask_files), status="", dim_match=None, affine_max_diff=None,
               fg_voxels=None, fg_per_file="", message="")
    if not mask_files:
        res["status"] = "no_mask"; return res
    img = read_header(image_path)
    masks, hdr0 = [], None
    per_file = []
    for mf in mask_files:
        h, m = read_uint8_mask(mf)
        if hdr0 is None:
            hdr0 = h
        elif (h["nx"], h["ny"], h["nz"]) != (hdr0["nx"], hdr0["ny"], hdr0["nz"]):
            res["status"] = "mask_grid_mismatch"; res["message"] = f"{os.path.basename(mf)} grid differs"; return res
        masks.append(m); per_file.append(sum(m))
    res["fg_per_file"] = ";".join(map(str, per_file))
    res["dim_match"] = (img["nx"], img["ny"], img["nz"]) == (hdr0["nx"], hdr0["ny"], hdr0["nz"])
    if not res["dim_match"]:
        res["status"] = "dim_mismatch"
        res["message"] = f"image {(img['nx'],img['ny'],img['nz'])} vs mask {(hdr0['nx'],hdr0['ny'],hdr0['nz'])}"
        return res
    fa = flipped_affine(hdr0["srow"], hdr0["ny"])
    res["affine_max_diff"] = round(max(abs(a - b) for ra, rb in zip(fa, img["srow"]) for a, b in zip(ra, rb)), 4)
    merged = or_merge(masks) if len(masks) > 1 else masks[0]
    res["fg_voxels"] = sum(merged)
    if res["affine_max_diff"] > affine_tol:
        res["status"] = "affine_mismatch"; res["message"] = "j-flip does not align mask to image"; return res
    if res["fg_voxels"] == 0:
        res["status"] = "empty_label"
    else:
        res["status"] = "ok"
    if out_path:
        write_label(img["header"], img["nx"], img["ny"], img["nz"], flip_j(merged, img["nx"], img["ny"], img["nz"]), out_path)
        res["label_path"] = out_path
    return res


def batch(mri_types: list[str] | None, patients: list[str] | None) -> int:
    index = json.load(open(os.path.join(ROOT, "data", "patient_series_index.json"), encoding="utf-8"))
    manifest_path = os.path.join(ROOT, "reports", "label_manifest.csv")
    prev: dict[tuple[str, str], dict] = {}
    if os.path.exists(manifest_path):  # keep rows of series not processed in this run
        for r in csv.DictReader(open(manifest_path, encoding="utf-8")):
            prev[(r["patient_id"], r["series_folder"])] = r
    rows: list[dict] = []
    t0 = time.time(); n = 0; counts: dict[str, int] = {}
    for p in index["patients"]:
        pid = p["patient_id"]
        if patients and pid not in patients:
            continue
        for s in p["series"]:
            if mri_types and s["mri_type"] not in mri_types:
                continue
            folder = s["folder"]; sno = folder.split("_")[0].lstrip("0") or "0"
            image = os.path.join(ROOT, "data", "processed", "Bladder_cancer", pid, "nifti", f"{sno}.nii.gz")
            mask_dir = os.path.join(ROOT, "data", "raw", "Bladder_cancer", pid, folder, "mask")
            out = os.path.join(ROOT, "data", "processed", "Bladder_cancer", pid, "labels", f"{sno}.nii.gz")
            row = dict(patient_id=pid, series_folder=folder, series_no=sno, mri_type=s["mri_type"],
                       image_path=os.path.relpath(image, ROOT), label_path="")
            if not os.path.exists(image):
                row.update(status="no_image", n_mask_files=len(glob.glob(os.path.join(mask_dir, "lesionAnnot3D-*.nii.gz"))),
                           dim_match="", affine_max_diff="", fg_voxels="", fg_per_file="", message="dcm2niix output missing")
            else:
                try:
                    r = build_label(mask_dir, image, out)
                except Exception as e:  # keep going, record the error
                    r = dict(status="error", n_mask_files="", dim_match="", affine_max_diff="", fg_voxels="", fg_per_file="", message=str(e)[:200])
                if r.get("label_path"):
                    r["label_path"] = os.path.relpath(r["label_path"], ROOT)
                row.update({k: ("" if v is None else v) for k, v in r.items()})
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            prev[(pid, folder)] = row; n += 1
            if n % 100 == 0:
                print(f"{n} series | {counts} | {time.time()-t0:.0f}s", flush=True)
    fields = ["patient_id", "series_folder", "series_no", "mri_type", "status", "n_mask_files", "dim_match",
              "affine_max_diff", "fg_voxels", "fg_per_file", "image_path", "label_path", "message"]
    rows = sorted(prev.values(), key=lambda r: (r["patient_id"], r["series_folder"]))
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(f"done: {n} series processed in {time.time()-t0:.0f}s | {counts}\nmanifest: {manifest_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mask_dir", nargs="?"); ap.add_argument("image_nifti", nargs="?"); ap.add_argument("out", nargs="?")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--mri-type", nargs="*", help="e.g. T2_AX (default: all)")
    ap.add_argument("--patients", nargs="*")
    a = ap.parse_args()
    if a.batch:
        return batch(a.mri_type, a.patients)
    if not (a.mask_dir and a.image_nifti and a.out):
        ap.error("single mode needs mask_dir image_nifti out (or use --batch)")
    r = build_label(a.mask_dir, a.image_nifti, a.out)
    print(json.dumps(r, ensure_ascii=False))
    return 0 if r["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
