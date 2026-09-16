#!/usr/bin/env python3
"""Assemble an nnU-Net v2 raw dataset from data/processed/Bladder_cancer/ (images + labels).

    data/nnUNet_raw/Dataset<ID>_<NAME>/
        dataset.json
        imagesTr/<NAME>_<patient>_0000.nii.gz   labelsTr/<NAME>_<patient>.nii.gz
        imagesTs/<NAME>_<patient>_0000.nii.gz   labelsTs/<NAME>_<patient>.nii.gz   (held-out test)

Policy (SEGMENTATION_PIPELINE.md 3단계 7, 0-5절, 8단계; NORMAL_COHORT.md):
  * one case per patient, one channel (default T2_AX)
  * label = data/processed/Bladder_cancer/<pid>/labels/<n>.nii.gz with 26-connected components
    of <= --min-component voxels removed (stray brush marks); removals are logged
  * patients whose T2_AX continuity status is REVIEW are excluded unless
    --include-review (they wait for radiologist confirmation)
  * patient-level split, fixed seed, test fraction stratified by lesion-volume tertile
  * --include-normal adds the normal cohort (data/processed/Bladder_normal/, all-zero labels).
    Case id = <NAME>_n<patientID>. The 48 patients also present in the cancer cohort (post-treatment
    follow-up) are forced into the SAME split as their cancer scan to avoid leakage; the rest are
    split independently with the same seed/test_frac.

Outputs reports/nnunet_dataset_manifest.csv (case -> patient, group, split, voxels, removed components).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import shutil
import struct
import sys
from collections import defaultdict, deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- nifti io
def read_nii(path: str) -> tuple[bytearray, int, int, int, bytearray]:
    raw = gzip.open(path, "rb").read()
    dim = struct.unpack_from("<8h", raw, 40)
    off = int(struct.unpack_from("<f", raw, 108)[0])
    return bytearray(raw[:352]), dim[1], dim[2], dim[3], bytearray(raw[off:])


def write_nii(hdr: bytearray, voxels: bytes, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(bytes(hdr) + bytes(4) + voxels)


# ------------------------------------------------- connected components
_NB = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]


def remove_small_components(vox: bytearray, nx: int, ny: int, nz: int, min_size: int) -> tuple[int, list[int]]:
    """Zero every 26-connected component with <= min_size voxels. Returns (n_components_total, removed_sizes)."""
    sl = nx * ny
    fg = {i for i, v in enumerate(vox) if v}
    seen: set[int] = set()
    removed: list[int] = []
    n_comp = 0
    for start in list(fg):
        if start in seen:
            continue
        n_comp += 1
        comp = [start]; seen.add(start); q = deque([start])
        while q:
            i = q.popleft()
            x, y, z = i % nx, (i // nx) % ny, i // sl
            for dx, dy, dz in _NB:
                X, Y, Z = x + dx, y + dy, z + dz
                if 0 <= X < nx and 0 <= Y < ny and 0 <= Z < nz:
                    j = X + Y * nx + Z * sl
                    if j in fg and j not in seen:
                        seen.add(j); q.append(j); comp.append(j)
        if len(comp) <= min_size:
            removed.append(len(comp))
            for j in comp:
                vox[j] = 0
    return n_comp, removed


# ---------------------------------------------------------------- helpers
def link_or_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def continuity_status(path: str) -> dict[tuple[str, str], str]:
    """Worst continuity status per (patient, series_folder) from reports/mask_continuity_summary.csv."""
    rank = {"PASS": 0, "REVIEW_MINOR": 1, "REVIEW_MULTIFOCAL": 2, "REVIEW": 3, "EXCLUDE": 0}
    worst: dict[tuple[str, str], str] = {}
    if not os.path.exists(path):
        return worst
    for r in csv.DictReader(open(path, encoding="utf-8")):
        k = (r["patient"], r["series"])
        if rank.get(r["status"], 0) >= rank.get(worst.get(k, "PASS"), 0):
            worst[k] = r["status"]
    return worst


def stratified_split(cases: list[dict], test_frac: float, seed: int) -> None:
    """Assign case['split'] in {'train','test'}; stratify by lesion-volume tertile."""
    vols = sorted(c["fg_voxels"] for c in cases)
    t1, t2 = vols[len(vols) // 3], vols[2 * len(vols) // 3]
    strata: dict[int, list[dict]] = defaultdict(list)
    for c in cases:
        strata[0 if c["fg_voxels"] < t1 else 1 if c["fg_voxels"] < t2 else 2].append(c)
    rng = random.Random(seed)
    for group in strata.values():
        rng.shuffle(group)
        n_test = round(len(group) * test_frac)
        for i, c in enumerate(group):
            c["split"] = "test" if i < n_test else "train"


def load_normal_cases(name: str) -> list[dict]:
    """Normal cohort rows (T2 AX, all-zero label) from scripts/process_normal.py output."""
    path = os.path.join(ROOT, "reports", "normal_conversion_manifest.csv")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found; run scripts/process_normal.py --index --convert first", file=sys.stderr)
        return []
    out = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["status"] not in ("ok", "ok_fallback"):
            continue
        out.append(dict(patient_id=r["patient_id"], case_id=f"{name}_{r['normal_id']}",
                        image=os.path.join(ROOT, r["image_path"]), label=os.path.join(ROOT, r["label_path"]),
                        also_in_cancer_cohort=r["also_in_cancer_cohort"] == "True", group="normal",
                        continuity="", fg_voxels_before=0))
    return out


# ------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-id", type=int, default=1)
    ap.add_argument("--name", default="BladderTumor")
    ap.add_argument("--mri-type", default="T2_AX")
    ap.add_argument("--channel-name", default="T2")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "data", "nnUNet_raw"))
    ap.add_argument("--min-component", type=int, default=10, help="remove 26-connected components with <= this many voxels")
    ap.add_argument("--include-review", action="store_true", help="also include patients whose continuity status is REVIEW")
    ap.add_argument("--max-voxels", type=int, default=0,
                    help="exclude labels with more foreground voxels than this; 0 = off (large masks were verified to be genuine tumors, 2026-09-15)")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-normal", action="store_true", help="add the normal cohort (NORMAL_COHORT.md)")
    a = ap.parse_args()

    ds_dir = os.path.join(a.out_root, f"Dataset{a.dataset_id:03d}_{a.name}")
    label_rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "reports", "label_manifest.csv"), encoding="utf-8"))
                  if r["mri_type"] == a.mri_type and r["status"] == "ok"]
    cont = continuity_status(os.path.join(ROOT, "reports", "mask_continuity_summary.csv"))

    cases: list[dict] = []
    excluded: list[dict] = []
    for r in label_rows:
        pid = r["patient_id"]
        c = dict(patient_id=pid, series_folder=r["series_folder"], case_id=f"{a.name}_{pid}", group="tumor",
                 image=os.path.join(ROOT, r["image_path"]), label=os.path.join(ROOT, r["label_path"]),
                 continuity=cont.get((pid, r["series_folder"]), "PASS"), fg_voxels_before=int(r["fg_voxels"]))
        if c["continuity"] == "REVIEW" and not a.include_review:
            c["split"] = "excluded_review"; excluded.append(c); continue
        if a.max_voxels and c["fg_voxels_before"] > a.max_voxels:
            c["split"] = "excluded_too_large"; excluded.append(c); continue
        cases.append(c)
    # one case per patient guaranteed? (T2_AX is unique per patient in this dataset)
    seen = set()
    for c in cases:
        if c["patient_id"] in seen:
            print(f"WARNING: patient {c['patient_id']} has >1 {a.mri_type} series; keeping first", file=sys.stderr)
        seen.add(c["patient_id"])
    cases = [c for i, c in enumerate(cases) if c["patient_id"] not in {x["patient_id"] for x in cases[:i]}]

    # clean labels + place files
    for d in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        shutil.rmtree(os.path.join(ds_dir, d), ignore_errors=True)

    def clean(clist: list[dict], label: str) -> None:
        for i, c in enumerate(clist, 1):
            hdr, nx, ny, nz, vox = read_nii(c["label"])
            n_comp, removed = remove_small_components(vox, nx, ny, nz, a.min_component)
            c.update(n_components=n_comp, removed_components=len(removed), removed_voxels=sum(removed),
                     fg_voxels=sum(vox))
            c["_hdr"], c["_vox"] = hdr, bytes(vox)
            if i % 50 == 0:
                print(f"cleaned {i}/{len(clist)} {label} labels", flush=True)

    clean(cases, "tumor")
    stratified_split(cases, a.test_frac, a.seed)
    for c in cases:
        c["split_source"] = "volume_stratified"

    normal_cases: list[dict] = []
    if a.include_normal:
        normal_cases = load_normal_cases(a.name)
        clean(normal_cases, "normal")
        patient_split = {c["patient_id"]: c["split"] for c in cases}  # only patients actually included as tumor cases
        locked = [c for c in normal_cases if c["also_in_cancer_cohort"] and c["patient_id"] in patient_split]
        locked_ids = {c["case_id"] for c in locked}
        free = [c for c in normal_cases if c["case_id"] not in locked_ids]
        for c in locked:
            c["split"] = patient_split[c["patient_id"]]; c["split_source"] = "inherited_from_cancer_scan"
        stratified_split(free, a.test_frac, a.seed)  # all fg_voxels==0 -> plain seeded shuffle-split
        for c in free:
            c["split_source"] = "independent_random"
        print(f"normal cohort: {len(normal_cases)} cases | locked to cancer-scan split: {len(locked)} "
              f"| independently split: {len(free)}")

    all_cases = cases + normal_cases
    for c in all_cases:
        img_dir, lab_dir = ("imagesTr", "labelsTr") if c["split"] == "train" else ("imagesTs", "labelsTs")
        link_or_copy(c["image"], os.path.join(ds_dir, img_dir, f"{c['case_id']}_0000.nii.gz"))
        write_nii(c["_hdr"], c["_vox"], os.path.join(ds_dir, lab_dir, f"{c['case_id']}.nii.gz"))
        del c["_hdr"], c["_vox"]

    n_train = sum(c["split"] == "train" for c in all_cases)
    desc = (f"Bladder tumor segmentation, {a.mri_type} single channel. "
            f"Labels: AVIEW lesion masks OR-merged, binarized, j-flipped to dcm2niix grid, "
            f"components <= {a.min_component} voxels removed. Patient-level split seed={a.seed}.")
    if a.include_normal:
        desc += (f" Includes {len(normal_cases)} normal (tumor-free) cases with all-zero labels "
                 f"(case id prefix 'n'; NORMAL_COHORT.md); patients also present in the cancer cohort "
                 f"share that patient's split to avoid leakage.")
    dataset_json = {
        "channel_names": {"0": a.channel_name},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": n_train,
        "file_ending": ".nii.gz",
        "name": a.name,
        "description": desc,
    }
    os.makedirs(ds_dir, exist_ok=True)
    json.dump(dataset_json, open(os.path.join(ds_dir, "dataset.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    man = os.path.join(ROOT, "reports", "nnunet_dataset_manifest.csv")
    fields = ["case_id", "patient_id", "group", "series_folder", "split", "split_source", "continuity",
              "fg_voxels_before", "fg_voxels", "n_components", "removed_components", "removed_voxels",
              "also_in_cancer_cohort"]
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval=""); w.writeheader()
        w.writerows(sorted(all_cases + excluded, key=lambda c: c["patient_id"]))
    n_rev = sum(c["split"] == "excluded_review" for c in excluded); n_big = len(excluded) - n_rev
    n_tumor_train = sum(c["split"] == "train" for c in cases)
    n_normal_train = sum(c["split"] == "train" for c in normal_cases)
    print(f"dataset: {ds_dir}"
          f"\n tumor:  train={n_tumor_train} test={len(cases)-n_tumor_train} excluded: REVIEW={n_rev} too_large={n_big}")
    if a.include_normal:
        print(f" normal: train={n_normal_train} test={len(normal_cases)-n_normal_train}")
    print(f" TOTAL:  train={n_train} test={len(all_cases)-n_train}"
          f"\n components removed: {sum(c['removed_components'] for c in all_cases)} "
          f"({sum(c['removed_voxels'] for c in all_cases)} voxels) across {sum(1 for c in all_cases if c['removed_components'])} cases"
          f"\n manifest: {man}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
