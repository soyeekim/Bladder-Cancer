#!/usr/bin/env python3
"""Batch-convert every series' dicom/ folder to NIfTI with dcm2niix.

Output layout (one folder per patient):
    data/processed/Bladder_cancer/<patient_id>/nifti/<seriesNumber>.nii.gz
    data/processed/Bladder_cancer/<patient_id>/nifti/<seriesNumber>.json      (dcm2niix sidecar)
    data/processed/Bladder_cancer/<patient_id>/nifti/<seriesNumber>.bval/.bvec (DWI/ADC only)

dcm2niix is called once per series dicom/ folder (never recursively over a
patient folder -- that crashes, see SEGMENTATION_PIPELINE.md 0-6절).

A manifest CSV and a log are written under reports/ and outputs/logs/.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DCM2NIIX = os.path.join(ROOT, "tools", "dcm2niix")
# v1.0.20260724 crashes (segfault / malloc corruption) on a handful of series; the
# 2024-02-02 release converts them identically, so it is used as a fallback.
DCM2NIIX_FALLBACK = os.path.join(ROOT, "tools", "dcm2niix_v20240202")


def convert_series(dicom_dir: str, out_dir: str, binary: str = DCM2NIIX) -> tuple[str, list[str], str]:
    """Run dcm2niix on one dicom/ folder. Returns (status, produced_files, message).

    If the primary binary crashes, retry once with DCM2NIIX_FALLBACK; the status
    is then "ok_fallback" so the manifest records which binary produced the file.
    """
    os.makedirs(out_dir, exist_ok=True)
    before = set(os.listdir(out_dir))
    cmd = [binary, "-z", "y", "-f", "%s", "-o", out_dir, dicom_dir]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=600)  # bytes: dcm2niix may print non-UTF8
    except subprocess.TimeoutExpired:
        return "timeout", [], "dcm2niix timed out"
    stdout = r.stdout.decode("utf-8", errors="replace")
    stderr = r.stderr.decode("utf-8", errors="replace")
    produced = sorted(set(os.listdir(out_dir)) - before)
    tail = "\n".join(l for l in stdout.splitlines() if l.startswith(("Convert", "Warning", "Error")))
    if r.returncode != 0 or not any(f.endswith(".nii.gz") for f in produced):
        msg = stderr.strip() or tail or f"rc={r.returncode}"
        if binary == DCM2NIIX and os.access(DCM2NIIX_FALLBACK, os.X_OK):
            for f in produced:  # clean partial output before retrying
                os.remove(os.path.join(out_dir, f))
            st, produced2, msg2 = convert_series(dicom_dir, out_dir, DCM2NIIX_FALLBACK)
            if st == "ok":
                return "ok_fallback", produced2, f"primary failed ({msg[:80]}); converted with {os.path.basename(DCM2NIIX_FALLBACK)}"
            return "error", produced2, f"primary: {msg[:120]} | fallback: {msg2[:120]}"
        return "error", produced, msg
    return "ok", produced, tail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "raw", "Bladder_cancer"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "processed", "Bladder_cancer"))
    ap.add_argument("--patients", nargs="*", help="subset of patient IDs (default: all)")
    ap.add_argument("--skip-existing", action="store_true", help="skip series whose <seriesNumber>.nii.gz already exists")
    args = ap.parse_args()

    if not os.access(DCM2NIIX, os.X_OK):
        print(f"dcm2niix not found/executable at {DCM2NIIX}", file=sys.stderr)
        return 2

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(ROOT, "outputs", "logs"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    log_path = os.path.join(ROOT, "outputs", "logs", f"dcm2niix_batch_{stamp}.log")
    manifest_path = os.path.join(ROOT, "reports", "dcm2niix_manifest.csv")

    patient_dirs = sorted(glob.glob(os.path.join(args.raw, "*", "")))
    if args.patients:
        keep = set(args.patients)
        patient_dirs = [p for p in patient_dirs if os.path.basename(p.rstrip("/")) in keep]

    rows: list[dict] = []
    t0 = time.time()
    n_ok = n_err = n_skip = 0
    with open(log_path, "w", encoding="utf-8") as log:
        for pi, pdir in enumerate(patient_dirs, 1):
            pid = os.path.basename(pdir.rstrip("/"))
            out_dir = os.path.join(args.out, pid, "nifti")
            for sdir in sorted(glob.glob(os.path.join(pdir, "*", ""))):
                series = os.path.basename(sdir.rstrip("/"))
                dicom_dir = os.path.join(sdir, "dicom")
                n_dcm = len(glob.glob(os.path.join(dicom_dir, "*.dcm")))
                series_no = series.split("_")[0].lstrip("0") or "0"
                expected = os.path.join(out_dir, f"{series_no}.nii.gz")
                if n_dcm == 0:
                    status, produced, msg = "no_dicom", [], "no .dcm files"
                elif args.skip_existing and os.path.exists(expected):
                    status, produced, msg = "skipped", [os.path.basename(expected)], "exists"
                else:
                    status, produced, msg = convert_series(dicom_dir, out_dir)
                n_ok += status in ("ok", "ok_fallback")
                n_err += status in ("error", "timeout", "no_dicom")
                n_skip += status == "skipped"
                niis = [f for f in produced if f.endswith(".nii.gz")]
                rows.append(dict(patient_id=pid, series_folder=series, n_dicom=n_dcm, status=status,
                                 n_nifti=len(niis), nifti_files=";".join(niis), all_files=";".join(produced),
                                 message=msg.replace("\n", " | ")[:500]))
                log.write(f"[{pi}/{len(patient_dirs)}] {pid}/{series} dcm={n_dcm} -> {status} {produced}\n{msg}\n\n")
                log.flush()
            if pi % 20 == 0 or pi == len(patient_dirs):
                print(f"{pi}/{len(patient_dirs)} patients | ok={n_ok} err={n_err} skip={n_skip} | {time.time()-t0:.0f}s", flush=True)

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"done. series ok={n_ok} err={n_err} skipped={n_skip}")
    print(f"manifest: {manifest_path}\nlog: {log_path}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
