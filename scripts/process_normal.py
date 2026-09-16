#!/usr/bin/env python3
"""Index and convert the NORMAL (tumor-free) cohort.

Source: data/raw/Bladder_normal/Bladder_cancer_AI_normal_patient_FULL Series/MRI/<pid>/<studydate>/*.dcm
        (all ~24 series of a study mixed in one folder; no mask JSON -> DICOM headers only)
        + "Bladder MRI_Normal 210여명_final.xlsx" (enroll flag, study date per ID)

Outputs
  data/patient_series_index_normal.json        (like patient_series_index.json; IDs prefixed with 'n')
  data/processed/Bladder_normal/n<pid>/nifti/<seriesNo>.nii.gz + .json     (T2 AX TSE only, dcm2niix)
  data/processed/Bladder_normal/n<pid>/labels/<seriesNo>.nii.gz            (all-zero uint8, image header)
  reports/normal_conversion_manifest.csv

Usage
  python3 scripts/process_normal.py --index            # build the index only
  python3 scripts/process_normal.py --index --convert  # index + convert T2 AX + zero labels
  python3 scripts/process_normal.py --convert --patients 00251784 ...

Only the one series used for training (T2 AX TSE) is converted: a normal study has ~1,300 DICOMs
across 24 series (dynamic contrast T1 VIBE, kidney HASTE, DIXON, ...) that the model never sees.
When a patient has two studies, the one whose date matches the Excel sheet is used (else the latest).
Standard library only.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import glob
import gzip
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "Bladder_normal", "Bladder_cancer_AI_normal_patient_FULL Series")
MRI = os.path.join(RAW, "MRI")
XLSX = os.path.join(RAW, "Bladder MRI_Normal 210여명_final.xlsx")
OUT = os.path.join(ROOT, "data", "processed", "Bladder_normal")
INDEX = os.path.join(ROOT, "data", "patient_series_index_normal.json")
MANIFEST = os.path.join(ROOT, "reports", "normal_conversion_manifest.csv")
DCM2NIIX = os.path.join(ROOT, "tools", "dcm2niix")
DCM2NIIX_FALLBACK = os.path.join(ROOT, "tools", "dcm2niix_v20240202")

TAGS = {(0x0008, 0x0008): "imgtype", (0x0008, 0x103E): "desc", (0x0018, 0x0010): "contrast",
        (0x0018, 0x0020): "scanseq", (0x0018, 0x0024): "seqname", (0x0018, 0x0080): "TR", (0x0018, 0x0081): "TE",
        (0x0020, 0x000E): "seriesUID", (0x0020, 0x0011): "seriesNo", (0x0020, 0x0037): "orient"}
_LONG = (b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN")


# ----------------------------------------------------------------- DICOM header
def read_tags(path: str, nbytes: int = 16384) -> dict:
    with open(path, "rb") as f:
        b = f.read(nbytes)
    if b[128:132] != b"DICM":
        return {}
    p = 132; ts = None; out = {}

    def elem(p, explicit):
        g, e = struct.unpack_from("<HH", b, p); p += 4
        if explicit:
            vr = b[p:p + 2]; p += 2
            if vr in _LONG:
                p += 2; ln = struct.unpack_from("<I", b, p)[0]; p += 4
            else:
                ln = struct.unpack_from("<H", b, p)[0]; p += 2
        else:
            vr = None; ln = struct.unpack_from("<I", b, p)[0]; p += 4
        return (g, e), vr, ln, p

    while p < len(b) - 8:
        tag, vr, ln, p = elem(p, True)
        if tag[0] != 0x0002:
            p -= 8 if vr is None else (12 if vr in _LONG else 8)
            break
        if tag == (0x0002, 0x0010):
            ts = b[p:p + ln].decode("ascii", "ignore").strip("\x00 ")
        p += ln
    explicit = ts != "1.2.840.10008.1.2"
    while p < len(b) - 8:
        tag, vr, ln, p = elem(p, explicit)
        if tag > (0x0020, 0x0040) or ln == 0xFFFFFFFF or p + ln > len(b):
            break
        if tag in TAGS:
            out[TAGS[tag]] = b[p:p + ln].decode("latin1", "ignore").strip("\x00 ")
        p += ln
    return out


def plane_from_orient(orient: str) -> str:
    try:
        r = [float(x) for x in orient.split("\\")]
        n = (r[1] * r[5] - r[2] * r[4], r[2] * r[3] - r[0] * r[5], r[0] * r[4] - r[1] * r[3])
        return ("SAG", "COR", "AX")[max(range(3), key=lambda i: abs(n[i]))]
    except Exception:
        return "?"


def classify(desc: str, imgtype: str, plane: str) -> str:
    d, t = (desc or "").upper(), (imgtype or "").upper()
    txt_plane = "SAG" if ("SAG" in d or "_SA_" in d) else "COR" if ("COR" in d or "_CO_" in d) else \
                "AX" if ("AX" in d or "TRA" in d) else plane
    if "LOCALIZER" in d or "LOCAZLIER" in d or "SCOUT" in d or "LOCAL(" in d or "LOCALIZER" in t:
        return "LOCALIZER"
    if "ADC" in d or "ADC" in t:
        return "ADC"
    if "DIFFUSION" in t or "DWI" in d or "DIFF" in d or "TRACEW" in d:
        return "DWI_CALC_BVAL" if "CALC" in d else f"DWI_{txt_plane}"
    if "DIXON" in d or "WATER" in t or "FAT" in t:
        return "T1_DIXON"
    if "VIBE" in d or "LAVA" in d:  # Siemens VIBE / GE LAVA: dynamic contrast 3D T1
        return "T1_VIBE_SUB" if ("SUB" in d or "\\SUB" in t) else "T1_VIBE"
    if "T1" in d:
        return f"T1_{txt_plane}"
    if "KIDNEY" in d and ("HASTE" in d or "SSFSE" in d or "T2" in d):
        return "T2_HASTE_KIDNEY"  # single-shot kidney T2 (Siemens HASTE / GE SSFSE)
    if "T2" in d or "HASTE" in d or "TSE" in d or "FSE" in d:
        return f"T2_{txt_plane}"
    return "OTHER"


# ----------------------------------------------------------------- Excel
def read_excel() -> dict[str, dict]:
    """ID -> {enroll, study_date(YYYYMMDD), accession} from sheet2 of the normal roster."""
    if not os.path.exists(XLSX):
        return {}
    z = zipfile.ZipFile(XLSX)
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    ss = [''.join(t.itertext()) for t in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(".//m:si", ns)]
    sheet = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet"))[1]
    rows = ET.fromstring(z.read(sheet)).findall(".//m:row", ns)

    def val(c):
        v = c.find("m:v", ns); s = v.text if v is not None else ""
        return ss[int(s)] if c.get("t") == "s" and s else s

    header = None; out = {}
    for r in rows:
        cells = {''.join(ch for ch in c.get("r") if ch.isalpha()): val(c) for c in r.findall("m:c", ns)}
        if header is None:
            header = cells; continue
        pid = cells.get("C", "").strip()
        if not pid:
            continue
        date = ""
        try:
            serial = float(cells.get("D", ""))
            date = (dt.date(1899, 12, 30) + dt.timedelta(days=int(serial))).strftime("%Y%m%d")
        except ValueError:
            pass
        out[pid.zfill(8)] = dict(enroll=cells.get("B", "").strip(), study_date=date, accession=cells.get("A", "").strip())
    return out


# ----------------------------------------------------------------- index
def build_index(cancer_ids: set[str]) -> dict:
    excel = read_excel()
    patients = []; type_counter = collections.Counter(); t0 = time.time()
    pdirs = sorted(glob.glob(os.path.join(MRI, "*", "")))
    for i, pdir in enumerate(pdirs, 1):
        pid = os.path.basename(pdir.rstrip("/"))
        studies = []
        for sdir in sorted(glob.glob(os.path.join(pdir, "*", ""))):
            sdate = os.path.basename(sdir.rstrip("/"))
            files = glob.glob(os.path.join(sdir, "*.dcm"))
            groups: dict[str, list] = collections.defaultdict(list)
            for f in files:
                h = read_tags(f)
                if h.get("seriesUID"):
                    groups[h["seriesUID"]].append((f, h))
            series = []
            for uid, items in groups.items():
                h = items[0][1]
                plane = plane_from_orient(h.get("orient", ""))
                mt = classify(h.get("desc", ""), h.get("imgtype", ""), plane)
                series.append(dict(series_no=h.get("seriesNo", ""), seriesDesc=h.get("desc", ""), mri_type=mt,
                                   plane=plane, n_dicom=len(items), imgtype=h.get("imgtype", ""),
                                   scanseq=h.get("scanseq", ""), seqname=h.get("seqname", ""),
                                   TR=h.get("TR", ""), TE=h.get("TE", ""), contrast=h.get("contrast", ""),
                                   seriesUID=uid, files=sorted(x[0] for x in items)))
            series.sort(key=lambda s: int(s["series_no"] or 0))
            for s in series:
                type_counter[s["mri_type"]] += 1
            studies.append(dict(study_folder=sdate, n_dicom=len(files), n_series=len(series), series=series))
        ex = excel.get(pid, {})
        # pick the study to use: Excel study date if it matches a folder, else the latest
        selected = None
        if studies:
            match = [s for s in studies if s["study_folder"] == ex.get("study_date")]
            selected = (match or [studies[-1]])[0]["study_folder"]
        for s in studies:
            s["selected"] = s["study_folder"] == selected
            s["t2_ax_tse"] = [x["series_no"] for x in s["series"] if x["mri_type"] == "T2_AX"]
        patients.append(dict(patient_id=pid, normal_id="n" + pid, also_in_cancer_cohort=pid in cancer_ids,
                             enroll=ex.get("enroll", ""), excel_study_date=ex.get("study_date", ""),
                             n_studies=len(studies), selected_study=selected, studies=studies))
        if i % 25 == 0:
            print(f"indexed {i}/{len(pdirs)} patients ({time.time()-t0:.0f}s)", flush=True)
    index = dict(generated_at=dt.datetime.now().astimezone().isoformat(),
                 description="정상군(FULL Series) 환자별 study/series 목록. series 는 DICOM 헤더(SeriesInstanceUID)로 묶고 "
                             "SeriesDescription/ImageType/orientation 으로 mri_type 분류. normal_id = 'n'+환자ID. "
                             "selected_study: 엑셀 검사일과 일치하는 study(없으면 최신). t2_ax_tse: 학습에 쓸 T2 AX TSE series 번호.",
                 source=os.path.relpath(MRI, ROOT), excel=os.path.relpath(XLSX, ROOT),
                 total_patients=len(patients), total_studies=sum(p["n_studies"] for p in patients),
                 total_series=sum(type_counter.values()), mri_type_counts=dict(type_counter.most_common()),
                 patients=patients)
    # file lists are large; keep them out of the JSON (they are recomputed at convert time)
    slim = json.loads(json.dumps(index))
    for p in slim["patients"]:
        for st in p["studies"]:
            for s in st["series"]:
                s.pop("files", None)
    with open(INDEX, "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)
    print(f"index: {INDEX} | patients {len(patients)} | series {slim['total_series']} | {time.time()-t0:.0f}s")
    return index


# ----------------------------------------------------------------- convert
def run_dcm2niix(files: list[str], out_dir: str, tag: str) -> tuple[str, list[str], str]:
    """Convert exactly `files` (symlinked into a temp dir) with dcm2niix (+fallback). Returns (status, produced, msg)."""
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"n_{tag}_") as tmp:
        for f in files:
            os.symlink(os.path.abspath(f), os.path.join(tmp, os.path.basename(f)))
        for binary in (DCM2NIIX, DCM2NIIX_FALLBACK):
            before = set(os.listdir(out_dir))
            r = subprocess.run([binary, "-z", "y", "-f", "%s", "-o", out_dir, tmp], capture_output=True)
            produced = sorted(set(os.listdir(out_dir)) - before)
            if r.returncode == 0 and any(x.endswith(".nii.gz") for x in produced):
                return ("ok" if binary == DCM2NIIX else "ok_fallback"), produced, ""
            for x in produced:
                os.remove(os.path.join(out_dir, x))
            msg = r.stderr.decode("utf-8", "replace").strip() or f"rc={r.returncode}"
    return "error", [], msg[:200]


def write_zero_label(image_path: str, out_path: str) -> tuple[int, int, int]:
    with gzip.open(image_path, "rb") as f:
        hdr = bytearray(f.read(352))
    dim = struct.unpack_from("<8h", hdr, 40); nx, ny, nz = dim[1], dim[2], dim[3]
    struct.pack_into("<8h", hdr, 40, 3, nx, ny, nz, 1, 1, 1, 1)
    struct.pack_into("<hh", hdr, 70, 2, 8)
    struct.pack_into("<f", hdr, 108, 352.0)
    struct.pack_into("<ff", hdr, 112, 1.0, 0.0)
    struct.pack_into("<ff", hdr, 124, 1.0, 0.0)
    hdr[148:228] = b"normal cohort: all-zero label on dcm2niix grid".ljust(80, b"\x00")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, "wb") as f:
        f.write(bytes(hdr) + bytes(4) + bytes(nx * ny * nz))
    return nx, ny, nz


def convert(index: dict, patients: list[str] | None, skip_existing: bool) -> None:
    rows = []; t0 = time.time(); counts = collections.Counter()
    for p in index["patients"]:
        pid = p["patient_id"]
        if patients and pid not in patients:
            continue
        nid = p["normal_id"]
        study = next((s for s in p["studies"] if s["selected"]), None)
        row = dict(patient_id=pid, normal_id=nid, study_folder=study["study_folder"] if study else "",
                   enroll=p["enroll"], also_in_cancer_cohort=p["also_in_cancer_cohort"],
                   series_no="", seriesDesc="", n_dicom=0, status="", image_path="", label_path="", dim="", message="")
        t2 = [s for s in study["series"] if s["mri_type"] == "T2_AX"] if study else []
        if not t2:
            row["status"] = "no_t2_ax"; rows.append(row); counts["no_t2_ax"] += 1; continue
        if len(t2) > 1:  # prefer plain TSE/FSE; push breath-hold / single-shot variants back; then most slices
            def pref(x):
                u = x["seriesDesc"].upper()
                return ("B H" in u or "BH" in u.replace(" ", "") and "(DL B H)" in u, "SSFSE" in u or "HASTE" in u,
                        not ("TSE" in u or "FSE" in u), -x["n_dicom"])
            t2.sort(key=pref)
            row["message"] = f"{len(t2)} T2_AX series; picked {t2[0]['series_no']}"
        s = t2[0]; sno = s["series_no"].lstrip("0") or "0"
        row.update(series_no=sno, seriesDesc=s["seriesDesc"], n_dicom=s["n_dicom"])
        out_dir = os.path.join(OUT, nid, "nifti"); image = os.path.join(out_dir, f"{sno}.nii.gz")
        label = os.path.join(OUT, nid, "labels", f"{sno}.nii.gz")
        if skip_existing and os.path.exists(image) and os.path.exists(label):
            row["status"] = "skipped"
        else:
            st, produced, msg = run_dcm2niix(s["files"], out_dir, nid)
            row["status"] = st; row["message"] = (row["message"] + " " + msg).strip()
            if st.startswith("ok"):
                niis = [x for x in produced if x.endswith(".nii.gz")]
                if os.path.basename(image) not in niis:  # dcm2niix may add a suffix; normalise the name
                    src = os.path.join(out_dir, niis[0]); os.replace(src, image)
                    for ext in (".json", ".bval", ".bvec"):
                        q = src[:-7] + ext
                        if os.path.exists(q):
                            os.replace(q, image[:-7] + ext)
                    if len(niis) > 1:
                        row["message"] += f" | extra outputs: {niis[1:]}"
                row["dim"] = "x".join(map(str, write_zero_label(image, label)))
        if row["status"] != "error":
            row["image_path"] = os.path.relpath(image, ROOT); row["label_path"] = os.path.relpath(label, ROOT)
        counts[row["status"]] += 1; rows.append(row)
        if len(rows) % 25 == 0:
            print(f"converted {len(rows)} | {dict(counts)} | {time.time()-t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"done: {dict(counts)} | manifest: {MANIFEST} | {time.time()-t0:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", action="store_true"); ap.add_argument("--convert", action="store_true")
    ap.add_argument("--patients", nargs="*"); ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    cancer_ids = {os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "data", "raw", "Bladder_cancer", "*"))}
    index = build_index(cancer_ids) if (a.index or not os.path.exists(INDEX)) else None
    if a.convert:
        if index is None:  # rebuild in memory to get file lists
            index = build_index(cancer_ids)
        convert(index, a.patients, a.skip_existing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
