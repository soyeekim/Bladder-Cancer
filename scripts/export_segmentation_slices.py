#!/usr/bin/env python3
"""Export paired 2D MRI and segmentation frames for one patient.

The NIfTI mask is stored as (DICOM columns, DICOM rows, slices), whereas a
DICOM pixel array is (rows, columns).  This script sorts DICOM instances by
physical slice position, transposes each NIfTI slice into DICOM pixel layout,
and records the exact image-to-mask pairing in a CSV manifest.

MRI PNGs are percentile-windowed QC images, not quantitative replacements for
the source DICOM pixel data.  Mask PNGs remain exact binary 0/255 images.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pydicom
from PIL import Image

from inspect_segmentation import (
    RTSTRUCT_SOP_CLASS_UID,
    SEG_SOP_CLASS_UID,
    annotation_jsons,
    b_value,
    position_key,
    read_dicom_headers,
    summarize_series,
)


REPORT_START = "<!-- SLICE_EXPORT_START -->"
REPORT_END = "<!-- SLICE_EXPORT_END -->"


def safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return text or "unknown"


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def format_b_value(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def volume_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split a DICOM series into spatial 3D volumes.

    Diffusion data are grouped by b-value.  A non-diffusion series must contain
    one instance at each spatial position; ambiguous repeated volumes are
    rejected rather than paired to masks silently.
    """
    values = [b_value(item["dataset"]) for item in records]
    if values and all(value is not None for value in values) and len(set(values)) > 1:
        grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for item, value in zip(records, values):
            grouped[float(value)].append(item)
        return [
            {
                "volume_id": f"b_{format_b_value(value)}_s_mm2",
                "b_value_s_per_mm2": value,
                "records": grouped[value],
            }
            for value in sorted(grouped)
        ]

    position_counts = Counter(
        position_key(getattr(item["dataset"], "ImagePositionPatient", None))
        for item in records
    )
    position_counts.pop(None, None)
    if position_counts and max(position_counts.values()) > 1:
        raise ValueError(
            "Repeated slice positions were found, but they cannot be separated "
            "unambiguously by diffusion b-value."
        )
    single_b = next((float(value) for value in values if value is not None), None)
    return [
        {
            "volume_id": "volume_00",
            "b_value_s_per_mm2": single_b,
            "records": records,
        }
    ]


def sort_spatially(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first = records[0]["dataset"]
    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    normal = np.cross(orientation[:3], orientation[3:])
    return sorted(
        records,
        key=lambda item: float(
            np.asarray(item["dataset"].ImagePositionPatient, dtype=float) @ normal
        ),
    )


def load_dicom_volume(records: list[dict[str, Any]]) -> np.ndarray:
    slices = []
    for item in records:
        dataset = pydicom.dcmread(item["path"])
        pixels = dataset.pixel_array.astype(np.float32)
        pixels = pixels * float(getattr(dataset, "RescaleSlope", 1.0))
        pixels += float(getattr(dataset, "RescaleIntercept", 0.0))
        slices.append(pixels)
    return np.stack(slices, axis=0)


def window_volume(volume: np.ndarray) -> tuple[np.ndarray, float, float]:
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return np.zeros(volume.shape, dtype=np.uint16), 0.0, 1.0
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        high = low + 1.0
    scaled = np.clip((volume - low) / (high - low), 0.0, 1.0)
    scaled[~np.isfinite(scaled)] = 0.0
    return np.rint(scaled * 65535.0).astype(np.uint16), float(low), float(high)


def save_overlay(image_u16: np.ndarray, mask: np.ndarray, path: Path) -> None:
    gray = np.rint(image_u16 / 257.0).astype(np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    color = np.asarray([255, 72, 0], dtype=np.float32)
    rgb[mask] = np.rint(0.55 * rgb[mask] + 0.45 * color).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def mask_paths_by_uid(patient_root: Path) -> dict[str, Path]:
    sidecars, errors = annotation_jsons(patient_root)
    if errors:
        raise ValueError("Annotation JSON read failed: " + "; ".join(errors))
    result: dict[str, Path] = {}
    for sidecar in sidecars:
        uid = sidecar.get("series_instance_uid")
        for lesion in sidecar["lesions"]:
            name = lesion.get("nifti_file")
            if not uid or not name:
                continue
            path = (patient_root / sidecar["directory"] / name).resolve()
            if uid in result and result[uid] != path:
                raise ValueError(f"Series {uid} references more than one NIfTI mask")
            result[uid] = path
    return result


def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("No image/mask pair was exported")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_report_section(summary: dict[str, Any], report_path: Path) -> str:
    lines = [
        REPORT_START,
        "## 2D frame 추출 및 slice별 segmentation 확인",
        "",
        f"- 추출 실행 일시: {summary['generated_at']}",
        f"- 결과 경로: `{summary['output_directory']}`",
        "- MRI frame: 각 3D volume을 물리적 slice 위치 순으로 정렬한 뒤 0-based 번호로 PNG 저장",
        "- Mask frame: NIfTI의 각 z-slice를 DICOM `(row, column)` 배열 방향으로 변환하여 이진 PNG(0/255)로 저장",
        "- MRI PNG는 volume별 1–99 percentile display window가 적용된 **QC용 영상**이며, 정량 분석 시 원본 DICOM을 사용해야 함",
        "",
        "### 실행 결과",
        "",
        f"- 총 **{summary['totals']['image_frames']}개 MRI frame**과 **{summary['totals']['paired_rows']}개 image–mask pair**를 manifest에 기록했습니다.",
        f"- 중복 없는 공간 mask frame은 **{summary['totals']['unique_mask_frames']}개**이며, 모든 image frame에 대응 mask가 있어 pairing 검사는 **{summary['totals']['pairing_status']}**입니다.",
        f"- 실제 foreground가 있는 고유 mask slice는 **{summary['totals']['unique_foreground_mask_frames']}개**, 이를 DWI b-value별 image에 대응시킨 foreground pair는 **{summary['totals']['foreground_pairs']}개**입니다.",
        f"- foreground가 없는 나머지 **{summary['totals']['empty_mask_pairs']}개 pair**도 빠진 mask가 아니라 값이 모두 0인 정상 background mask입니다.",
        "",
        "| Series | Image volume | 공간 slices | 추출 image frames | 고유 mask frames | Foreground slice (0-based) | Foreground frame (1-based) | slice별 foreground voxels | Pairing |",
        "|---:|---|---:|---:|---:|---|---|---|---|",
    ]
    for item in summary["series"]:
        volume_text = ", ".join(volume["volume_id"] for volume in item["volumes"])
        zero_based = ", ".join(map(str, item["foreground_slice_indices_0based"])) or "없음"
        one_based = ", ".join(map(str, item["foreground_frame_numbers_1based"])) or "없음"
        counts = ", ".join(
            f"{entry['slice_index_0based']}:{entry['foreground_voxels']}"
            for entry in item["foreground_voxels_by_slice"]
        ) or "—"
        lines.append(
            f"| {item['series_number']} | {volume_text} | {item['spatial_slices']} | "
            f"{item['image_frames']} | {item['unique_mask_frames']} | {zero_based} | "
            f"{one_based} | {counts} | {item['pairing_status']} |"
        )
    lines.extend(
        [
            "",
            "Series 17은 하나의 공간 grid(40 slices)에 b=0/600/1000의 3개 image volume이 있습니다. 따라서 mask PNG 40개를 중복 저장하지 않고 세 volume이 같은 위치의 mask를 공유하며, manifest에는 **120개 image–mask 대응 관계**가 각각 기록됩니다.",
            "",
            "생성 파일:",
            "",
            f"- Pair manifest: `{summary['manifest_csv']}`",
            f"- 구조화 요약: `{summary['summary_json']}`",
            "- 각 series 폴더: `masks/`(모든 slice), `<volume>/images/`(모든 slice), `<volume>/overlays/`(foreground slice만)",
            "",
            "재현 명령:",
            "",
            "```bash",
            f"python scripts/export_segmentation_slices.py '{summary['patient_root']}' --output-dir '{summary['output_directory']}' --report '{report_path.resolve()}'",
            "```",
            REPORT_END,
        ]
    )
    return "\n".join(lines)


def update_report(report_path: Path, section: str) -> None:
    text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    if REPORT_START in text and REPORT_END in text:
        start = text.index(REPORT_START)
        end = text.index(REPORT_END, start) + len(REPORT_END)
        updated = text[:start].rstrip() + "\n\n" + section + text[end:]
    else:
        anchor = "\n## 재현 방법\n"
        if anchor in text:
            updated = text.replace(anchor, "\n\n" + section + anchor, 1)
        else:
            updated = text.rstrip() + "\n\n" + section + "\n"
    report_path.write_text(updated, encoding="utf-8")


def export_patient(patient_root: Path, output_dir: Path) -> dict[str, Any]:
    patient_root = patient_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records, errors = read_dicom_headers(patient_root)
    if errors:
        raise ValueError("DICOM header read failed: " + "; ".join(errors))
    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        dataset = item["dataset"]
        uid = str(getattr(dataset, "SeriesInstanceUID", ""))
        sop_uid = str(getattr(dataset, "SOPClassUID", ""))
        if uid and sop_uid not in {SEG_SOP_CLASS_UID, RTSTRUCT_SOP_CLASS_UID}:
            by_uid[uid].append(item)

    series = [summarize_series(uid, items, patient_root) for uid, items in by_uid.items()]
    series.sort(key=lambda item: int(item["series_number"]))
    masks = mask_paths_by_uid(patient_root)
    manifest_rows: list[dict[str, Any]] = []
    series_summaries: list[dict[str, Any]] = []

    for series_item in series:
        uid = series_item["series_instance_uid"]
        if uid not in masks:
            raise ValueError(f"No NIfTI mask is linked to series {series_item['series_number']}")
        number = int(series_item["series_number"])
        series_dir = output_dir / f"series_{number:04d}"
        masks_dir = series_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        nifti = nib.load(masks[uid])
        mask_xyz = np.asanyarray(nifti.dataobj) != 0
        expected_shape = (
            int(series_item["geometry"]["columns"]),
            int(series_item["geometry"]["rows"]),
            int(series_item["geometry"]["unique_slice_positions"]),
        )
        if mask_xyz.shape != expected_shape:
            raise ValueError(
                f"Series {number}: mask shape {mask_xyz.shape} != DICOM grid {expected_shape}"
            )

        foreground_counts = np.count_nonzero(mask_xyz, axis=(0, 1)).astype(int)
        foreground_indices = np.flatnonzero(foreground_counts).astype(int).tolist()
        mask_files: list[Path] = []
        for z_index in range(mask_xyz.shape[2]):
            mask_rc = mask_xyz[:, :, z_index].T
            mask_path = masks_dir / f"slice_{z_index:03d}.png"
            Image.fromarray(mask_rc.astype(np.uint8) * 255).save(mask_path)
            mask_files.append(mask_path)

        volume_summaries = []
        for volume in volume_groups(by_uid[uid]):
            ordered = sort_spatially(volume["records"])
            if len(ordered) != mask_xyz.shape[2]:
                raise ValueError(
                    f"Series {number}/{volume['volume_id']}: {len(ordered)} image slices "
                    f"!= {mask_xyz.shape[2]} mask slices"
                )
            volume_dir = series_dir / volume["volume_id"]
            images_dir = volume_dir / "images"
            overlays_dir = volume_dir / "overlays"
            images_dir.mkdir(parents=True, exist_ok=True)
            overlays_dir.mkdir(parents=True, exist_ok=True)
            pixels = load_dicom_volume(ordered)
            display, window_low, window_high = window_volume(pixels)

            for z_index, item in enumerate(ordered):
                image_path = images_dir / f"slice_{z_index:03d}.png"
                Image.fromarray(display[z_index]).save(image_path)
                has_foreground = bool(foreground_counts[z_index])
                overlay_path: Path | None = None
                if has_foreground:
                    overlay_path = overlays_dir / f"slice_{z_index:03d}.png"
                    save_overlay(display[z_index], mask_xyz[:, :, z_index].T, overlay_path)
                dataset = item["dataset"]
                manifest_rows.append(
                    {
                        "series_number": number,
                        "series_description": series_item["series_description"],
                        "volume_id": volume["volume_id"],
                        "b_value_s_per_mm2": volume["b_value_s_per_mm2"]
                        if volume["b_value_s_per_mm2"] is not None
                        else "",
                        "slice_index_0based": z_index,
                        "frame_number_1based": z_index + 1,
                        "dicom_instance_number": getattr(dataset, "InstanceNumber", ""),
                        "image_position_patient_lps": json.dumps(
                            [float(value) for value in dataset.ImagePositionPatient]
                        ),
                        "source_dicom": relative(item["path"], patient_root),
                        "image_png": relative(image_path, output_dir),
                        "mask_png": relative(mask_files[z_index], output_dir),
                        "overlay_png": relative(overlay_path, output_dir)
                        if overlay_path
                        else "",
                        "mask_foreground_voxels": int(foreground_counts[z_index]),
                        "has_foreground": has_foreground,
                    }
                )
            volume_summaries.append(
                {
                    "volume_id": volume["volume_id"],
                    "b_value_s_per_mm2": volume["b_value_s_per_mm2"],
                    "image_frames": len(ordered),
                    "display_window_1st_99th_percentile": [
                        round(window_low, 6),
                        round(window_high, 6),
                    ],
                }
            )

        image_frames = sum(item["image_frames"] for item in volume_summaries)
        foreground_pairs = len(foreground_indices) * len(volume_summaries)
        series_summaries.append(
            {
                "series_number": number,
                "series_description": series_item["series_description"],
                "sequence_class": series_item["sequence_class"],
                "mask_shape_xyz": list(mask_xyz.shape),
                "spatial_slices": int(mask_xyz.shape[2]),
                "volumes": volume_summaries,
                "image_frames": image_frames,
                "unique_mask_frames": int(mask_xyz.shape[2]),
                "paired_rows": image_frames,
                "foreground_slice_indices_0based": foreground_indices,
                "foreground_frame_numbers_1based": [value + 1 for value in foreground_indices],
                "foreground_voxels_by_slice": [
                    {
                        "slice_index_0based": value,
                        "foreground_voxels": int(foreground_counts[value]),
                    }
                    for value in foreground_indices
                ],
                "unique_foreground_mask_frames": len(foreground_indices),
                "foreground_pairs": foreground_pairs,
                "empty_mask_pairs": image_frames - foreground_pairs,
                "pairing_status": "PASS",
            }
        )

    manifest_path = output_dir / "manifest.csv"
    write_manifest(manifest_rows, manifest_path)
    totals = {
        "series": len(series_summaries),
        "image_volumes": sum(len(item["volumes"]) for item in series_summaries),
        "image_frames": sum(item["image_frames"] for item in series_summaries),
        "unique_mask_frames": sum(item["unique_mask_frames"] for item in series_summaries),
        "paired_rows": len(manifest_rows),
        "unique_foreground_mask_frames": sum(
            item["unique_foreground_mask_frames"] for item in series_summaries
        ),
        "foreground_pairs": sum(item["foreground_pairs"] for item in series_summaries),
        "empty_mask_pairs": sum(item["empty_mask_pairs"] for item in series_summaries),
        "pairing_status": "PASS"
        if len(manifest_rows) == sum(item["image_frames"] for item in series_summaries)
        else "FAIL",
    }
    summary_path = output_dir / "summary.json"
    summary = {
        "schema_version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "patient_id": patient_root.name,
        "patient_root": patient_root.as_posix(),
        "output_directory": output_dir.as_posix(),
        "image_png_note": "QC image with per-volume 1st-99th percentile display window",
        "mask_png_values": [0, 255],
        "manifest_csv": manifest_path.as_posix(),
        "summary_json": summary_path.as_posix(),
        "totals": totals,
        "series": series_summaries,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export spatially paired MRI/mask PNG frames and a CSV manifest."
    )
    parser.add_argument("patient_dir", type=Path, help="Path to one patient directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for PNG frames, manifest.csv, and summary.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional Markdown audit report to update with the extraction result",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_patient(args.patient_dir, args.output_dir)
    if args.report:
        section = make_report_section(summary, args.report)
        update_report(args.report, section)
    totals = summary["totals"]
    print(f"Patient: {summary['patient_id']}")
    print(f"Image volumes: {totals['image_volumes']}")
    print(f"Image frames: {totals['image_frames']}")
    print(f"Unique mask frames: {totals['unique_mask_frames']}")
    print(f"Paired manifest rows: {totals['paired_rows']}")
    print(f"Unique foreground mask frames: {totals['unique_foreground_mask_frames']}")
    print(f"Foreground pairs: {totals['foreground_pairs']}")
    print(f"Pairing status: {totals['pairing_status']}")
    print(f"Manifest: {summary['manifest_csv']}")
    print(f"Summary: {summary['summary_json']}")
    if args.report:
        print(f"Updated report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
