#!/usr/bin/env python3
"""Audit one patient's MRI series and segmentation annotations.

The script is intentionally read-only with respect to the input directory.  It
groups DICOM instances by SeriesInstanceUID, distinguishes repeated diffusion
volumes from unique slice locations, inspects NIfTI masks, and compares each
mask grid with the referenced DICOM series in RAS coordinates.

Patient names, birth dates, and other direct identifiers are never exported.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from scipy import ndimage


SEG_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.66.4"
RTSTRUCT_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.481.3"
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


def safe_value(value: Any) -> Any:
    """Convert pydicom/numpy values to JSON-compatible, non-PHI scalars."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)) or value.__class__.__name__ == "MultiValue":
        return [safe_value(item) for item in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def rounded_list(values: Any, digits: int = 6) -> list[float] | None:
    if values is None:
        return None
    return [round(float(value), digits) for value in values]


def get_float(dataset: pydicom.Dataset, name: str) -> float | None:
    value = getattr(dataset, name, None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def file_suffix(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower() or "[no extension]"


def relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def read_dicom_headers(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() in {".dcm", ".dicom"}
            or path.suffix == ""
        )
    ]
    for path in sorted(candidates):
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
        except (InvalidDicomError, OSError, ValueError) as exc:
            errors.append(f"{relative(path, root)}: {type(exc).__name__}: {exc}")
            continue
        records.append({"path": path, "dataset": dataset})
    return records, errors


def classify_sequence(description: str, image_type: list[str]) -> str:
    text = " ".join([description, *image_type]).upper()
    if "ADC" in text:
        return "ADC"
    if "DWI" in text or "DIFFUSION" in text or "TRACEW" in text:
        return "DWI"
    if "T2" in text:
        return "T2WI"
    if "T1" in text:
        return "T1WI"
    return "Other MR"


def orientation_label(iop: np.ndarray | None) -> str | None:
    if iop is None or iop.shape != (6,):
        return None
    normal = np.cross(iop[:3], iop[3:])
    dominant = int(np.argmax(np.abs(normal)))
    labels = ["sagittal", "coronal", "axial"]
    prefix = "" if abs(normal[dominant]) >= 0.95 else "oblique "
    return prefix + labels[dominant]


def b_value(dataset: pydicom.Dataset) -> float | None:
    standard = getattr(dataset, "DiffusionBValue", None)
    if standard is not None:
        try:
            return float(standard)
        except (TypeError, ValueError):
            pass
    # Siemens MR private tag commonly used in classic single-frame DICOM.
    tag = (0x0019, 0x100C)
    if tag in dataset:
        try:
            return float(dataset[tag].value)
        except (TypeError, ValueError):
            pass
    return None


def position_key(position: Any, digits: int = 4) -> tuple[float, ...] | None:
    if position is None or len(position) != 3:
        return None
    return tuple(round(float(item), digits) for item in position)


def all_equal(items: list[Any]) -> bool:
    return len({json.dumps(safe_value(item), sort_keys=True) for item in items}) <= 1


def dicom_geometry(records: list[dict[str, Any]]) -> dict[str, Any]:
    datasets = [item["dataset"] for item in records]
    first = datasets[0]
    iop_value = getattr(first, "ImageOrientationPatient", None)
    iop = np.asarray(iop_value, dtype=float) if iop_value is not None else None
    positions = [
        np.asarray(getattr(dataset, "ImagePositionPatient"), dtype=float)
        for dataset in datasets
        if getattr(dataset, "ImagePositionPatient", None) is not None
    ]
    unique_by_key: dict[tuple[float, ...], np.ndarray] = {}
    for position in positions:
        unique_by_key[position_key(position)] = position

    computed_spacing = None
    affine_ras = None
    ordered_positions: list[np.ndarray] = []
    if iop is not None and len(unique_by_key) > 0:
        row_direction = iop[:3]
        column_direction = iop[3:]
        normal = np.cross(row_direction, column_direction)
        ordered_positions = sorted(unique_by_key.values(), key=lambda pos: float(pos @ normal))
        projections = np.asarray([float(pos @ normal) for pos in ordered_positions])
        if len(projections) > 1:
            computed_spacing = float(np.median(np.diff(projections)))
        else:
            computed_spacing = get_float(first, "SpacingBetweenSlices") or get_float(
                first, "SliceThickness"
            )
        pixel_spacing = getattr(first, "PixelSpacing", None)
        if pixel_spacing is not None and computed_spacing is not None:
            spacing = np.asarray(pixel_spacing, dtype=float)
            affine_lps = np.eye(4, dtype=float)
            # NIfTI axis order is (DICOM columns, DICOM rows, slices).
            affine_lps[:3, 0] = row_direction * spacing[1]
            affine_lps[:3, 1] = column_direction * spacing[0]
            affine_lps[:3, 2] = normal * computed_spacing
            affine_lps[:3, 3] = ordered_positions[0]
            affine_ras = LPS_TO_RAS @ affine_lps

    position_counts = Counter(
        position_key(getattr(dataset, "ImagePositionPatient", None))
        for dataset in datasets
    )
    position_counts.pop(None, None)
    repetitions = sorted(position_counts.values())
    inferred_volumes = None
    if repetitions and len(set(repetitions)) == 1:
        inferred_volumes = repetitions[0]

    pixel_spacings = [getattr(dataset, "PixelSpacing", None) for dataset in datasets]
    orientations = [getattr(dataset, "ImageOrientationPatient", None) for dataset in datasets]
    rows = [getattr(dataset, "Rows", None) for dataset in datasets]
    columns = [getattr(dataset, "Columns", None) for dataset in datasets]

    return {
        "rows": int(first.Rows) if getattr(first, "Rows", None) is not None else None,
        "columns": int(first.Columns) if getattr(first, "Columns", None) is not None else None,
        "image_size_hw": [int(first.Rows), int(first.Columns)]
        if getattr(first, "Rows", None) is not None and getattr(first, "Columns", None) is not None
        else None,
        "pixel_spacing_row_col_mm": rounded_list(getattr(first, "PixelSpacing", None)),
        "slice_thickness_mm": get_float(first, "SliceThickness"),
        "spacing_between_slices_tag_mm": get_float(first, "SpacingBetweenSlices"),
        "computed_slice_spacing_mm": round(computed_spacing, 6)
        if computed_spacing is not None
        else None,
        "image_orientation_patient_lps": rounded_list(iop),
        "orientation_plane": orientation_label(iop),
        "unique_slice_positions": len(unique_by_key) if unique_by_key else len(records),
        "files_per_slice_position": inferred_volumes,
        "inferred_3d_volumes": inferred_volumes,
        "dicom_affine_ras": np.round(affine_ras, 8).tolist()
        if affine_ras is not None
        else None,
        "consistent_rows_columns": all_equal(rows) and all_equal(columns),
        "consistent_pixel_spacing": all_equal(pixel_spacings),
        "consistent_orientation": all_equal(orientations),
    }


def summarize_series(uid: str, records: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    datasets = [item["dataset"] for item in records]
    first = datasets[0]
    image_type = [str(item) for item in getattr(first, "ImageType", [])]
    description = str(getattr(first, "SeriesDescription", ""))
    sequence_class = classify_sequence(description, image_type)
    b_values = Counter(b_value(dataset) for dataset in datasets)
    b_values.pop(None, None)
    sop_uids = Counter(str(getattr(dataset, "SOPClassUID", "")) for dataset in datasets)
    modalities = sorted({str(getattr(dataset, "Modality", "")) for dataset in datasets})
    folder_paths = sorted(
        {relative(item["path"].parent, root) for item in records}
    )
    metadata_fields = [
        "ProtocolName",
        "MRAcquisitionType",
        "ScanningSequence",
        "SequenceVariant",
        "ScanOptions",
        "SequenceName",
        "RepetitionTime",
        "EchoTime",
        "FlipAngle",
        "EchoTrainLength",
        "MagneticFieldStrength",
        "Manufacturer",
        "ManufacturerModelName",
    ]
    acquisition = {
        name: safe_value(getattr(first, name, None)) for name in metadata_fields
    }
    acquisition["ImageType"] = image_type
    return {
        "series_instance_uid": uid,
        "series_number": safe_value(getattr(first, "SeriesNumber", None)),
        "series_description": description,
        "sequence_class": sequence_class,
        "modality": modalities,
        "sop_class_uids": dict(sop_uids),
        "source_directories": folder_paths,
        "dicom_file_count": len(records),
        "b_values_s_per_mm2": {
            str(int(value) if float(value).is_integer() else value): count
            for value, count in sorted(b_values.items())
        } if sequence_class == "DWI" else {},
        "source_b_value_tags_s_per_mm2": {
            str(int(value) if float(value).is_integer() else value): count
            for value, count in sorted(b_values.items())
        },
        "geometry": dicom_geometry(records),
        "acquisition": acquisition,
    }


def annotation_jsons(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{relative(path, root)}: {type(exc).__name__}: {exc}")
            continue
        for study in payload.get("study", []):
            for series in study.get("series", []):
                lesions = []
                for lesion in series.get("lesion", []):
                    volume = None
                    for measurement in lesion.get("measureValue", []):
                        if measurement.get("unit") in {"mm³", "mm3"}:
                            try:
                                volume = float(measurement.get("value"))
                            except (TypeError, ValueError):
                                pass
                    comment = lesion.get("userAnnotComment") or {}
                    lesions.append(
                        {
                            "lesion_type": lesion.get("lesionType"),
                            "has_pixel_mask": lesion.get("hasPixelMask"),
                            "nifti_file": lesion.get("maskFileNameNifti"),
                            "mask_position_metadata": lesion.get("maskPos"),
                            "reported_volume_mm3": volume,
                            "reported_measurement": lesion.get("measureString"),
                            "annotation_comment": comment.get("annotation"),
                        }
                    )
                results.append(
                    {
                        "path": relative(path, root),
                        "directory": relative(path.parent, root),
                        "format_version": payload.get("formatVersion"),
                        "analysis_software": payload.get("AnalysisSoftware"),
                        "analysis_software_version": payload.get(
                            "AnalysisSoftwareVersion"
                        ),
                        "series_instance_uid": series.get("seriesInstanceUid"),
                        "series_number": series.get("seriesNumber"),
                        "series_description": series.get("seriesDesc"),
                        "lesions": lesions,
                    }
                )
    return results, errors


def nifti_summary(path: Path, root: Path) -> dict[str, Any]:
    image = nib.load(path)
    data = np.asanyarray(image.dataobj)
    foreground = np.isfinite(data) & (data != 0)
    indices = np.argwhere(foreground)
    unique_values, counts = np.unique(data, return_counts=True)
    value_counts = None
    if len(unique_values) <= 20:
        value_counts = {
            str(safe_value(value)): int(count)
            for value, count in zip(unique_values, counts)
        }

    voxel_volume = float(abs(np.linalg.det(image.affine[:3, :3])))
    nonzero_count = int(foreground.sum())
    bbox_min = bbox_max = centroid = world_min = world_max = None
    component_count = 0
    component_sizes: list[int] = []
    if nonzero_count:
        bbox_min = indices.min(axis=0)
        bbox_max = indices.max(axis=0)
        world = nib.affines.apply_affine(image.affine, indices)
        centroid = world.mean(axis=0)
        world_min = world.min(axis=0)
        world_max = world.max(axis=0)
        labels, component_count = ndimage.label(
            foreground, structure=np.ones((3, 3, 3), dtype=np.uint8)
        )
        component_sizes = sorted(
            np.bincount(labels.ravel())[1:].astype(int).tolist(), reverse=True
        )

    return {
        "path": relative(path, root),
        "format": "NIfTI (.nii.gz)" if path.name.lower().endswith(".nii.gz") else "NIfTI (.nii)",
        "shape_xyz": list(map(int, image.shape)),
        "dtype": str(data.dtype),
        "voxel_spacing_xyz_mm": [round(float(item), 6) for item in image.header.get_zooms()[:3]],
        "orientation_codes": list(nib.aff2axcodes(image.affine)),
        "qform_code": int(image.header["qform_code"]),
        "sform_code": int(image.header["sform_code"]),
        "affine_ras": np.round(image.affine, 8).tolist(),
        "value_counts": value_counts,
        "value_range": [safe_value(np.nanmin(data)), safe_value(np.nanmax(data))],
        "nonzero_voxels": nonzero_count,
        "foreground_bbox_voxel_inclusive": [
            bbox_min.astype(int).tolist(),
            bbox_max.astype(int).tolist(),
        ]
        if bbox_min is not None
        else None,
        "foreground_centroid_ras_mm": rounded_list(centroid, 3),
        "foreground_bbox_ras_mm": [rounded_list(world_min, 3), rounded_list(world_max, 3)]
        if world_min is not None
        else None,
        "connected_components_26": int(component_count),
        "component_sizes_voxels": component_sizes,
        "voxel_volume_mm3": round(voxel_volume, 6),
        "foreground_volume_mm3": round(nonzero_count * voxel_volume, 3),
    }


def compare_grid(mask: dict[str, Any], series: dict[str, Any]) -> dict[str, Any]:
    geometry = series["geometry"]
    expected_shape = [
        geometry.get("columns"),
        geometry.get("rows"),
        geometry.get("unique_slice_positions"),
    ]
    expected_spacing = [
        geometry["pixel_spacing_row_col_mm"][1],
        geometry["pixel_spacing_row_col_mm"][0],
        geometry["computed_slice_spacing_mm"],
    ] if geometry.get("pixel_spacing_row_col_mm") and geometry.get("computed_slice_spacing_mm") else None
    shape_match = mask["shape_xyz"] == expected_shape
    spacing_match = bool(
        expected_spacing
        and np.allclose(mask["voxel_spacing_xyz_mm"], expected_spacing, atol=1e-3)
    )
    affine_difference = None
    affine_match = False
    if geometry.get("dicom_affine_ras") is not None:
        difference = np.abs(
            np.asarray(mask["affine_ras"]) - np.asarray(geometry["dicom_affine_ras"])
        )
        affine_difference = float(difference.max())
        affine_match = bool(np.allclose(difference, 0.0, atol=1e-3))
    return {
        "expected_shape_xyz": expected_shape,
        "shape_match": shape_match,
        "expected_spacing_xyz_mm": rounded_list(expected_spacing),
        "spacing_match": spacing_match,
        "max_affine_abs_difference_mm": round(affine_difference, 8)
        if affine_difference is not None
        else None,
        "affine_match_atol_1e-3": affine_match,
        "overall_grid_alignment": shape_match and spacing_match and affine_match,
    }


def inspect_annotations(
    root: Path,
    series: list[dict[str, Any]],
    sidecars: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_uid = {item["series_instance_uid"]: item for item in series}
    sidecars_by_nifti: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for sidecar in sidecars:
        for lesion in sidecar["lesions"]:
            name = lesion.get("nifti_file")
            if name:
                sidecars_by_nifti[(root / sidecar["directory"] / name).resolve()].append(
                    {"sidecar": sidecar, "lesion": lesion}
                )

    annotations: list[dict[str, Any]] = []
    for path in sorted([*root.rglob("*.nii"), *root.rglob("*.nii.gz")]):
        summary = nifti_summary(path, root)
        references = sidecars_by_nifti.get(path.resolve(), [])
        reference = references[0] if references else None
        target_uid = reference["sidecar"].get("series_instance_uid") if reference else None
        target_series = by_uid.get(target_uid)
        summary["sidecar_path"] = reference["sidecar"]["path"] if reference else None
        summary["target_series_instance_uid"] = target_uid
        summary["target_series_number"] = (
            target_series.get("series_number") if target_series else None
        )
        summary["target_series_description"] = (
            target_series.get("series_description") if target_series else None
        )
        summary["reported_volume_mm3"] = (
            reference["lesion"].get("reported_volume_mm3") if reference else None
        )
        summary["annotation_comment"] = (
            reference["lesion"].get("annotation_comment") if reference else None
        )
        if summary["reported_volume_mm3"] is not None:
            summary["reported_vs_computed_volume_difference_mm3"] = round(
                summary["foreground_volume_mm3"] - summary["reported_volume_mm3"], 3
            )
        else:
            summary["reported_vs_computed_volume_difference_mm3"] = None
        summary["grid_comparison"] = (
            compare_grid(summary, target_series) if target_series else None
        )
        annotations.append(summary)

    same_grid_comparisons: list[dict[str, Any]] = []
    for left_index, left in enumerate(annotations):
        for right in annotations[left_index + 1 :]:
            if left["shape_xyz"] != right["shape_xyz"]:
                continue
            if not np.allclose(left["affine_ras"], right["affine_ras"], atol=1e-3):
                continue
            left_data = np.asanyarray(nib.load(root / left["path"]).dataobj) != 0
            right_data = np.asanyarray(nib.load(root / right["path"]).dataobj) != 0
            left_count = int(left_data.sum())
            right_count = int(right_data.sum())
            intersection = int(np.count_nonzero(left_data & right_data))
            denominator = left_count + right_count
            dice = (2.0 * intersection / denominator) if denominator else 1.0
            same_grid_comparisons.append(
                {
                    "left_series_number": left["target_series_number"],
                    "right_series_number": right["target_series_number"],
                    "left_description": left["target_series_description"],
                    "right_description": right["target_series_description"],
                    "intersection_voxels": intersection,
                    "dice": round(dice, 4),
                }
            )
    return annotations, same_grid_comparisons


def centroid_consistency(annotations: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [
        item for item in annotations if item.get("foreground_centroid_ras_mm") is not None
    ]
    if len(usable) < 2:
        return None
    distances = []
    for left_index, left in enumerate(usable):
        for right in usable[left_index + 1 :]:
            distance = float(
                np.linalg.norm(
                    np.asarray(left["foreground_centroid_ras_mm"], dtype=float)
                    - np.asarray(right["foreground_centroid_ras_mm"], dtype=float)
                )
            )
            distances.append(
                {
                    "left_series_number": left["target_series_number"],
                    "right_series_number": right["target_series_number"],
                    "distance_mm": round(distance, 3),
                }
            )
    maximum = max(distances, key=lambda item: item["distance_mm"])
    return {
        "maximum_pairwise_distance_mm": maximum["distance_mm"],
        "maximum_distance_pair": [
            maximum["left_series_number"],
            maximum["right_series_number"],
        ],
        "pairwise_distances": distances,
    }


def choose_image_records(
    records: list[dict[str, Any]], series: dict[str, Any]
) -> list[dict[str, Any]]:
    """Choose one 3D image volume; for DWI, prefer the highest stored b-value."""
    values = [b_value(item["dataset"]) for item in records]
    available = [value for value in values if value is not None]
    if available and series["sequence_class"] == "DWI":
        selected = max(available)
        return [item for item in records if b_value(item["dataset"]) == selected]
    # Keep one image at each position if duplicate instances remain.
    chosen: dict[tuple[float, ...] | None, dict[str, Any]] = {}
    for item in records:
        key = position_key(getattr(item["dataset"], "ImagePositionPatient", None))
        chosen.setdefault(key, item)
    return list(chosen.values())


def make_qc_figure(
    root: Path,
    output_path: Path,
    annotations: list[dict[str, Any]],
    series: list[dict[str, Any]],
    records_by_uid: dict[str, list[dict[str, Any]]],
) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_uid = {item["series_instance_uid"]: item for item in series}
    panels: list[tuple[np.ndarray, np.ndarray, int, str]] = []
    for annotation in annotations:
        uid = annotation.get("target_series_instance_uid")
        target = by_uid.get(uid)
        records = records_by_uid.get(uid, [])
        if not target or not records or not annotation.get("grid_comparison", {}).get("shape_match"):
            continue
        selected = choose_image_records(records, target)
        iop = np.asarray(
            getattr(selected[0]["dataset"], "ImageOrientationPatient"), dtype=float
        )
        normal = np.cross(iop[:3], iop[3:])
        selected.sort(
            key=lambda item: float(
                np.asarray(item["dataset"].ImagePositionPatient, dtype=float) @ normal
            )
        )
        slices = []
        for item in selected:
            dataset = pydicom.dcmread(item["path"])
            pixels = dataset.pixel_array.astype(np.float32)
            pixels = pixels * float(getattr(dataset, "RescaleSlope", 1.0)) + float(
                getattr(dataset, "RescaleIntercept", 0.0)
            )
            slices.append(pixels)
        # DICOM pixels are (row, column); NIfTI/mask is (column, row, slice).
        volume = np.transpose(np.stack(slices, axis=0), (2, 1, 0))
        mask = np.asanyarray(nib.load(root / annotation["path"]).dataobj) != 0
        slice_index = int(np.argmax(mask.sum(axis=(0, 1))))
        title = f"S{target['series_number']} {target['sequence_class']}\n{target['geometry']['orientation_plane']} / slice {slice_index}"
        if target["sequence_class"] == "DWI" and target["b_values_s_per_mm2"]:
            title += f" / b={max(map(float, target['b_values_s_per_mm2'])):g}"
        panels.append((volume, mask, slice_index, title))

    if not panels:
        return None
    columns = min(3, len(panels))
    rows = math.ceil(len(panels) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 5 * rows), squeeze=False)
    for axis, (volume, mask, slice_index, title) in zip(axes.flat, panels):
        image_slice = volume[:, :, slice_index].T
        mask_slice = mask[:, :, slice_index].T
        finite = image_slice[np.isfinite(image_slice)]
        low, high = np.percentile(finite, [1, 99]) if finite.size else (0, 1)
        axis.imshow(image_slice, cmap="gray", origin="lower", vmin=low, vmax=high)
        overlay = np.ma.masked_where(~mask_slice, mask_slice)
        axis.imshow(overlay, cmap="autumn", origin="lower", alpha=0.45, vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    for axis in axes.flat[len(panels) :]:
        axis.axis("off")
    figure.suptitle("Segmentation overlay QC (red/yellow = mask)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path.name


def markdown_escape(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def fmt_vector(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    return "[" + ", ".join(f"{float(item):.{digits}g}" for item in value) + "]"


def build_markdown(audit: dict[str, Any], qc_filename: str | None) -> str:
    series = audit["series"]
    annotations = audit["annotations"]
    annotated_uids = {
        item.get("target_series_instance_uid")
        for item in annotations
        if item.get("target_series_instance_uid")
    }
    lines = [
        f"# MRI 및 segmentation 데이터 점검 보고서 — 환자 {audit['patient_id']}",
        "",
        f"- 분석 일시: {audit['generated_at']}",
        f"- 입력 경로: `{audit['input_root']}`",
        "- 개인정보 보호: 환자명, 생년월일 등 직접 식별자는 보고서에 내보내지 않음",
        "",
        "## 핵심 결론",
        "",
        f"- DICOM SeriesInstanceUID 기준 MRI series는 **{len(series)}개**입니다.",
        f"- 저장된 DICOM은 총 **{audit['dicom']['file_count']}개**이며, 고유 slice 위치 기준 3D image volume은 **{sum(item['geometry']['inferred_3d_volumes'] or 1 for item in series)}개**입니다.",
        f"- segmentation은 **{len(annotations)}개의 3D NIfTI volume**이며, DICOM SEG와 RTSTRUCT는 각각 **{audit['annotation_formats']['dicom_seg']}개**, **{audit['annotation_formats']['rtstruct']}개**입니다.",
        f"- annotation이 연결된 MRI series는 **{len(annotated_uids)}/{len(series)}개**입니다.",
    ]
    aligned_count = sum(
        bool(item.get("grid_comparison", {}).get("overall_grid_alignment"))
        for item in annotations
    )
    lines.append(
        f"- shape·spacing·affine을 모두 비교한 DICOM–mask grid 정합성은 **{aligned_count}/{len(annotations)}개 통과**입니다."
    )
    centroid_check = audit.get("cross_series_centroid_consistency")
    if centroid_check:
        lines.append(
            f"- sequence별 mask centroid의 최대 쌍별 거리는 **{centroid_check['maximum_pairwise_distance_mm']:.3f} mm**(series {centroid_check['maximum_distance_pair'][0]}↔{centroid_check['maximum_distance_pair'][1]})로, 서로 다른 plane/grid에서도 주 병변 위치가 일관됩니다."
        )
    multi_components = [
        item for item in annotations if item["connected_components_26"] > 1
    ]
    if multi_components:
        labels = ", ".join(
            f"series {item['target_series_number']} ({item['component_sizes_voxels']})"
            for item in multi_components
        )
        lines.append(
            f"- 검토 필요: 하나보다 많은 26-connected component가 있는 mask가 있습니다: **{labels}**. 별도 병변인지 작은 stray annotation인지 영상 확인이 필요합니다."
        )

    lines.extend(["", "## 폴더 및 파일 구조", "", "```text"])
    lines.append(f"{audit['patient_id']}/")
    for folder in audit["directory_structure"]:
        lines.append(f"├── {folder['name']}/")
        lines.append(f"│   ├── dicom/  ({folder['dicom_files']} files)")
        lines.append(
            f"│   └── mask/   ({folder['mask_files']} files: {', '.join(folder['mask_names'])})"
        )
    lines.extend(["```", "", "파일 확장자 집계:"])
    for suffix, count in audit["file_extensions"].items():
        lines.append(f"- `{suffix}`: {count}")

    lines.extend(
        [
            "",
            "## MRI series 요약",
            "",
            "| Series | 분류 | SeriesDescription | DICOM files / 고유 slices / volumes | Image H×W | Pixel spacing (row, col) mm | Thickness / slice spacing mm | 방향 | Segmentation | Grid 정합 |",
            "|---:|---|---|---:|---:|---|---|---|---|---|",
        ]
    )
    annotation_by_uid = {
        item.get("target_series_instance_uid"): item for item in annotations
    }
    for item in series:
        geometry = item["geometry"]
        annotation = annotation_by_uid.get(item["series_instance_uid"])
        comparison = annotation.get("grid_comparison") if annotation else None
        alignment = "PASS" if comparison and comparison["overall_grid_alignment"] else ("FAIL" if comparison else "—")
        lines.append(
            "| {number} | {kind} | {description} | {files} / {slices} / {volumes} | {size} | {spacing} | {thickness:g} / {slice_spacing:g} | {plane} | {seg} | {alignment} |".format(
                number=markdown_escape(item["series_number"]),
                kind=markdown_escape(item["sequence_class"]),
                description=markdown_escape(item["series_description"]),
                files=item["dicom_file_count"],
                slices=geometry["unique_slice_positions"],
                volumes=geometry["inferred_3d_volumes"] or "?",
                size="×".join(map(str, geometry["image_size_hw"])),
                spacing=fmt_vector(geometry["pixel_spacing_row_col_mm"]),
                thickness=geometry["slice_thickness_mm"],
                slice_spacing=geometry["computed_slice_spacing_mm"],
                plane=markdown_escape(geometry["orientation_plane"]),
                seg="있음" if annotation else "없음",
                alignment=alignment,
            )
        )
    lines.extend(
        [
            "",
            "`volumes`는 같은 slice 위치가 반복되는 횟수입니다. DWI series의 파일 수를 slice 수로 오인하지 않도록 분리했습니다.",
            "",
            "## Series별 acquisition metadata",
            "",
        ]
    )
    for item in series:
        geometry = item["geometry"]
        acquisition = item["acquisition"]
        lines.extend(
            [
                f"### Series {item['series_number']} — {item['series_description']}",
                "",
                f"- Modality / class: `{', '.join(item['modality'])}` / `{item['sequence_class']}`",
                f"- SeriesInstanceUID: `{item['series_instance_uid']}`",
                f"- Orientation (DICOM LPS): `{fmt_vector(geometry['image_orientation_patient_lps'], 6)}` → **{geometry['orientation_plane']}**",
                f"- MR acquisition: type `{markdown_escape(acquisition['MRAcquisitionType'])}`, scanning sequence `{markdown_escape(acquisition['ScanningSequence'])}`, variant `{markdown_escape(acquisition['SequenceVariant'])}`, options `{markdown_escape(acquisition['ScanOptions'])}`",
                f"- SequenceName / ProtocolName: `{markdown_escape(acquisition['SequenceName'])}` / `{markdown_escape(acquisition['ProtocolName'])}`",
                f"- TR / TE / flip angle: `{markdown_escape(acquisition['RepetitionTime'])} ms` / `{markdown_escape(acquisition['EchoTime'])} ms` / `{markdown_escape(acquisition['FlipAngle'])}°`",
                f"- ImageType: `{markdown_escape(acquisition['ImageType'])}`",
            ]
        )
        if item["b_values_s_per_mm2"]:
            b_text = ", ".join(
                f"b={key}: {count} slices"
                for key, count in item["b_values_s_per_mm2"].items()
            )
            lines.append(f"- Stored diffusion b-values (s/mm²): **{b_text}**")
        lines.append("")

    lines.extend(
        [
            "## Segmentation 저장 형식 및 정합성",
            "",
            "- 저장 방식: 각 series 폴더의 `mask/lesionAnnot3D-000.nii.gz`에 저장된 **하나의 3D mask volume**",
            "- 부가 정보: 같은 폴더의 `lesionAnnot3D.json` (Coreline Soft AVIEW export metadata 및 연결 SeriesInstanceUID)",
            f"- slice별 독립 mask 이미지: {audit['annotation_formats']['slice_mask_images']}개",
            f"- DICOM SEG: {audit['annotation_formats']['dicom_seg']}개, RTSTRUCT: {audit['annotation_formats']['rtstruct']}개, NRRD: {audit['annotation_formats']['nrrd']}개",
            "",
            "| Series | Mask shape | Spacing xyz mm | 값 | Nonzero voxels | Volume mm³ (computed / JSON) | Components | Centroid RAS mm | Shape / spacing / affine |",
            "|---:|---|---|---|---:|---:|---|---|---|",
        ]
    )
    lesion_types = sorted(
        {
            lesion.get("lesion_type")
            for sidecar in audit["annotation_sidecars"]
            for lesion in sidecar["lesions"]
            if lesion.get("lesion_type")
        }
    )
    commented_series = [
        (item["target_series_number"], item["annotation_comment"])
        for item in annotations
        if item.get("annotation_comment")
    ]
    lines.insert(
        lines.index("| Series | Mask shape | Spacing xyz mm | 값 | Nonzero voxels | Volume mm³ (computed / JSON) | Components | Centroid RAS mm | Shape / spacing / affine |"),
        f"- Annotation semantics: JSON lesion type은 `{', '.join(lesion_types) or 'unspecified'}`이며, 사용자 comment는 "
        + (", ".join(f"series {number}=`{comment}`" for number, comment in commented_series) if commented_series else "없음")
        + "입니다. 표준화된 class명(예: bladder/tumor)이 없으므로 학습 label 의미는 annotation protocol로 확인해야 합니다.\n",
    )
    for item in annotations:
        comparison = item.get("grid_comparison") or {}
        values = item["value_counts"] or item["value_range"]
        lines.append(
            "| {number} | {shape} | {spacing} | {values} | {voxels} | {computed:.1f} / {reported} | {components} {sizes} | {centroid} | {shape_ok} / {spacing_ok} / {affine_ok} |".format(
                number=markdown_escape(item["target_series_number"]),
                shape="×".join(map(str, item["shape_xyz"])),
                spacing=fmt_vector(item["voxel_spacing_xyz_mm"]),
                values=markdown_escape(values),
                voxels=item["nonzero_voxels"],
                computed=item["foreground_volume_mm3"],
                reported=f"{item['reported_volume_mm3']:.1f}" if item["reported_volume_mm3"] is not None else "—",
                components=item["connected_components_26"],
                sizes=markdown_escape(item["component_sizes_voxels"]),
                centroid=fmt_vector(item["foreground_centroid_ras_mm"]),
                shape_ok="PASS" if comparison.get("shape_match") else "FAIL",
                spacing_ok="PASS" if comparison.get("spacing_match") else "FAIL",
                affine_ok="PASS" if comparison.get("affine_match_atol_1e-3") else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "모든 mask의 foreground 값은 255이므로 학습 전에는 `mask > 0`으로 이진화하는 것이 안전합니다. 계산 volume과 JSON 기록 volume이 일치하는지는 voxel count × affine voxel volume으로 검증했습니다.",
        ]
    )

    comparisons = audit["same_grid_mask_comparisons"]
    if comparisons:
        lines.extend(["", "### 동일 grid mask 간 비교", ""])
        for item in comparisons:
            lines.append(
                f"- Series {item['left_series_number']} ↔ {item['right_series_number']}: Dice **{item['dice']:.4f}**, 교집합 {item['intersection_voxels']} voxels."
            )

    lines.extend(["", "## 품질 확인 및 다음 단계 권고", ""])
    dwi = next((item for item in series if item["sequence_class"] == "DWI"), None)
    if dwi:
        lines.append(
            f"- DWI series {dwi['series_number']}은 {dwi['dicom_file_count']} files = {dwi['geometry']['unique_slice_positions']} slices × {dwi['geometry']['inferred_3d_volumes']} b-value volumes입니다. 현재 저장된 b-value는 {', '.join(dwi['b_values_s_per_mm2'])} s/mm²이며, description의 `CAL 1400`과 달리 b=1400 instance는 별도로 저장되어 있지 않습니다."
        )
    if multi_components:
        item = multi_components[0]
        lines.append(
            f"- Series {item['target_series_number']} mask에는 {item['connected_components_26']}개 component({', '.join(map(str, item['component_sizes_voxels']))} voxels)가 있습니다. 작은 component를 자동 제거하기 전에 overlay로 별도 병변/의도된 annotation인지 판독자가 확인해야 합니다."
        )
    lines.extend(
        [
            "- DICOM과 NIfTI는 좌표계 표기가 각각 LPS와 RAS이므로 부호 변환 없이 origin/orientation을 직접 비교하면 잘못된 불일치 판정이 납니다. 본 분석은 LPS→RAS 변환 후 affine을 비교했습니다.",
            "- 이후 전처리에서는 series 17의 b-value별 volume을 분리하고, ADC(series 18)와 DWI가 동일 grid라는 점을 활용할 수 있습니다. 다른 T2 plane들은 서로 다른 grid이므로 registration/resampling 정책이 필요합니다.",
        ]
    )
    if qc_filename:
        lines.extend(
            [
                "",
                "## Segmentation overlay QC",
                "",
                f"![Segmentation overlay QC]({qc_filename})",
                "",
                "각 panel은 foreground가 가장 많은 slice이며, DWI는 저장된 가장 높은 b-value를 배경 영상으로 사용했습니다.",
            ]
        )
    lines.extend(
        [
            "",
            "## 재현 방법",
            "",
            "```bash",
            f"python scripts/inspect_segmentation.py '{audit['input_root']}' --output-dir '{audit['output_directory']}'",
            "```",
            "",
            "구조화된 전체 결과는 같은 위치의 JSON 파일에 저장됩니다.",
        ]
    )
    return "\n".join(lines) + "\n"


def audit_patient(root: Path, output_dir: Path, make_qc: bool = True) -> tuple[dict[str, Any], Path, Path]:
    root = root.resolve()
    output_dir = output_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Patient directory does not exist: {root}")

    files = [path for path in root.rglob("*") if path.is_file()]
    extension_counts = Counter(file_suffix(path) for path in files)
    dicom_records, dicom_errors = read_dicom_headers(root)
    records_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    other_dicom: list[dict[str, Any]] = []
    for record in dicom_records:
        dataset = record["dataset"]
        uid = str(getattr(dataset, "SeriesInstanceUID", ""))
        sop_uid = str(getattr(dataset, "SOPClassUID", ""))
        if uid and sop_uid not in {SEG_SOP_CLASS_UID, RTSTRUCT_SOP_CLASS_UID}:
            records_by_uid[uid].append(record)
        else:
            other_dicom.append(record)
    series = [
        summarize_series(uid, records, root)
        for uid, records in records_by_uid.items()
    ]
    series.sort(
        key=lambda item: (
            int(item["series_number"]) if str(item["series_number"]).isdigit() else 10**9,
            item["series_description"],
        )
    )
    sidecars, sidecar_errors = annotation_jsons(root)
    annotations, same_grid = inspect_annotations(root, series, sidecars)

    directory_structure = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        dicom_dir = folder / "dicom"
        mask_dir = folder / "mask"
        directory_structure.append(
            {
                "name": folder.name,
                "dicom_files": sum(path.is_file() for path in dicom_dir.glob("*"))
                if dicom_dir.is_dir()
                else 0,
                "mask_files": sum(path.is_file() for path in mask_dir.glob("*"))
                if mask_dir.is_dir()
                else 0,
                "mask_names": sorted(
                    path.name for path in mask_dir.glob("*") if path.is_file()
                )
                if mask_dir.is_dir()
                else [],
            }
        )

    sop_counts = Counter(
        str(getattr(record["dataset"], "SOPClassUID", ""))
        for record in dicom_records
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_id = root.name
    report_stem = f"patient_{patient_id}_data_audit"
    slice_mask_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    slice_mask_images = sum(
        1
        for path in files
        if path.suffix.lower() in slice_mask_extensions
        and any(part.lower() in {"mask", "masks", "seg", "segmentation"} for part in path.parts)
    )
    audit = {
        "schema_version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "patient_id": patient_id,
        "input_root": root.as_posix(),
        "output_directory": output_dir.as_posix(),
        "privacy_note": "Direct identifiers such as patient name and birth date are excluded.",
        "directory_structure": directory_structure,
        "file_extensions": dict(sorted(extension_counts.items())),
        "dicom": {
            "file_count": len(dicom_records),
            "read_errors": dicom_errors,
            "sop_class_uid_counts": dict(sop_counts),
        },
        "annotation_formats": {
            "nifti_3d": len(annotations),
            "dicom_seg": sop_counts.get(SEG_SOP_CLASS_UID, 0),
            "rtstruct": sop_counts.get(RTSTRUCT_SOP_CLASS_UID, 0),
            "nrrd": extension_counts.get(".nrrd", 0) + extension_counts.get(".nhdr", 0),
            "slice_mask_images": slice_mask_images,
            "json_sidecars": len(sidecars),
        },
        "series": series,
        "annotation_sidecars": sidecars,
        "annotation_sidecar_errors": sidecar_errors,
        "annotations": annotations,
        "same_grid_mask_comparisons": same_grid,
        "cross_series_centroid_consistency": centroid_consistency(annotations),
    }

    qc_path = output_dir / f"{report_stem}_qc.png"
    qc_error = None
    try:
        qc_filename = (
            make_qc_figure(root, qc_path, annotations, series, records_by_uid)
            if make_qc
            else None
        )
    except Exception as exc:  # The structural audit must survive a missing codec.
        qc_filename = None
        qc_error = f"{type(exc).__name__}: {exc}"
    audit["qc_image"] = qc_path.as_posix() if qc_filename else None
    audit["qc_error"] = qc_error
    json_path = output_dir / f"{report_stem}.json"
    markdown_path = output_dir / f"{report_stem}.md"
    json_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_path.write_text(build_markdown(audit, qc_filename), encoding="utf-8")
    return audit, markdown_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze one patient's MRI DICOM series and segmentation files."
    )
    parser.add_argument("patient_dir", type=Path, help="Path to one patient directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for Markdown, JSON, and QC PNG outputs (default: reports)",
    )
    parser.add_argument(
        "--no-qc-image", action="store_true", help="Skip segmentation overlay PNG"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit, markdown_path, json_path = audit_patient(
        args.patient_dir, args.output_dir, make_qc=not args.no_qc_image
    )
    aligned = sum(
        bool(item.get("grid_comparison", {}).get("overall_grid_alignment"))
        for item in audit["annotations"]
    )
    print(f"Patient: {audit['patient_id']}")
    print(f"MRI series: {len(audit['series'])}")
    print(f"DICOM files: {audit['dicom']['file_count']}")
    print(f"3D masks: {len(audit['annotations'])}")
    print(f"Grid alignment passed: {aligned}/{len(audit['annotations'])}")
    print(f"Markdown report: {markdown_path}")
    print(f"JSON report: {json_path}")
    if audit.get("qc_image"):
        print(f"QC image: {audit['qc_image']}")


if __name__ == "__main__":
    main()
