"""Per-slice DICOM header extraction (headers only, no pixel decode).

Used by the offline metadata cache builder (scripts/scan_metadata.py) and by
the online submission notebook (build one study's slice table from raw
DICOMs). Column names match the metadata cache so the rest of the pipeline
(geometry/slots/laterality/crop) consumes either source identically.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

COLUMNS = [
    "study_uid", "series_uid", "sop_uid", "rel_path", "file_size_bytes",
    "rows", "columns", "samples_per_pixel", "photometric", "bits_allocated",
    "bits_stored", "high_bit", "pixel_representation", "number_of_frames",
    "transfer_syntax",
    "iop_x", "iop_y", "iop_z", "iop_cx", "iop_cy", "iop_cz",
    "ipp_x", "ipp_y", "ipp_z",
    "spacing_row", "spacing_col", "slice_thickness", "spacing_between_slices",
    "slice_location", "instance_number", "series_number", "acquisition_number",
    "in_stack_position_number", "stack_id",
    "laterality", "patient_position", "patient_id", "body_part",
    "manufacturer", "model", "series_description", "sequence_name",
    "scanning_sequence", "sequence_variant", "scan_options",
    "mr_acquisition_type", "magnetic_field_strength",
    "echo_time", "repetition_time", "flip_angle", "echo_train_length",
    "echo_numbers", "number_of_averages",
    "rescale_slope", "rescale_intercept", "window_center", "window_width",
]

_TAG_MAP = {
    "Rows": ("rows", int), "Columns": ("columns", int),
    "SamplesPerPixel": ("samples_per_pixel", int),
    "PhotometricInterpretation": ("photometric", str),
    "BitsAllocated": ("bits_allocated", int), "BitsStored": ("bits_stored", int),
    "HighBit": ("high_bit", int), "PixelRepresentation": ("pixel_representation", int),
    "NumberOfFrames": ("number_of_frames", int),
    "SliceThickness": ("slice_thickness", float),
    "SpacingBetweenSlices": ("spacing_between_slices", float),
    "SliceLocation": ("slice_location", float),
    "InstanceNumber": ("instance_number", int),
    "SeriesNumber": ("series_number", int),
    "AcquisitionNumber": ("acquisition_number", int),
    "InStackPositionNumber": ("in_stack_position_number", int),
    "StackID": ("stack_id", str),
    "Laterality": ("laterality", str), "PatientPosition": ("patient_position", str),
    "PatientID": ("patient_id", str), "BodyPartExamined": ("body_part", str),
    "Manufacturer": ("manufacturer", str), "ManufacturerModelName": ("model", str),
    "SeriesDescription": ("series_description", str), "SequenceName": ("sequence_name", str),
    "ScanningSequence": ("scanning_sequence", str), "SequenceVariant": ("sequence_variant", str),
    "ScanOptions": ("scan_options", str), "MRAcquisitionType": ("mr_acquisition_type", str),
    "MagneticFieldStrength": ("magnetic_field_strength", float),
    "EchoTime": ("echo_time", float), "RepetitionTime": ("repetition_time", float),
    "FlipAngle": ("flip_angle", float), "EchoTrainLength": ("echo_train_length", float),
    "EchoNumbers": ("echo_numbers", float), "NumberOfAverages": ("number_of_averages", float),
    "RescaleSlope": ("rescale_slope", float), "RescaleIntercept": ("rescale_intercept", float),
    "WindowCenter": ("window_center", float), "WindowWidth": ("window_width", float),
}


def read_slice_meta(path: str, root: str = "") -> dict:
    """Header-only metadata row for one .dcm file (NaN for absent tags)."""
    import pydicom
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
    except Exception:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    return dataset_meta(ds, path, root)


def dataset_meta(ds, path: str, root: str = "") -> dict:
    """Extract the same metadata from an already opened DICOM dataset."""
    row = {c: np.nan for c in COLUMNS}
    row.update({
        "study_uid": str(getattr(ds, "StudyInstanceUID", "")),
        "series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
        "sop_uid": str(getattr(ds, "SOPInstanceUID", "")),
        "rel_path": os.path.relpath(path, root) if root else path,
        "file_size_bytes": os.path.getsize(path),
        "transfer_syntax": str(getattr(ds.file_meta, "TransferSyntaxUID", "")),
    })
    for kw, (col, cast) in _TAG_MAP.items():
        v = getattr(ds, kw, None)
        if v is None:
            continue
        if cast is str:
            row[col] = str(v)
        else:
            try:
                value = float(v)
                if np.isfinite(value):
                    row[col] = cast(value)
            except (TypeError, ValueError, OverflowError):
                pass
    # Treat malformed geometry as missing, so the normal geometry fallback
    # can handle this slice without discarding every series in the study.
    for tag, names in (
        ("ImageOrientationPatient", ("iop_x", "iop_y", "iop_z", "iop_cx", "iop_cy", "iop_cz")),
        ("ImagePositionPatient", ("ipp_x", "ipp_y", "ipp_z")),
        ("PixelSpacing", ("spacing_row", "spacing_col")),
    ):
        try:
            values = np.asarray(getattr(ds, tag), dtype=float)
            if values.shape == (len(names),) and np.isfinite(values).all():
                row.update(zip(names, values.tolist()))
        except (AttributeError, TypeError, ValueError):
            pass
    if row['laterality'] not in ('L', 'R'):
        side = str(getattr(ds, 'ImageLaterality', '')).upper()
        if side in ('L', 'R'):
            row['laterality'] = side
    return row


def study_frame(study_dir: str) -> "pd.DataFrame":
    """Per-slice metadata frame for one study directory (study/series/*.dcm)."""
    import glob

    import pandas as pd
    rows = []
    for path in sorted(glob.glob(os.path.join(study_dir, "*", "*.dcm"))):
        try:
            row = read_slice_meta(path, root=study_dir)
            if not row['study_uid'] or not row['series_uid']:
                raise ValueError('missing study/series UID')
            rows.append(row)
        except Exception as exc:
            warnings.warn(f'skipped unreadable DICOM header {path}: {type(exc).__name__}: {exc}',
                          RuntimeWarning, stacklevel=2)
    return pd.DataFrame(rows, columns=COLUMNS)
