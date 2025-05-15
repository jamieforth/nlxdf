from pathlib import Path

import mne


def export_set(xdf_data_path, base_dir, stream_id, raw):
    set_dir = base_dir / "set"
    set_dir.mkdir(exist_ok=True)
    file_name = Path(xdf_data_path).stem
    file_name = f"{file_name}-{stream_id}.set"
    file_name = set_dir / file_name
    print(file_name)
    mne.export.export_raw(file_name, raw)


def export_fif(xdf_data_path, base_dir, stream_id, raw):
    fif_dir = base_dir / "fif"
    fif_dir.mkdir(exist_ok=True)
    file_name = Path(xdf_data_path).stem
    file_name = f"{file_name}-{stream_id}.raw.fif"
    file_name = fif_dir / file_name
    print(file_name)
    raw.save(file_name)


def export_marker_csv(xdf_data_path, base_dir, stream_id, marker):
    csv_dir = base_dir / "csv"
    csv_dir.mkdir(exist_ok=True)
    marker_file_name = Path(xdf_data_path).stem
    marker_file_name = f"{marker_file_name}-{stream_id}.csv"
    marker_file_name = csv_dir / marker_file_name
    print(marker_file_name)
    marker.to_csv(marker_file_name)


def export_marker_tsv(xdf_data_path, base_dir, stream_id, marker):
    tsv_dir = base_dir / "tsv"
    tsv_dir.mkdir(exist_ok=True)
    marker_file_name = Path(xdf_data_path).stem
    marker_file_name = f"{marker_file_name}-{stream_id}.tsv"
    marker_file_name = tsv_dir / marker_file_name
    print(marker_file_name)
    marker.to_csv(marker_file_name, sep="\t")
