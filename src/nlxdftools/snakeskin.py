import argparse
from pathlib import Path
import datetime
import mne

from nlxdftools import NlXdf


def markers_to_matlab(markers):
    df = markers["marker-ts"].copy()

    # Matlab style event file.
    matlab = df[0].str.extract(r"(^.*)( T:.*)", expand=True)
    matlab.columns = ["type", "values"]
    matlab = matlab.rename_axis(index="latency")
    return matlab


def markers_to_annot(markers, orig_time):
    df = markers["marker-ts"].copy()

    # Remove time-stamps from P300 to be used as events.
    p300 = df[0].str.extract(r"(^P300:\d{4})").dropna()
    df.loc[p300.index] = p300
    annotations = mne.Annotations(df.index, 0, df.iloc[:, 0], orig_time=orig_time)
    return annotations


def main():
    parser = argparse.ArgumentParser(
        description="""Resample ~snakeskin in the wild~ XDF files."""
    )

    parser.add_argument(
        "-i",
        nargs="+",
        help="Source XDF files.",
    )

    parser.add_argument(
        "-o",
        help="Destination directory for resampled files.",
    )

    parser.add_argument(
        "--label",
        default=None,
        help="Label to append to destination directory.",
    )

    args = parser.parse_args()

    xdf_data_paths = [Path(path) for path in args.i]
    print(xdf_data_paths)

    start_time = datetime.datetime.now()
    label = args.label
    dest_dir = (
        f"{start_time.isoformat(timespec='seconds')}{'' if not label else f'-{label}'}"
    )
    dest_dir = Path(args.o) / dest_dir
    dest_dir.mkdir()

    for xdf_data_path in xdf_data_paths:
        subdir = dest_dir / Path(xdf_data_path).parent.stem
        subdir.mkdir()
        xdf = NlXdf(xdf_data_path).load()
        raws, markers = xdf.raw_mne(fs_max_ratio=1, annotation_fn=markers_to_annot)
        for stream_id, raw in raws.items():
            file_name = Path(xdf_data_path).stem
            file_name = f"{file_name}-{stream_id}.set"
            file_name = subdir / file_name
            print(file_name)
            mne.export.export_raw(file_name, raw)
        marker_file_name = Path(xdf_data_path).stem
        marker_file_name = f"{marker_file_name}-markers.tsv"
        marker_file_name = subdir / marker_file_name
        print(marker_file_name)
        markers_to_matlab(markers).to_csv(marker_file_name, sep="\t")
