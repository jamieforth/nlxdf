import argparse
import datetime
from pathlib import Path

import git
import mne
import pandas as pd

from nlxdftools import NlXdf
from nlxdftools.export import (
    check_markers,
    export_fif,
    export_marker_csv,
    export_marker_tsv,
    export_set,
)


def markers_to_csv(markers):
    """CSV style event file."""

    # Mirko's markers
    mirko = markers["marker-ts"]
    # Keep only type and unix_time (nanoseconds).
    mirko = mirko[0].str.extract(r"(^.*)[ _][Tt]:(\d*)").dropna()
    mirko = pd.DataFrame({"type": mirko[0], "unix_time": mirko[1]})
    check_markers(len(markers["marker-ts"]), len(mirko))
    csv = {
        "marker-ts": mirko,
    }
    return csv

def markers_to_matlab(markers):
    """Matlab style event file."""

    # Mirko's markers
    mirko = markers["marker-ts"]
    # Keep only type and unix_time (nanoseconds).
    mirko = mirko[0].str.extract(r"(^.*)[ _][Tt]:(\d*)").dropna()
    mirko.columns = ["type", "unix_time"]
    mirko = mirko.rename_axis(index="latency")
    check_markers(len(markers["marker-ts"]), len(mirko))
    matlab = {
        "marker-ts": mirko,
    }
    return matlab


def markers_to_annot(markers, orig_time):
    df = markers["marker-ts"]

    # Extract heartbeats - don't include as annotations.
    heartbeats = df[0].str.extract(r"^(H)[ _][Tt]:(\d*)").dropna()
    heartbeats.columns = ["type", "unix_time"]

    # Separate timestamps from other message types.
    df = df[0].str.extract(r"(?!H[ _])^(.*)[ _][Tt]:(\d*)").dropna()
    check_markers(len(markers["marker-ts"]), len(heartbeats) + len(df))
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

    parser.add_argument(
        "--fs",
        default=512,
        type=int,
        help="Resample rate.",
    )

    args = parser.parse_args()

    xdf_data_paths = [Path(path) for path in args.i]
    for path in xdf_data_paths:
        print(path, flush=True)

    start_time = datetime.datetime.now()
    label = args.label
    repo = git.Repo(__file__, search_parent_directories=True)
    hexsha = repo.head.commit.hexsha[0:8]
    batch_dir = f"{start_time.isoformat(timespec='seconds')}{'' if not label else f'-{label}-{hexsha}'}"
    batch_dir = Path(args.o) / batch_dir
    batch_dir.mkdir()

    for xdf_data_path in xdf_data_paths:
        performance = Path(xdf_data_path).parent.stem
        perf_dir = batch_dir / performance
        perf_dir.mkdir(exist_ok=True)
        # Messages recordings.
        filename = Path(xdf_data_path).name
        if filename == "messages-1.xdf":
            xdf = NlXdf(xdf_data_path).load(2, 5, 9)
        elif filename == "messages-2.xdf":
            xdf = NlXdf(xdf_data_path).load(2, 5)
        elif filename == "messages-3.xdf":
            xdf = NlXdf(xdf_data_path).load(1, 3, 6)
        else:
            # Performance recordings.
            xdf = NlXdf(xdf_data_path).load()

        raws, markers = xdf.raw_mne(
            fs_new=args.fs, annotation_fn=markers_to_annot, with_stream_id=True
        )
        # Export data as SET and FIF.
        eeg_dir = perf_dir / "eeg"
        eeg_dir.mkdir(exist_ok=True)
        for stream_id, raw in raws.items():
            export_fif(xdf_data_path, eeg_dir, stream_id, raw)
            export_set(xdf_data_path, eeg_dir, stream_id, raw)
        # Export markers as CSV
        marker_dir = perf_dir / "markers"
        marker_dir.mkdir(exist_ok=True)
        csv_markers = markers_to_csv(markers)
        for stream_id, marker in csv_markers.items():
            export_marker_csv(xdf_data_path, marker_dir, stream_id, marker)
        # Export markers as matlab TSV
        matlab_markers = markers_to_matlab(markers)
        for stream_id, marker in matlab_markers.items():
            export_marker_tsv(xdf_data_path, marker_dir, stream_id, marker)
        del xdf
        del raws
        del markers
        del csv_markers
        del matlab_markers
