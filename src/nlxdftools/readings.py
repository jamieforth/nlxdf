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


class ReadingsXdf(NlXdf):
    """Custom hacks for parsing Readings data.

    """
    def _parse_info(self, data, nl_id_as_index=True, info_fix_fn=None, **kwargs):
        df = super()._parse_info(data, **kwargs)
        if 'marker-video' in df.index:
            # Fix incorrect marker-stream sample rate.
            df.loc['marker-video', 'nominal_srate'] = 0
        return df

    def data(
        self,
        *stream_ids,
        exclude=[],
        cols=None,
        ignore_missing_cols=False,
        with_stream_id=False,
        concat=False,
    ):
        data = super().data(
            *stream_ids,
            exclude=exclude,
            cols=cols,
            ignore_missing_cols=ignore_missing_cols,
            with_stream_id=True,
            concat=False,
        )
        # Fix marker-audio.
        if "marker-audio" in data:
            if self.header()["datetime"] == pd.Timestamp(
                "2024-11-09 19:45:52+0000", tz="UTC"
            ):
                # Extra special hack for Saturday. The audio recording started
                # before LabRecorder stated, and stopped afterwards - so we
                # have no markers to synchronise the audio! This number is the
                # estimated start time derived by measuring the latency between
                # markers, audio recordings and phase-aligned voice samples
                # from Thursday and Friday, and applying that to align Saturday
                # audio and markers.
                start_t = 56220.307308375
                data["marker-audio"] = pd.DataFrame(
                    {
                        "type": ["AUDIO_START"],
                        "seconds": [0],
                    },
                    index=pd.MultiIndex.from_tuples(
                        [(0, 0, start_t)],
                        names=["segment", "sample", "time_stamp"],
                    ),
                )
            else:
                df = data["marker-audio"]
                df = df.droplevel("segment")
                start_t = (
                    df.loc[df[0].str.contains('"state": 2')]
                    .reset_index("time_stamp")
                    .iloc[-1][
                        "time_stamp"
                    ]  # last occurrence (Friday had a false start)
                )
                data["marker-audio"] = pd.DataFrame(
                    {
                        "type": ["AUDIO_START"],
                        "seconds": [0],
                    },
                    index=pd.MultiIndex.from_tuples(
                        [(0, 0, start_t)],
                        names=["segment", "sample", "time_stamp"],
                    ),
                )
            data["marker-audio"].attrs.update({"load_params": self.load_params})
        if concat:
            data = pd.concat(data, axis=0, names=["stream_id"]).sort_index()
            data.attrs.update({"load_params": self.load_params})
            return data
        else:
            return self._single_or_multi_stream_data(data, with_stream_id)


def markers_to_csv(markers):
    """CSV style event file."""
    csv = {}

    # Mirko's markers
    if "marker-ts" in markers:
        mirko = markers["marker-ts"]
        mirko = mirko[0].str.extract(r"^T:(\d*)_M:(\w*)").dropna()
        mirko.columns = ["unix_time", "type"]
        mirko = pd.DataFrame({"type": mirko["type"], "unix_time": mirko["unix_time"]})
        check_markers(len(markers["marker-ts"]), len(mirko))
        csv["marker-ts"] = mirko

    # Audio
    if "marker-audio" in markers:
        audio = markers["marker-audio"]
        csv["marker-audio"] = audio

    # Video: Select only one of the duplicate columns.
    if "marker-video" in markers:
        video = markers["marker-video"][[0]]
        video = pd.DataFrame({"type": "H", "milliseconds": video[0]})
        csv["marker-video"] = video
    return csv


def markers_to_matlab(markers):
    """Matlab style event file."""
    matlab = {}

    # Mirko's markers
    if "marker-ts" in markers:
        mirko = markers["marker-ts"]
        mirko = mirko[0].str.extract(r"^T:(\d*)_M:(\w*)").dropna()
        mirko.columns = ["unix_time", "type"]
        mirko = pd.DataFrame({"type": mirko["type"], "unix_time": mirko["unix_time"]})
        mirko = mirko.rename_axis(index="latency")
        check_markers(len(markers["marker-ts"]), len(mirko))
        matlab["marker-ts"] = mirko

    # Audio
    if "marker-audio" in markers:
        audio = markers["marker-audio"]
        audio = audio.rename_axis(index="latency")
        matlab["marker-audio"] = audio

    # Video
    if "marker-video" in markers:
        video = markers["marker-video"][[0]]
        video = pd.DataFrame({"type": "H", "milliseconds": video[0]})
        video.rename_axis(index="latency", inplace=True)
        matlab["marker-video"] = video
    return matlab


def markers_to_annot(markers, orig_time):
    df = markers["marker-ts"]

    # Extract heartbeats - don't include as annotations.
    heartbeats = df[0].str.extract(r"^T:(\d*)_M:(H)").dropna()
    heartbeats.columns = ["unix_time", "type"]

    # Separate timestamps from other message types.
    df = df[0].str.extract(r"^T:(\d*)_M:(?!H)(\w*)").dropna()
    df.columns = ["unix_time", "type"]
    check_markers(len(markers["marker-ts"]), len(heartbeats) + len(df))
    annotations = mne.Annotations(df.index, 0, df.loc[:, "type"], orig_time=orig_time)

    # Audio
    if "marker-audio" in markers:
        df = markers["marker-audio"]
        annotations.append(df.index, 0, df.iloc[:, 0])

    # Video
    # Don't add video as annotations.
    # if "marker-video" in markers:
    #     df = markers["marker-video"]
    #     annotations.append(df.index, 0, df.iloc[:, 0])
    return annotations


def main():
    parser = argparse.ArgumentParser(
        description="""Resample 'Readings of what was never written' XDF
        files."""
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
        if "fri" in performance:
            exclude = [
                "test-ref",
                "test-ctrl",
                "test-eeg-n",
            ]
        elif "sat" in performance:
            exclude = [
                "test-ref",
                "test-ctrl",
                "test-eeg-k",
                "test-eeg-k-2",
            ]
        else:
            exclude = []

        perf_dir = batch_dir / performance
        perf_dir.mkdir()
        xdf = ReadingsXdf(xdf_data_path).load()
        raws, markers = xdf.raw_mne(
            fs_new=args.fs,
            annotation_fn=markers_to_annot,
            exclude=exclude,
            with_stream_id=True,
        )
        # Export data as SET and FIF.
        eeg_dir = perf_dir / "eeg"
        eeg_dir.mkdir()
        for stream_id, raw in raws.items():
            export_fif(xdf_data_path, eeg_dir, stream_id, raw)
            export_set(xdf_data_path, eeg_dir, stream_id, raw)
        # Export markers as CSV
        marker_dir = perf_dir / "markers"
        marker_dir.mkdir()
        csv_markers = markers_to_csv(markers)
        for stream_id, marker in csv_markers.items():
            if stream_id == "marker-ts":
                comment = "Unix timestamps drift, do not use!"
            elif stream_id == "marker-video":
                comment = "Millisecond timestamps drift randomly?"
            else:
                comment = None
            export_marker_csv(xdf_data_path, marker_dir, stream_id, marker, comment)
        # Export markers as matlab TSV
        matlab_markers = markers_to_matlab(markers)
        for stream_id, marker in matlab_markers.items():
            if stream_id == "marker-ts":
                comment = "Unix timestamps drift, do not use!"
            elif stream_id == "marker-video":
                comment = "Millisecond timestamps drift randomly?"
            else:
                comment = None
            export_marker_tsv(xdf_data_path, marker_dir, stream_id, marker, comment)
        del xdf
        del raws
        del markers
        del csv_markers
        del matlab_markers
