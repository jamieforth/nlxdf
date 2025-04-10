"""Class for working with sets of Neurolive/AntNeuro XDF data files."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from nlxdftools import NlXdf, plotting


class NlXdfDataset(Mapping):
    """Class for working with sets of Neurolive/AntNeuro XDF data files."""

    def __init__(self, xdf_data_paths, verbose=False):
        """Dataset is a dictionary of {label, nlxdf} pairs."""
        self.dataset = {key: NlXdf(xdf_data_path, verbose)
                        for key, xdf_data_path in xdf_data_paths.items()}

    def __getitem__(self, key):
        """Get XDF file from dataset."""
        return self.dataset.__getitem__(key)

    def __len__(self):
        """Return the number of XDF files in data.."""
        return self.dataset.__len__

    def __iter__(self):
        """Iterate over datasets."""
        return self.dataset.__iter__()

    def count_devices(self):
        """Count the number of devices in each XDF file."""
        data = {}
        for recording, nlxdf in self.items():
            streams = nlxdf.resolve_streams()
            data[recording] = len(streams.index)
        df = pd.DataFrame(data, index=['num device']).T
        df.index.rename('recording', inplace=True)
        return df

    def stream_ids(self):
        """Tabulate stream-ids in each XDF file."""
        data = {}
        for recording, nlxdf in self.items():
            streams = nlxdf.resolve_streams()
            data[recording] = pd.Series(
                list(streams.index), index=range(1, len(streams) + 1)
            )
        df = pd.DataFrame(data)
        df.columns.rename("recording", inplace=True)
        return df

    def hostnames(self):
        """Tabulate device hostnames in each XDF file."""
        data = {}
        for recording, nlxdf in self.items():
            streams = nlxdf.resolve_streams()
            data[recording] = pd.Series(
                list(streams["hostname"]), index=range(1, len(streams) + 1)
            )
        df = pd.DataFrame(data)
        df.columns.rename("recording", inplace=True)
        return df

    def source_ids(self, warn_changed=True):
        """Tabulate stream-ids and their source-id for each XDF file."""
        data = {}
        for recording, nlxdf in self.items():
            streams = nlxdf.resolve_streams()
            data[recording] = streams["source_id"]
        df = pd.DataFrame(data)
        if warn_changed:
            stream_source_constant = df.apply(
                lambda row: row.dropna().nunique() == 1, axis=1
            )
            source_changed = stream_source_constant.loc[~stream_source_constant]
            for src in source_changed.index:
                print(f"Source changed {src}: {df.loc[src].dropna().unique()}")
        df.index.rename("stream_id", inplace=True)
        df.columns.rename("recording", inplace=True)
        return df

    def count_stream_types(self):
        """Return number of streams per type in each XDF file."""
        data = {}
        for recording, nlxdf in self.items():
            streams = nlxdf.resolve_streams()
            data[recording] = streams.loc[:, 'type'].value_counts()
        df = pd.DataFrame(data).T
        df.index.rename('recording', inplace=True)
        df.replace(np.nan, 0, inplace=True)
        return df

    def count_channels_per_type(self):
        """Return number of channels per type per device for each XDF file."""
        data = {}
        for recording, nlxdf in self.items():
            streams = nlxdf.resolve_streams()
            data[recording] = streams.loc[:, ["type", "channel_count"]]
        df = pd.concat(data, names=["recording", "stream_id"])
        return df

    def time_stamp_info(self, *select_streams, **kwargs):
        """Summarise loaded time-stamp data across recordings."""
        data = {}
        for recording, nlxdf in self.items():
            try:
                nlxdf.load(*select_streams, **kwargs)
            except ValueError as exc:
                print(exc)
                continue
            data[recording] = nlxdf.time_stamp_info()
            nlxdf.unload()
        df = pd.concat(data)
        df.index.rename('recording', level=0, inplace=True)
        return df

    def plot_sample_counts(self, *select_streams, **kwargs):
        df = self.time_stamp_info(*select_streams, **kwargs)
        ax = plotting.plot_sample_counts_df(df)
        return ax

    def max_sample_count_diff(self, *select_streams, **kwargs):
        """Compute maximum sample count difference across recordings."""
        df = self.time_stamp_info(*select_streams, **kwargs)
        max_samples = df['sample_count'].groupby(level=0, sort=False).max()
        min_samples = df['sample_count'].groupby(level=0, sort=False).min()
        count_diff = max_samples - min_samples
        count_diff.name = 'max_sample_count_diff'
        return count_diff

    def time_stamp_intervals(self, *select_streams, **kwargs):
        """Return time-stamp intervals across recordings."""
        data = {}
        for recording, nlxdf in self.items():
            try:
                nlxdf.load(*select_streams, **kwargs)
            except ValueError as exc:
                print(exc)
                continue
            intervals = nlxdf.time_stamp_intervals()
            if intervals is not None:
                data[recording] = intervals
            nlxdf.unload()
        df = pd.concat(data)
        df.index.rename('recording', level=0, inplace=True)
        return df

    def plot_time_stamp_intervals(self, *select_streams, units='nanoseconds',
                                  showfliers=True, **kwargs):
        df = self.time_stamp_intervals(*select_streams, **kwargs)
        axes = plotting.plot_time_stamp_intervals_df(df, units, showfliers)
        return axes

    def segment_info(self, *select_streams, **kwargs):
        """Summarise loaded segment data across recordings."""
        data = {}
        for recording, nlxdf in self.items():
            try:
                nlxdf.load(*select_streams, **kwargs)
            except ValueError as exc:
                print(exc)
                continue
            data[recording] = nlxdf.segment_info()
            nlxdf.unload()
        df = pd.concat(data)
        df.index.rename('recording', level=0, inplace=True)
        return df

    def check_channels(self, expected, *select_streams, **kwargs):
        """Check all devices have expected channels across recordings."""
        data = {}
        for recording, nlxdf in self.items():
            try:
                nlxdf.load(*select_streams, **kwargs)
            except ValueError as exc:
                print(exc)
                continue
            different = nlxdf.check_channels(expected)
            if different is None:
                print(f"{recording} channels correct")
            else:
                data[recording] = different
            nlxdf.unload()
        df = pd.concat(data)
        df.index.rename('recording', level=0, inplace=True)
        return df
