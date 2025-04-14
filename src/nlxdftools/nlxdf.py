"""Class for working with Neurolive/AntNeuro XDF data files."""

from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pdxdf import Xdf
from pdxdf.errors import NoLoadableStreamsError, XdfAlreadyLoadedError

from nlxdftools.plotting import format_load_params, format_title


class NlXdf(Xdf):
    """Class for working with individual Neurolive/AntNeuro XDF data files.

    Provides a pandas-based layer of abstraction over raw XDF data to
    simplify data processing.
    """

    hostname_device_mapper = {
        'DESKTOP-3R7C1PH': 'eeg-a',
        'DESKTOP-2TI6RBU': 'eeg-b',
        'DESKTOP-MN7K6RM': 'eeg-c',
        'DESKTOP-URRV98M': 'eeg-d',
        'DESKTOP-DATOEVU': 'eeg-e',
        'TABLET-9I44R1AR': 'eeg-f',
        'DESKTOP-SLAKFQE': 'eeg-g',
        'DESKTOP-6FJTJJN': 'eeg-h',
        'DESKTOP-HDOESKS': 'eeg-i',
        'DESKTOP-LIA3G09': 'eeg-j',
        'DESKTOP-V6779I4': 'eeg-k',
        'DESKTOP-PLV2A7L': 'eeg-l',
        'DESKTOP-SSSOE1L': 'eeg-m',
        'DESKTOP-RM16J67': 'eeg-n',
        'DESKTOP-N2RA68S': 'eeg-o',
        'DESKTOP-S597Q21': 'eeg-p',
        'DESKTOP-OE9298C': 'eeg-q',  # Snakeskin
        'TABLET-06K1PLD4': 'eeg-q',  # Readings
        'DESKTOP-MK0GQFM': 'eeg-r',
        'DESKTOP-7GV3RJU': 'eeg-s',
        'DESKTOP-S5A1PPK': 'eeg-t',
        'TABLET-3BS4NTP2': 'eeg-u',
        'DESKTOP-QG4CNEV': 'eeg-v',
        'TABLET-STDTE3Q6': 'eeg-w',  # Snakeskin
        'DESKTOP-OAF4OCM': 'eeg-w',  # Readings
        'DESKTOP-T3RKRMH': 'eeg-x',
        'TABLET-47TCFCEB': 'eeg-cs-v',
        'CGS-PCD-26098': 'tabarnak',
        'CGS-PCL-38928': 'laura',
        'cgs-macl-39034.campus.goldsmiths.ac.uk': 'mirko',
        'kassia': 'jamief',
    }

    metadata_mapper = {
        'type': {
            'EEG': 'eeg',
            #'marker': 'Timestamp',
        },
    }

    channel_metadata_mapper = {
        'label': {
            '0': 'Fp1',
            '1': 'Fpz',
            '2': 'Fp2',
            '3': 'F7',
            '4': 'F3',
            '5': 'Fz',
            '6': 'F4',
            '7': 'F8',
            '8': 'FC5',
            '9': 'FC1',
            '10': 'FC2',
            '11': 'FC6',
            '12': 'M1',
            '13': 'T7',
            '14': 'C3',
            '15': 'Cz',
            '16': 'C4',
            '17': 'T8',
            '18': 'M2',
            '19': 'CP5',
            '20': 'CP1',
            '21': 'CP2',
            '22': 'CP6',
            '23': 'P7',
            '24': 'P3',
            '25': 'Pz',
            '26': 'P4',
            '27': 'P8',
            '28': 'POz',
            '29': 'O1',
            '30': 'Oz',
            '31': 'O2',
            '32': 'Resp',  # EEG 101 with 34 channels?
            '67': 'CPz',
            '33': 'trigger',
            '34': 'counter',
        },
        'type': {
            'ref': 'eeg',
            'aux': 'misc',
            'bip': 'misc',
            'trigger': 'stim',
            'counter': 'misc',
            'trg': 'stim',
            'exg': 'ecg',
        },
    }

    def resolve_streams(self, nl_id_as_index=True):
        """Resolve XDF streams from file using pyxdf.resolve_stream().

        Apply custom device name mapping for Neurolive analysis.
        """
        df = super().resolve_streams()
        nl_ids = self._create_nl_ids(df)
        df['nl_id'] = nl_ids
        if nl_id_as_index:
            # Set nl_id as the index.
            df.reset_index(inplace=True)
            df.set_index('nl_id', inplace=True, verify_integrity=True)
            df.sort_index(inplace=True)
        else:
            # Append nl_id as a new column.
            cols = df.columns.tolist()
            # Move nl_id to first column.
            cols = cols[-1:] + cols[0:-1]
            df = df[cols]
        return df

    def load(
            self,
            *select_streams,
            channel_scale_field='unit',
            channel_name_field='label',
            synchronize_clocks=True,
            dejitter_timestamps=True,
            handle_clock_resets=True,
            handle_non_monotonic=True,
            **kwargs):
        """Load XDF data from file using pyxdf.load_xdf().

        Apply custom defaults for Neurolive analysis.
        """
        try:
            self._load(*select_streams,
                       channel_scale_field=channel_scale_field,
                       channel_name_field=channel_name_field,
                       synchronize_clocks=synchronize_clocks,
                       dejitter_timestamps=dejitter_timestamps,
                       handle_clock_resets=handle_clock_resets,
                       handle_non_monotonic=handle_non_monotonic,
                       **kwargs)
        except (NoLoadableStreamsError, XdfAlreadyLoadedError) as exc:
            print(exc)
            return self

        # Map stream-IDs to neurolive IDs.
        self._loaded_stream_ids = self._map_stream_ids(self.loaded_stream_ids)
        self._desc = self._map_stream_ids(self._desc)
        self._segments = self._map_stream_ids(self._segments)
        self._clock_segments = self._map_stream_ids(self._clock_segments)
        self._channel_info = self._map_stream_ids(self._channel_info)
        self._footer = self._map_stream_ids(self._footer)
        self._clock_offsets = self._map_stream_ids(self._clock_offsets)
        self._time_series = self._map_stream_ids(self._time_series)
        self._time_stamps = self._map_stream_ids(self._time_stamps)
        return self

    def check_channels(self, expected):
        if not isinstance(expected, pd.Series):
            expected = pd.Series(expected, name="expected")
        ch = self.channel_info(cols='label', concat=True)
        ch = ch.droplevel(1, axis=1)
        same = ch.eq(expected, axis=0)
        if same.all().all():
            return "All channels correct"
        else:
            different = ch.loc[:, ~same.all()]
            different = different.fillna('missing')
            different = different[~different.eq(expected, axis=0)]
            different['expected'] = expected
            return different.dropna()

    def _parse_info(self, data, nl_id_as_index=True, **kwargs):
        """Map neurolive stream ids and types."""
        df = super()._parse_info(data, **kwargs)
        # Lowercase types following MNE convention.
        df['type'] = df['type'].str.lower()
        # Fix-up metadata types.
        df.replace(self.metadata_mapper, inplace=True)
        nl_ids = self._create_nl_ids(df)
        df['nl_id'] = nl_ids
        if nl_id_as_index:
            # Set nl_id as the index.
            df.reset_index(inplace=True)
            df.set_index('nl_id', inplace=True, verify_integrity=True)
            df.sort_index(inplace=True)
        else:
            # Append nl_id as a new column.
            cols = df.columns.tolist()
            # Move nl_id to first column.
            cols = cols[-1:] + cols[0:-1]
            df = df[cols]
        return df

    def _parse_channel_info(self, data, **kwargs):
        """Map AntNeuro stream ids and channel names."""
        data = super()._parse_channel_info(data, **kwargs)
        if data is not None:
            for df in data.values():
                df['type'] = df['type'].str.lower()
                # For AntNeuro App which doesn't include channel labels.
                if 'index' in df and 'label' not in df:
                    df['label'] = df['index']
                df.replace(self.channel_metadata_mapper,
                           inplace=True)
        return data

    def _create_nl_ids(self, df):

        unique_id_counter = Counter()

        def make_nl_id(row, unique_id_counter, hostname_map):
            # Fallback to stream_id as string.
            nl_id = str(row.name)

            # EEG devices.
            if row['type'].lower() == 'eeg':
                if row['hostname'] in hostname_map:
                    # Map tablet hostname to eeg-* labels.
                    nl_id = hostname_map[row['hostname']]
                else:
                    # Unknown EEG device.
                    nl_id = 'eeg-?'

            # Eye tracking.
            elif row['name'].lower().startswith('pupil'):
                # Map Pupil Labs device/streams.
                if row['type'].lower() == 'event':
                    nl_id = f'pl-{row["source_id"]}-event'
                elif row['type'].lower() == 'gaze':
                    nl_id = f'pl-{row["source_id"]}-gaze'

            # Marker streams.
            elif row['name'] == 'TABARNAK V3':
                nl_id = 'marker-ts'
            elif row['name'] == 'TimestampStream':
                nl_id = 'marker-ts'
            elif row['name'] == 'CameraRecordingTime':
                nl_id = 'marker-video'
            elif row['name'] == 'audio':
                nl_id = 'marker-audio'
            elif row['name'] == 'Keyboard_Marker_Stream':
                nl_id = 'marker-kb'
            elif row['name'] == 'FrameNumber_Stream':
                nl_id = 'marker-video'

            # Simulated sync test streams.
            elif row['type'] == 'data' and row['name'].startswith('Test'):
                if row['hostname'] in ['neurolive', 'bobby']:
                    # Sync test running on the LabRecorder host -- the
                    # closest thing we have to a ground truth with
                    # simulated data.
                    nl_id = 'test-ref'
                elif row['hostname'] in hostname_map:
                    # Sync test running on an EEG tablet or known host.
                    nl_id = f'test-{hostname_map[row["hostname"]]}'
                else:
                    # Sync test running on another device.
                    nl_id = 'test'
            elif row['type'] == 'control':
                # Simulated sync test control stream.
                nl_id = 'test-ctrl'

            # Generic mappings.
            elif row['type'].lower() == 'markers':
                # Catch-all marker stream mapper.
                nl_id = 'marker-' + nl_id
            elif row['name'].startswith('_relay_'):
                # Catch-all relayed streams.
                nl_id = 'relay-' + nl_id

            # Automatically increment ID for duplicate stream IDs.
            unique_id_counter.update([nl_id])
            if unique_id_counter[nl_id] > 1:
                nl_id = f'{nl_id}-{unique_id_counter[nl_id]}'
            return nl_id

        nl_ids = df.apply(
            make_nl_id,
            axis='columns',
            unique_id_counter=unique_id_counter,
            hostname_map=self.hostname_device_mapper,
        )
        return nl_ids

    def _map_stream_ids(self, data):
        if data is None:
            return data
        if isinstance(data, list):
            data = [self._stream_id_to_nl_id(stream_id)
                    for stream_id in data]
            data.sort()
        elif isinstance(data, dict):
            data = {self._stream_id_to_nl_id(stream_id): df
                    for stream_id, df in data.items()}
            data = dict(sorted(data.items()))
        elif isinstance(data, pd.DataFrame):
            data.rename(index=self._stream_id_to_nl_id, inplace=True)
            data.sort_index(inplace=True)
        return data

    def _stream_id_to_nl_id(self, stream_id):
        nl_id = self._info.index[
            self._info['stream_id'] == stream_id
        ][0]
        return nl_id

    def plot_time_stamps(
        self,
        *stream_ids,
        exclude=[],
        title="Time-stamps",
        subplots=False,
        non_monotonic=False,
        downsample_non_monotonic=True,
    ):
        data = self.time_stamps(*stream_ids, exclude=exclude, with_stream_id=True)
        with mpl.rc_context(
            {"axes.prop_cycle": plt.cycler("color", plt.cm.tab20.colors)}
        ):
            n = len(data)
            if n > 1 and subplots:
                fig, axes = plt.subplots(
                    n, figsize=(8, 4 * (0.8 * n)), sharex=True, sharey=True
                )
            else:
                fig, ax = plt.subplots(1)
                axes = [ax]
            for i, (stream_id, ts) in zip(range(n), data.items()):
                ax = axes[i % len(axes)]
                if non_monotonic:
                    non_mono = ts.loc[ts.diff() < 0]
                    if downsample_non_monotonic:
                        downsample = max(int(non_mono.shape[0] / 500), 1)
                    else:
                        downsample = 1
                    if non_mono.any():
                        non_mono.to_frame()[::downsample].plot.scatter(
                            "time_stamp",
                            "time_stamp",
                            marker="_",
                            linewidth=0.5,
                            ax=ax,
                            s=500,
                            color=plt.cm.tab20.colors[7],
                            alpha=0.8,
                            label='non-monotonic',
                        )
                ts.to_frame().plot.scatter("time_stamp", "time_stamp", ax=ax, label=stream_id, s=1)
                ax.legend(bbox_to_anchor=(1, 1), loc=2)
            title = format_title(title, ts)
            axes[0].set_title(title)
            axes[0].text(
                x=1.0,
                y=1.0,
                s=format_load_params(ts),
                fontsize=7,
                transform=axes[0].transAxes,
                horizontalalignment="left",
                verticalalignment="bottom",
            )
        return axes

    def plot_data(
        self, *stream_ids, exclude=[], cols=None, title="XDF data", subplots=False
    ):
        data = self.data(*stream_ids, exclude=exclude, cols=cols, with_stream_id=True)
        with mpl.rc_context(
            {"axes.prop_cycle": plt.cycler("color", plt.cm.tab20.colors)}
        ):
            n = len(data)
            if n > 1 and subplots:
                fig, axes = plt.subplots(
                    n, figsize=(10, 4 * (0.5 * n)), sharex=True, sharey=True
                )
            else:
                fig, ax = plt.subplots(1)
                axes = [ax]
            for i, (stream_id, df), segments, clock_segments in zip(
                range(n),
                data.items(),
                self.segments(
                    *stream_ids, exclude=exclude, with_stream_id=True
                ).values(),
                self.clock_segments(
                    *stream_ids, exclude=exclude, with_stream_id=True
                ).values(),
            ):
                df = pd.concat({stream_id: df}, axis=1)
                ax = axes[i % len(axes)]
                for i, (seg_start, seg_end) in zip(range(len(segments)), segments):
                    # Plot start of segments.
                    if i == 0:
                        ax.axvline(
                            df.index[seg_start],
                            color=plt.cm.tab20.colors[2],
                            alpha=0.5,
                            label="segments",
                        )
                    else:
                        ax.axvline(
                            df.index[seg_start],
                            color=plt.cm.tab20.colors[2],
                            alpha=0.5,
                        )
                for i, (c_seg_start, c_seg_end) in zip(
                    range(len(clock_segments)), clock_segments
                ):
                    # Plot end of clock segments.
                    if i == 0:
                        ax.axvline(
                            df.index[c_seg_end],
                            color=plt.cm.tab20.colors[6],
                            alpha=0.5,
                            label="clock segments",
                        )
                    else:
                        ax.axvline(
                            df.index[c_seg_end],
                            color=plt.cm.tab20.colors[6],
                            alpha=0.5,
                        )
                df.plot(ax=ax)
                ax.legend(bbox_to_anchor=(1, 1), loc=2)
            title = format_title(title, df)
            axes[0].set_title(title)
            axes[0].text(
                x=1.0,
                y=1.0,
                s=format_load_params(df),
                fontsize=7,
                transform=axes[0].transAxes,
                horizontalalignment="left",
                verticalalignment="bottom",
            )
        return axes

    def plot_data_box(self, *stream_ids, exclude=[], cols=None, title="XDF data"):
        if cols is not None and not isinstance(cols, list):
            cols = [cols]
        df = self.data(
            *stream_ids,
            exclude=exclude,
            cols=cols,
            with_stream_id=True,
            as_single_df=True,
        )
        with mpl.rc_context(
            {"axes.prop_cycle": plt.cycler("color", plt.cm.tab20.colors)}
        ):
            ax = df.plot.box(vert=False)
            ax.invert_yaxis()
            title = format_title(title, df)
            ax.set_title(title)
            ax.text(
                x=1.0,
                y=1.0,
                s=format_load_params(df),
                fontsize=7,
                transform=ax.transAxes,
                horizontalalignment="left",
                verticalalignment="bottom",
            )
            ax.set_xlabel("value")
        return ax
