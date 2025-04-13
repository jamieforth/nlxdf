import textwrap

import matplotlib.pyplot as plt
import numpy as np


# Plotting
def scale_seconds(df, units="seconds"):
    if units == "seconds":
        return df
    if units == "hours":
        return df / 60 / 60
    if units == "minutes":
        return df / 60
    if units == "milliseconds":
        return df * 1000
    if units == "microseconds":
        return df * 1e6
    if units == "nanoseconds":
        return df * 1e9
    print(f"Unknown units {units}")
    return None


def format_load_params(df):
    if "load_params" in df.attrs:
        params = "\n".join(
            [
                f"{k}={v}"
                for (k, v) in df.attrs["load_params"].items()
                if k != "select_streams"
            ]
        )
    return params


def format_title(title, df):
    if "load_params" in df.attrs and "select_streams" in df.attrs["load_params"]:
        select_streams = ", ".join(
            [
                f"{list(select.keys())[0]}={list(select.values())[0]}"
                if isinstance(select, dict)
                else f"stream-id={select}"
                for select in df.attrs["load_params"]["select_streams"]
            ]
        )
        title = textwrap.fill(title + ": " + select_streams, 80)
    return title


def plot_sample_counts_df(df):
    n = df.shape[0]
    ax = (
        df["sample_count"]
        .unstack(level=1)
        .plot.barh(
            colormap="tab20",
            figsize=(6, 4 * max(n * 0.05, 1)),
            width=0.9,
        )
    )
    ax.legend(bbox_to_anchor=(1, 1), loc=2)
    ax.set_xlabel("sample count")
    title = "Sample counts across recordings"
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
    ax.invert_yaxis()
    return ax


# FIXME by segment?
def plot_time_stamp_intervals_df(df, units="milliseconds", showfliers=True):
    df = scale_seconds(df, units)
    n = df.index.levels[0].size
    cols = 2
    rows = np.ceil(n / cols).astype(int)
    fig, axes = plt.subplots(
        rows, cols, sharex=True, sharey=False, figsize=(6 + cols, (4 + rows))
    )
    axes = axes.ravel()
    for (rec, data), i in zip(df.groupby(level=0, sort=False), range(0, n)):
        data = data.dropna(axis=1, how="all").sort_index(axis=1)
        data.plot.box(vert=False, ax=axes[i], showfliers=showfliers)
        axes[i].set_title(rec)
        axes[i].set_xlabel(units)
        axes[i].invert_yaxis()
    title = "Time-stamp interval per device across recordings"
    axes[0].text(
        x=1.0,
        y=1.0,
        s=format_load_params(df),
        fontsize=7,
        transform=axes[0].transAxes,
        horizontalalignment="left",
        verticalalignment="bottom",
    )
    fig.suptitle(title)
    fig.tight_layout()
    return axes


def plot_first_time_stamps_df(df, units="seconds"):
    df = scale_seconds(df, units)
    earliest = df["first_timestamp"].groupby(level=0, sort=False).min()
    df = df["first_timestamp"] - earliest
    n = df.index.levels[0].size
    cols = 2
    rows = np.ceil(n / cols).astype(int)
    fig, axes = plt.subplots(
        rows, cols, sharex=True, sharey=False, figsize=(6 + cols, 4 + rows)
    )
    axes = axes.ravel()
    for (rec, data), i in zip(df.groupby(level=0, sort=False), range(0, n)):
        # First segment only.
        data = data.xs(0, level='segment')
        data.reset_index().plot.scatter(
            x="first_timestamp", y="stream_id", marker="|", ax=axes[i]
        )
        axes[i].set_title(rec)
        axes[i].set_xlabel(units)
        axes[i].invert_yaxis()
    title = "First time-stamp per device across recordings"
    axes[0].text(
        x=1.0,
        y=1.0,
        s=format_load_params(df),
        fontsize=7,
        transform=axes[0].transAxes,
        horizontalalignment="left",
        verticalalignment="bottom",
    )
    fig.suptitle(title)
    fig.tight_layout()
    return axes


def plot_first_time_stamps_dist_df(df, units="seconds"):
    # First segment only.
    df = df.xs(0, level='segment')
    df = scale_seconds(df, units)
    earliest = df["first_timestamp"].groupby(level=0, sort=False).min()
    df = df["first_timestamp"] - earliest
    ax = df.unstack(0).plot.box(vert=False)
    ax.set_xlabel(units)
    ax.set_ylabel("recording")
    ax.invert_yaxis()
    title = "First time-stamp distribtion across recordings"
    ax.text(
        x=1.0,
        y=1.0,
        s=format_load_params(df),
        fontsize=7,
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="bottom",
    )

    ax.set_title(title)
    return ax


def plot_clock_offsets(data, normalise=False):
    n = len(data)
    cols = 2
    rows = np.ceil(n / cols).astype(int)
    fig, axes = plt.subplots(
        rows, cols, sharex=False, sharey=False, figsize=(6 + cols, 4 + rows)
    )
    axes = axes.ravel()
    for (stream_id, offsets), i in zip(data.items(), range(0, n)):
        if normalise:
            offsets["time"] = offsets["time"] - offsets["time"].median()
            offsets["value"] = offsets["value"] - offsets["value"].median()
        offsets = offsets.set_index("time")
        offsets.plot(ax=axes[i], legend=None)
        axes[i].set_title(stream_id)
    title = "Clock offsets per stream"
    fig.suptitle(title)
    fig.tight_layout()
    return axes
