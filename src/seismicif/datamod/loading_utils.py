import numpy as np
import obspy
import pandas as pd
import subprocess
import os
from pathlib import Path
from datetime import timedelta, datetime
from seismicif.datamod.preproc_utils import preproc_stream
from seismicif.metrics import iou


def sliding_windows_from_trace(inp, window_size=10000, stride=5000):
    start_time = inp.stats.starttime
    sr = inp.stats.sampling_rate

    inp = np.array(inp)

    if inp.shape[0] < window_size:
        return np.zeros([0, window_size]), []

    up = int((inp.shape[0] - window_size) / stride) + 1
    X = np.zeros([up, window_size])
    T = []

    for i in range(up):
        t0 = stride * i
        t1 = t0 + window_size
        X[i, :] = inp[t0:t1]
        T.append(
            [
                start_time + timedelta(seconds=t0 / sr),
                start_time + timedelta(seconds=t1 / sr),
            ]
        )

    return X, T


def sliding_windows_from_stream(st, window_size=10000, stride=5000):
    X = np.zeros([0, window_size])
    T = []

    for tr in st:
        Xn, Tn = sliding_windows_from_trace(tr, window_size=window_size, stride=stride)
        if Xn.shape[0] == 0:
            continue
        T.append(Tn)
        X = np.vstack([X, Xn])

    if X.shape[0] == 0:
        return X, []

    T = np.vstack(T)

    return X, T


def remove_duplicate_traces(stream):
    unique_traces = []
    seen = set()

    for trace in stream:
        # Create a unique identifier for the trace based on start time and data
        identifier = (trace.stats.starttime.timestamp, trace.data.tobytes())

        if identifier not in seen:
            seen.add(identifier)
            unique_traces.append(trace)

    return obspy.Stream(traces=unique_traces)


def remove_overlaps(stream):
    # Sort traces by starttime
    stream.sort(keys=["starttime"])

    # List to hold the final non-overlapping traces
    non_overlapping_traces = obspy.Stream()

    # Iterate through sorted traces and handle overlaps
    for i, trace in enumerate(stream):
        if i == 0:
            non_overlapping_traces.append(trace)
        else:
            prev_trace = non_overlapping_traces[-1]
            if trace.stats.starttime < prev_trace.stats.endtime:
                # If there's an overlap, trim the current trace
                overlap_duration = prev_trace.stats.endtime - trace.stats.starttime
                overlap_samples = int(overlap_duration * trace.stats.sampling_rate)

                # Ensure we don't try to access more samples than available
                if overlap_samples >= trace.data.shape[0]:
                    # Skip this trace if the overlap removes all data
                    continue

                trace.data = trace.data[overlap_samples:]
                trace.stats.starttime = prev_trace.stats.endtime

            if (
                trace.data.shape[0] > 0
            ):  # Only append if there are data points left after trimming
                non_overlapping_traces.append(trace)

    return non_overlapping_traces


def preproc_flow_annotations(flows):
    start = []
    for i in flows["start"]:
        t0 = i

        if "T" in t0:
            t0 = t0.replace("Z", "")
        else:
            t0 = t0.replace(" ", "T")
        start.append(t0)

    flows["start"] = [obspy.UTCDateTime(i) for i in start]

    stop = []
    for i in flows["stop"]:
        t0 = i

        if "T" in t0:
            t0 = t0.replace("Z", "")
        else:
            t0 = t0.replace(" ", "T")
        stop.append(t0)

    flows["stop"] = [obspy.UTCDateTime(i) for i in stop]

    return flows


def date_to_year_day(date_str):
    date = datetime.strptime(str(date_str), "%Y-%m-%d")
    year = date.year
    day_of_year = date.timetuple().tm_yday
    return f"{year}.{day_of_year:03d}"


def find_date_in_strings(strings, date_str):
    target = date_to_year_day(date_str)
    return [s for s in strings if target in str(s)]


def read_segment(t0, t1, paths):
    d1 = str(t0)[:10]
    d2 = str(t1)[:10]

    t0 = obspy.UTCDateTime(str(t0))
    t1 = obspy.UTCDateTime(str(t1))

    st = obspy.Stream()
    for i in pd.date_range(start=d1, end=d2, freq="D"):
        new_st = obspy.read(find_date_in_strings(paths, str(i)[:10])[0])
        st += new_st

    preproc_stream(st)
    st = remove_overlaps(remove_duplicate_traces(st))
    st.trim(t0, t1)

    x = np.array([])
    for tr in st:
        x = np.append(x, tr.data)

    new_tr = obspy.Trace()
    new_tr.data = x
    new_tr.stats.sampling_rate = tr.stats.sampling_rate
    new_tr.stats.starttime = t0

    return new_tr


def extract_template(t0, t1, if_mod, paths, window_size=10000, stride=5000):
    tr = read_segment(t0, t1, paths)
    X, _ = sliding_windows_from_trace(tr, window_size=window_size, stride=stride)

    if_scores = -if_mod.score_samples(X)
    tem = X[np.argmax(if_scores), :]

    return (tem - np.mean(tem)) / np.std(tem)


def get_git_root() -> Path:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL
        )
        return Path(output.decode("utf-8").strip())
    except subprocess.CalledProcessError:
        raise RuntimeError("Not inside a Git repository")


def get_data_subpath(*subpath_parts: str) -> Path:
    git_root = get_git_root()
    data_path = git_root / "data" / Path(*subpath_parts)
    return data_path


def find_paths(network, station, channel, start=2018, stop=2020):
    paths = []
    for i in range(start, stop + 1):
        path = get_data_subpath(network, str(i), station, channel)
        stream_paths = [path / f for f in os.listdir(path) if not f.startswith("._")]
        paths.append(
            np.array(sorted(stream_paths, key=lambda f: int(str(f).rsplit(".", 1)[-1])))
        )
    paths = np.concatenate(paths)

    return paths


def extract_split_flows(df, station, start, stop):
    start = obspy.UTCDateTime(f"{start}-01-01")
    stop = obspy.UTCDateTime(f"{stop}-12-31T23:59:59.999999")

    df = df[df["station"] == station].reset_index(drop=True)
    df = df[(df["start"] > start) & (df["stop"] < stop)].reset_index(drop=True)
    df = df[df["confidence"] != "earthquake"].reset_index(drop=True)
    df = df[df["confidence"] != "rockfall"].reset_index(drop=True)
    df = df[df["confidence"] != "quarrywork"].reset_index(drop=True)

    high_conf_flows = df[df["confidence"] == "high"].reset_index(drop=True)
    lower_conf_flows = df[
        (df["confidence"] == "low") | (df["confidence"] == "med")
    ].reset_index(drop=True)

    return lower_conf_flows, high_conf_flows, df


def extract_valid_segments(segments, lower_conf_flows, high_conf_flows):
    if len(segments) == 0:
        return segments

    intersections, _, _ = iou(segments, lower_conf_flows)
    non_lower_conf = np.append(intersections[0], np.diff(intersections)) == 0
    intersections, _, _ = iou(segments, high_conf_flows)
    high_conf = np.append(intersections[0], np.diff(intersections)) > 0

    return segments[non_lower_conf | high_conf].reset_index(drop=True)
