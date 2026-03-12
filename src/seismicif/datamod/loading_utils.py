import numpy as np
import obspy
import pandas as pd
import subprocess
import os
from pathlib import Path
from datetime import timedelta, datetime
from seismicif.datamod.preproc_utils import preproc_stream


def sliding_windows_from_trace(inp, window_size=10000, stride=5000):
    """
    Generates sliding windows from a seismic trace and returns the windowed data along with their corresponding time intervals.
    Parameters
    ----------
    inp : obspy.Trace
        Trace object containing the signal to extract windows from, its metadata (starttime, sampling_rate) will be used.
    window_size : int, optional
        Number of samples in each window. Default is 10,000.
    stride : int, optional
        Number of samples to move the window at each step. Default is 5,000.
    Returns
    -------
    X : np.ndarray
        Array of shape (num_windows, window_size) containing the sliding windws of the input trace.
    T : list of list
        List of [start_time, end_time] pairs (as datetime objects) corresponding to each window.
    Notes
    -----
    - If the input trace is shorter than `window_size`, returns an empty array and empty list.
    - Time intervals are computed using the trace's start time and sampling rate.
    """
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
    """
    Generates sliding windows from a obspy stream of seismic traces.
    This function iterates over each trace in the input stream, applies the
    `sliding_windows_from_trace` function to extract windows of data, and aggregates
    the results into a single array.
    Parameters
    ----------
    st : obspy.Stream
        Obspy stream of traces containing the signals of interest.
    window_size : int, optional
        Number of samples in each window. Default is 10,000.
    stride : int, optional
        Number of samples to move the window at each step. Default is 5,000.
    Returns
    -------
    X : np.ndarray
        Array of shape (n_windows, window_size) containing the windowed data from all traces.
    T : np.ndarray
        Array of shape (n_windows, 2) containing the start- and stop time for each sliding window.
    """
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
    """
    Removes duplicate traces from an ObsPy Stream object.
    A trace is considered a duplicate if it has the same start time and identical data as another trace in the stream.
    Parameters
    ----------
    stream : obspy.Stream
        The ObsPy Stream object containing traces to remove duplicates from.
    Returns
    -------
    obspy.Stream
        A new ObsPy Stream object containing only unique traces.
    """
    unique_traces = []
    seen = set()

    for trace in stream:
        identifier = (trace.stats.starttime.timestamp, trace.data.tobytes())

        if identifier not in seen:
            seen.add(identifier)
            unique_traces.append(trace)

    return obspy.Stream(traces=unique_traces)


def remove_overlaps(stream):
    """
    Removes overlapping segments from traces contained in an ObsPy Stream.
    This function sorts the traces in the input stream by start time and iterates through each trace,
    removing any overlapping samples between consecutive traces. If a trace is completely
    overlapped by the previous trace, it is skipped. Otherwise, the overlapping portion is
    trimmed from the start of the trace, and its start time is updated accordingly.
    Parameters
    ----------
    stream : obspy.Stream
        A stream of seismic traces to process for overlaps.
    Returns
    -------
    obspy.Stream
        A new stream containing only non-overlapping traces.
    """
    stream.sort(keys=["starttime"])
    non_overlapping_traces = obspy.Stream()

    for i, trace in enumerate(stream):
        if i == 0:
            non_overlapping_traces.append(trace)
        else:
            prev_trace = non_overlapping_traces[-1]
            if trace.stats.starttime < prev_trace.stats.endtime:
                overlap_duration = prev_trace.stats.endtime - trace.stats.starttime
                overlap_samples = int(overlap_duration * trace.stats.sampling_rate)

                if overlap_samples >= trace.data.shape[0]:
                    continue

                trace.data = trace.data[overlap_samples:]
                trace.stats.starttime = prev_trace.stats.endtime

            if trace.data.shape[0] > 0:
                non_overlapping_traces.append(trace)

    return non_overlapping_traces


def preproc_flow_annotations(flows):
    """
    Preprocesses the 'start' and 'stop' timestamp annotations in a flows pandas DataFrame.
    This function normalizes the timestamp strings in the 'start' and 'stop' columns of the input DataFrame
    to a consistent format, handling both space-separated and 'T'-separated date-times, and removes
    trailing 'Z' if present. The normalized strings are then converted to `obspy.UTCDateTime` objects.
    Parameters
    ----------
    flows: pandas.DataFrame
        A pandas dataframe containing 'start' and 'stop' columns containing timestamp strings.
    Returns:
    ----------
    pandas.DataFrame: The input dataframe with 'start' and 'stop' timestamp strings converted to `obspy.UTCDateTime` objects.
    """
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
    """
    Converts a date string in 'YYYY-MM-DD' format to a string representing the year and the day of the year.
    Parameters
    ----------
    date_str: str
        Date string in 'YYYY-MM-DD' format.
    Returns:
    ----------
    str: String in the format 'YYYY.DDD', where 'YYYY' is the year and 'DDD' is the zero-padded day of the year.
    """
    date = datetime.strptime(str(date_str), "%Y-%m-%d")
    year = date.year
    day_of_year = date.timetuple().tm_yday
    return f"{year}.{day_of_year:03d}"


def find_date_in_strings(strings, date_str):
    """
    Filters a list of strings, returning those that contain a specific date in 'year-day' format,
    see date_to_year_day function.
    Parameters
    ----------
    strings: list
        List of strings to search.
    date_str: str
        Date string to match, which will be converted to 'year-day' format.
    Returns:
    ----------
    list: Strings from the input list that contain the target date in 'year-day' format.
    """
    target = date_to_year_day(date_str)
    return [s for s in strings if target in str(s)]


def read_segment(t0, t1, paths):
    """
    Extracts a segment from a seismic trace starting- and ending at times t0 and t1 respecively.
    This function first indentifies the path containing the obspy Streams containing the segment
    and extracts it after preprocessing is perfomred.
    Parameters
    ----------
    t0 : datetime or str
        Start time of the segment to read. Has to be convertible by obspy.UTCDateTime().
    t1 : datetime or str
        End time of the segment to read. Has to be convertible by obspy.UTCDateTime().
    paths : list of str
        List of file paths containing the seismic miniseed recordings.
    Returns
    -------
    obspy.Trace
        A single ObsPy Trace containg the desired preprocessed segment.
    """
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
    """
    Extracts and normalizes the sliding window over the segment with the largest IF anomaly score.
    Parameters:
    ----------
    t0 : obspy.UTCDateTime
        Start time of the segment to read.
    t1 : obspy.UTCDateTime
        End time of the segment to read.
    if_mod: sklearn.ensemble.IsolationForest
        Trained Isolation Forest model for anomaly scoring.
    paths : list of str
        List of file paths containing the seismic miniseed recordings.
    window_size : int, optional
        Number of samples in each window. Default is 10,000.
    stride : int, optional
        Number of samples to move the window at each step. Default is 5,000.
    Returns:
    ----------
    np.ndarray: The segment template.
    Notes:
    ----------
    This sliding window can be used as a template to compare two segments via dynamic
    time warping.
    """
    tr = read_segment(t0, t1, paths)
    X, _ = sliding_windows_from_trace(tr, window_size=window_size, stride=stride)

    if_scores = -if_mod.score_samples(X)
    tem = X[np.argmax(if_scores), :]

    return (tem - np.mean(tem)) / np.std(tem)


def get_git_root() -> Path:
    """
    Returns the root directory of the current Git repository.
    Uses the `git rev-parse --show-toplevel` command to determine the repository's root.
    Raises a RuntimeError if the current working directory is not inside a Git repository.
    Returns:
    ----------
    Path: The absolute path to the root of the Git repository.
    Raises:
    ----------
    RuntimeError: If the current directory is not inside a Git repository.
    """
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL
        )
        return Path(output.decode("utf-8").strip())
    except subprocess.CalledProcessError:
        raise RuntimeError("Not inside a Git repository")


def get_data_subpath(*subpath_parts: str) -> Path:
    """
    Constructs a path to a data subdirectory within the project's git root.
    Parameters:
    ----------
    *subpath_parts: str
            Variable length argument list representing subdirectories or file names
            to append to the 'data' directory within the git root.
    Returns:
    ----------
    Path: The full path to the specified data subdirectory or file.
    Raises:
    ----------
    Any exception raised by get_git_root() if the git root cannot be determined.
    """
    git_root = get_git_root()
    data_path = git_root / "data" / Path(*subpath_parts)
    return data_path


def find_paths(network, station, channel, start=2018, stop=2020):
    """
    Finds and returns a sorted list of file paths for seismic data streams for a given network, station, and channel over a specified year range.
    Parameters:
    ----------
    network: str
        The seismic network.
    station: str
        The station identifier.
    channel: str
        The channel.
    start: int, optional
        The starting year (inclusive). Defaults to 2018.
    stop: int, optional
        The ending year (inclusive). Defaults to 2020.
    Returns:
    ----------
        np.ndarray: A concatenated and sorted array of file paths for the specified parameters and year range.
    Notes:
    ----------
        - Files starting with "._" are ignored.
        - Sorting is performed based on the integer value of the file extension.
    """
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
    """
    Filters the DataFrame to debris flow segments for a specified station  over a specified year range.
    Splits the DataFrame into lower- and high confidence segments.
    Parameters:
    ----------
    df: pd.DataFrame
        Input DataFrame including columns for start, stop, station and confidence
    station: str
        Station for which to extract segments for.
    start: int, optional
        The starting year (inclusive). Defaults to 2018.
    stop: int, optional
        The ending year (inclusive). Defaults to 2020.
    Returns:
    ----------
    lower_conf_flows: pd.DataFrame
        DataFrame of flows with 'low' or 'med' confidence.
    high_conf_flows: pd.DataFrame
        DataFrame of flows with 'high' confidence.
    df: pd.DataFrame
        Filtered DataFrame containing all debris flow segments.
    """
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


def limit_segment_length(
    t0, t1, paths, if_mod, max_len=1800, window_size=10000, stride=5000, return_tr=False
):
    """
    Limits the length of a segment based on IF anomaly scores.
    This function reads a segment between times `t0` and `t1` and finds the sliding window
    with the higest anomaly score. This window is then iteratively expanding by adding the sliding window
    in the direction of the larger score, until a max length is reached.
    Parameters
    ----------
    t0 : obspy.UTCDateTime
        Start time of the segment to read.
    t1 : obspy.UTCDateTime
        End time of the segment to read.
    paths : list of str
        List of file paths containing the seismic miniseed recordings.
    if_mod : sklearn.ensemble.IsolationForest
        Trained Isolation Forest model used to score sliding windows.
    max_len : int, optional
        Maximum allowed length of the segment (default is 1800 seconds or 30 minutes).
    window_size : int, optional
        Number of samples in each window. Default is 10,000.
    stride : int, optional
        Number of samples to move the window at each step. Default is 5,000.
    Returns
    ----------
    X : np.ndarray
        Sliding windows from the limited segments.
    if_scores : np.ndarray
        Isolation Forest anomaly scores for each window in the segment.
    start : obspy.UTCDateTime
        Start time of the limited segment.
    stop : obspy.UTCDateTime
        End time of the limited segment.
    """
    tr = read_segment(t0, t1, paths)
    X, T = sliding_windows_from_trace(tr, window_size=window_size, stride=stride)
    if_scores = -if_mod.score_samples(X)

    idx_low = np.argmax(if_scores)
    idx_high = np.argmax(if_scores)

    while True:
        seg_length = T[idx_high][1] - T[idx_low][0]
        if seg_length >= max_len:
            break

        if idx_low == 0 and idx_high == len(if_scores) - 1:
            break

        if idx_low == 0:
            idx_high += 1
        elif idx_high == len(if_scores) - 1:
            idx_low -= 1
        elif if_scores[idx_high + 1] > if_scores[idx_low - 1]:
            idx_high += 1
        else:
            idx_low -= 1

    X = X[idx_low : idx_high + 1]
    if_scores = if_scores[idx_low : idx_high + 1]
    start = T[idx_low][0]
    stop = T[idx_high][1]

    # Todo: update docstring
    if return_tr:
        return tr.trim(start, stop)

    return X, if_scores, start, stop
