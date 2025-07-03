import numpy as np
import obspy
from datetime import timedelta


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
