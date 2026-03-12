import pandas as pd
import numpy as np
from obspy.signal.trigger import trigger_onset
from datetime import timedelta


def trace_trigger_detections(tr, onset_thres, offset_thres, lag=100, max_len=1800):
    """
    Extracts segments flagged by a triggering algorithm over a seismic trace for a specified onset-
    and offset threshold.
    Parameters:
    ----------
    tr: obspy.Trace
        Seismic trace object containing the signal.
    onset_thres: float
        Threshold value for activating the trigger.
    offset_thres: float
        Threshold value for deactivating the trigger.
    lag: float, optional
        Time added to the end time of the flagged segment.
    Returns:
    ----------
    dtc: dict
        Consists of lists:
        - 'start' (UTCDateTime): Start times of the flagged segments.
        - 'stop' (UTCDateTime): Stop times of the flagged segments.
        - 'scores' (float): Maximum value of the trace data within the segment intervals.
    Notes:
    ----------
    The lag is used to accomodate the IF trigger where:
        - the IF anomaly score for a sliding window is given w.r.t the start of the time
            window.
        - the flagged segment is marked from the starting point of the onset window,
            until the starting point of the offset window.
    If the trigger flags no segments, return an empty list.
    """
    sr = tr.stats.sampling_rate
    dtc = []

    try:
        locs = trigger_onset(tr.data, onset_thres, offset_thres)
    except:
        return dtc

    sr = tr.stats.sampling_rate

    for i in locs:
        t0 = tr.stats.starttime + timedelta(seconds=i[0] / sr)
        t1 = tr.stats.starttime + timedelta(seconds=i[1] / sr) + timedelta(seconds=lag)

        scores = tr.data[i[0] : (i[1] + 1)]

        most_anomalous_idx = np.argmax(scores)
        most_anomalous_start = tr.stats.starttime + timedelta(
            seconds=(i[0] + most_anomalous_idx) / sr
        )
        segment_score = scores[most_anomalous_idx]

        idx_low = most_anomalous_idx
        idx_high = most_anomalous_idx

        while True:
            seg_length = (idx_high - idx_low) / sr
            if seg_length >= max_len:
                break

            if idx_low == 0 and idx_high == len(scores) - 1:
                break

            if idx_low == 0:
                idx_high += 1
            elif idx_high == len(scores) - 1:
                idx_low -= 1
            elif scores[idx_high + 1] > scores[idx_low - 1]:
                idx_high += 1
            else:
                idx_low -= 1

        segment_of_interest_start = tr.stats.starttime + timedelta(
            seconds=(i[0] + idx_low) / sr
        )
        segment_of_interest_stop = tr.stats.starttime + timedelta(
            seconds=(i[0] + idx_high) / sr
        )

        dtc.append(
            {
                "start": t0,
                "stop": t1,
                "most_anomalous_start": most_anomalous_start,
                "segment_of_interest_start": segment_of_interest_start,
                "segment_of_interest_stop": segment_of_interest_stop,
                "scores": segment_score,
            }
        )

    return dtc


#  tr = read_segment(t0, t1, paths)
#     X, T = sliding_windows_from_trace(tr, window_size=window_size, stride=stride)
#     if_scores = -if_mod.score_samples(X)

#     idx_low = np.argmax(if_scores)
#     idx_high = np.argmax(if_scores)

#     while True:
#         seg_length = T[idx_high][1] - T[idx_low][0]
#         if seg_length >= max_len:
#             break

#         if idx_low == 0 and idx_high == len(if_scores) - 1:
#             break

#         if idx_low == 0:
#             idx_high += 1
#         elif idx_high == len(if_scores) - 1:
#             idx_low -= 1
#         elif if_scores[idx_high + 1] > if_scores[idx_low - 1]:
#             idx_high += 1
#         else:
#             idx_low -= 1

#     X = X[idx_low : idx_high + 1]
#     if_scores = if_scores[idx_low : idx_high + 1]
#     start = T[idx_low][0]
#     stop = T[idx_high][1]


def stream_trigger_detections(st, onset_thres, offset_thres, lag=100):
    """
    Extracts segments flagged by a triggering algorithm over a stream of seismic traces for a specified onset-
    and offset threshold and returns a sorted DataFrame of the flagged segements.
    Parameters
    ----------
    st : obspy.Stream
        An iterable of seismic trace objects to process.
    onset_thres : float
        Threshold value for event onset detection.
    offset_thres : float
        Threshold value for event offset detection.
    lag : int, optional
        Time added to the end time of the flagged segment.
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the flagged segmebts for all traces in the stream,
        sorted by detection scores in descending order.
    """
    dtc = []

    for tr in st:
        dtc.append(
            pd.DataFrame(
                trace_trigger_detections(tr, onset_thres, offset_thres, lag=lag)
            )
        )

    dtc = pd.concat(dtc).reset_index(drop=True)
    ord = np.flip(np.argsort(np.array(dtc["scores"])))
    dtc = dtc.iloc[ord, :].reset_index(drop=True)

    return dtc
