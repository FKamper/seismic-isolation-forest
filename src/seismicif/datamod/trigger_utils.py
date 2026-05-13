import pandas as pd
import numpy as np
from obspy.signal.trigger import trigger_onset
from datetime import timedelta


def trace_trigger_detections(tr, onset_thres, offset_thres, lag=100):
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

        dtc.append(
            {
                "start": t0,
                "stop": t1,
                "most_anomalous_start": most_anomalous_start,
                "scores": segment_score,
            }
        )

    return dtc


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
