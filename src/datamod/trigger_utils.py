import pandas as pd
import numpy as np
from obspy.signal.trigger import trigger_onset
from datetime import timedelta


def trace_trigger_detections(tr, onset_thres, offset_thres, lag=100):
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

        sc = np.max(tr.data[i[0] : (i[1] + 1)])
        dtc.append({"start": t0, "stop": t1, "scores": sc})

    return dtc


def stream_trigger_detections(st, onset_thres, offset_thres, lag=100):
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
