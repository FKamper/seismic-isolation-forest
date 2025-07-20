import obspy
import pandas as pd
import numpy as np
from obspy.signal.trigger import classic_sta_lta
from seismicif.metrics.utils import iou
from datetime import timedelta
from seismicif.datamod.loading_utils import remove_duplicate_traces, remove_overlaps
from seismicif.datamod.preproc_utils import preproc_stream
from seismicif.datamod.trigger_utils import trace_trigger_detections
from tqdm import tqdm


def create_slta_trace(tr, sw, lw, station=""):
    cft = classic_sta_lta(tr, sw, lw)
    new_trace = obspy.Trace(data=cft)
    new_trace.stats.starttime = tr.stats.starttime
    new_trace.stats.sampling_rate = tr.stats.sampling_rate
    new_trace.stats.channel = "STA-LTA"
    new_trace.stats.station = station

    return new_trace


def create_slta_stream(st, sw, lw):
    traces = []
    for tr in st:
        try:
            cft = classic_sta_lta(tr, sw, lw)
        except:
            continue

        new_trace = obspy.Trace(data=cft)
        new_trace.stats.starttime = tr.stats.starttime
        new_trace.stats.sampling_rate = tr.stats.sampling_rate

        traces.append(new_trace)

    return obspy.Stream(traces=traces)


def detections_to_df(dtc):
    df = []

    if len(dtc) == 0:
        return df

    else:
        for i in dtc:
            start = [j["start"] for j in i]
            stop = [j["stop"] for j in i]
            scores = [j["scores"] for j in i]
            df.append(pd.DataFrame({"start": start, "stop": stop, "scores": scores}))

        df = pd.concat(df).reset_index(drop=True)

        return df


def slta_stream_detections(tr_in, st_in, sw, lw, onset_thres, offset_thres):
    # make copies for safety
    if tr_in is not None:
        res_tr = tr_in.copy()
    else:
        res_tr = None

    st = st_in.copy()

    dtc = []
    slta_tr = None
    t0 = st[0].stats.starttime

    if (
        res_tr is not None
        and res_tr.stats.endtime + timedelta(seconds=1 / res_tr.stats.sampling_rate)
        >= st[0].stats.starttime
    ):
        t0 = res_tr.stats.endtime + timedelta(seconds=1 / res_tr.stats.sampling_rate)
        st.trim(t0, st[-1].stats.endtime)
        st[0].data = np.append(res_tr.data, st[0].data)
        st[0].stats.starttime = res_tr.stats.starttime
        # trim to relevant data
        st.trim(
            t0 - timedelta(seconds=lw / res_tr.stats.sampling_rate),
            st[-1].stats.endtime,
        )

    for tr in st:
        if tr.data.shape[0] >= lw:
            slta_tr = create_slta_trace(tr, sw, lw)
            # trim to start time of stream, does nothing if slta_tr.stats.starttime > t0
            slta_tr.trim(t0, slta_tr.stats.endtime)
            new_dtc = trace_trigger_detections(
                slta_tr, onset_thres, offset_thres, lag=0
            )
            if len(new_dtc) > 0:
                dtc.append(new_dtc)

    res_tr = st[-1].copy()

    return detections_to_df(dtc), res_tr, slta_tr


def slta_compute_iou(flow_streams, gt, sw, lw, onset_thres, offset_thres):
    df = []

    for flow in flow_streams:
        new_df, _, _ = slta_stream_detections(
            flow["res_tr"], flow["st"], sw, lw, onset_thres, offset_thres
        )

        if len(new_df) > 0:
            df.append(new_df)

    if len(df) == 0:
        return 0

    else:
        df = pd.concat(df).reset_index(drop=True)
        intersection, union, _ = iou(df, gt)
        return intersection[-1] / union[-1]


def slta_detections_from_paths(paths, sw, lw, onset_thres, offset_thres, preproc=True):
    res_tr = None
    dtc = []

    for p in tqdm(paths.tolist()):
        st = obspy.read(p)
        if preproc:
            preproc_stream(st)
        st = remove_duplicate_traces(st)
        st = remove_overlaps(st)

        new_dtc, res_tr, _ = slta_stream_detections(
            res_tr, st, sw, lw, onset_thres, offset_thres
        )

        if len(new_dtc) > 0:
            dtc.append(new_dtc)

    dtc = pd.concat(dtc).reset_index(drop=True)
    reorder = np.flip(np.argsort(dtc["scores"]))

    return dtc.iloc[reorder, :].reset_index(drop=True)
