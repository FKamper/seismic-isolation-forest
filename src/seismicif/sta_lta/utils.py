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
    """
    Generates a new ObsPy Trace containing the classic STA-LTA characteristic function for a given input trace.
    Parameters:
    ----------
    tr: obspy.Trace
        Input seismic trace.
    sw: int
        Short window length for STA calculation.
    lw: int
        Long window length for LTA calculation.
    station: str, optional
        Station name to assign to the output trace. Defaults to "".
    Returns:
    ----------
    obspy.Trace: A new trace object containing the STA-LTA characteristic function, with updated metadata.
    """
    cft = classic_sta_lta(tr, sw, lw)
    new_trace = obspy.Trace(data=cft)
    new_trace.stats.starttime = tr.stats.starttime
    new_trace.stats.sampling_rate = tr.stats.sampling_rate
    new_trace.stats.channel = "STA-LTA"
    new_trace.stats.station = station

    return new_trace


def create_slta_stream(st, sw, lw):
    """
    Generates a new ObsPy Stream containing STA/LTA characteristic functions for each trace in the input stream.
    Parameters:
    ----------
    st: obspy.Stream
        Input ObsPy Stream containing seismic traces.
    sw: int
        Short window length for STA calculation.
    lw: int
        Long window length for LTA calculation.
    Returns:
    ----------
    obspy.Stream: Stream of traces where each trace contains the STA-LTA characteristic function of the corresponding input trace.
    Notes:
    ----------
        - Traces for which the STA-LTA calculation fails are skipped.
        - The output traces retain the original start time and sampling rate.
    """
    traces = []
    for tr in st:
        try:
            cft = classic_sta_lta(tr, sw, lw)
        except Exception:
            continue

        new_trace = obspy.Trace(data=cft)
        new_trace.stats.starttime = tr.stats.starttime
        new_trace.stats.sampling_rate = tr.stats.sampling_rate

        traces.append(new_trace)

    return obspy.Stream(traces=traces)


def detections_to_df(dtc):
    """
    Converts a list of detection dictionaries into a single pandas DataFrame.
    Each detection in the input list should be an iterable of dictionaries with keys "start", "stop", and "scores".
    The function extracts these values and constructs a DataFrame for each detection, then concatenates them into one DataFrame.
    Parameters
    ----------
    dtc : list
        A list of detections, where each detection is an iterable of dictionaries containing "start", "stop", and "scores" keys.
    Returns
    -------
    pandas.DataFrame or list
        A concatenated DataFrame of all detections if input is not empty, otherwise an empty list.
    """
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
    """
    Extract segments flagged by the STA-LTA algorithm from a stream of seismic traces.
    This function appends a previous trace (if provided and no breakages present) to the current stream,
    removing overlaps, and extracts segments by applying the onset and offset thresholds to the STA-LTA
    characteristic function.
    Parameters
    ----------
    tr_in: obspy.Trace or None
        Previous trace to be appended to the current stream, or None.
    st_in: obspy.Stream
        Stream of traces to process.
    sw: int
        Short window length for STA calculation.
    lw: int
        Long window length for LTA calculation.
    onset_thres : float
        Threshold value for event onset detection.
    offset_thres : float
        Threshold value for event offset detection.
    Returns
    ----------
    Tuple:
        - DataFrame containing flagged segments.
        - The last trace processed (for use in subsequent calls).
        - The last STA-LTA trace computed.
    """
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

        st.trim(
            t0 - timedelta(seconds=lw / res_tr.stats.sampling_rate),
            st[-1].stats.endtime,
        )

    for tr in st:
        if tr.data.shape[0] >= lw:
            slta_tr = create_slta_trace(tr, sw, lw)

            slta_tr.trim(t0, slta_tr.stats.endtime)
            new_dtc = trace_trigger_detections(
                slta_tr, onset_thres, offset_thres, lag=0
            )
            if len(new_dtc) > 0:
                dtc.append(new_dtc)

    res_tr = st[-1].copy()

    return detections_to_df(dtc), res_tr, slta_tr


def slta_compute_iou(flow_streams, gt, sw, lw, onset_thres, offset_thres):
    """
    Computes the Intersection over Union (IoU) between segments flagged by the STA-LTA trigger
    in seismic data streams and ground truth events.
    Parameters
    ----------
    flow_streams : list of dict
        List of dictionaries, each containing seismic data streams and associated metadata.
    gt : pandas.DataFrame
        Ground truth (debris flow) events as a DataFrame.
    sw : int
        Short window length for STA calculation.
    lw : int
        Long window length for LTA calculation.
    onset_thres : float
        Threshold value for event onset detection.
    offset_thres : float
        Threshold value for event offset detection.
    Returns
    ----------
    float
        The IoU score between flagged segments and ground truth events. Returns 0 if no detections are found.
    """
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
    """
    Extracts segments flagged by the STA-LTA algorithm from a list of seismic data file paths.
    For each file path, reads the seismic data, optionally applies preprocessing, removes duplicate traces and overlaps,
    and performs STA-LTA event detection. Flagged segments are aggregated, sorted by their detection scores in descending order,
    and returned as a pandas DataFrame.
    Parameters
    ----------
    paths: array-like
        List or array of file paths to seismic data files.
    sw : int
        Short window length for STA calculation.
    lw : int
        Long window length for LTA calculation.
    onset_thres : float
        Threshold value for event onset detection.
    offset_thres : float
        Threshold value for event offset detection.
    preproc : bool, optional
        If True, applies preprocessing to the seismic data. Defaults to True.
    Returns:
    ----------
        pandas.DataFrame: DataFrame containing flagged segments, sorted by detection scores in descending order.
    """
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
