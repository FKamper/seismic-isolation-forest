import obspy
import numpy as np


def preproc_stream(st: obspy.Stream, taper=False, target_sr=100) -> None:
    """
    Find and remove the response from the input obspy Stream. The response removal happens inplace,
    so no value is returned. Pre-processing and pre-filtering is applied, as well as a tapering.

    :param st: :class:`~obspy.core.Stream` object.
    :param resp_path: Path to the response files
    :type resp_path: str

    :rtype: None
    :return: Nothing
    """
    # Loop over all traces and apply pre-processing
    for tr in st:
        tr.detrend("linear")
        tr.detrend("demean")
        if taper:
            tr.taper(0.1)

        if tr.stats.sampling_rate != target_sr:
            tr.stats.orig_sample_rate = tr.stats.sampling_rate
            if tr.stats.sampling_rate > target_sr:
                tr.resample(target_sr, window="hann", no_filter=False)
            else:
                tr.resample(target_sr, window="hann", no_filter=True)
        else:
            tr.stats.orig_sample_rate = tr.stats.sampling_rate

    st.filter("highpass", freq=0.3, zerophase=True)
    # If the stream is not empty, and if the data was tapered, remove the tapered time-windows from the trace.
    # This means the start-time is after, and the endtime before the initial start- and end-times respectively.
    if len(st) > 0:
        if taper:
            st.trim(
                starttime=tr.stats.starttime
                + 0.1 * (tr.stats.endtime - tr.stats.starttime),
                endtime=tr.stats.endtime
                - 0.1 * (tr.stats.endtime - tr.stats.starttime),
            )

    return st


def normalize_stream(st: obspy.Stream) -> None:
    arr = np.zeros(0)
    for tr in st:
        arr = np.append(arr, tr.data)

    m = np.median(arr)
    s = np.mean(np.abs(arr - m))

    for tr in st:
        tr.data = (tr.data - m) / s

    return st
