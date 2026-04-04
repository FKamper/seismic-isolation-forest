import obspy
import numpy as np


def preproc_stream(
    st: obspy.Stream, taper=False, target_sr=100, filter_freq=0.3
) -> None:
    """
    Preprocesses an ObsPy Stream object by detrending, tapering, resampling, and filtering.
    Parameters
    ----------
    st: obspy.Stream
        The seismic data stream to preprocess.
    taper: bool, optional
        If True, applies a taper to each trace and trims the stream edges. Default is False.
    target_sr: int, optional
        The target sampling rate (Hz) for resampling. Default is 100.
    filter_freq: float, optional
        The frequency (Hz) for the highpass filter. Default is 0.3.
    Returns:
    ----------
    obspy.Stream: The preprocessed seismic data stream.
    Notes:
    ----------
    - Each trace is detrended (linear and demean).
    - Tapering is applied to each trace if `taper` is True.
    - Traces are resampled to `target_sr` if their sampling rate differs.
    - A highpass filter (`filter_freq` Hz) is applied to the entire stream.
    - If `taper` is True and the stream is not empty, the stream is trimmed at both ends by 10% of its duration.
    """
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

    st.filter("highpass", freq=filter_freq, zerophase=True)

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
    """
    Normalizes the data in each trace of an ObsPy Stream object.
    The normalization is performed by subtracting the median of all trace data
    from each data point and dividing by the mean absolute deviation from the median.
    This operation is applied in-place to each trace in the stream.
    Parameters
    ----------
    st : obspy.Stream
        The ObsPy Stream object containing one or more traces to be normalized.
    Returns
    ----------
    obspy.Stream
        The normalized ObsPy Stream object.
    """
    arr = np.zeros(0)
    for tr in st:
        arr = np.append(arr, tr.data)

    m = np.median(arr)
    s = np.mean(np.abs(arr - m))

    for tr in st:
        tr.data = (tr.data - m) / s

    return st
