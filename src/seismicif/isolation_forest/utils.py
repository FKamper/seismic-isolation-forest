import obspy
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import IsolationForest as IF
from seismicif.datamod.preproc_utils import preproc_stream, normalize_stream
from seismicif.datamod.loading_utils import (
    sliding_windows_from_stream,
    remove_duplicate_traces,
    remove_overlaps,
)
from datetime import timedelta


def train_if(
    stream_paths,
    norm=False,
    preproc=True,
    if_mod=None,
    max_val=np.inf,
    max_samples=256,
    n_jobs=-1,
    if_random_state=0,
    np_seed=42,
):
    """
    Trains an Isolation Forest model to a collection of seismic data streams. An IF tree is trained
    to sliding windows extracted from each miniseed recording, after preprocessing.
    Parameters
    ----------
    stream_paths : array-like
        Array of file paths to seismic data streams.
    norm : bool, optional
        If True, normalize each stream before training. Default is False.
    preproc : bool, optional
        If True, apply preprocessing to each stream before training. Default is True.
    if_mod : sklearn.ensemble.IsolationForest or None, optional
        An existing Isolation Forest model to continue training. If None, a new model is created. Default is None.
    max_val : float, optional
        Maximum allowed value in a miniseed recording. Streams with values exceeding this are skipped. Default is np.inf.
    max_samples : int, optional
        Maximum number of samples to used to train each tree. If a stream has fewer samples than `max_samples`, samples are drawn with replacement. Default is 256.
    n_jobs : int, optional
        Number of jobs to run in parallel for training the Isolation Forest. Default is -1 (use all available cores).
    if_random_state : int, optional
        Random state for the Isolation Forest. Default is 0.
    np_seed : int, optional
        Seed for NumPy random number generator. Default is 42.
    Returns
    -------
    if_mod : sklearn.ensemble.IsolationForest
        The trained Isolation Forest model.
    """
    if if_mod is None:
        if_mod = IF(
            n_estimators=0,
            max_features=1.0,
            warm_start=True,
            n_jobs=n_jobs,
            random_state=if_random_state,
            max_samples=max_samples,
        )

    else:
        max_samples = if_mod.max_samples_

    for i in tqdm(range(stream_paths.shape[0])):
        st = obspy.read(stream_paths[i])
        if preproc:
            preproc_stream(st)
        if norm:
            normalize_stream(st)

        X, _ = sliding_windows_from_stream(st)

        if X.shape[0] == 0:
            continue

        if X.shape[0] < max_samples:
            rng = np.random.default_rng(np_seed)
            idx = rng.choice(X.shape[0], max_samples, replace=True)
            X = X[idx]

        if np.max(X) > max_val:
            print(stream_paths[i])
            continue

        if_mod.n_estimators = if_mod.n_estimators + 1
        if_mod.fit(X)

    return if_mod


# def compute_c(n):
#     """
#     Compute the average path length of an unsuccessful search in a binary search tree.
#     n can be an integer or a NumPy array.
#     """
#     gamma = 0.5772156649
#     n = np.array(n)
#     c = np.zeros_like(n, dtype=float)
#     mask = n > 1
#     c[mask] = 2 * (np.log(n[mask] - 1) + gamma) - 2 * (n[mask] - 1) / n[mask]
#     return c


# def reweighted_score_samples(model, X):
#     """
#     Compute Isolation Forest anomaly scores manually, reproducing model.score_samples.

#     Parameters:
#         model: trained sklearn IsolationForest
#         X: array-like of shape (n_samples, n_features)

#     Returns:
#         scores: array of anomaly scores for each sample
#     """
#     scores = []

#     for tree in model.estimators_:
#         node_counts = tree.decision_path(X).toarray().sum(axis=1)
#         leaf_nodes = tree.apply(X)
#         n_leaf = tree.tree_.n_node_samples[leaf_nodes]
#         edges = node_counts - 1

#         h = edges + compute_c(n_leaf)
#         scores.append(h / compute_c(tree.tree_.n_node_samples[0]))

#     scores = np.array(scores).T
#     return 2 ** (-scores.mean(axis=1))


def compute_scores(
    stream_paths,
    if_mod,
    norm=False,
    preproc=True,
    max_val=np.inf,
    window_size=10000,
    stride_frac=0.5,
):
    """
    Computes anomaly scores for seismic data streams using an Isolation Forest model and
    returns a sorted DataFrame by time.
    Parameters
    ----------
    stream_paths : array-like
        Array of file paths to seismic data streams.
    norm : bool, optional
        If True, normalize each stream before training. Default is False.
    preproc : bool, optional
        If True, apply preprocessing to each stream before training. Default is True.
    if_mod : sklearn.ensemble.IsolationForest or None, optional
        An existing Isolation Forest model to continue training. If None, a new model is created. Default is None.
    max_val : float, optional
        Maximum allowed value in a miniseed recording. Streams with values exceeding this are skipped. Default is np.inf.
    window_size : int, optional
        Number of samples in each window. Default is 10,000.
    stride_frac : float, optional
        Number of samples to move the window at each step expressed as a fraction of the window size. Default = 0.5.
    Returns
    -------
    scores_df : pandas.DataFrame
        DataFrame containing the following columns:
            - 'start': Start time of each window.
            - 'stop': Stop time of each window.
            - 'anomaly_score': Anomaly score for each window (negative Isolation Forest score).
            - 'std': Standard deviation of each window.
    Notes
    -----
    - Streams are optionally preprocessed and normalized.
    - Duplicate and overlapping traces are removed.
    - Overlaps between subsequent miniseed recordings are removed.
    - Sliding windows are extracted from each stream for scoring.
    - Streams with no valid windows or exceeding `max_val` are skipped.
    - The function concatenates results from all streams and sorts them by start time.
    """
    scores = []
    times = []
    stdevs = []
    res_tr = None

    for i in tqdm(range(stream_paths.shape[0])):
        st = obspy.read(stream_paths[i])
        if preproc:
            preproc_stream(st)
        if norm:
            normalize_stream(st)
        st = remove_duplicate_traces(st)
        st = remove_overlaps(st)

        if (
            res_tr is not None
            and res_tr.stats.endtime + timedelta(seconds=1 / res_tr.stats.sampling_rate)
            >= st[0].stats.starttime
        ):
            st.trim(
                res_tr.stats.endtime
                + timedelta(seconds=1 / res_tr.stats.sampling_rate),
                st[-1].stats.endtime,
            )
            st[0].data = np.append(res_tr.data, st[0].data)
            st[0].stats.starttime = res_tr.stats.starttime

        X, T = sliding_windows_from_stream(
            st, window_size=window_size, stride=int(stride_frac * window_size)
        )
        if X.shape[0] == 0:
            continue

        if np.max(X) > max_val:
            print(stream_paths[i])
            continue

        scores.append(-if_mod.score_samples(X))
        # scores.append(reweighted_score_samples(if_mod, X))

        stdevs.append(np.std(X, axis=1))
        times.append(T)

        res_tr = st[-1].copy()
        res_tr.trim(
            T[T.shape[0] - 1, 1]
            - timedelta(seconds=stride_frac * res_tr.stats.sampling_rate),
            st[-1].stats.endtime,
        )

    if len(times) == 0:
        return pd.DataFrame({"start": [], "stop": [], "anomaly_score": [], "std": []})

    times = np.concatenate(times)
    sc = np.concatenate(scores)
    stdevs = np.concatenate(stdevs)
    scores_df = pd.DataFrame(
        {"start": times[:, 0], "stop": times[:, 1], "anomaly_score": sc, "std": stdevs}
    )
    scores_df = scores_df.iloc[np.argsort(scores_df["start"]), :].reset_index(drop=True)

    return scores_df


def create_if_stream(scores_df, station="?", sr=0.02, network="?"):
    """
    Creates an ObsPy Stream object from a DataFrame of scores, grouping consecutive scores into traces.
    Parameters
    ----------
    scores_df : pandas.DataFrame
        The first column should be timestamps, and the third column should be scores.
    station : str, optional
        Station code to assign to each trace (default is "?").
    sr : float, optional
        Sampling rate in Hz for the traces (default is 0.02).
    network : str, optional
        Network code to assign to each trace (default is "?").
    Returns
    ----------
    obspy.Stream
        An ObsPy Stream object containing traces of scores grouped by consecutive timestamps.
    Notes
    ----------
    To do: Refer to columns by name rather than index.
    """
    if_stream = obspy.Stream()
    j = 0

    while j < scores_df.shape[0] - 1:
        t0 = scores_df.iloc[j, 0]
        sc = [scores_df.iloc[j, 2]]

        for i in range(j + 1, scores_df.shape[0]):
            if scores_df.iloc[i, 0] - scores_df.iloc[i - 1, 0] > 1 / sr:
                break
            sc.append(scores_df.iloc[i, 2])

        j = i

        tr = obspy.Trace(data=np.array(sc))
        tr.stats.sampling_rate = sr
        tr.stats.starttime = t0
        tr.stats.channel = "IF"
        tr.stats.station = station
        tr.stats.network = network
        if_stream += tr

    return if_stream
