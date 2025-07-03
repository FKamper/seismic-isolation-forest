import obspy
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest as IF
from datamod.preproc_utils import preproc_stream, normalize_stream
from datamod.loading_utils import (
    sliding_windows_from_stream,
    remove_duplicate_traces,
    remove_overlaps,
)
from datetime import timedelta


def train_if(stream_paths, norm=False, preproc=True, if_mod=None, max_val=np.inf):
    if if_mod is None:
        if_mod = IF(
            n_estimators=0, max_features=1.0, warm_start=True, n_jobs=16, random_state=0
        )

    for i in range(stream_paths.shape[0]):
        st = obspy.read(stream_paths[i])
        if preproc:
            preproc_stream(st)
        if norm:
            normalize_stream(st)

        X, T = sliding_windows_from_stream(st)
        if X.shape[0] == 0:
            continue

        if np.max(X) > max_val:
            print(stream_paths[i])
            continue

        if_mod.n_estimators = if_mod.n_estimators + 1
        if_mod.fit(X)
        print(np.round(100 * (i / stream_paths.shape[0]), 2), "%", end=" \r")

    return if_mod


def compute_scores(
    stream_paths, if_mod, norm=False, preproc=True, max_val=np.inf, stride_frac=0.5
):
    scores = []
    times = []
    stdevs = []
    res_tr = None

    for i in range(stream_paths.shape[0]):
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

        X, T = sliding_windows_from_stream(st)
        if X.shape[0] == 0:
            continue

        if np.max(X) > max_val:
            print(stream_paths[i])
            continue

        scores.append(-if_mod.score_samples(X))
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
