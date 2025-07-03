import obspy
import numpy as np
from sklearn.ensemble import IsolationForest as IF
from datamod.preproc_utils import preproc_stream, normalize_stream
from datamod.loading_utils import sliding_windows_from_stream


def train_if(stream_paths, norm=True, preproc=True, if_mod=None, max_val=np.inf):
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
