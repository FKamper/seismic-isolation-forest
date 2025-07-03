import numpy as np
from datetime import timedelta


def sliding_windows_from_trace(inp, window_size=10000, stride=5000):
    start_time = inp.stats.starttime
    sr = inp.stats.sampling_rate

    inp = np.array(inp)

    if inp.shape[0] < window_size:
        return np.zeros([0, window_size]), []

    up = int((inp.shape[0] - window_size) / stride) + 1
    X = np.zeros([up, window_size])
    T = []

    for i in range(up):
        t0 = stride * i
        t1 = t0 + window_size
        X[i, :] = inp[t0:t1]
        T.append(
            [
                start_time + timedelta(seconds=t0 / sr),
                start_time + timedelta(seconds=t1 / sr),
            ]
        )

    return X, T


def sliding_windows_from_stream(st, window_size=10000, stride=5000):
    X = np.zeros([0, window_size])
    T = []

    for tr in st:
        Xn, Tn = sliding_windows_from_trace(tr, window_size=window_size, stride=stride)
        if Xn.shape[0] == 0:
            continue
        T.append(Tn)
        X = np.vstack([X, Xn])

    if X.shape[0] == 0:
        return X, []

    T = np.vstack(T)

    return X, T
