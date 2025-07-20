from fastdtw import fastdtw
import numpy as np
from seismicif.datamod.loading_utils import extract_template, sliding_windows_from_trace
from datetime import timedelta


def compute_dtw_distance(x, template):
    distance, _ = fastdtw(x, template)
    return distance


def compute_wrapper(args):
    template_row, segment_template = args
    return compute_dtw_distance(template_row, segment_template)


def template_dtw(t0, t1, paths, templates, if_mod, window_size=10000, stride=5000):
    segment_template = extract_template(
        t0, t1 + timedelta(seconds=50), if_mod, paths, window_size, stride
    )

    return np.array(
        [
            compute_dtw_distance(templates[i, :], segment_template)
            for i in range(templates.shape[0])
        ]
    )


def segment_dtw(seg1, seg2, if_mod, window_size=10000, stride=5000):
    X1, _ = sliding_windows_from_trace(seg1, window_size, stride)
    scores_seg1 = -if_mod.score_samples(X1)

    X2, _ = sliding_windows_from_trace(seg2, window_size, stride)
    scores_seg2 = -if_mod.score_samples(X2)
    _, path = fastdtw(scores_seg1, scores_seg2)

    dtw_dists = []

    for i in range(len(path)):
        print(np.round(100 * (i / len(path))), "%", end=" \r")
        x = X1[path[i][0], :]
        x = (x - np.mean(x)) / np.std(x)

        y = X2[path[i][1], :]
        y = (y - np.mean(y)) / np.std(y)

        dtw_dists.append(fastdtw(x, y)[0])

    return dtw_dists
