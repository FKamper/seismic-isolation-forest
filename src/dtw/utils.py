from fastdtw import fastdtw
import numpy as np
from datamod.loading_utils import extract_template
from datetime import timedelta


def compute_dtw_distance(x, template):
    distance, _ = fastdtw(x, template)
    return distance


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
