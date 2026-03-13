from fastdtw import fastdtw
import numpy as np
from seismicif.datamod.loading_utils import extract_template
from datetime import timedelta


def compute_dtw_distance(x, template):
    """
    Computes the Dynamic Time Warping (DTW) distance between two sequences.
    Parameters:
    ----------
    x: array-like
        The first input sequence.
    template: array-like
        The second input sequence to compare against.
    Returns:
    ----------
    float: The DTW distance between the input sequences.
    Notes:
    ----------
        This function uses the `fastdtw` algorithm for efficient computation.
    """
    distance, _ = fastdtw(x, template)
    return distance


def template_dtw(t0, t1, paths, templates, if_mod, window_size=10000, stride=5000):
    """
    Computes Dynamic Time Warping (DTW) distances between a template extracted from a segment
    and a set of pre-extracted templates extracted from a set of (other) segments.
    Parameters:
    ----------
    t0: datetime
        Start time of the segment.
    t1: datetime
        End time of the segment.
    paths: list or dict
        Data source paths or identifiers required for template extraction.
    templates: np.ndarray
        Array of template sequences to compare against, shape (n_templates, sequence_length).
    if_mod: object
        Isolation Forest model or related object used in template extraction.
    window_size: int, optional
        Size of the window for segment extraction. Defaults to 10000.
    stride: int, optional
        Stride for segment extraction. Defaults to 5000.
    Returns:
    ----------
        np.ndarray: Array of DTW distances between the segment template and each of the pre-extracted templates.
    """
    segment_template = extract_template(
        t0, t1 + timedelta(seconds=50), if_mod, paths, window_size, stride
    )

    return np.array(
        [
            compute_dtw_distance(templates[i, :], segment_template)
            for i in range(templates.shape[0])
        ]
    )


def segment_dtw(seg1, seg2):
    """
    Computes the segment Dynamic Time Warping (DTW) distances between two segments.
    The function first aligns sliding windows from the two segments by performing DTW between their
    corresponding anomaly scores. Then DTW is performed between all pairs of matched sliding windows.
    All sliding windows are normalized before DTW computation.
    Parameters:
    ----------
    seg1: tuple
        A tuple (X1, scores_seg1, _, _), where X1 contains sliding windows extracted from the first segment,
        and scores_seg1 is a 1D numpy array containing the corresponding IF anomaly scores.
    seg2: tuple
        A tuple (X2, scores_seg2, _, _), where X2 contains sliding windows extracted from the second segment,
        and scores_seg2 is a 1D numpy array containing the corresponding IF anomaly scores.
    Returns:
    ----------
    tuple:
        dtw_dists (list of float): List of DTW distances between normalized feature vectors along the alignment path.
        path (list of tuple): List of index pairs representing the alignment path between the two score arrays.
    """
    X1, scores_seg1, _, _ = seg1
    X2, scores_seg2, _, _ = seg2

    _, path = fastdtw(
        scores_seg1, scores_seg2, radius=max(scores_seg1.shape[0], scores_seg2.shape[0])
    )
    dtw_dists = []

    for i in range(len(path)):
        x = X1[path[i][0], :]
        x = (x - np.mean(x)) / np.std(x)

        y = X2[path[i][1], :]
        y = (y - np.mean(y)) / np.std(y)

        dtw_dists.append(fastdtw(x, y, radius=1)[0])

    return dtw_dists, path


def distribute_pairwise_segment_dtw(args):
    i, j, segments = args
    seg1 = segments[i]
    seg2 = segments[j]
    return segment_dtw(seg1, seg2)


def distribute_reference_segment_dtw(args):
    target_segment, ref_segments = args

    dtw_dists = []

    for i in range(len(ref_segments)):
        dtw_dists.append(segment_dtw(target_segment, ref_segments[i]))

    return dtw_dists
