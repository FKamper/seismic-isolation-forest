import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def get_merge_height(Z, obj_index):
    """
    Returns the first height at which the object with index `obj_index`
    is merged into any cluster.
    """
    for row in Z:
        if obj_index in row[:2]:
            return row[2]
    return None


def remove_singleton_merges(D):
    nseg = D.shape[0]
    Dsqf = squareform(D)
    Z = linkage(Dsqf, method="complete")

    max_merge_height = np.max(Z[:, 2])
    merge_heights = np.array([get_merge_height(Z, i) for i in range(nseg)])

    idx = np.arange(nseg)
    while np.max(merge_heights) == max_merge_height:
        idx = np.array(idx[idx != idx[np.argmax(merge_heights)]])
        Dr = D[np.ix_(idx, idx)]
        Dsqf = squareform(Dr)
        Z = linkage(Dsqf, method="complete")
        merge_heights = np.array([get_merge_height(Z, i) for i in range(len(idx))])
        max_merge_height = np.max(Z[:, 2])

    return idx
