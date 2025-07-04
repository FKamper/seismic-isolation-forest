import numpy as np
import pandas as pd


def iou(df, flows):
    intersection = np.zeros(df.shape[0])
    union = np.zeros(df.shape[0])

    for i in range(df.shape[0]):
        t0 = pd.Timestamp(str(df["start"][i]))
        t1 = pd.Timestamp(str(df["stop"][i]))

        union[i] = (t1 - t0).total_seconds()

        for j in range(flows.shape[0]):
            flow_t0 = pd.Timestamp(str(flows["start"][j]))
            flow_t1 = pd.Timestamp(str(flows["stop"][j]))

            intersection[i] = intersection[i] + max(
                0, (min(t1, flow_t1) - max(t0, flow_t0)).total_seconds()
            )

    intersection = np.cumsum(intersection)

    flow_len = 0

    for j in range(flows.shape[0]):
        flow_t0 = pd.Timestamp(str(flows["start"][j]))
        flow_t1 = pd.Timestamp(str(flows["stop"][j]))
        flow_len = flow_len + (flow_t1 - flow_t0).total_seconds()

    union = np.cumsum(union) + flow_len - intersection

    return intersection, union, flow_len
