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


def est_thresholds(df, gt, mode="lower"):
    df = df.reset_index(drop=True)

    start_times = np.array([pd.to_datetime(str(i)) for i in df["start"]])
    end_times = np.array([pd.to_datetime(str(i)) for i in df["stop"]])

    det_lens = np.array([i.total_seconds() for i in end_times - start_times])
    df = df.assign(det_lens=det_lens)

    unique_lens = np.sort(np.append(0, np.unique(det_lens)))

    iou_scores = []
    score_thres = []

    for i in range(1, unique_lens.shape[0]):
        df = df[df["det_lens"] > unique_lens[i - 1]].reset_index(drop=True)
        intersection, union, _ = iou(df, gt)
        iou_vals = intersection / union
        k = np.argmax(iou_vals)

        if k + 1 >= df.shape[0]:
            iou_scores.append(0)
            if mode == "lower":
                score_thres.append(0)
            else:
                score_thres.append(np.inf)

        else:
            iou_scores.append(iou_vals[k])
            score_thres.append(df.iloc[k + 1, 2])

    iou_vals = np.array(iou_scores)
    score_thres = np.array(score_thres)
    k = np.argmax(iou_scores)

    return iou_scores[k], unique_lens[k], score_thres[k]
