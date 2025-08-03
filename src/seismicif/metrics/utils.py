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


def compute_statistics(detections, gt):
    if len(detections) == 0:
        return {
            "iou": np.nan,
            "recall": 0,
            "FN": gt.shape[0],
            "precision": np.nan,
            "FP": 0,
            "TP": 0,
        }

    intersection, union, _ = iou(detections, gt)
    iou_val = 100 * (intersection / union)[-1]

    FP = np.sum(np.append(intersection[0], np.diff(intersection)) == 0)
    intersection, _, _ = iou(gt, detections)
    TP = np.sum(np.append(intersection[0], np.diff(intersection)) > 0)
    FN = np.sum(np.append(intersection[0], np.diff(intersection)) == 0)

    recall = 100 * TP / (TP + FN)
    precision = 100 * (TP / (TP + FP))

    return {
        "iou": iou_val,
        "recall": recall,
        "FN": FN,
        "precision": precision,
        "FP": FP,
        "TP": TP,
    }


def find_fp_fn(df, gt):
    if len(df) == 0:
        FP = pd.DataFrame([])
        FN = gt

    else:
        intersection, _, _ = iou(df, gt)
        FP = df.iloc[
            np.where(np.append(intersection[0], np.diff(intersection)) == 0)[0], :
        ].reset_index(drop=True)

        intersection, _, _ = iou(gt, df)
        FN = gt.iloc[
            np.where(np.append(intersection[0], np.diff(intersection)) == 0)[0], :
        ].reset_index(drop=True)

    return FP, FN


def find_unconfirmed_fp(detections, station_flows, confirmed_fp, station, source):
    fp = find_fp_fn(detections, station_flows)[0]
    fp = find_fp_fn(fp, confirmed_fp)[0]
    if len(fp) > 0:
        fp.drop(columns=["scores", "det_lens"], inplace=True)
        fp.insert(0, "station", np.repeat(station, len(fp)))
        fp.insert(len(fp.columns), "source", np.repeat(source, len(fp)))
    return fp


def extract_valid_segments(segments, lower_conf_flows, high_conf_flows):
    if len(segments) == 0:
        return segments

    intersections, _, _ = iou(segments, lower_conf_flows)
    non_lower_conf = np.append(intersections[0], np.diff(intersections)) == 0
    intersections, _, _ = iou(segments, high_conf_flows)
    high_conf = np.append(intersections[0], np.diff(intersections)) > 0

    return segments[non_lower_conf | high_conf].reset_index(drop=True)


def compute_station_metrics(
    sta_lta_detections,
    if_detections,
    dtw_detections,
    lower_conf_flows,
    high_conf_flows,
):
    """
    Compute metrics for the mining methods for a given station.
    """

    valid_detections = extract_valid_segments(
        sta_lta_detections, lower_conf_flows, high_conf_flows
    )
    sta_lta_metrics = compute_statistics(valid_detections, high_conf_flows)
    valid_detections = extract_valid_segments(
        if_detections, lower_conf_flows, high_conf_flows
    )
    if_metrics = compute_statistics(valid_detections, high_conf_flows)
    valid_detections = extract_valid_segments(
        dtw_detections, lower_conf_flows, high_conf_flows
    )
    dtw_metrics = compute_statistics(valid_detections, high_conf_flows)

    return {
        "sta_lta_iou": f"{np.round(sta_lta_metrics['iou'], 2)}",
        "if_iou": f"{np.round(if_metrics['iou'], 2)}",
        "dtw_iou": f"{np.round(dtw_metrics['iou'], 2)}",
        "sta_lta_recall": f"{np.round(sta_lta_metrics['recall'], 2)} ({sta_lta_metrics['FN']})",
        "if_recall": f"{np.round(if_metrics['recall'], 2)} ({if_metrics['FN']})",
        "dtw_recall": f"{np.round(dtw_metrics['recall'], 2)} ({dtw_metrics['FN']})",
        "sta_lta_precision": f"{np.round(sta_lta_metrics['precision'], 2)} ({sta_lta_metrics['FP']})",
        "if_precision": f"{np.round(if_metrics['precision'], 2)} ({if_metrics['FP']})",
        "dtw_precision": f"{np.round(dtw_metrics['precision'], 2)} ({dtw_metrics['FP']})",
    }


def compute_lower_conf_recall(
    sta_lta_detections, if_detections, dtw_detections, lower_conf_flows
):
    output = {}

    med_conf_flows = lower_conf_flows[
        lower_conf_flows["confidence"] == "med"
    ].reset_index(drop=True)
    low_conf_flows = lower_conf_flows[
        lower_conf_flows["confidence"] == "low"
    ].reset_index(drop=True)

    output["#_low_conf"] = len(low_conf_flows)
    output["#_med_conf"] = len(med_conf_flows)

    try:
        output["sta_lta_lc_recall"] = compute_statistics(
            sta_lta_detections, low_conf_flows
        )["recall"]
    except Exception:
        output["sta_lta_lc_recall"] = 0.0
    try:
        output["if_lc_recall"] = compute_statistics(if_detections, low_conf_flows)[
            "recall"
        ]
    except Exception:
        output["if_lc_recall"] = 0.0
    try:
        output["dtw_lc_recall"] = compute_statistics(dtw_detections, low_conf_flows)[
            "recall"
        ]
    except Exception:
        output["dtw_lc_recall"] = 0.0

    try:
        output["sta_lta_mc_recall"] = compute_statistics(
            sta_lta_detections, med_conf_flows
        )["recall"]
    except Exception:
        output["sta_lta_mc_recall"] = 0.0
    try:
        output["if_mc_recall"] = compute_statistics(if_detections, med_conf_flows)[
            "recall"
        ]
    except Exception:
        output["if_mc_recall"] = 0.0
    try:
        output["dtw_mc_recall"] = compute_statistics(dtw_detections, med_conf_flows)[
            "recall"
        ]
    except Exception:
        output["dtw_mc_recall"] = 0.0

    return output
