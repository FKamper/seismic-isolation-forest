import argparse
import os
import numpy as np
import pandas as pd
import obspy
import json
import sys

from joblib import dump, load
from seismicif.datamod.loading_utils import preproc_flow_annotations
from seismicif.isolation_forest import create_if_stream
from seismicif.datamod.trigger_utils import stream_trigger_detections
from seismicif.metrics import iou, est_thresholds

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate thresholds and produce detections."
    )
    parser.add_argument(
        "--network", type=str, required=True, help="Network where station is located."
    )
    parser.add_argument(
        "--station",
        type=str,
        required=True,
        help="Target station.",
    )

    parser.add_argument(
        "--tr_start",
        type=int,
        required=True,
        help="Training start year.",
    )

    parser.add_argument(
        "--tr_stop",
        type=int,
        required=True,
        help="Training end year.",
        )

    parser.add_argument(
        "--te_start",
        type=int,
        required=True,
        help="Testing start year.",
    )

    parser.add_argument(
        "--te_stop",
        type=int,
        required=True,
        help="Testing end year.",
        )

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Scoring method used to rank segments.",
        )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.method == "IF":
        try:
            pd.read_csv(f"../output/if_detections/tr_{args.station}.csv")
            pd.read_csv(f"../output/if_detections/te_{args.station}.csv")
            print("Detections already generated")
            sys.exit()

        except (FileNotFoundError, OSError):
            print("Generating Detections")

        segments = pd.read_csv(f"../output/if_segments/{args.station}.csv",index_col=0)
        mode = "lower"

    tr_start = obspy.UTCDateTime(f"{args.tr_start}-01-01")
    tr_end = obspy.UTCDateTime(f"{args.tr_stop}-12-31")

    te_start = obspy.UTCDateTime(f"{args.te_start}-01-01")
    te_end = obspy.UTCDateTime(f"{args.te_stop}-12-31")

    tr_segments = segments[(segments["start"] > tr_start) & (segments["stop"] < tr_end)].reset_index(drop=True)
    te_segments = segments[(segments["start"] > te_start) & (segments["stop"] < te_end)].reset_index(drop=True)

    flows = preproc_flow_annotations(pd.read_csv("../catalogs/calibration_catalog.csv"))
    flows = flows[flows["station"] == args.station].reset_index(drop=True)
    tr_flows = flows[(flows["start"] > tr_start) & (flows["stop"] < tr_end)].reset_index(drop=True)
    te_flows = flows[(flows["start"] > te_start) & (flows["stop"] < te_end)].reset_index(drop=True)
    tr_high_conf_flows = tr_flows[tr_flows["confidence"] == "1"].reset_index(drop=True)
    tr_lower_conf_flows = tr_flows[tr_flows["confidence"] != "1"].reset_index(drop=True)

    intersections, _, _ = iou(tr_segments, tr_lower_conf_flows)
    non_lower_conf = np.append(intersections[0], np.diff(intersections)) == 0
    intersections, _, _ = iou(tr_segments, tr_high_conf_flows)
    high_conf = np.append(intersections[0], np.diff(intersections)) > 0
    valid_segments = tr_segments[non_lower_conf | high_conf].reset_index(drop=True)

    _, min_len, score_thres = est_thresholds(valid_segments, tr_high_conf_flows, mode=mode)

    if args.method == "IF":
        filename = "../output/if_detections/threshold_params.json"

    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append({"station":args.station,"min_len":min_len,"score_thres":score_thres})

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    tr_detections = tr_segments.copy()
    start_times = np.array([pd.to_datetime(str(i)) for i in tr_detections["start"]])
    end_times = np.array([pd.to_datetime(str(i)) for i in tr_detections["stop"]])
    det_lens = np.array([i.total_seconds() for i in end_times - start_times])
    tr_detections = tr_detections.assign(det_lens=det_lens)
    tr_detections = tr_detections[
        (tr_detections["scores"] > score_thres) & (tr_detections["det_lens"] > min_len)
    ].reset_index(drop=True)

    te_detections = te_segments.copy()
    start_times = np.array([pd.to_datetime(str(i)) for i in te_detections["start"]])
    end_times = np.array([pd.to_datetime(str(i)) for i in te_detections["stop"]])
    det_lens = np.array([i.total_seconds() for i in end_times - start_times])
    te_detections = te_detections.assign(det_lens=det_lens)
    te_detections = te_detections[
        (te_detections["scores"] > score_thres) & (te_detections["det_lens"] > min_len)
    ].reset_index(drop=True)


    if args.method == "IF":
        tr_detections.to_csv(f"../output/if_detections/tr_{args.station}.csv")
        te_detections.to_csv(f"../output/if_detections/te_{args.station}.csv")



if __name__ == "__main__":
    main()
