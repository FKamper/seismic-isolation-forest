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
from seismicif.metrics import iou, compute_statistics

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

    return parser.parse_args()

def main():
    args = parse_args()

    tr_start = obspy.UTCDateTime(f"{args.tr_start}-01-01")
    tr_end = obspy.UTCDateTime(f"{args.tr_stop}-12-31")

    te_start = obspy.UTCDateTime(f"{args.te_start}-01-01")
    te_end = obspy.UTCDateTime(f"{args.te_stop}-12-31")

    tr_detections = pd.read_csv(f"../output/if_detections/tr_{args.station}.csv",index_col=0)
    te_detections = pd.read_csv(f"../output/if_detections/te_{args.station}.csv",index_col=0)

    flows = preproc_flow_annotations(pd.read_csv("../catalogs/flow_catalog.csv"))
    flows = flows[flows["station"] == args.station].reset_index(drop=True)
    tr_flows = flows[(flows["start"] > tr_start) & (flows["stop"] < tr_end)].reset_index(drop=True)
    te_flows = flows[(flows["start"] > te_start) & (flows["stop"] < te_end)].reset_index(drop=True)
    tr_high_conf_flows = tr_flows[tr_flows["confidence"] == 1].reset_index(drop=True)
    tr_lower_conf_flows = tr_flows[tr_flows["confidence"] != 1].reset_index(drop=True)
    te_high_conf_flows = te_flows[te_flows["confidence"] == 1].reset_index(drop=True)
    te_lower_conf_flows = te_flows[te_flows["confidence"] != 1].reset_index(drop=True)

    intersections, _, _ = iou(tr_detections, tr_lower_conf_flows)
    non_lower_conf = np.append(intersections[0], np.diff(intersections)) == 0
    intersections, _, _ = iou(tr_detections, tr_high_conf_flows)
    high_conf = np.append(intersections[0], np.diff(intersections)) > 0
    valid_detections = tr_detections[non_lower_conf | high_conf].reset_index(drop=True)
    print("Training:",compute_statistics(valid_detections, tr_high_conf_flows))

    intersections, _, _ = iou(te_detections, te_lower_conf_flows)
    non_lower_conf = np.append(intersections[0], np.diff(intersections)) == 0
    intersections, _, _ = iou(te_detections, te_high_conf_flows)
    high_conf = np.append(intersections[0], np.diff(intersections)) > 0
    valid_detections = te_detections[non_lower_conf | high_conf].reset_index(drop=True)
    print("Testing:",compute_statistics(valid_detections, te_high_conf_flows))

if __name__ == "__main__":
    main()
