import argparse
import os
import numpy as np
import pandas as pd
import obspy
import json

from joblib import dump, load
from seismicif.datamod.loading_utils import preproc_flow_annotations
from seismicif.isolation_forest import create_if_stream
from seismicif.datamod.trigger_utils import stream_trigger_detections
from seismicif.metrics import iou

def parse_args():
    parser = argparse.ArgumentParser(
        description="Deploy IF trigger."
    )
    parser.add_argument(
        "--network", type=str, required=True, help="Network where station is located."
    )
    parser.add_argument(
        "--station",
        type=str,
        required=True,
        help="Target station for deployment",
    )

    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Start year of deployment.",
    )

    parser.add_argument(
            "--stop",
            type=int,
            required=True,
            help="End year of deployment.",
        )

    parser.add_argument(
        "--onset",
        type=float,
        required=True,
        help="Onset threshold for IF trigger.",
    )

    parser.add_argument(
        "--offset",
        type=float,
        required=True,
        help="Offset threshold for IF trigger.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    start = obspy.UTCDateTime(f"{args.start}-01-01")
    end = obspy.UTCDateTime(f"{args.stop}-12-31")

    scores_df = pd.read_csv(f"../output/if_scores/{args.station}.csv",index_col=0)
    scores_df["start"] = [obspy.UTCDateTime(str(i)) for i in scores_df["start"]]
    scores_df["stop"] = [obspy.UTCDateTime(str(i)) for i in scores_df["stop"]]
    scores_df = scores_df[(scores_df['start'] > start) & (scores_df['stop'] < end)]

    if_st = create_if_stream(scores_df, station=args.station, sr=0.02, network=args.network)
    if_segments = stream_trigger_detections(if_st, args.onset, args.offset)
    if_segments.to_csv(f"../output/if_segments/{args.station}.csv")

if __name__ == "__main__":
    main()
