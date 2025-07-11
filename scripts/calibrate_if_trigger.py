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
        description="Calibrate IF trigger to Initial catalog."
    )
    parser.add_argument(
        "--network", type=str, required=True, help="Network where station is located."
    )
    parser.add_argument(
        "--station",
        type=str,
        required=True,
        help="Target station for calibration.",
    )

    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Start year of calibration period.",
    )

    parser.add_argument(
            "--stop",
            type=int,
            required=True,
            help="End of calibration period.",
        )

    return parser.parse_args()


def main():
    args = parse_args()
    start = obspy.UTCDateTime(f"{args.start}-01-01")
    end = obspy.UTCDateTime(f"{args.stop}-12-31")

    flows = preproc_flow_annotations(pd.read_csv("../catalogs/initial_catalog.csv",index_col=0))
    flows = flows[(flows["confidence"] != "earthquake")]
    flows = flows[(flows["confidence"] != "rockfall")]
    flows = flows[(flows["confidence"] != "quarrywork")]
    flows = flows[flows["station"] == args.station].reset_index(drop=True)
    flows = flows[(flows["start"] > start) & (flows["stop"] < end)].reset_index(
        drop=True
    )

    high_conf_flows = flows[flows["confidence"] == "1"].reset_index(drop=True)

    scores_df = pd.read_csv(f"../output/if_scores/{args.station}.csv",index_col=0)
    scores_df["start"] = [obspy.UTCDateTime(str(i)) for i in scores_df["start"]]
    scores_df["stop"] = [obspy.UTCDateTime(str(i)) for i in scores_df["stop"]]
    scores_df = scores_df[(scores_df['start'] > start) & (scores_df['stop'] < end)]

    if_st = create_if_stream(scores_df, station=args.station, sr=0.02, network=args.network)
    st = obspy.Stream()
    for i in range(high_conf_flows.shape[0]):
        dt = high_conf_flows["start"].iloc[i]
        start = obspy.UTCDateTime(f"{dt.date}T00:00:00")
        dt = high_conf_flows["stop"].iloc[i]
        end = obspy.UTCDateTime(f"{dt.date}T23:59:59.999999")

        st += if_st.copy().trim(start, end)

    onset_grid = np.array([0.55, 0.60, 0.65, 0.70])
    offset_grid = np.array([0.50, 0.55, 0.60, 0.65])
    iou_table = np.zeros([len(onset_grid), len(offset_grid)])


    for i in range(onset_grid.shape[0]):
        for j in range(offset_grid.shape[0]):
            if onset_grid[i] < offset_grid[j]:
                continue

            segments = stream_trigger_detections(st, onset_grid[i], offset_grid[j])
            inter, union, _ = iou(segments, high_conf_flows)
            iou_table[i, j] = inter[-1] / union[-1]

    max_idx = np.unravel_index(np.argmax(iou_table), iou_table.shape)
    onset_thres, offset_thres = onset_grid[max_idx[0]], offset_grid[max_idx[1]]

    output_dict = {"station":args.station,"onset_thres":onset_thres,"offset_thres":offset_thres}
    filename = "../output/if_segments/trigger_params.json"

    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        data = []


    data.append(output_dict)

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
