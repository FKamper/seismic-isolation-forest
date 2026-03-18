import argparse
import os
import pandas as pd
import numpy as np
import obspy
import json

from seismicif.datamod.loading_utils import preproc_flow_annotations, extract_split_flows
from seismicif.datamod.trigger_utils import stream_trigger_detections
from seismicif.isolation_forest import create_if_stream
from seismicif.metrics import iou

def main():
    parser = argparse.ArgumentParser(description="Train Isolation Forest and Compute Anomaly Scores")
    parser.add_argument("-network", type=str, required=True, help="Network contain the IF scores.")
    parser.add_argument("-station", type=str, required=True, help="Station to calibrate IF trigger to.")
    parser.add_argument("-tr_start", type=int, required=True, help="Year to start Training")
    parser.add_argument("-tr_end", type=int, required=True, help="Year to end Training")

    args = parser.parse_args()

    onset_grid = np.array([0.55, 0.60, 0.65, 0.70])
    offset_grid = np.array([0.50, 0.55, 0.60, 0.65])

    catalog_path = f"../catalogs/{args.network}/"

    if not os.path.isdir(catalog_path):
        print(f"Folder does not exist. Please create ../catalogs/{args.network}/")
        return

    try:
        flows = preproc_flow_annotations(pd.read_csv(f"{catalog_path}/initial_catalog.csv"))
        _, _, all_flows = extract_split_flows(flows,args.station,args.tr_start,args.tr_end)

    except (FileNotFoundError, OSError):
        print(f"No initial catalog found for network {args.network}.")
        return

    segments_path = f"../output/{args.network}/if/segments/"

    if not os.path.isdir(segments_path):
        print(f"Folder does not exist. Please create ../output/{args.network}/if/segments/")
        return

    print(f"Calibrating trigger from {args.tr_start} to {args.tr_end}")

    trigger_params_path = f"../output/{args.network}/if/segments/trigger_params.json"

    if not os.path.isfile(trigger_params_path):
        trigger_params = [{"station": args.station,"onset_thres": np.nan,"offset_thres": np.nan}]

    else:
        with open(trigger_params_path, "r") as f:
            trigger_params = json.load(f)

    if args.station not in (d["station"] for d in trigger_params):
        trigger_params.append({"station": args.station,"onset_thres": np.nan,"offset_thres": np.nan})

    entry = next((d for d in trigger_params if d.get("station") == args.station), None)

    if entry["onset_thres"] is not np.nan and entry["offset_thres"] is not np.nan:
        print(f"{args.station}: IF trigger already calibrated with onset threshold {entry['onset_thres']} and offset threshold {entry['offset_thres']}. Delete to recalibrate.")
        return

    scores_path = f"../output/{args.network}/if/scores/{args.station}.csv"

    try:
        print("Loading IF scores. Can take a few seconds.")
        scores_df = pd.read_csv(scores_path, index_col=0)
        scores_df["start"] = [obspy.UTCDateTime(str(i)) for i in scores_df["start"]]
        scores_df["stop"] = [obspy.UTCDateTime(str(i)) for i in scores_df["stop"]]


    except (FileNotFoundError, OSError):
        print(f"{args.station}: No IF scores found. Please run run_if.py to compute scores before calibrating trigger.")
        return

    print(f"{args.station}: Proceeding with Calibration")

    dates = []
    for i in range(all_flows.shape[0]):
        dates.append(all_flows["start"].iloc[i].date)
        dates.append(all_flows["stop"].iloc[i].date)
    dates = np.array(dates)
    dates = np.unique(dates)

    if_st = create_if_stream(scores_df, station=args.station, sr=0.02, network=args.network)
    st = obspy.Stream()

    for dt in dates:
        start = obspy.UTCDateTime(f"{dt}T00:00:00")
        end = obspy.UTCDateTime(f"{dt}T23:59:59.999999")
        st += if_st.copy().trim(start, end)

    iou_table = np.zeros([len(onset_grid), len(offset_grid)])
    for i in range(onset_grid.shape[0]):
        for j in range(offset_grid.shape[0]):
            if onset_grid[i] < offset_grid[j]:
                continue

            segments = stream_trigger_detections(st, onset_grid[i], offset_grid[j])
            inter, union, _ = iou(segments, all_flows)
            iou_table[i, j] = inter[-1] / union[-1]

    max_idx = np.unravel_index(np.argmax(iou_table), iou_table.shape)
    onset_thres, offset_thres = onset_grid[max_idx[0]], offset_grid[max_idx[1]]
    entry["onset_thres"] = onset_thres
    entry["offset_thres"] = offset_thres

    with open(trigger_params_path, "w") as f:
        json.dump(trigger_params, f)


if __name__ == "__main__":
    main()
