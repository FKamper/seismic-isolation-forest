"""
This script extracts IF segments from time series of IF anomaly scores for a given station and evaluation period.
The resulting segments are saved as CSV files in the output directory.

Example Usage:
python get_if_segments.py -network XP -station ILL11 -eval_start 2018 -eval_end 2020

This script performs the following steps:
1. Parses command-line arguments for network, station, and evaluation time period.
2. Loads the time series of IF scores for the specified station.
3. Extracts the IF segments.
4. Saves the resulting segments to a CSV file in the output directory.

Notes:
- The script assumes that run_if.py has already been run to compute the IF scores for the specified station and training period. If not, it prompts the user to run run_if.py before proceeding.
- The script assumes that calibrate_if_trigger.py has already been run to calibrate the trigger thresholds for the specified station. If not, it uses rule of thumb thresholds.
"""
import argparse
import os
import pandas as pd
import numpy as np
import obspy
import json

from seismicif.datamod.trigger_utils import stream_trigger_detections
from seismicif.isolation_forest import create_if_stream

def main():
    parser = argparse.ArgumentParser(description="Extract IF segments from time series of IF scores for a given station and evaluation period.")
    parser.add_argument("-network", type=str, required=True, help="Network contain the IF scores.")
    parser.add_argument("-station", type=str, required=True, help="Station to calibrate IF trigger to.")
    parser.add_argument("-eval_start", type=int, required=True, help="Year to start Evaluation")
    parser.add_argument("-eval_end", type=int, required=True, help="Year to end Evaluation")

    args = parser.parse_args()

    segments_path = f"../output/{args.network}/if/segments/"

    if not os.path.isdir(segments_path):
        print(f"Folder does not exist. Please create ../output/{args.network}/if/segments/")
        return

    trigger_params_path = f"../output/{args.network}/if/segments/trigger_params.json"

    if not os.path.isfile(trigger_params_path):
        print(f"No trigger parameters found for network. Running with rule of thumb thresholds.")
        onset_thres, offset_thres = 0.60,0.55

    else:
        with open(trigger_params_path, "r") as f:
            trigger_params = json.load(f)

        if args.station not in (d["station"] for d in trigger_params):
            print(f"No trigger parameters found for station. Running with rule of thumb thresholds.")
            onset_thres, offset_thres = 0.60,0.55

        else:
            entry = next((d for d in trigger_params if d.get("station") == args.station), None)
            onset_thres, offset_thres = entry["onset_thres"], entry["offset_thres"]
            print(f"{args.station}: Using onset threshold {onset_thres} and offset threshold {offset_thres} from calibration.")

    try:
        scores_df = pd.read_csv(f"../output/{args.network}/if/scores/{args.station}.csv",index_col=0)
        scores_df["start"] = [obspy.UTCDateTime(str(i)) for i in scores_df["start"]]
        scores_df["stop"] = [obspy.UTCDateTime(str(i)) for i in scores_df["stop"]]

    except (FileNotFoundError, OSError):
        print(f"{args.station}: No IF scores found. Please run run_if.py to compute scores before extracting segments.")
        return

    eval_start = obspy.UTCDateTime(args.eval_start, 1, 1)
    eval_end   = obspy.UTCDateTime(args.eval_end, 12, 31, 23, 59, 59)

    mask = (scores_df["start"] <= eval_end) & (scores_df["stop"] >= eval_start)
    scores_df = scores_df[mask].reset_index(drop=True)

    if_st = create_if_stream(scores_df, station=args.station, sr=0.02, network=args.network)
    if_segments = stream_trigger_detections(if_st, onset_thres, offset_thres)
    if_segments.to_csv(f"../output/{args.network}/if/segments/{args.station}.csv")

if __name__ == "__main__":
    main()
