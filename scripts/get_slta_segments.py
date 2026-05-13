"""
This script extracts STA-LTA segments from the miniseed recordings for a given station and evaluation period.
The resulting segments are saved as CSV files in the output directory.

Example usage ~ should be run from the scripts/ directory:
python get_slta_segments.py -network XP -station ILL11 -channel HHZ.D -eval_start 2018 -eval_end 2022

This script performs the following steps:
1. Parses command-line arguments for network, station, and evaluation time period.
2. Extracts the STA-LTA segments.
3. Saves the resulting segments to a CSV file in the output directory.

Notes:
- The script assumes that calibrate_sta_lta_trigger.py has already been run to calibrate the trigger thresholds for the specified station. If not, it uses rule of thumb thresholds.
- As far as is feasible (i.e. available data and no breakages) we append the seismic waveform of the previous day so that the STA-LTA can immediately compute its characteristic function.
"""
import argparse
import os
import pandas as pd
import numpy as np
import obspy
import json

from seismicif.datamod.loading_utils import find_paths
from seismicif.sta_lta import slta_detections_from_paths


def main():
    parser = argparse.ArgumentParser(description="Extract STA-LTA trigger segments.")
    parser.add_argument("-network", type=str, required=True, help="Network containing the waveforms.")
    parser.add_argument("-station", type=str, required=True, help="Station to calibrate trigger to.")
    parser.add_argument("-channel", type=str, required=True, help="Channel to collect waveforms from.")
    parser.add_argument("-eval_start", type=int, required=True, help="Year to start Evaluation")
    parser.add_argument("-eval_end", type=int, required=True, help="Year to end Evaluation")

    args = parser.parse_args()

    try:
        paths = find_paths(args.network, args.station, args.channel, args.eval_start, args.eval_end)

    except Exception as e:
        print("Cannot find paths for station in the given channel and time range.")
        return


    args = parser.parse_args()

    segments_path = f"../output/{args.network}/sta_lta/segments/"

    if not os.path.isdir(segments_path):
        print(f"Folder does not exist. Please create ../output/{args.network}/sta_lta/segments/")
        return

    trigger_params_path = f"../output/{args.network}/sta_lta/segments/trigger_params.json"

    if not os.path.isfile(trigger_params_path):
        print(f"No trigger parameters found for network. Running with rule of thumb thresholds.")


    else:
        with open(trigger_params_path, "r") as f:
            trigger_params = json.load(f)

        if args.station not in (d["station"] for d in trigger_params):
            print(f"No trigger parameters found for {args.station}. Running with rule of thumb thresholds.")
            onset_thres, offset_thres, sw, lw = 6.0 , 0.125, 50000, 500000

        else:
            entry = next((d for d in trigger_params if d.get("station") == args.station), None)
            onset_thres, offset_thres = entry["onset_thres"], entry["offset_thres"]
            sw, lw = entry["sw"], entry["lw"]

            print(f"{args.station}: Using onset threshold {onset_thres}, offset threshold {offset_thres}, short window length {sw/100}s, and long window length {lw/100}s from calibration.")


    try:
        sta_lta_segments = pd.read_csv(f"{segments_path}/{args.station}.csv")
        print(f"Trigger segments already exist for {args.station}. Delete to recompute.")


    except (FileNotFoundError, OSError):
        sta_lta_segments = slta_detections_from_paths(paths, sw, lw, onset_thres, offset_thres)
        sta_lta_segments.to_csv(f"{segments_path}/{args.station}.csv")


if __name__ == "__main__":
    main()
