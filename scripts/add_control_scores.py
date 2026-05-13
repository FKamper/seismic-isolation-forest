"""
Script to add time series of IF anomaly scores from a control station to time series of IF anomaly scores from a target station.

Example Usage:
python add_control_scores.py -network DK -target_station KARAT -control_station KARAT_CONT -channel HHZ.D -eval_start 2022 -eval_end 2023

This script performs the following steps:
    1. Parses command-line arguments for network, target station, control station, channel and evaluation period.
    2. Loads the IF segments for the target station and extracts the 30 minutes most anomalous sub-segment for each.
    3. For each limited IF segment, extracts the maximum of the corresponding IF anomaly scores from the control station for the same time period.
    4. Adds the control station IF anomaly scores as a new column to the target station IF segments dataframe and saves the updated dataframe to a CSV file in the output directory.

Notes:
- The script assumes that run_if.py has already been run for both the target and control stations. If not, it prompts the user to run run_if.py before adding control scores.
- The script assumes that get_if_segments.py has already been run to compute the IF segments for the target station. If not, it prompts the user to run get_if_segments.py before adding control scores.
"""

import argparse
import os
import pandas as pd
import numpy as np
import joblib

from seismicif.datamod.loading_utils import find_paths, limit_segment_length, preproc_flow_annotations
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Add IF anomaly scores from control to target station.")
    parser.add_argument("-network", type=str, required=True, help="Network containing the stations.")
    parser.add_argument("-target_station", type=str, required=True, help="Target station.")
    parser.add_argument("-control_station", type=str, required=True, help="Control station.")
    parser.add_argument("-channel", type=str, required=True, help="Channel.")
    parser.add_argument("-eval_start", type=int, required=True, help="Evaluation start time.")
    parser.add_argument("-eval_end", type=int, required=True, help="Evaluation end time.")

    args = parser.parse_args()

    try:
        paths = find_paths(args.network, args.target_station, args.channel, args.eval_start, args.eval_end)
    except Exception as e:
        print("Cannot find paths for target station in the given channel.")
        return

    try:
        control_paths = find_paths(args.network, args.control_station, args.channel, args.eval_start, args.eval_end)
    except Exception as e:
        print("Cannot find paths for control station in the given channel.")
        return


    model_path = f"../output/{args.network}/if/models/"

    try:
        if_mod = joblib.load(f"{model_path}/{args.target_station}.joblib")
        print("Target station isolation forest model loaded")

    except (FileNotFoundError, OSError):
        print("Isolation forest does not exist for target station. Please run run_if.py.")
        return

    control_model_path = f"../output/{args.network}/if/models/"

    try:
        control_if_mod = joblib.load(f"{control_model_path}/{args.control_station}.joblib")
        print("Control station isolation forest model loaded")

    except (FileNotFoundError, OSError):
        print("Isolation forest does not exist for control station. Please run run_if.py.")
        return

    segments_path = f"../output/{args.network}/if/segments/{args.target_station}.csv"

    if not os.path.isfile(segments_path):
        print(f"Segments for target station do not exist. Please generate and store in ..{segments_path}")
        return

    else:
        print("Target station segments loaded.")
        segments =  preproc_flow_annotations(pd.read_csv(segments_path, index_col=0))


    if_segments = []
    for i in tqdm(range(segments.shape[0]),desc="Limiting IF segments"):
        if_segments.append(
            limit_segment_length(segments["start"][i], segments["stop"][i], paths, if_mod)
        )

    control_scores = []


    for i in tqdm(range(len((if_segments))),desc="Extracting control IF scores"):
        try:
            t0 = if_segments[i][2]
            t1 = if_segments[i][3]
            control_segment = limit_segment_length(t0, t1, control_paths, control_if_mod)
            control_scores.append(np.max(control_segment[1]))

        except:
            control_scores.append(np.nan)
            continue

    segments["control_scores"] = control_scores
    segments.to_csv(segments_path)

if __name__ == "__main__":
    main()
