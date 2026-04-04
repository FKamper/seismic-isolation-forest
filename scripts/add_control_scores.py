import argparse
import os
import pandas as pd
import numpy as np
import obspy
import joblib
# import json

#from seismicif.datamod.loading_utils import preproc_flow_annotations, extract_split_flows, find_paths, limit_segment_length
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


    # try:
    #     print(f"Loading IF scores for control station. Can take a few seconds...")
    #     control_df = pd.read_csv(f"../output/{args.network}/if/scores/{args.control_station}.csv",index_col=0)
    #     control_df["start"] = [obspy.UTCDateTime(str(i)) for i in control_df["start"]]
    #     control_df["stop"] = [obspy.UTCDateTime(str(i)) for i in control_df["stop"]]

    # except (FileNotFoundError, OSError):
    #     print(f"No IF scores found for control station. Please run run_if.py to compute scores.")
    #     return

    # control_scores = []
    # #Todo: this loop can probably be optimized
    # for i in tqdm(range(segments.shape[0]),desc="Adding control scores to target station IF scores..."):
    #     _, _ , t0, t1 = limit_segment_length(segments["start"][i], segments["stop"][i], paths, if_mod)
    #     subset = control_df.loc[(control_df["start"] >= t0) & (control_df["stop"] <= t1),"anomaly_score"]
    #     max_score = subset.max() if not subset.empty else np.nan
    #     control_scores.append(max_score)

    # segments["control_scores"] = control_scores


if __name__ == "__main__":
    main()
