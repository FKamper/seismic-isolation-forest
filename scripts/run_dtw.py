"""
This script performs Dynamic Time Warping (DTW) between segments and reference segments for a given station and evaluation period.

Example usage ~ should be run from the scripts/ directory:
python run_dtw.py -network XP -station ILL11 -channel HHZ.D -eval_start 2018 -eval_end 2020 -score yes

This script performs the following steps:
1. Parses command-line arguments for network, station, channel, evaluation time period, and whether to score the segments.
2. Loads the IF segments and reference segments for the specified station and evaluation period.
3. Performs segment DTW between each IF segment and the reference segments, storing the results.
4. If the score argument is set to "yes":
   - compute the mean segment DTW distance to the reference segments for each IF segment, ignoring any reference segments that overlap with the IF segment.
   - rerank the IF segments by their mean segment DTW distance to the reference segments and save the resulting scored segments to a CSV file in the output directory.

Notes:
- The script assumes that run_if.py has already been run to compute the IF model for the specified station. If not, it prompts the user to run run_if.py before performing DTW.
- The script assumes that the IF segments have already been extracted and stored in the output directory for the specified station. If not, it prompts the user to run get_if_segments.py before performing DTW.
- The script assumes that the reference segments have already been extracted and stored in the output directory for the specified station. If not, it prompts the user to run get_ref_segments.py before performing DTW.
- If the score argument is set to "yes", the script assumes that the output directory for the scored segments exists. If not, it prompts the user to create the output directory before scoring the segments.
"""
import argparse
import os
import pickle
import pandas as pd
import joblib
import multiprocessing as mp
import numpy as np

from tqdm import tqdm
from seismicif.datamod.loading_utils import preproc_flow_annotations, find_paths, limit_segment_length
from seismicif.dtw import distribute_reference_segment_dtw
from seismicif.metrics import iou

def main():
    parser = argparse.ArgumentParser(description="Perform DTW between segments and reference segments.")
    parser.add_argument("-network", type=str, required=True, help="Network where segments were extracted from.")
    parser.add_argument("-station", type=str, required=True, help="Station where segments were extracted from.")
    parser.add_argument("-channel", type=str, required=True, help="Channel of the segments.")
    parser.add_argument("-eval_start", type=int, required=True, help="Year to start performing dtw")
    parser.add_argument("-eval_end", type=int, required=True, help="Year to end dtw")
    parser.add_argument("-score",type=str,default="no",help="should segments be scored?")

    args = parser.parse_args()

    dtw_path = f"../output/{args.network}/dtw/distances/"
    if not os.path.isdir(dtw_path):
        print(f"Folder does not exist. Please create ../output/{args.network}/dtw/distances/")
        return

    try:
        paths = find_paths(args.network, args.station, args.channel, args.eval_start, args.eval_end)
    except Exception as e:
        print(f"Error finding paths: {e}")
        return

    model_path = f"../output/{args.network}/if/models/"

    try:
        if_mod = joblib.load(f"{model_path}/{args.station}.joblib")
        print(f"{args.station}: Isolation Forest Model Loaded")

    except (FileNotFoundError, OSError):
        print(f"{args.station}: Isolation forest does not exist. Please run run_if.py.")

    segments_path = f"../output/{args.network}/if/segments/"

    if not os.path.isdir(segments_path):
        print(f"IF segments do not exist. Please generate and store in ../output/{args.network}/if/segments/")
        return

    else:
        print("Loaded IF segments")
        if_segments_df =  preproc_flow_annotations(pd.read_csv(f"{segments_path}/{args.station}.csv",index_col=0))
        dtw_segments_df = if_segments_df.copy()

    ref_segments_path = f"../output/{args.network}/dtw/reference_segments/"

    try:
        ref_segments_df = pd.read_csv(f"{ref_segments_path}/{args.station}.csv")
        ref_segments_df = ref_segments_df[ref_segments_df["include"].values == "yes"].reset_index(drop=True)
        ref_segments_df.drop(columns=["include"], inplace=True)
        print("Loaded reference segments")
    except:
        print(f"Reference segments do not exist. Please generate and store in {ref_segments_path}")
        return

    try:
        with open(f"{dtw_path}/{args.station}.pkl", 'rb') as f:
            results = pickle.load(f)
            print(f"{args.station}: Loaded Existing Reference DTW Results")

    except FileNotFoundError:
        ref_segments = []
        for i in tqdm(range(ref_segments_df.shape[0]),desc=f"{args.station}: Limiting reference segment lengths"):
            ref_segments.append(limit_segment_length(ref_segments_df["start"][i], ref_segments_df["stop"][i], paths, if_mod))

        paired_segments = []
#Todo: if there is an error below it is probably because the input eval period does not cover the if segment period. Need to add error handling for this.
        for i in tqdm(range(if_segments_df.shape[0]),desc=f"{args.station}: Limiting IF segment lengths"):
            paired_segments.append((limit_segment_length(if_segments_df["start"][i], if_segments_df["stop"][i], paths, if_mod),ref_segments))

        with mp.Pool(mp.cpu_count()) as pool:
            results = list(tqdm(pool.imap(distribute_reference_segment_dtw, paired_segments), total=len(paired_segments),desc=f"{args.station}: Performing Reference Segment DTW"))

        with open(f"{dtw_path}/{args.station}.pkl", 'wb') as f:
            pickle.dump(results, f)

    if args.score == "yes":
        segment_storage_path = f"../output/{args.network}/dtw/segments/"
        if not os.path.isdir(segment_storage_path):
            print(f"\nFolder to store DTW scored segments do not exist. Please create ../output/{args.network}/dtw/segments/")
            return

        try:
            dtw_segments_df = pd.read_csv(f"{segment_storage_path}/{args.station}.csv", index_col=0)
            print("\nDTW scored segments exist. Please delete to recompute.")

        except:
            print("\nScoring Segments")

            with open(f"{dtw_path}/{args.station}.pkl", 'rb') as f:
                results = pickle.load(f)
                print("Loaded reference pairwise DTW distances.")

            X = np.zeros((len(results), len(results[0])))

            for i in range(len(results)):
                X[i] = np.array([np.median(results[i][k][0]) for k in range(len(results[i]))])
                cummulative_intersections, _, _ = iou(ref_segments_df, dtw_segments_df.iloc[[i]].reset_index(drop=True))
                intersection_lengths = np.append(cummulative_intersections[0], np.diff(cummulative_intersections))
                overlap_idx = np.where(intersection_lengths > 0)[0]
                X[i,overlap_idx] = np.nan

            print("Loaded segment DTW distances.")

            dtw_segments_df["scores"] = np.nanmean(X,axis=1)
            dtw_segments_df.sort_values(by="scores", inplace=True)
            dtw_segments_df.reset_index(drop=True, inplace=True)
            dtw_segments_df.to_csv(f"{segment_storage_path}/{args.station}.csv")

if __name__ == "__main__":
    main()
