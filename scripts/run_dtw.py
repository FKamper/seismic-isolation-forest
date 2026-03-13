import argparse
import os
import pickle
import pandas as pd
import joblib
import multiprocessing as mp
import numpy as np

from tqdm import tqdm
from seismicif.datamod.loading_utils import preproc_flow_annotations, find_paths, limit_segment_length
from seismicif.dtw import distribute_reference_segment_dtw, get_dtw_score

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
        if_segments_df.drop(columns=["start", "stop","most_anomalous_start"], inplace=True)
        if_segments_df.rename(columns={"segment_of_interest_start": "start","segment_of_interest_stop": "stop"}, inplace=True)

    ref_segments_path = f"../output/{args.network}/dtw/reference_segments/"

    try:
        ref_segments_df = pd.read_csv(f"{ref_segments_path}/{args.station}.csv")
        idx = np.where(ref_segments_df["include"].values == "yes")[0]
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
            with open(f"{ref_segments_path}/{args.station}_pairwise", 'rb') as f:
                indices, results = pickle.load(f)

            nseg = max(indices)[1] + 1
            D = np.zeros([nseg,nseg])

            for k in range(len(indices)):
                i,j = indices[k]
                D[i,j] = np.median(results[k][0])
                D[j,i] = D[i,j]

            D = D[np.ix_(idx, idx)]

            print("Loaded reference pairwise DTW distances.")

            with open(f"{dtw_path}/{args.station}.pkl", 'rb') as f:
                results = pickle.load(f)

            X = np.zeros((len(results), len(results[0])))
            for i in range(len(results)):
                X[i] = np.array([np.median(results[i][k][0]) for k in range(len(results[i]))])

            print("Loaded segment DTW distances.")

            dendo_height_scores =[]
            for i in range(len(if_segments_df)):
                score = get_dtw_score(if_segments_df.iloc[[i],:].reset_index(drop=True), ref_segments_df, X[i,:], D)
                dendo_height_scores.append(score)
            dendo_height_scores = np.array(dendo_height_scores)

            print("Computed merge heights.")

            dtw_segments_df["scores"] = dendo_height_scores
            dtw_segments_df.sort_values(by="scores", inplace=True)
            dtw_segments_df.reset_index(drop=True, inplace=True)
            dtw_segments_df.to_csv(f"{segment_storage_path}/{args.station}.csv")

if __name__ == "__main__":
    main()
