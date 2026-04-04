import argparse
import os
import joblib
import pickle
import pandas as pd
import multiprocessing as mp
import numpy as np

from tqdm import tqdm
from seismicif.datamod.loading_utils import preproc_flow_annotations, extract_split_flows, find_paths, limit_segment_length
from seismicif.dtw import  distribute_pairwise_segment_dtw, remove_singleton_merges

def main():
    parser = argparse.ArgumentParser(description="Perform DTW between segments and reference segments.")
    parser.add_argument("-network", type=str, required=True, help="Network where regerence segments are to be extracted from.")
    parser.add_argument("-station", type=str, required=True, help="Station for which reference segments are to be extracted.")
    parser.add_argument("-channel", type=str, required=True, help="Channel of the waveforms.")
    parser.add_argument("-tr_start", type=int, required=True, help="Start year of period to extract from.")
    parser.add_argument("-tr_end", type=int, required=True, help="Stop year of period to extract from.")
    parser.add_argument("-use_catalog",type=str,default="yes",help="should a catalog be used to generate the reference segments?")
    parser.add_argument("-num_seg",type=int,default=200,help="number of segments to cluster and extract segments from")

    args = parser.parse_args()

    try:
        paths = find_paths(args.network, args.station, args.channel, args.tr_start, args.tr_end)
    except Exception as e:
        print(f"Error finding paths: {e}")
        return

    model_path = f"../output/{args.network}/if/models/"

    try:
        if_mod = joblib.load(f"{model_path}/{args.station}.joblib")
        print(f"{args.station}: Isolation Forest Model Loaded")

    except (FileNotFoundError, OSError):
        print(f"{args.station}: Isolation forest does not exist. Please run run_if.py.")

    dtw_path = f"../output/{args.network}/dtw/reference_segments"
    if not os.path.isdir(dtw_path):
        print(f"Folder does not exist. Please create ../output/{args.network}/dtw/reference_segments")
        return

    if args.use_catalog == "yes":
        catalog_path = f"../catalogs/{args.network}/"

        if not os.path.isdir(catalog_path):
            print(f"Folder does not exist. Please create ../catalogs/{args.network}/")
            return

        try:
            events = preproc_flow_annotations(pd.read_csv(f"{catalog_path}/initial_catalog.csv"))
            _, high_conf_events, _ = extract_split_flows(events,args.station,args.tr_start,args.tr_end)
        except FileNotFoundError:
            print(f"Initial catalog does not exists. Please create one in ../catalogs/{args.network}/")
            return

        ref_segments = []
        for i in tqdm(range(high_conf_events.shape[0]),desc=f"{args.station}: Limiting reference segment lengths"):
             ref_segments.append(limit_segment_length(high_conf_events["start"][i],high_conf_events["stop"][i],paths,if_mod))

        n = len(ref_segments)
        indices = [(i, j, ref_segments) for i in range(n) for j in range(i + 1, n)]

        try:
            with open(f"{dtw_path}/{args.station}_pairwise", 'rb') as f:
                indices, results = pickle.load(f)
                print(f"{args.station}: Loaded Existing Pairwise DTW Results")

        except FileNotFoundError:
            with mp.Pool(mp.cpu_count()) as pool:
                results = list(tqdm(pool.imap(distribute_pairwise_segment_dtw, indices), total=len(indices),desc=f"{args.station}: Performing Pairwise Segment DTW"))

            indices = [(i, j) for i in range(n) for j in range(i + 1, n)]

            with open(f"{dtw_path}/{args.station}_pairwise", 'wb') as f:
                pickle.dump([indices,results], f)

        nseg = max(indices)[1] + 1
        D = np.zeros([nseg,nseg])

        for k in range(len(indices)):
            i,j = indices[k]
            D[i,j] = np.median(results[k][0])
            D[j,i] = D[i,j]

        idx = remove_singleton_merges(D)

        ref_segments = {"start": [i[2] for i in ref_segments], "stop": [i[3] for i in ref_segments],"include": np.repeat("no",n)}
        ref_segments = pd.DataFrame(ref_segments)
        ref_segments.loc[idx, "include"] = "yes"
        ref_segments.to_csv(f"{dtw_path}/{args.station}.csv", index=False)

    else:
        if_segments_path = f"../output/{args.network}/if/segments/"

        try:
            if_segments_df = pd.read_csv(f"{if_segments_path}/{args.station}.csv",index_col = 0)
            print(f"{args.station}: Loaded IF segments")

        except (FileNotFoundError, OSError):
            print(f"{args.station}: Isolation forest segments do not exist. Please run get_if_segments.py.")

        if_segments = []
        n = min(args.num_seg, if_segments_df.shape[0])

        for i in tqdm(range(n),desc=f"{args.station}: Limiting IF segment lengths"):
              if_segments.append(limit_segment_length(if_segments_df["start"][i],if_segments_df["stop"][i],paths,if_mod))

        indices = [(i, j, if_segments) for i in range(n) for j in range(i + 1, n)]

        try:
            with open(f"{dtw_path}/{args.station}_pairwise", 'rb') as f:
                indices, results = pickle.load(f)
                print(f"{args.station}: Loaded Existing Pairwise DTW Results")

        except FileNotFoundError:
            with mp.Pool(mp.cpu_count()) as pool:
                results = list(tqdm(pool.imap(distribute_pairwise_segment_dtw, indices), total=len(indices),desc=f"{args.station}: Performing Pairwise Segment DTW"))

            with open(f"{dtw_path}/{args.station}_pairwise", 'wb') as f:
                pickle.dump([indices,results], f)

        # ref_segments = {"start":  if_segments["start"][:args.num_seg] , "stop":  if_segments["stop"][:args.num_seg] ,"include": np.repeat("yes",args.num_seg)}
        # ref_segments = pd.DataFrame(ref_segments)
        # ref_segments.to_csv(f"{dtw_path}/{args.station}.csv", index=False)

if __name__ == "__main__":
    main()
