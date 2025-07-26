import os
import numpy as np
import pandas as pd
import pickle
import obspy
import json
import warnings
from joblib import dump, load
from tqdm import tqdm
from seismicif.datamod.loading_utils import find_paths, preproc_flow_annotations, read_segment, limit_segment_length
from seismicif.dtw import segment_dtw
import multiprocessing as mp
from functools import partial

warnings.filterwarnings("ignore", message="Resampled trace would have less than one sample.*")

network = "DK"
stations = ["NUUG","KARAT"]
channel = "HHZ.D"
num_segments = 50

def distribute_segment_dtw(args):
    i, j, segments = args
    seg1 = segments[i]
    seg2 = segments[j]
    return segment_dtw(seg1, seg2)


if __name__ == "__main__":

    for station in stations:
        if station == "NUUG":
            start, stop = 2017, 2017

        if station == "KARAT":
            start, stop = 2022, 2023

        if_mod = load(f"../output/DK/if/models/{station}.joblib")
        if_segments = preproc_flow_annotations(pd.read_csv(f"../output/DK/if/segments/{station}.csv",index_col=0)).iloc[:num_segments,:]
        paths = find_paths(network, station, channel, start, stop)

        try:
            with open(f'../output/DK/dtw/info/{station}.pkl', 'rb') as f:
                pickle.load(f)

        except:
            print(f"{station}: Limiting Segment Lengths")

            segments = []
            for i in tqdm(range(if_segments.shape[0])):
                segments.append(limit_segment_length(if_segments["start"][i],if_segments["stop"][i],paths,if_mod))

            n = len(segments)
            indices = [(i, j, segments) for i in range(n) for j in range(i + 1, n)]

            print(f"{station}: Performing Pairwise Segment DTW")

            with mp.Pool(mp.cpu_count()) as pool:
                results = list(tqdm(pool.imap(distribute_segment_dtw, indices), total=len(indices)))

            indices = [(i, j) for i in range(n) for j in range(i + 1, n)]

            with open(f'../output/DK/dtw/info/{station}.pkl', 'wb') as f:
                pickle.dump([indices,results], f)
