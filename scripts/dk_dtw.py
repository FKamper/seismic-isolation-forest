import os
import numpy as np
import pandas as pd
import pickle
import obspy
import json
from joblib import dump, load
from tqdm import tqdm
from seismicif.datamod.loading_utils import find_paths, preproc_flow_annotations, read_segment
from seismicif.dtw import segment_dtw
import multiprocessing as mp
from functools import partial

network = "DK"
stations = ["NUUG","KARAT"]
channel = "HHZ.D"


def distribute_segment_dtw(args):
    i, j, segments, paths, if_mod = args
    seg1 = read_segment(segments["start"][i], segments["stop"][i], paths)
    seg2 = read_segment(segments["start"][j], segments["stop"][j], paths)
    return segment_dtw(seg1, seg2, if_mod)


if __name__ == "__main__":

    for station in stations:
        if station == "NUUG":
            start, stop = 2017, 2017

        if station == "KARAT":
            start, stop = 2022, 2023

        if_mod = load(f"../output/DK/if/models/{station}.joblib")
        segments = preproc_flow_annotations(pd.read_csv(f"../output/DK/if/segments/{station}.csv",index_col=0)).iloc[:50,:]
        paths = find_paths(network, station, channel, start, stop)

        n = segments.shape[0]
        indices = [(i, j, segments, paths, if_mod) for i in range(n) for j in range(i + 1, n)]

        print(f"{station}: Performing Pairwise Segment DTW")

        with mp.Pool(mp.cpu_count()) as pool:
            results = list(tqdm(pool.imap(distribute_segment_dtw, indices), total=len(indices)))

        indices = [(i, j) for i in range(n) for j in range(i + 1, n)]

        with open(f'../output/DK/dtw/info/{station}.pkl', 'wb') as f:
            pickle.dump([indices,results], f)

        break
