import os
import numpy as np
import pandas as pd
import obspy
from joblib import dump, load
import json
import warnings
from tqdm import tqdm
from seismicif.datamod.loading_utils import find_paths, limit_segment_length
from seismicif.datamod.trigger_utils import stream_trigger_detections
from seismicif.isolation_forest import train_if, compute_scores, create_if_stream

warnings.filterwarnings("ignore", message="Resampled trace would have less than one sample.*")

onset_thres, offset_thres = 0.60, 0.55
network = "DK"
stations = ["NUUG","KARAT"]
control_station = "ILULI"
channel = "HHZ.D"

for station in stations:

    if station == "NUUG":
        start, stop = 2017, 2017

    if station == "KARAT":
        start, stop = 2022, 2023

    print(f"{station}: Training IF")

    paths = find_paths(network, station, channel, start, stop)

    try:
        if_mod = load(f"../output/DK/if/models/{station}.joblib")

    except (FileNotFoundError, OSError):
        if_mod = train_if(paths)
        dump(if_mod,f"../output/DK/if/models/{station}.joblib")

    print(f"{station}: Computing Anomaly Scores")

    try:
        scores_df = pd.read_csv(f"../output/DK/if/scores/{station}.csv",index_col=0)
        scores_df["start"] = [obspy.UTCDateTime(str(i)) for i in scores_df["start"]]
        scores_df["stop"] = [obspy.UTCDateTime(str(i)) for i in scores_df["stop"]]

    except (FileNotFoundError, OSError):
        scores_df = compute_scores(paths, if_mod)
        scores_df.to_csv(f"../output/DK/if/scores/{station}.csv")

    control_paths = find_paths(network, control_station, channel, start, stop)

    print(f"{station}: Training Control IF")

    try:
        control_if_mod = load(f"../output/DK/if/models/{station}_control.joblib")

    except (FileNotFoundError, OSError):
        control_if_mod = train_if(control_paths)
        dump(control_if_mod,f"../output/DK/if/models/{station}_control.joblib")

    print(f"{station}: Computing Control Anomaly Scores")

    try:
        control_scores_df = pd.read_csv(f"../output/DK/if/scores/{station}_control.csv",index_col=0)
        control_scores_df["start"] = [obspy.UTCDateTime(str(i)) for i in control_scores_df["start"]]
        control_scores_df["stop"] = [obspy.UTCDateTime(str(i)) for i in control_scores_df["stop"]]

    except (FileNotFoundError, OSError):
        control_scores_df = compute_scores(control_paths, control_if_mod)
        control_scores_df.to_csv(f"../output/DK/if/scores/{station}_control.csv")

    try:
        pd.read_csv(f"../output/DK/if/segments/{station}.csv")

    except (FileNotFoundError, OSError):
        print(f"{station}: Creating IF Streams")
        if_st = create_if_stream(scores_df, station=station, sr=0.02, network=network)
        control_if_st = create_if_stream(control_scores_df, station=station, sr=0.02, network=network)
        if_segments = stream_trigger_detections(if_st, onset_thres, offset_thres)

        print(f"{station}: Exctracting Segments")
        control_scores = []
        for i in tqdm(range(if_segments.shape[0])):
            _, _, seg_start, seg_stop = limit_segment_length(if_segments["start"][i],if_segments["stop"][i],paths,if_mod)
            st = control_if_st.copy().trim(seg_start, seg_stop)
            max_val = -np.inf
            for tr in st:
                max_val = max(max_val,np.max(tr.data))
            control_scores.append(max_val)
        if_segments["control_scores"] = control_scores
        if_segments.to_csv(f"../output/DK/if/segments/{station}.csv")
