import os
import numpy as np
import pandas as pd
import obspy
from joblib import dump, load
import json
from seismicif.datamod.loading_utils import find_paths
from seismicif.datamod.trigger_utils import stream_trigger_detections
from seismicif.isolation_forest import train_if, compute_scores, create_if_stream

onset_thres, offset_thres = 0.6, 0.6
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

    print(f"{station}: Exctracting Segments")

    if_st = create_if_stream(scores_df, station=station, sr=0.02, network=network)
    if_segments = stream_trigger_detections(if_st, onset_thres, offset_thres)
    if_segments.to_csv(f"../output/DK/if/segments/{station}.csv")

    paths = find_paths(network, control_station, channel, start, stop)

    print(f"{station}: Training Control IF")

    paths = find_paths(network, control_station, channel, start, stop)

    try:
        if_mod = load(f"../output/DK/if/models/{station}_control.joblib")

    except (FileNotFoundError, OSError):
        if_mod = train_if(paths)
        dump(if_mod,f"../output/DK/if/models/{station}_control.joblib")

    print(f"{station}: Computing Control Anomaly Scores")

    try:
        scores_df = pd.read_csv(f"../output/DK/if/scores/{station}_control.csv",index_col=0)
        scores_df["start"] = [obspy.UTCDateTime(str(i)) for i in scores_df["start"]]
        scores_df["stop"] = [obspy.UTCDateTime(str(i)) for i in scores_df["stop"]]

    except (FileNotFoundError, OSError):
        scores_df = compute_scores(paths, if_mod)
        scores_df.to_csv(f"../output/DK/if/scores/{station}_control.csv")

    print(f"{station}: Exctracting Control Segments")

    if_st = create_if_stream(scores_df, station=station, sr=0.02, network=network)
    if_segments = stream_trigger_detections(if_st, onset_thres, offset_thres)
    if_segments.to_csv(f"../output/DK/if/segments/{station}_control.csv")
