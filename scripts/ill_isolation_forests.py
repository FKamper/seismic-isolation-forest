import os
import numpy as np
import pandas as pd
import obspy
from joblib import dump, load
from seismicif.datamod.loading_utils import find_paths
from seismicif.isolation_forest import train_if, compute_scores


stations = ["ILL11","ILL12","ILL13","ILL14","ILL15","ILL16","ILL17","ILL18"]
tr_start, tr_stop, te_start, te_stop = 2018, 2020, 2021, 2022

for station in stations:
    channel = "EHZ.D"
    if station == "ILL11": channel = "HHZ.D"

    tr_paths, te_paths = find_paths("XP",station,channel, tr_start, tr_stop), find_paths("XP",station,channel, te_start, te_stop)

    print(f"{station}: Training IF")
    try:
        if_mod = load(f"../output/if_models/{station}.joblib")

    except (FileNotFoundError, OSError):
        if_mod = train_if(tr_paths)
        dump(if_mod,f"../output/if_models/{station}.joblib")
    print(f"{station}: Training IF Completed")

    print(f"{station}: Computing Anomaly Scores")
    try:
        scores_df = pd.read_csv(f"../output/if_scores/{station}.csv",index_col=0)
        scores_df["start"] = [obspy.UTCDateTime(str(i)) for i in scores_df["start"]]
        scores_df["stop"] = [obspy.UTCDateTime(str(i)) for i in scores_df["stop"]]

    except (FileNotFoundError, OSError):
        scores_df = compute_scores(np.concatenate([tr_paths, te_paths]), if_mod)
        scores_df.to_csv(f"../output/if_scores/{station}.csv")
    print(f"{station}: Anomaly Scores Computed")

    break
