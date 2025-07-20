import os
import numpy as np
import pandas as pd
import obspy
from joblib import dump, load
import json
from seismicif.datamod.loading_utils import find_paths, preproc_flow_annotations, extract_split_flows, extract_valid_segments
from seismicif.datamod.trigger_utils import stream_trigger_detections
from seismicif.isolation_forest import train_if, compute_scores, create_if_stream
from seismicif.metrics import iou, est_thresholds


tr_start, tr_stop, te_start, te_stop = 2018, 2020, 2021, 2022
onset_grid = np.array([0.55, 0.60, 0.65, 0.70])
offset_grid = np.array([0.50, 0.55, 0.60, 0.65])
network = "XP"
stations = ["ILL11","ILL12","ILL13","ILL14","ILL15","ILL16","ILL17","ILL18"]

for station in stations:
    channel = "EHZ.D"
    if station == "ILL11": channel = "HHZ.D"

    tr_paths, te_paths = find_paths(network, station, channel, tr_start, tr_stop), find_paths("XP", station, channel, te_start, te_stop)

    print(f"{station}: Training IF")

    try:
        if_mod = load(f"../output/if_models/{station}.joblib")

    except (FileNotFoundError, OSError):
        if_mod = train_if(tr_paths)
        dump(if_mod,f"../output/if_models/{station}.joblib")

    print(f"{station}: Computing Anomaly Scores")

    try:
        scores_df = pd.read_csv(f"../output/if_scores/{station}.csv",index_col=0)
        scores_df["start"] = [obspy.UTCDateTime(str(i)) for i in scores_df["start"]]
        scores_df["stop"] = [obspy.UTCDateTime(str(i)) for i in scores_df["stop"]]

    except (FileNotFoundError, OSError):
        scores_df = compute_scores(np.concatenate([tr_paths, te_paths]), if_mod)
        scores_df.to_csv(f"../output/if_scores/{station}.csv")

    print(f"{station}: Calibrating and Deploying Trigger")

    try:
        if_segments = preproc_flow_annotations(pd.read_csv(f"../output/if_segments/{station}.csv",index_col=0))

    except (FileNotFoundError, OSError):
        flows = preproc_flow_annotations(pd.read_csv("../catalogs/initial_catalog.csv",index_col=0))
        lower_conf_flows, high_conf_flows, all_flows = extract_split_flows(flows,station,tr_start,tr_stop)

        dates = []
        for i in range(all_flows.shape[0]):
            dates.append(all_flows["start"].iloc[i].date)
            dates.append(all_flows["stop"].iloc[i].date)
        dates = np.array(dates)
        dates = np.unique(dates)

        if_st = create_if_stream(scores_df, station=station, sr=0.02, network=network)
        st = obspy.Stream()

        for dt in dates:
            start = obspy.UTCDateTime(f"{dt}T00:00:00")
            end = obspy.UTCDateTime(f"{dt}T23:59:59.999999")
            st += if_st.copy().trim(start, end)

        iou_table = np.zeros([len(onset_grid), len(offset_grid)])
        for i in range(onset_grid.shape[0]):
            for j in range(offset_grid.shape[0]):
                if onset_grid[i] < offset_grid[j]:
                    continue

                segments = stream_trigger_detections(st, onset_grid[i], offset_grid[j])
                inter, union, _ = iou(segments, all_flows)
                iou_table[i, j] = inter[-1] / union[-1]

        max_idx = np.unravel_index(np.argmax(iou_table), iou_table.shape)
        onset_thres, offset_thres = onset_grid[max_idx[0]], offset_grid[max_idx[1]]

        output_dict = {"station":station,"onset_thres":onset_thres,"offset_thres":offset_thres}
        filename = "../output/if_segments/trigger_params.json"

        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(output_dict)

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

        if_segments = stream_trigger_detections(if_st, onset_thres, offset_thres)
        if_segments.to_csv(f"../output/if_segments/{station}.csv")

    print(f"{station}: Generating Detections")

    try:
        pd.read_csv(f"../output/if_detections/{station}.csv")

    except (FileNotFoundError, OSError):
        flows = preproc_flow_annotations(pd.read_csv("../catalogs/calibration_catalog.csv",index_col=0))
        lower_conf_flows, high_conf_flows, _ = extract_split_flows(flows,station,tr_start,tr_stop)
        tr_start_UTC = obspy.UTCDateTime(f"{tr_start}-01-01")
        tr_stop_UTC = obspy.UTCDateTime(f"{tr_stop}-12-31T23:59:59.999999")
        tr_segments =  if_segments[(if_segments["start"] >  tr_start_UTC) & (if_segments["stop"] < tr_stop_UTC)].reset_index(drop=True)

        valid_segments = extract_valid_segments(tr_segments,lower_conf_flows,high_conf_flows)
        _, min_len, score_thres = est_thresholds(valid_segments, high_conf_flows)

        filename = "../output/if_detections/threshold_params.json"

        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append({"station":station,"min_len":min_len,"score_thres":score_thres})

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

        detections = if_segments.copy()
        start_times = np.array([pd.to_datetime(str(i)) for i in detections["start"]])
        end_times = np.array([pd.to_datetime(str(i)) for i in detections["stop"]])
        det_lens = np.array([i.total_seconds() for i in end_times - start_times])
        detections = detections.assign(det_lens=det_lens)
        detections = detections[
            (detections["scores"] > score_thres) & (detections["det_lens"] > min_len)
        ].reset_index(drop=True)
        detections.to_csv(f"../output/if_detections/{station}.csv")
