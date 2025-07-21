import os
import numpy as np
import pandas as pd
import obspy
from joblib import dump, load
import json
from seismicif.datamod.loading_utils import find_paths, preproc_flow_annotations, extract_split_flows, extract_valid_segments
from seismicif.datamod.trigger_utils import stream_trigger_detections
from seismicif.isolation_forest import train_if, compute_scores, create_if_stream
from seismicif.metrics import iou, compute_statistics

stations = ["ILL11","ILL12","ILL13","ILL14","ILL15","ILL16","ILL17","ILL18"]
tr_start, tr_stop, te_start, te_stop = 2018, 2020, 2021, 2022
network = "XP"

tr_start_UTC = obspy.UTCDateTime(f"{tr_start}-01-01")
tr_stop_UTC = obspy.UTCDateTime(f"{tr_stop}-12-31T23:59:59.999999")
te_start_UTC = obspy.UTCDateTime(f"{te_start}-01-01")
te_stop_UTC = obspy.UTCDateTime(f"{te_stop}-12-31T23:59:59.999999")


for station in stations:
    flows = preproc_flow_annotations(pd.read_csv("../catalogs/XP/flow_catalog.csv",index_col=0))
    flows = flows[["2022-09-08" not in str(i) for i in flows["start"]]].reset_index(drop=True)

    tr_lower_conf_flows, tr_high_conf_flows,_ = extract_split_flows(flows,station,tr_start,tr_stop)
    te_lower_conf_flows, te_high_conf_flows,_ = extract_split_flows(flows,station,te_start,te_stop)

    # sta_lta_detections = preproc_flow_annotations(pd.read_csv(f"../output/sta_lta_detections/{station}.csv",index_col=0))
    # tr_sta_lta_detections =  sta_lta_detections[(sta_lta_detections["start"] >  tr_start_UTC) & (sta_lta_detections["stop"] < tr_stop_UTC)].reset_index(drop=True)
    # te_sta_lta_detections =  sta_lta_detections[(sta_lta_detections["start"] >  te_start_UTC) & (sta_lta_detections["stop"] < te_stop_UTC)].reset_index(drop=True)

    if_detections = preproc_flow_annotations(pd.read_csv(f"../output/XP/if/detections/{station}.csv",index_col=0))
    tr_if_detections =  if_detections[(if_detections["start"] >  tr_start_UTC) & (if_detections["stop"] < tr_stop_UTC)].reset_index(drop=True)
    te_if_detections =  if_detections[(if_detections["start"] >  te_start_UTC) & (if_detections["stop"] < te_stop_UTC)].reset_index(drop=True)

    # dtw_detections = preproc_flow_annotations(pd.read_csv(f"../output/dtw_detections/{station}.csv",index_col=0))
    # tr_dtw_detections =  dtw_detections[(dtw_detections["start"] >  tr_start_UTC) & (dtw_detections["stop"] < tr_stop_UTC)].reset_index(drop=True)
    # te_dtw_detections =  dtw_detections[(dtw_detections["start"] >  te_start_UTC) & (dtw_detections["stop"] < te_stop_UTC)].reset_index(drop=True)

    print("Training Metrics")

    # valid_detections = extract_valid_segments(tr_sta_lta_detections,tr_lower_conf_flows,tr_high_conf_flows)
    # print(compute_statistics(valid_detections,tr_high_conf_flows))

    valid_detections = extract_valid_segments(tr_if_detections,tr_lower_conf_flows,tr_high_conf_flows)
    print(compute_statistics(valid_detections,tr_high_conf_flows))

    # valid_detections = extract_valid_segments(tr_dtw_detections,tr_lower_conf_flows,tr_high_conf_flows)
    # print(compute_statistics(valid_detections,tr_high_conf_flows))

    print("Testing Metrics")
    # valid_detections = extract_valid_segments(te_sta_lta_detections,te_lower_conf_flows,te_high_conf_flows)
    # print(compute_statistics(valid_detections,te_high_conf_flows))

    valid_detections = extract_valid_segments(te_if_detections,te_lower_conf_flows,te_high_conf_flows)
    print(compute_statistics(valid_detections,te_high_conf_flows))

    # try:
    #     valid_detections = extract_valid_segments(te_dtw_detections,te_lower_conf_flows,te_high_conf_flows)
    #     print(compute_statistics(valid_detections,te_high_conf_flows))

    # except:
    #     continue
