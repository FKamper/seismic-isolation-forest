import os
import numpy as np
import pandas as pd
import obspy
import sys
import json
from joblib import dump, load
from seismicif.datamod.loading_utils import find_paths, preproc_flow_annotations, extract_split_flows, extract_valid_segments
from seismicif.datamod.trigger_utils import stream_trigger_detections
from seismicif.isolation_forest import train_if, compute_scores, create_if_stream
from seismicif.metrics import iou, compute_statistics, find_unconfirmed_fp

stations = ["ILL11","ILL12","ILL13","ILL14","ILL15","ILL16","ILL17","ILL18"]
tr_start, tr_stop, te_start, te_stop = 2018, 2020, 2021, 2022
network = "XP"

tr_start_UTC = obspy.UTCDateTime(f"{tr_start}-01-01")
tr_stop_UTC = obspy.UTCDateTime(f"{tr_stop}-12-31T23:59:59.999999")
te_start_UTC = obspy.UTCDateTime(f"{te_start}-01-01")
te_stop_UTC = obspy.UTCDateTime(f"{te_stop}-12-31T23:59:59.999999")

flows = preproc_flow_annotations(pd.read_csv("../catalogs/XP/flow_catalog.csv",index_col=0))
confirmed_fp = preproc_flow_annotations(pd.read_csv("../catalogs/XP/confirmed_FP.csv"))
flows = flows[["2022-09-08" not in str(i) for i in flows["start"]]].reset_index(drop=True)
all_detections = {}
tr_metrics = {}
te_metrics = {}

#collect all detections
for station in stations:
    all_detections[station] = {}
    all_detections[station]["tr_detections"] = {}
    all_detections[station]["te_detections"] = {}

    sta_lta_detections = preproc_flow_annotations(pd.read_csv(f"../output/XP/sta_lta/detections/{station}.csv",index_col=0))
    if_detections = preproc_flow_annotations(pd.read_csv(f"../output/XP/if/detections/{station}.csv",index_col=0))
    dtw_detections = preproc_flow_annotations(pd.read_csv(f"../output/XP/if_dtw/detections/{station}.csv",index_col=0))

    all_detections[station]["tr_detections"]["sta_lta"] = sta_lta_detections[(sta_lta_detections["start"] >  tr_start_UTC) & (sta_lta_detections["stop"] < tr_stop_UTC)].reset_index(drop=True)
    all_detections[station]["tr_detections"]["if"] = if_detections[(if_detections["start"] >  tr_start_UTC) & (if_detections["stop"] < tr_stop_UTC)].reset_index(drop=True)
    all_detections[station]["tr_detections"]["dtw"] = dtw_detections[(dtw_detections["start"] >  tr_start_UTC) & (dtw_detections["stop"] < tr_stop_UTC)].reset_index(drop=True)

    all_detections[station]["te_detections"]["sta_lta"] = sta_lta_detections[(sta_lta_detections["start"] >  te_start_UTC) & (sta_lta_detections["stop"] < te_stop_UTC)].reset_index(drop=True)
    all_detections[station]["te_detections"]["if"] = if_detections[(if_detections["start"] >  te_start_UTC) & (if_detections["stop"] < te_stop_UTC)].reset_index(drop=True)
    all_detections[station]["te_detections"]["dtw"] = dtw_detections[(dtw_detections["start"] >  te_start_UTC) & (dtw_detections["stop"] < te_stop_UTC)].reset_index(drop=True)

#look for unconfirmed false positives
tr_unconfirmed_fp = {}
te_unconfirmed_fp = {}

confirmed_fp_dict = {}
for station in ["ILL11", "ILL12", "ILL13", "ILL14", "ILL16", "ILL17", "ILL18"]:
    confirmed_fp_dict[station] = pd.concat(
        [
            confirmed_fp[confirmed_fp["station"] == station].iloc[:, [0, 1, 2, 5]],

        ]
    ).reset_index(drop=True)

for station in ["ILL11", "ILL12", "ILL13", "ILL18"]:
    tr_unconfirmed_fp[station] = []
    te_unconfirmed_fp[station] = []
    _, _, station_flows = extract_split_flows(flows, station, tr_start, te_stop)

    fp = find_unconfirmed_fp(all_detections[station]["tr_detections"]["if"], station_flows, confirmed_fp_dict[station], station, "IF")
    tr_unconfirmed_fp[station].append(fp)
    fp = find_unconfirmed_fp(all_detections[station]["te_detections"]["if"], station_flows, confirmed_fp_dict[station], station, "IF")
    te_unconfirmed_fp[station].append(fp)

    fp = find_unconfirmed_fp(all_detections[station]["tr_detections"]["sta_lta"], station_flows, confirmed_fp_dict[station], station, "STA-LTA")
    tr_unconfirmed_fp[station].append(fp)
    fp = find_unconfirmed_fp(all_detections[station]["te_detections"]["sta_lta"], station_flows, confirmed_fp_dict[station], station, "STA-LTA")
    te_unconfirmed_fp[station].append(fp)

    fp = find_unconfirmed_fp(all_detections[station]["tr_detections"]["dtw"], station_flows, confirmed_fp_dict[station], station, "DTW")
    tr_unconfirmed_fp[station].append(fp)
    fp = find_unconfirmed_fp(all_detections[station]["te_detections"]["dtw"], station_flows, confirmed_fp_dict[station], station, "DTW")
    te_unconfirmed_fp[station].append(fp)

    tr_unconfirmed_fp[station] = pd.concat(tr_unconfirmed_fp[station], ignore_index=True)
    te_unconfirmed_fp[station] = pd.concat(te_unconfirmed_fp[station], ignore_index=True)

for station in ["ILL14","ILL16","ILL17"]:
    _, _, station_flows = extract_split_flows(flows, station, tr_start, te_stop)
    fp = find_unconfirmed_fp(all_detections[station]["tr_detections"]["dtw"], station_flows, confirmed_fp_dict[station], station, "DTW")
    tr_unconfirmed_fp[station] = fp
    fp = find_unconfirmed_fp(all_detections[station]["te_detections"]["dtw"], station_flows, confirmed_fp_dict[station], station, "DTW")
    te_unconfirmed_fp[station] = fp

tr_unconfirmed_fp = pd.concat(tr_unconfirmed_fp, ignore_index=True)
te_unconfirmed_fp = pd.concat(te_unconfirmed_fp, ignore_index=True)


if len(tr_unconfirmed_fp) > 0:
    tr_unconfirmed_fp.to_csv("../catalogs/XP/tr_unconfirmed_fp.csv", index=False)
    sys.exit("Unconfirmed false positives found in training detections. Please check the output file: ../catalogs/XP/tr_unconfirmed_fp.csv")
if len(te_unconfirmed_fp) > 0:
    te_unconfirmed_fp.to_csv("../catalogs/XP/te_unconfirmed_fp.csv", index=False)
    sys.exit("Unconfirmed false positives found in testing detections. Please check the output file: ../catalogs/XP/te_unconfirmed_fp.csv")



for station in stations:
    sta_lta_detections = all_detections[station]["tr_detections"]["sta_lta"]
    if_detections = all_detections[station]["tr_detections"]["if"]
    dtw_detections = all_detections[station]["tr_detections"]["dtw"]
    tr_metrics[station] = {}
    tr_lower_conf_flows, tr_high_conf_flows,_ = extract_split_flows(flows,station,tr_start,tr_stop)

    valid_detections = extract_valid_segments(sta_lta_detections,tr_lower_conf_flows,tr_high_conf_flows)
    sta_lta_metrics = compute_statistics(valid_detections,tr_high_conf_flows)
    valid_detections = extract_valid_segments(if_detections,tr_lower_conf_flows,tr_high_conf_flows)
    if_metrics = compute_statistics(valid_detections,tr_high_conf_flows)
    valid_detections = extract_valid_segments(dtw_detections,tr_lower_conf_flows,tr_high_conf_flows)
    dtw_metrics = compute_statistics(valid_detections,tr_high_conf_flows)

    tr_metrics[station]["sta_lta_iou"] = f"{np.round(sta_lta_metrics['iou'],2)}"
    tr_metrics[station]["if_iou"] = f"{np.round(if_metrics['iou'],2)}"
    tr_metrics[station]["dtw_iou"] = f"{np.round(dtw_metrics['iou'],2)}"
    tr_metrics[station]["sta_lta_recall"] = f"{np.round(sta_lta_metrics['recall'],2)} ({sta_lta_metrics['FN']})"
    tr_metrics[station]["if_recall"] = f"{np.round(if_metrics['recall'],2)} ({if_metrics['FN']})"
    tr_metrics[station]["dtw_recall"] = f"{np.round(dtw_metrics['recall'],2)} ({dtw_metrics['FN']})"
    tr_metrics[station]["sta_lta_precision"] = f"{np.round(sta_lta_metrics['precision'],2)} ({sta_lta_metrics['FP']})"
    tr_metrics[station]["if_precision"] = f"{np.round(if_metrics['precision'],2)} ({if_metrics['FP']})"
    tr_metrics[station]["dtw_precision"] = f"{np.round(dtw_metrics['precision'],2)} ({dtw_metrics['FP']})"

    sta_lta_detections = all_detections[station]["te_detections"]["sta_lta"]
    if_detections = all_detections[station]["te_detections"]["if"]
    dtw_detections = all_detections[station]["te_detections"]["dtw"]
    te_metrics[station] = {}
    te_lower_conf_flows, te_high_conf_flows,_ = extract_split_flows(flows,station,te_start,te_stop)

    valid_detections = extract_valid_segments(sta_lta_detections,te_lower_conf_flows,te_high_conf_flows)
    sta_lta_metrics = compute_statistics(valid_detections,te_high_conf_flows)
    valid_detections = extract_valid_segments(if_detections,te_lower_conf_flows,te_high_conf_flows)
    if_metrics = compute_statistics(valid_detections,te_high_conf_flows)
    valid_detections = extract_valid_segments(dtw_detections,te_lower_conf_flows,te_high_conf_flows)
    dtw_metrics = compute_statistics(valid_detections,te_high_conf_flows)

    te_metrics[station]["sta_lta_iou"] = f"{np.round(sta_lta_metrics['iou'],2)}"
    te_metrics[station]["if_iou"] = f"{np.round(if_metrics['iou'],2)}"
    te_metrics[station]["dtw_iou"] = f"{np.round(dtw_metrics['iou'],2)}"
    te_metrics[station]["sta_lta_recall"] = f"{np.round(sta_lta_metrics['recall'],2)} ({sta_lta_metrics['FN']})"
    te_metrics[station]["if_recall"] = f"{np.round(if_metrics['recall'],2)} ({if_metrics['FN']})"
    te_metrics[station]["dtw_recall"] = f"{np.round(dtw_metrics['recall'],2)} ({dtw_metrics['FN']})"
    te_metrics[station]["sta_lta_precision"] = f"{np.round(sta_lta_metrics['precision'],2)} ({sta_lta_metrics['FP']})"
    te_metrics[station]["if_precision"] = f"{np.round(if_metrics['precision'],2)} ({if_metrics['FP']})"
    te_metrics[station]["dtw_precision"] = f"{np.round(dtw_metrics['precision'],2)} ({dtw_metrics['FP']})"


print(f"\n===== Training Metrics =====\n")
print(pd.DataFrame(tr_metrics).T)

print(f"\n===== Testing Metrics =====\n")
print(pd.DataFrame(te_metrics).T)
