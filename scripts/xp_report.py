import os
import numpy as np
import pandas as pd
import obspy
import sys
import json
from joblib import dump, load
from seismicif.datamod.loading_utils import find_paths, preproc_flow_annotations, extract_split_flows
from seismicif.datamod.trigger_utils import stream_trigger_detections
from seismicif.isolation_forest import train_if, compute_scores, create_if_stream
from seismicif.metrics import iou, compute_statistics, find_unconfirmed_fp, compute_lower_conf_recall, extract_valid_segments, compute_station_metrics, find_fp_fn

stations = ["ILL11","ILL12","ILL13","ILL14","ILL15","ILL16","ILL17","ILL18"]
tr_start, tr_stop, te_start, te_stop = 2018, 2020, 2021, 2022
network = "XP"

tr_start_UTC = obspy.UTCDateTime(f"{tr_start}-01-01")
tr_stop_UTC = obspy.UTCDateTime(f"{tr_stop}-12-31T23:59:59.999999")
te_start_UTC = obspy.UTCDateTime(f"{te_start}-01-01")
te_stop_UTC = obspy.UTCDateTime(f"{te_stop}-12-31T23:59:59.999999")

flows = preproc_flow_annotations(pd.read_csv("../catalogs/XP/flow_catalog.csv"))
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
    dtw_detections = preproc_flow_annotations(pd.read_csv(f"../output/XP/dtw/detections/{station}.csv",index_col=0))

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
for station in ["ILL11", "ILL12", "ILL13", "ILL14", "ILL15", "ILL16", "ILL17", "ILL18"]:
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

for station in ["ILL14","ILL15","ILL16","ILL17"]:
    _, _, station_flows = extract_split_flows(flows, station, tr_start, te_stop)

    fp = find_unconfirmed_fp(all_detections[station]["tr_detections"]["dtw"], station_flows, confirmed_fp_dict[station], station, "DTW")
    tr_unconfirmed_fp[station] = fp
    fp = find_unconfirmed_fp(all_detections[station]["te_detections"]["dtw"], station_flows, confirmed_fp_dict[station], station, "DTW")
    te_unconfirmed_fp[station] = fp


tr_unconfirmed_fp = pd.concat(tr_unconfirmed_fp, ignore_index=True)
te_unconfirmed_fp = pd.concat(te_unconfirmed_fp, ignore_index=True)
tr_unconfirmed_fp.to_csv("../catalogs/XP/tr_unconfirmed_fp.csv", index=False)
te_unconfirmed_fp.to_csv("../catalogs/XP/te_unconfirmed_fp.csv", index=False)

if len(tr_unconfirmed_fp) > 0:
    sys.exit("Unconfirmed false positives found in training detections. Please check the output file: ../catalogs/XP/tr_unconfirmed_fp.csv")
if len(te_unconfirmed_fp) > 0:
    sys.exit("Unconfirmed false positives found in testing detections. Please check the output file: ../catalogs/XP/te_unconfirmed_fp.csv")

for station in stations:
    sta_lta_detections = all_detections[station]["tr_detections"]["sta_lta"]
    if_detections = all_detections[station]["tr_detections"]["if"]
    dtw_detections = all_detections[station]["tr_detections"]["dtw"]
    lower_conf_flows, high_conf_flows, _ = extract_split_flows(flows, station, tr_start, tr_stop)
    tr_metrics[station] =  compute_station_metrics(sta_lta_detections, if_detections, dtw_detections, lower_conf_flows, high_conf_flows)

    sta_lta_detections = all_detections[station]["te_detections"]["sta_lta"]
    if_detections = all_detections[station]["te_detections"]["if"]
    dtw_detections = all_detections[station]["te_detections"]["dtw"]
    lower_conf_flows, high_conf_flows, _ = extract_split_flows(flows, station, te_start, te_stop)
    te_metrics[station] = compute_station_metrics(sta_lta_detections, if_detections, dtw_detections, lower_conf_flows, high_conf_flows)

print(f"\n===== Training Metrics =====\n")
tr_metrics_df = {}
for station in stations:
    tr_metrics_df[station] = tr_metrics[station]["for_printing"]
print(pd.DataFrame(tr_metrics_df).T)

print(f"\n===== Testing Metrics =====\n")
te_metrics_df = {}
for station in stations:
    te_metrics_df[station] = te_metrics[station]["for_printing"]
print(pd.DataFrame(te_metrics_df).T)

print(f"\n===== Condensed Testing Metrics =====\n")

for station in stations:
    if np.isnan(te_metrics[station]["sta_lta_metrics"]["precision"]):
        te_metrics[station]["sta_lta_metrics"]["precision"] = 0.0
    if np.isnan(te_metrics[station]["if_metrics"]["precision"]):
        te_metrics[station]["if_metrics"]["precision"] = 0.0
    if np.isnan(te_metrics[station]["dtw_metrics"]["precision"]):
        te_metrics[station]["dtw_metrics"]["precision"] = 0.0

condensed_te_metrics ={}
good_stations = ["ILL11","ILL12","ILL13","ILL18"]
condensed_te_metrics["good"] = {}
condensed_te_metrics["good"]["sta_lta_iou"] = np.mean([te_metrics[station]["sta_lta_metrics"]["iou"] for station in good_stations])
condensed_te_metrics["good"]["if_iou"] = np.mean([te_metrics[station]["if_metrics"]["iou"] for station in good_stations])
condensed_te_metrics["good"]["dtw_iou"] = np.mean([te_metrics[station]["dtw_metrics"]["iou"] for station in good_stations])
condensed_te_metrics["good"]["sta_lta_recall"] = np.mean([te_metrics[station]["sta_lta_metrics"]["recall"] for station in good_stations])
condensed_te_metrics["good"]["if_recall"] = np.mean([te_metrics[station]["if_metrics"]["recall"] for station in good_stations])
condensed_te_metrics["good"]["dtw_recall"] = np.mean([te_metrics[station]["dtw_metrics"]["recall"] for station in good_stations])
condensed_te_metrics["good"]["sta_lta_precision"] = np.mean([te_metrics[station]["sta_lta_metrics"]["precision"] for station in good_stations])
condensed_te_metrics["good"]["if_precision"] = np.mean([te_metrics[station]["if_metrics"]["precision"] for station in good_stations])
condensed_te_metrics["good"]["dtw_precision"] = np.mean([te_metrics[station]["dtw_metrics"]["precision"] for station in good_stations])

bad_stations = ["ILL14","ILL15","ILL16","ILL17"]
condensed_te_metrics["bad"] = {}
condensed_te_metrics["bad"]["sta_lta_iou"] = np.mean([te_metrics[station]["sta_lta_metrics"]["iou"] for station in bad_stations])
condensed_te_metrics["bad"]["if_iou"] = np.mean([te_metrics[station]["if_metrics"]["iou"] for station in bad_stations])
condensed_te_metrics["bad"]["dtw_iou"] = np.mean([te_metrics[station]["dtw_metrics"]["iou"] for station in bad_stations])
condensed_te_metrics["bad"]["sta_lta_recall"] = np.mean([te_metrics[station]["sta_lta_metrics"]["recall"] for station in bad_stations])
condensed_te_metrics["bad"]["if_recall"] = np.mean([te_metrics[station]["if_metrics"]["recall"] for station in bad_stations])
condensed_te_metrics["bad"]["dtw_recall"] = np.mean([te_metrics[station]["dtw_metrics"]["recall"] for station in bad_stations])
condensed_te_metrics["bad"]["sta_lta_precision"] = np.mean([te_metrics[station]["sta_lta_metrics"]["precision"] for station in bad_stations])
condensed_te_metrics["bad"]["if_precision"] = np.mean([te_metrics[station]["if_metrics"]["precision"] for station in bad_stations])
condensed_te_metrics["bad"]["dtw_precision"] = np.mean([te_metrics[station]["dtw_metrics"]["precision"] for station in bad_stations])

all_stations = stations
condensed_te_metrics["all"] = {}
condensed_te_metrics["all"]["sta_lta_iou"] = np.mean([te_metrics[station]["sta_lta_metrics"]["iou"] for station in all_stations])
condensed_te_metrics["all"]["if_iou"] = np.mean([te_metrics[station]["if_metrics"]["iou"] for station in all_stations])
condensed_te_metrics["all"]["dtw_iou"] = np.mean([te_metrics[station]["dtw_metrics"]["iou"] for station in all_stations])
condensed_te_metrics["all"]["sta_lta_recall"] = np.mean([te_metrics[station]["sta_lta_metrics"]["recall"] for station in all_stations])
condensed_te_metrics["all"]["if_recall"] = np.mean([te_metrics[station]["if_metrics"]["recall"] for station in all_stations])
condensed_te_metrics["all"]["dtw_recall"] = np.mean([te_metrics[station]["dtw_metrics"]["recall"] for station in all_stations])
condensed_te_metrics["all"]["sta_lta_precision"] = np.mean([te_metrics[station]["sta_lta_metrics"]["precision"] for station in all_stations])
condensed_te_metrics["all"]["if_precision"] = np.mean([te_metrics[station]["if_metrics"]["precision"] for station in all_stations])
condensed_te_metrics["all"]["dtw_precision"] = np.mean([te_metrics[station]["dtw_metrics"]["precision"] for station in all_stations])


print(pd.DataFrame(condensed_te_metrics).T.round(2))

# for station in good_stations:
#     print()



tr_lower_conf_recall_dict = {}
te_lower_conf_recall_dict = {}
for station in stations:
    #tr_lower_conf_recall_dict[station] = {}
    lower_conf_flows, _,_ = extract_split_flows(flows,station,tr_start,tr_stop)
    sta_lta_detections = all_detections[station]["tr_detections"]["sta_lta"]
    if_detections = all_detections[station]["tr_detections"]["if"]
    dtw_detections = all_detections[station]["tr_detections"]["dtw"]
    tr_lower_conf_recall_dict[station] = compute_lower_conf_recall(sta_lta_detections, if_detections, dtw_detections, lower_conf_flows)

    lower_conf_flows, _ , _ = extract_split_flows(flows,station,te_start,te_stop)
    sta_lta_detections = all_detections[station]["te_detections"]["sta_lta"]
    if_detections = all_detections[station]["te_detections"]["if"]
    dtw_detections = all_detections[station]["te_detections"]["dtw"]
    te_lower_conf_recall_dict[station] = compute_lower_conf_recall(sta_lta_detections, if_detections, dtw_detections, lower_conf_flows)

print(f"\n===== Training Lower Confidence Recall =====\n")
df = pd.DataFrame(tr_lower_conf_recall_dict).T
overall = np.zeros(df.shape[1])
overall[0] = df["#_low_conf"].sum()
overall[1] = df["#_med_conf"].sum()
overall[2:] = df.iloc[:, 2:].mean(axis=0).values
df.loc["overall"] = overall
print(df.round(2))

print(f"\n===== Testing Lower Confidence Recall =====\n")
df = pd.DataFrame(te_lower_conf_recall_dict).T
overall = np.zeros(df.shape[1])
overall[0] = df["#_low_conf"].sum()
overall[1] = df["#_med_conf"].sum()
overall[2:] = df.iloc[:, 2:].mean(axis=0).values
df.loc["overall"] = overall
print(df.round(2))


print(f"\n===== Trigger Parameters =====\n")

trigger_params_tab = {}

filename = "../output/XP/if/segments/trigger_params.json"
with open(filename, "r") as f:
    if_trigger_params = json.load(f)
if_trigger_params = sorted(if_trigger_params, key=lambda d: stations.index(d['station']))

trigger_params_tab["if-onset"] = [i["onset_thres"] for i in if_trigger_params]
trigger_params_tab["if-offset"] = [i["offset_thres"] for i in if_trigger_params]

filename = "../output/XP/sta_lta/segments/trigger_params.json"
with open(filename, "r") as f:
    slta_trigger_params = json.load(f)
slta_trigger_params = sorted(slta_trigger_params, key=lambda d: stations.index(d['station']))

trigger_params_tab["slta-onset"] = [i["onset_thres"] for i in slta_trigger_params]
trigger_params_tab["slta-offset"] = [i["offset_thres"] for i in slta_trigger_params]
trigger_params_tab["slta-sw"] = [i["sw"]/100 for i in slta_trigger_params]
trigger_params_tab["slta-lw"] = [i["lw"]/100 for i in slta_trigger_params]

df = pd.DataFrame(trigger_params_tab).round(4)
df.index = stations

print(df)

print(f"\n===== Thresholds =====\n")
threshold_tab = {}

filename = "../output/XP/if/detections/threshold_params.json"

with open(filename, "r") as f:
    if_threshold_params = json.load(f)
if_threshold_params = sorted(if_threshold_params, key=lambda d: stations.index(d['station']))

threshold_tab["if-thres"] = [i["score_threshold"] for i in if_threshold_params]
threshold_tab["if-mdl"] = [i["minimum_detection_length"] for i in if_threshold_params]


filename = "../output/XP/dtw/detections/threshold_params.json"

with open(filename, "r") as f:
    dtw_threshold_params = json.load(f)
dtw_threshold_params = sorted(dtw_threshold_params, key=lambda d: stations.index(d['station']))

threshold_tab["dtw-thres"] = [i["score_threshold"] for i in dtw_threshold_params]
threshold_tab["dtw-mdl"] = [i["minimum_detection_length"] for i in dtw_threshold_params]

filename = "../output/XP/sta_lta/detections/threshold_params.json"

with open(filename, "r") as f:
    slta_threshold_params = json.load(f)
slta_threshold_params = sorted(slta_threshold_params, key=lambda d: stations.index(d['station']))

threshold_tab["slta-thres"] = [i["score_threshold"] for i in slta_threshold_params]
threshold_tab["slta-mdl"] = [i["minimum_detection_length"] for i in slta_threshold_params]

df = pd.DataFrame(threshold_tab).round(4)
df.index = stations

print(df)
