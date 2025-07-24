import pandas as pd
import numpy as np
import obspy
import pickle
import multiprocessing as mp
import tqdm as tqdm
from seismicif.datamod.loading_utils import (
    find_paths,
    preproc_flow_annotations,
    find_date_in_strings,
    extract_split_flows,extract_template, read_segment
)
from seismicif.datamod.preproc_utils import preproc_stream
from seismicif.metrics import iou
from seismicif.dtw import segment_dtw
from joblib import dump, load
from itertools import combinations


def find_fp_fn(df, gt):
    if len(df) == 0:
        FP = np.nan
        FN = gt

    else:
        I, U, _ = iou(df, gt)
        FP = df.iloc[np.where(np.append(I[0], np.diff(I)) == 0)[0], :]

        I, U, _ = iou(gt, df)
        FN = gt.iloc[np.where(np.append(I[0], np.diff(I)) == 0)[0], :]

    return FP, FN


def compare_pair(args):
    i, j, if_mod, paths, segments = args
    seg1 = read_segment(segments["start"][i], segments["stop"][i], paths)
    seg2 = read_segment(segments["start"][j], segments["stop"][j], paths)
    return segment_dtw(seg1, seg2, if_mod, paths)

def main():
    flows = preproc_flow_annotations(pd.read_csv("../catalogs/XP/flow_catalog.csv",index_col=0))

    IF_FP = {}
    STA_LTA_FP = {}
    DTW_FP = {}

    stations = ["ILL11","ILL12","ILL13","ILL18"]
    if_detections = {}
    sta_lta_detections = {}
    dtw_detections = {}

    for station in stations:
        if_detections[station] = pd.read_csv(f"../output/XP/if/detections/{station}.csv",index_col=0)
        sta_lta_detections[station] = pd.read_csv(f"../output/XP/sta_lta/detections/{station}.csv",index_col=0)

    stations = ["ILL11","ILL12","ILL13","ILL14","ILL16","ILL17","ILL18"]

    for station in stations:
        dtw_detections[station] = pd.read_csv(f"../output/XP/if_dtw/detections/{station}.csv",index_col=0)

    for station in ["ILL11","ILL12","ILL13","ILL18"]:
        _,_,station_flows = extract_split_flows(flows,station,2018,2022)
        IF_FP[station] = find_fp_fn(if_detections[station],station_flows)[0].reset_index(drop=True)
        STA_LTA_FP[station] = find_fp_fn(sta_lta_detections[station],station_flows)[0].reset_index(drop=True)

    for station in ["ILL11","ILL12","ILL13","ILL14","ILL16","ILL17","ILL18"]:
        _,_,station_flows = extract_split_flows(flows,station,2018,2022)
        DTW_FP[station] = find_fp_fn(dtw_detections[station],station_flows)[0].reset_index(drop=True)

    FP = preproc_flow_annotations(pd.read_csv("../catalogs/XP/confirmed_FP.csv"))

    confirmed_FP = {}

    for station in ["ILL11","ILL12","ILL13","ILL14","ILL16","ILL17","ILL18"]:
        confirmed_FP[station] = FP[FP["station"] == station].reset_index(drop=True)

    dtw_unconfirmed = {}
    for station in ["ILL11","ILL12","ILL13","ILL14","ILL16","ILL17","ILL18"]:
        FP = find_fp_fn(DTW_FP[station],confirmed_FP[station])[0]

        if FP is np.nan:
            continue

        if len(FP) == 0:
            continue

        dtw_unconfirmed[station] = FP

    FP = preproc_flow_annotations(pd.read_csv("../catalogs/XP/confirmed_FP.csv"))
    flows = flows[["2022-09-08" not in str(i) for i in flows["start"]]].reset_index(drop=True)
    ILL17_flows = flows[flows["station"] == "ILL17"].reset_index(drop=True)
    ILL17_FP = FP[FP["station"] == "ILL17"].reset_index(drop=True)
    FP_labels = np.array(ILL17_FP["Labels"])

    FP_labels[1] = "IN"
    FP_labels[2] = "N"
    FP_labels[3] = "N"
    FP_labels[5] = "N"
    FP_labels[6] = "N"
    FP_labels[9] = "IN"
    FP_labels[10] = "IN"
    FP_labels[11] = "AN"
    FP_labels[13] = "U"
    FP_labels[14] = "U"

    flow_labels = np.array(ILL17_flows["confidence"])
    flow_labels[flow_labels == "high"] = "DF"
    flow_labels[flow_labels == "med"] = "DF"
    flow_labels[pd.isna(ILL17_flows["Notes"])] = "DF"

    flow_labels[2]  = "SF"
    flow_labels[3]  = "CA"
    flow_labels[8]  = "SF"
    flow_labels[25]  = "SF"
    flow_labels[32]  = "TA"
    flow_labels[34]  = "CA"
    flow_labels[36]  = "CA"
    flow_labels[37]  = "CA"
    flow_labels[49]  = "RF"
    flow_labels[51]  = "SA"
    flow_labels[54]  = "RF"
    flow_labels[55]  = "CA"
    flow_labels[56]  = "CA"
    flow_labels[57]  = "SA"
    flow_labels[58]  = "RF"
    flow_labels[59]  = "CA"
    flow_labels[61]  = "CA"
    flow_labels[62]  = "SA"
    flow_labels[63]  = "CA"
    flow_labels[64]  = "SA"
    flow_labels[65]  = "CA"
    flow_labels[67]  = "CA"
    flow_labels[69]  = "CA"

    segments = pd.concat([ILL17_flows[["start","stop"]],ILL17_FP[["start","stop"]],dtw_unconfirmed["ILL17"][["start","stop"]]]).reset_index(drop=True)
    segments["labels"] = np.concatenate([flow_labels,FP_labels,np.repeat("?",dtw_unconfirmed["ILL17"].shape[0])])
    if_mod = load("../output/XP/if/models/ILL17.joblib")
    paths = find_paths("XP", "ILL17", "EHZ.D", 2018, 2022)

    args_list = [(i, j, if_mod, paths, segments) for i, j in combinations(range(segments.shape[0]), 2)]

    print("DTW")

    results = []
    with mp.Pool() as pool:
        for result in tqdm(pool.imap_unordered(compare_pair, args_list), total=len(args_list)):
            results.append(result)

    with open("../output/XP/dendrogram/ILL17.pkl", "wb") as f:
        pickle.dump(results, f)



if __name__ == "__main__":
    main()
