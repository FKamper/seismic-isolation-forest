import os
import numpy as np
import pandas as pd
import obspy
import json
from joblib import dump, load
from tqdm import tqdm
from seismicif.datamod.loading_utils import find_paths, preproc_flow_annotations, extract_split_flows, extract_valid_segments, extract_template
from seismicif.datamod.trigger_utils import stream_trigger_detections
from seismicif.dtw import template_dtw
from seismicif.metrics import iou, est_thresholds
from multiprocessing import Pool
from functools import partial

stations = ["ILL11","ILL12","ILL13","ILL14","ILL15","ILL16","ILL17","ILL18"]
tr_start, tr_stop, te_start, te_stop = 2018, 2020, 2021, 2022
network = "XP"

def distribute_template_dtw(i, segments, tr_paths, te_paths, templates, if_mod):
    return template_dtw(
        segments["start"][i],
        segments["stop"][i],
        np.concatenate([tr_paths, te_paths]),
        templates,
        if_mod,
    )

if __name__ == "__main__":

    for station in stations:
        channel = "EHZ.D"
        if station == "ILL11": channel = "HHZ.D"

        tr_paths, te_paths = find_paths(network, station, channel, tr_start, tr_stop), find_paths("XP", station, channel, te_start, te_stop)

        if_mod = load(f"../output/XP/if/models/{station}.joblib")
        segments = preproc_flow_annotations(pd.read_csv(f"../output/XP/if/segments/{station}.csv",index_col=0))

        flows = preproc_flow_annotations(pd.read_csv("../XP/catalogs/initial_catalog.csv",index_col=0))
        lower_conf_flows, high_conf_flows, _ = extract_split_flows(flows,station,tr_start,tr_stop)

        print(f"{station}: Extracting High-confidence Templates")

        loaded = False
        try:
            dtw_dists = np.load(f"../output/XP/dtw/distances/{station}.npy")
            loaded = True

        except (FileNotFoundError, OSError):
            templates = []
            for i in tqdm(range(high_conf_flows.shape[0])):
                t0 = high_conf_flows["start"][i]
                t1 = high_conf_flows["stop"][i]
                templates.append(extract_template(t0, t1, if_mod, tr_paths))
            templates = np.array(templates)

        print(f"{station}: Performing Template DTW")

        if not loaded:
            fn = partial(distribute_template_dtw, segments=segments, tr_paths=tr_paths,
                        te_paths=te_paths, templates=templates, if_mod=if_mod)

            with Pool() as pool:
                dtw_dists = list(tqdm(pool.imap(fn, range(segments.shape[0])), total=segments.shape[0]))

            dtw_dists = np.array(dtw_dists)
            np.save(f"../output/XP/dtw/distances/{station}.npy", dtw_dists)

        print(f"{station}: Generating Segments")

        try:
            dtw_segments = preproc_flow_annotations(pd.read_csv(f"../output/XP/dtw/segments/{station}.csv",index_col=0))

        except (FileNotFoundError, OSError):
            dtw_scores = []

            for i in range(dtw_dists.shape[0]):
                intersection, _, _ = iou(
                    high_conf_flows, segments.loc[[i]].reset_index(drop=True)
                )
                high_conf_overlap = np.append(intersection[0], np.diff(intersection)) > 0
                dtw_scores.append(np.mean(dtw_dists[i, ~high_conf_overlap]))

            dtw_segments = segments.copy()
            dtw_segments["scores"] = dtw_scores
            dtw_segments = dtw_segments.iloc[np.argsort(dtw_segments["scores"])].reset_index(
                drop=True
            )

            dtw_segments.to_csv(f"../output/XP/dtw/segments/{station}.csv")


        print(f"{station}: Generating Detections")

        try:
            pd.read_csv(f"../output/XP/dtw/detections/{station}.csv")

        except (FileNotFoundError, OSError):
            flows = preproc_flow_annotations(pd.read_csv("../XP/catalogs/calibration_catalog.csv",index_col=0))
            lower_conf_flows, high_conf_flows, _ = extract_split_flows(flows,station,tr_start,tr_stop)
            tr_start_UTC = obspy.UTCDateTime(f"{tr_start}-01-01")
            tr_stop_UTC = obspy.UTCDateTime(f"{tr_stop}-12-31T23:59:59.999999")
            tr_segments =  dtw_segments[(dtw_segments["start"] >  tr_start_UTC) & (dtw_segments["stop"] < tr_stop_UTC)].reset_index(drop=True)

            valid_segments = extract_valid_segments(tr_segments,lower_conf_flows,high_conf_flows)
            _, min_len, score_thres = est_thresholds(valid_segments, high_conf_flows,mode="upper")

            filename = "../output/XP/dtw/detections/threshold_params.json"

            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                with open(filename, "r") as f:
                    data = json.load(f)
            else:
                data = []

            data.append({"station":station,"min_len":min_len,"score_thres":score_thres})

            with open(filename, "w") as f:
                json.dump(data, f, indent=4)

            detections = dtw_segments.copy()
            start_times = np.array([pd.to_datetime(str(i)) for i in detections["start"]])
            end_times = np.array([pd.to_datetime(str(i)) for i in detections["stop"]])
            det_lens = np.array([i.total_seconds() for i in end_times - start_times])
            detections = detections.assign(det_lens=det_lens)
            detections = detections[
                (detections["scores"] < score_thres) & (detections["det_lens"] > min_len)
            ].reset_index(drop=True)
            detections.to_csv(f"../output/XP/dtw/detections/{station}.csv")
