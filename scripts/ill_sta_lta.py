import obspy
import os
import numpy as np
import pandas as pd
import json
from multiprocessing import Pool, cpu_count
from functools import partial
from itertools import product
from tqdm import tqdm
from datetime import timedelta
from seismicif.datamod.loading_utils import (
    preproc_flow_annotations,
    find_date_in_strings,
    remove_duplicate_traces,
    remove_overlaps,
    find_paths,
    extract_split_flows,
    extract_valid_segments,
)
from seismicif.datamod.preproc_utils import preproc_stream
from seismicif.sta_lta import slta_compute_iou, slta_detections_from_paths
from seismicif.metrics import est_thresholds

tr_start, tr_stop, te_start, te_stop = 2018, 2020, 2021, 2022
network = "XP"
stations = ["ILL11","ILL12","ILL13","ILL14","ILL15","ILL16","ILL17","ILL18"]

sampling_rate = 100
lw_grid = 5000 * np.power(2.0, np.arange(-5, 6))
sw_grid = 0.003125 * np.power(2.0, np.arange(11))
onset_grid = 0.1875 * np.power(2.0, np.arange(11))
offset_grid = 0.00390625 * np.power(2.0, np.arange(11))

for station in stations:
    channel = "EHZ.D"
    if station == "ILL11": channel = "HHZ.D"

    tr_paths, te_paths = find_paths(network, station, channel, tr_start, tr_stop), find_paths("XP", station, channel, te_start, te_stop)
    flows = preproc_flow_annotations(pd.read_csv("../catalogs/initial_catalog.csv",index_col=0))
    lower_conf_flows, high_conf_flows, all_flows = extract_split_flows(flows,station,tr_start,tr_stop)

    print(f"{station}: Calibrating Trigger")

    try:
        iou_table = np.load(f"../output/sta_lta_grids/{station}.npy")

    except (FileNotFoundError, OSError):
        print(f"{station}: Extracting Miniseed Recordings")

        dates = []
        for i in range(all_flows.shape[0]):
            dates.append(all_flows["start"].iloc[i].date)
            dates.append(all_flows["stop"].iloc[i].date)

        dates = np.array(dates)
        dates = np.unique(dates)

        prev_days = []

        for dt in dates:
            prev_days.append(dt - timedelta(days=1))

        prev_days = np.array(prev_days)

        st = []

        for i in tqdm(range(dates.shape[0])):
            flow_st = obspy.read(find_date_in_strings(tr_paths, dates[i])[0])
            flow_st = remove_duplicate_traces(flow_st)
            flow_st = remove_overlaps(flow_st)
            preproc_stream(flow_st)

            prev_day_st = obspy.read(find_date_in_strings(tr_paths, prev_days[i])[0])
            prev_day_st = remove_duplicate_traces(prev_day_st)
            prev_day_st = remove_overlaps(prev_day_st)
            preproc_stream(prev_day_st)

            st.append({"res_tr": prev_day_st[-1], "st": flow_st})

        iou_table = np.zeros([11, 11, 11, 11])
        iou_table[:] = np.nan
        ic, jc, kc, lc = 5, 5, 5, 5
        stop = False

        while not stop:
            print(f"{station}: Performing Local Search")
            stop = True

            i_range = range(max(0, ic - 1), min(10, ic + 2))
            j_range = range(max(0, jc - 1), min(10, jc + 2))
            k_range = range(max(0, kc - 1), min(10, kc + 2))
            l_range = range(max(0, lc - 1), min(10, lc + 2))

            for i, j, k, l in tqdm(product(i_range, j_range, k_range, l_range), total=len(i_range)*len(j_range)*len(k_range)*len(l_range)):
                if not np.isnan(iou_table[i, j, k, l]):
                    continue

                stop = False
                lw = int(sampling_rate * lw_grid[j])
                sw = sampling_rate * sw_grid[i] * lw_grid[j]
                onset_thres, offset_thres = onset_grid[k], offset_grid[l]

                iou_table[i, j, k, l] = slta_compute_iou(st, all_flows, sw, lw, onset_thres, offset_thres)

            #all the positions around the current one has been filled.
            if stop:
                break

            temp_table = iou_table.copy()
            temp_table[np.isnan(temp_table)] = 0

            # check if we found a better solution than the current...otherwise stop
            if np.max(temp_table) > iou_table[ic, jc, kc, lc]:
                stop = False
                ic, jc, kc, lc = np.unravel_index(np.argmax(temp_table), temp_table.shape)

            else:
                stop = True

        np.save(f"../output/sta_lta_grids/{station}.npy", iou_table)

    print(f"{station}: Extracting Trigger Segments")

    try:
        sta_lta_segments = pd.read_csv(f"../output/sta_lta_segments/{station}.csv", index_col=0)

    except (FileNotFoundError, OSError):
        iou_table[np.isnan(iou_table)] = 0
        iou_table[np.isnan(iou_table)] = 0
        ic, jc, kc, lc = np.unravel_index(np.argmax(iou_table), iou_table.shape)

        lw = int(sampling_rate * lw_grid[jc])
        sw = sampling_rate * sw_grid[ic] * lw_grid[jc]
        onset_thres, offset_thres = onset_grid[kc], offset_grid[lc]

        output_dict = {"station":station,"onset_thres":onset_thres,"offset_thres":offset_thres,"sw":sw,"lw":lw}
        filename = "../output/sta_lta_grids/trigger_params.json"

        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(output_dict)

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

        sta_lta_segments = slta_detections_from_paths(np.concatenate([tr_paths,te_paths]), sw, lw, onset_thres, offset_thres)
        sta_lta_segments.to_csv(f"../output/sta_lta_segments/{station}.csv")

    print(f"{station}: Generating Detections")

    try:
        pd.read_csv(f"../output/sta_lta_detections/{station}.csv")

    except (FileNotFoundError, OSError):
        flows = preproc_flow_annotations(pd.read_csv("../catalogs/calibration_catalog.csv",index_col=0))
        lower_conf_flows, high_conf_flows, _ = extract_split_flows(flows,station,tr_start,tr_stop)
        tr_start_UTC = obspy.UTCDateTime(f"{tr_start}-01-01")
        tr_stop_UTC = obspy.UTCDateTime(f"{tr_stop}-12-31T23:59:59.999999")
        tr_segments =  sta_lta_segments[(sta_lta_segments["start"] >  tr_start_UTC) & (sta_lta_segments["stop"] < tr_stop_UTC)].reset_index(drop=True)

        valid_segments = extract_valid_segments(tr_segments,lower_conf_flows,high_conf_flows)
        _, min_len, score_thres = est_thresholds(valid_segments, high_conf_flows)

        filename = "../output/sta_lta_detections/threshold_params.json"

        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append({"station":station,"min_len":min_len,"score_thres":score_thres})

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

        detections = sta_lta_segments.copy()
        start_times = np.array([pd.to_datetime(str(i)) for i in detections["start"]])
        end_times = np.array([pd.to_datetime(str(i)) for i in detections["stop"]])
        det_lens = np.array([i.total_seconds() for i in end_times - start_times])
        detections = detections.assign(det_lens=det_lens)
        detections = detections[
            (detections["scores"] > score_thres) & (detections["det_lens"] > min_len)
        ].reset_index(drop=True)
        detections.to_csv(f"../output/sta_lta_detections/{station}.csv")

        break
