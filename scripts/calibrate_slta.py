import argparse
import os
import pandas as pd
import numpy as np
import obspy
import json

from seismicif.datamod.loading_utils import find_paths, preproc_flow_annotations, extract_split_flows,remove_duplicate_traces, remove_overlaps, find_date_in_strings
from seismicif.datamod.preproc_utils import preproc_stream
from datetime import timedelta
from tqdm import tqdm
from itertools import product
from seismicif.sta_lta import slta_compute_iou

def main():
    parser = argparse.ArgumentParser(description="Calibrate SLTA trigger parameters to initial catalog.")
    parser.add_argument("-network", type=str, required=True, help="Network containing the waveforms.")
    parser.add_argument("-station", type=str, required=True, help="Station to calibrate trigger to.")
    parser.add_argument("-channel", type=str, required=True, help="Channel to collect waveforms from.")
    parser.add_argument("-tr_start", type=int, required=True, help="Year to start training")
    parser.add_argument("-tr_end", type=int, required=True, help="Year to end training")

    sampling_rate = 100
    lw_grid = 5000 * np.power(2.0, np.arange(-5, 6))
    sw_grid = 0.003125 * np.power(2.0, np.arange(11))
    onset_grid = 0.1875 * np.power(2.0, np.arange(11))
    offset_grid = 0.00390625 * np.power(2.0, np.arange(11))

    args = parser.parse_args()

    try:
        paths = paths = find_paths(args.network, args.station, args.channel, args.tr_start, args.tr_end)

    except Exception as e:
        print("Cannot find paths for station in the given channel and time range.")
        return

    try:
        flows = preproc_flow_annotations(pd.read_csv("../catalogs/XP/initial_catalog.csv"))
        _, _, all_flows = extract_split_flows(flows,args.station,args.tr_start,args.tr_end)

    except Exception as e:
        print("Cannot find initial catalog to calibrate trigger to.")
        return

    slta_iou_folder = f"../output/{args.network}/sta_lta/iou_tables/"

    if not os.path.isdir(slta_iou_folder):
        print(f"Folder does not exist. Please create {slta_iou_folder}")
        return

    slta_segment_folder = f"../output/{args.network}/sta_lta/segments/"

    if not os.path.isdir(slta_segment_folder):
        print(f"Folder does not exist. Please create {slta_segment_folder}")
        return

    try:
        iou_table = np.load(f"{slta_iou_folder}/{args.station}.npy")
        print(f"IoU table already exists in {slta_iou_folder}. Delete to recompute.")

    except:
        print(f"{args.station}: Extracting Miniseed Recordings")

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
            flow_st = obspy.read(find_date_in_strings(paths, dates[i])[0])
            flow_st = remove_duplicate_traces(flow_st)
            flow_st = remove_overlaps(flow_st)
            preproc_stream(flow_st)

            prev_day_st = obspy.read(find_date_in_strings(paths, prev_days[i])[0])
            prev_day_st = remove_duplicate_traces(prev_day_st)
            prev_day_st = remove_overlaps(prev_day_st)
            preproc_stream(prev_day_st)

            st.append({"res_tr": prev_day_st[-1], "st": flow_st})

            iou_table = np.zeros([11, 11, 11, 11])
            iou_table[:] = np.nan
            ic, jc, kc, lc = 5, 5, 5, 5
            stop = False

        while not stop:
            print(f"{args.station}: Performing Local Search")
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
                sw = int(sampling_rate * sw_grid[i] * lw_grid[j])
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

        np.save(f"{slta_iou_folder}/{args.station}.npy", iou_table)


    iou_table[np.isnan(iou_table)] = 0
    iou_table[np.isnan(iou_table)] = 0
    ic, jc, kc, lc = np.unravel_index(np.argmax(iou_table), iou_table.shape)

    lw = int(sampling_rate * lw_grid[jc])
    sw = int(sampling_rate * sw_grid[ic] * lw_grid[jc])
    onset_thres, offset_thres = onset_grid[kc], offset_grid[lc]


    trigger_params_path = f"../output/{args.network}/sta_lta/segments/trigger_params.json"

    if not os.path.isfile(trigger_params_path):
        trigger_params = [{"station": args.station,"onset_thres": np.nan,"offset_thres": np.nan, "sw": sw, "lw": lw}]

    else:
        with open(trigger_params_path, "r") as f:
            trigger_params = json.load(f)

    if args.station not in (d["station"] for d in trigger_params):
        trigger_params.append({"station": args.station,"onset_thres": np.nan,"offset_thres": np.nan, "sw": sw, "lw": lw})

    entry = next((d for d in trigger_params if d.get("station") == args.station), None)

    if entry["onset_thres"] is not np.nan and entry["offset_thres"] is not np.nan:
        print(f"{args.station}: STA-LTA trigger already calibrated. Delete to recalibrate.")
        return

    entry["onset_thres"] = onset_thres
    entry["offset_thres"] = offset_thres
    entry["sw"] = sw
    entry["lw"] = lw

    with open(trigger_params_path, "w") as f:
        json.dump(trigger_params, f)


if __name__ == "__main__":
    main()
