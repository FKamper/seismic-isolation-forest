import argparse
import os
import pandas as pd
import numpy as np
import obspy
import json

from seismicif.datamod.loading_utils import preproc_flow_annotations, extract_split_flows
from seismicif.metrics import iou, est_thresholds, extract_valid_segments

def main():
    parser = argparse.ArgumentParser(description="Generate Detections from scored Segments")
    parser.add_argument("-network", type=str, required=True, help="Network containing the segments.")
    parser.add_argument("-station", type=str, required=True, help="Station containing the segements.")
    parser.add_argument("-method",type=str, required=True, help="Method used to score segments. Either if, sta_lta or dtw.")
    parser.add_argument("-tr_start", type=int, required=True, help="Start year of training.")
    parser.add_argument("-tr_end", type=int, required=True, help="End year of training.")
    parser.add_argument("-eval_start", type=int, required=True, help="Year to start evaluation")
    parser.add_argument("-eval_end", type=int, required=True, help="Year to end evaluation")

    args = parser.parse_args()

    segments_path = f"../output/{args.network}/{args.method}/segments/{args.station}.csv"

    if not os.path.isfile(segments_path):
        print(f"Segments do not exist. Please generate and store in ..{segments_path}")
        return

    else:
        segments =  preproc_flow_annotations(pd.read_csv(segments_path, index_col=0))

    detections_path = f"../output/{args.network}/{args.method}/detections/"

    if not os.path.isdir(detections_path):
        print(f"Detections folder does not exist. Please create ../output/{args.network}/{args.method}/detections/")
        return

    thres_params_path = f"../output/{args.network}/{args.method}/detections/threshold_params.json"

    if not os.path.isfile(thres_params_path):
        thres_params = [{"station": args.station,"score_threshold": np.nan,"minimum_detection_length": np.nan,"best_iou": np.nan}]

    else:
        with open(thres_params_path, "r") as f:
            thres_params = json.load(f)

    if args.station not in (d["station"] for d in thres_params):
        thres_params.append({"station": args.station,"score_threshold": np.nan,"minimum_detection_length": np.nan,"best_iou": np.nan})


    entry = next((d for d in thres_params if d.get("station") == args.station), None)

    if entry["score_threshold"] is not np.nan and entry["minimum_detection_length"] is not np.nan:
        print(f"{args.station}: Thresholds already calibrated with score threshold {entry['score_threshold']} and minimum detection length {entry['minimum_detection_length']}. Delete to recalibrate.")

    else:
        catalog_path = f"../catalogs/{args.network}/"

        tr_start = obspy.UTCDateTime(args.tr_start, 1, 1)
        tr_end   = obspy.UTCDateTime(args.tr_end, 12, 31, 23, 59, 59)

        try:
            flows = preproc_flow_annotations(pd.read_csv(f"{catalog_path}/calibration_catalog.csv"))
            lower_conf_flows, high_conf_flows, _ = extract_split_flows(flows,args.station,args.tr_start,args.tr_end)
            tr_segments =  segments[(segments["start"] >  tr_start) & (segments["stop"] < tr_end)].reset_index(drop=True)
            valid_segments = extract_valid_segments(tr_segments,lower_conf_flows,high_conf_flows)

            if args.method == "dtw":
                mode = "upper"
            else:
                mode = "lower"

            entry["best_iou"], entry["minimum_detection_length"], entry["score_threshold"] = est_thresholds(valid_segments, high_conf_flows,mode=mode)
            with open(thres_params_path, "w") as f:
                json.dump(thres_params, f, indent=4)

        except (FileNotFoundError, OSError):
            print(f"No calibration catalog found for network {args.network}.")
            return

    detections = segments.copy()
    start_times = np.array([pd.to_datetime(str(i)) for i in detections["start"]])
    end_times = np.array([pd.to_datetime(str(i)) for i in detections["stop"]])
    det_lens = np.array([i.total_seconds() for i in end_times - start_times])
    detections = detections.assign(det_lens=det_lens)

    if args.method == "dtw":
        detections = detections[
                (detections["scores"] < entry["score_threshold"]) & (detections["det_lens"] > entry["minimum_detection_length"])
            ].reset_index(drop=True)

    else:
        detections = detections[
                (detections["scores"] > entry["score_threshold"]) & (detections["det_lens"] > entry["minimum_detection_length"])
            ].reset_index(drop=True)

    detections.to_csv(f"{detections_path}/{args.station}.csv")



if __name__ == "__main__":
    main()
