import argparse
import os
import numpy as np
import pandas as pd
import obspy
import json
import sys

from joblib import dump, load
from seismicif.datamod.loading_utils import preproc_flow_annotations, extract_template
from seismicif.dtw import template_dtw

def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform template DTW for target station."
    )

    parser.add_argument(
        "--network", type=str, required=True, help="Network where station is located."
    )

    parser.add_argument(
        "--station",
        type=str,
        required=True,
        help="Target station.",
    )

    parser.add_argument(
        "--channel",
        type=str,
        required=True,
        help="Target channel.",
    )

    parser.add_argument(
        "--tr_start",
        type=int,
        required=True,
        help="Training start year.",
    )

    parser.add_argument(
        "--tr_stop",
        type=int,
        required=True,
        help="Training end year.",
        )

    parser.add_argument(
        "--te_start",
        type=int,
        required=True,
        help="Testing start year.",
    )

    parser.add_argument(
        "--te_stop",
        type=int,
        required=True,
        help="Testing end year.",
        )

    return parser.parse_args()


def main():
    args = parse_args()

    paths = []
    for i in range(args.tr_start,args.tr_stop+1):
        folder = f"../data/{args.network}/{i}/{args.station}/{args.channel}/"
        stream_paths = [folder + f for f in os.listdir(folder) if not f.startswith("._")]
        paths.append(np.array(sorted(stream_paths, key=lambda f: int(f.rsplit(".", 1)[-1]))))


    segments = preproc_flow_annotations(pd.read_csv(f"../output/if_segments/{args.station}.csv",index_col=0))
    if_mod = load(f"../output/if_models/{args.station}.joblib")
    tr_start = obspy.UTCDateTime(f"{args.tr_start}-01-01")
    tr_end = obspy.UTCDateTime(f"{args.tr_stop}-12-31")

    flows = preproc_flow_annotations(pd.read_csv("../catalogs/initial_catalog.csv",index_col=0))
    flows = flows[(flows["confidence"] != "earthquake")]
    flows = flows[(flows["confidence"] != "rockfall")]
    flows = flows[(flows["confidence"] != "quarrywork")]
    flows = flows[flows["station"] == args.station].reset_index(drop=True)
    flows = flows[(flows["start"] > tr_start) & (flows["stop"] < tr_end)].reset_index(
        drop=True
    )
    high_conf_flows = flows[flows["confidence"] == "1"].reset_index(drop=True)

    templates = []
    for i in range(high_conf_flows.shape[0]):
        t0 = high_conf_flows["start"][i]
        t1 = high_conf_flows["stop"][i]
        templates.append(extract_template(t0, t1, if_mod, np.concatenate(paths)))
    templates = np.array(templates)

    for i in range(args.te_start,args.te_stop+1):
        folder = f"../data/{args.network}/{i}/{args.station}/{args.channel}/"
        stream_paths = [folder + f for f in os.listdir(folder) if not f.startswith("._")]
        paths.append(np.array(sorted(stream_paths, key=lambda f: int(f.rsplit(".", 1)[-1]))))

    try:
        dtw_dists = np.load(f"../output/dtw_distances/{args.station}.npy")

    except (FileNotFoundError, OSError):
        dtw_dists = []
        for i in range(segments.shape[0]):
            dtw_dists.append(
                template_dtw(
                    segments["start"][i],
                    segments["stop"][i],
                    np.concatenate(paths),
                    templates,
                    if_mod,
                )
            )
        dtw_dists = np.array(dtw_dists)
        np.save(f"../output/dtw_distances/{args.station}.npy", dtw_dists)







if __name__ == "__main__":
    main()
