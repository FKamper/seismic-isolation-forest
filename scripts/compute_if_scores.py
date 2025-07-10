import argparse
import os
import numpy as np
import pandas as pd

from joblib import dump, load
from seismicif.isolation_forest import compute_scores

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract anomaly scores time series from fitted model."
    )
    parser.add_argument(
        "--network", type=str, required=True, help="Network where station is located."
    )
    parser.add_argument(
        "--station",
        type=str,
        required=True,
        help="Station to compute the anomaly scores for.",
    )

    parser.add_argument(
        "--channel",
        type=str,
        required=True,
        help="Channel from which to read the seismic waveforms.",
    )

    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Start year of time series.",
    )

    parser.add_argument(
            "--stop",
            type=int,
            required=True,
            help="Final year  of time series.",
        )

    return parser.parse_args()


def main():
    args = parse_args()

    paths = []

    for i in range(args.start,args.stop+1):
        folder = f"../data/{args.network}/{i}/{args.station}/{args.channel}/"
        stream_paths = [folder + f for f in os.listdir(folder) if not f.startswith("._")]
        paths.append(np.array(sorted(stream_paths, key=lambda f: int(f.rsplit(".", 1)[-1]))))


    if_mod = load(f"../output/if_models/{args.station}.joblib")

    try:
        pd.read_csv(f"../output/if_scores/{args.station}.csv")

    except (FileNotFoundError, OSError):
        scores_df = compute_scores(np.concatenate(paths), if_mod)
        scores_df.to_csv(f"../output/if_scores/{args.station}.csv")





if __name__ == "__main__":
    main()
