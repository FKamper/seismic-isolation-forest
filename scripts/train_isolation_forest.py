import argparse
import os
import numpy as np

from joblib import dump, load
from seismicif.isolation_forest import train_if

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the isolation forest model."
    )
    parser.add_argument(
        "--network", type=str, required=True, help="Network where station is located/"
    )
    parser.add_argument(
        "--station",
        type=str,
        required=True,
        help="Station to train model to.",
    )

    parser.add_argument(
        "--channel",
        type=str,
        required=True,
        help="Channel from which to read the seismic waveforms/",
    )

    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Year to start training.",
    )

    parser.add_argument(
            "--stop",
            type=int,
            required=True,
            help="Final year for training.",
        )

    return parser.parse_args()


def main():
    args = parse_args()

    paths = []

    for i in range(args.start,args.stop+1):
        folder = f"../data/{args.network}/{i}/{args.station}/{args.channel}/"
        stream_paths = [folder + f for f in os.listdir(folder) if not f.startswith("._")]
        paths.append(np.array(sorted(stream_paths, key=lambda f: int(f.rsplit(".", 1)[-1]))))

    try:
        if_mod = load(f"../output/if_models/{args.station}.joblib")

    except (FileNotFoundError, OSError):

        if_mod = None

        for stream_paths in paths:
            if_mod = train_if(stream_paths,if_mod=if_mod)

        dump(if_mod,f"../output/if_models/{args.station}.joblib")





if __name__ == "__main__":
    main()
