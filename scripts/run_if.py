"""
Script to train an isolation forest to seismic waveforms extracted from a given network, station, channel and specified training time period,
and compute time series of the corresponding anomaly scores over a specified evaluation time period.

Example Usage:
python run_if.py -network XP -station ILL11 -channel HHZ.D -tr_start 2018 -tr_end 2020 -eval_start 2018 -eval_end 2022

This script performs the following steps:
1. Parses command-line arguments for network, station, channel, training and evaluation time periods.
2. Finds the paths for the training and evaluation data.
3. Trains the isolation forest model if it doesn't already exist.
4. Computes anomaly scores using the trained model if they don't already exist.
5. Saves the trained model and computed scores to the specified output directories.
"""
import argparse
import os
import pandas as pd
import numpy as np

from seismicif.datamod.loading_utils import find_paths
from seismicif.isolation_forest import train_if, compute_scores
from joblib import dump, load

def main():
    parser = argparse.ArgumentParser(description="Train Isolation Forest and Compute Anomaly Scores")
    parser.add_argument("-network", type=str, required=True, help="Network to Train On.")
    parser.add_argument("-station", type=str, required=True, help="Station to Train On.")
    parser.add_argument("-channel", type=str, required=True, help="Channel to Train On.")
    parser.add_argument("-tr_start", type=int, required=True, help="Year to start Training")
    parser.add_argument("-tr_end", type=int, required=True, help="Year to end Training")
    parser.add_argument("-eval_start", type=int, required=True, help="Year to start Evaluation")
    parser.add_argument("-eval_end", type=int, required=True, help="Year to end Evaluation")

    args = parser.parse_args()

    try:
        tr_paths = find_paths(args.network, args.station, args.channel, args.tr_start, args.tr_end)
    except Exception as e:
        print(f"Error finding training paths: {e}")
        return

    model_path = f"../output/{args.network}/if/models/"

    if not os.path.isdir(model_path):
        print(f"Folder does not exist. Please create ../output/{args.network}/if/models/")
        return

    try:
        if_mod = load(f"../output/{args.network}/if/models/{args.station}.joblib")

    except (FileNotFoundError, OSError):
        print(f"{args.station}: Training IF")
        if_mod = train_if(tr_paths)
        dump(if_mod,f"../output/{args.network}/if/models/{args.station}.joblib")


    scores_path = f"../output/{args.network}/if/scores/"

    if not os.path.isdir(scores_path):
        print(f"Folder does not exist. Please create ../output/{args.network}/if/scores/")
        return

    try:
        eval_paths = find_paths(args.network, args.station, args.channel, args.eval_start, args.eval_end)
    except Exception as e:
        print(f"Error finding evaluation paths: {e}")
        return

    try:
        scores_df = pd.read_csv(f"../output/{args.network}/if/scores/{args.station}.csv",index_col=0)

    except (FileNotFoundError, OSError):
        print(f"{args.station}: Computing Scores")
        scores_df = compute_scores(eval_paths, if_mod)
        scores_df.to_csv(f"../output/{args.network}/if/scores/{args.station}.csv")


if __name__ == "__main__":
    main()
