import argparse
import os
import pandas as pd

from seismicif.datamod.loading_utils import preproc_flow_annotations

def main():
    parser = argparse.ArgumentParser(description="Perform DTW between segments and reference segments.")
    parser.add_argument("-network", type=str, required=True, help="Network where segments were extracted from.")
    parser.add_argument("-station", type=str, required=True, help="Station where segments were extracted from.")
    parser.add_argument("-channel", type=str, required=True, help="Channel of the segments.")
    # parser.add_argument("-tr_start", type=int, required=True, help="Year to start Training")
    # parser.add_argument("-tr_end", type=int, required=True, help="Year to end Training")
    # parser.add_argument("-eval_start", type=int, required=True, help="Year to start Evaluation")
    # parser.add_argument("-eval_end", type=int, required=True, help="Year to end Evaluation")

    args = parser.parse_args()

    segments_path = f"../output/{args.network}/if/segments/"

    if not os.path.isdir(segments_path):
        print(f"IF segments do not exist. Please generate and store in ../output/{args.network}/if/segments/")
        return

    else:
        print("Loaded IF segments")
        if_segments =  preproc_flow_annotations(pd.read_csv(f"{segments_path}/{args.station}.csv",index_col=0))

    print(if_segments.head())
    # try:
    #     tr_paths = find_paths(args.network, args.station, args.channel, args.tr_start, args.tr_end)
    # except Exception as e:
    #     print(f"Error finding training paths: {e}")
    #     return

    # model_path = f"../output/{args.network}/if/models/"

    # if not os.path.isdir(model_path):
    #     print(f"Folder does not exist. Please create ../output/{args.network}/if/models/")
    #     return

    # try:
    #     if_mod = load(f"../output/{args.network}/if/models/{args.station}.joblib")

    # except (FileNotFoundError, OSError):
    #     print(f"{args.station}: Training IF")
    #     if_mod = train_if(tr_paths)
    #     dump(if_mod,f"../output/{args.network}/if/models/{args.station}.joblib")


    # scores_path = f"../output/{args.network}/if/scores/"

    # if not os.path.isdir(scores_path):
    #     print(f"Folder does not exist. Please create ../output/{args.network}/if/scores/")
    #     return

    # try:
    #     eval_paths = find_paths(args.network, args.station, args.channel, args.eval_start, args.eval_end)
    # except Exception as e:
    #     print(f"Error finding evaluation paths: {e}")
    #     return

    # try:
    #     scores_df = pd.read_csv(f"../output/{args.network}/if/scores/{args.station}.csv",index_col=0)

    # except (FileNotFoundError, OSError):
    #     print(f"{args.station}: Computing Scores")
    #     scores_df = compute_scores(eval_paths, if_mod)
    #     scores_df.to_csv(f"../output/{args.network}/if/scores/{args.station}.csv")


if __name__ == "__main__":
    main()
