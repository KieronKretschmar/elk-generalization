import argparse
from pathlib import Path
import os
import pandas as pd
from distutils.util import strtobool
from itertools import combinations
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from elk_utils import aggregate_segments, DiversifyTrainingConfig


def earliest_informative_layer_index(aurocs_per_layer, metric):
    max_auroc = max(aurocs_per_layer)
    informative_layers = [i for i, auroc in enumerate(aurocs_per_layer) if auroc - 0.5 >= 0.95 * (max_auroc - 0.5)]
    if len(informative_layers):
        earliest_informative_layer = informative_layers[0]
    else:
        earliest_informative_layer = int(len(aurocs_per_layer)/2)
    return earliest_informative_layer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize the test results from diversify experiments regarding transfer performance of probes."
    )
    parser.add_argument("--data-dir", type=str, help="Path to the directory containing directories for each dataset")
    parser.add_argument("--models", nargs="+", type=str, help="List of model names.")
    parser.add_argument("--reporters", type=str, nargs="+", default="lr", help="Which reporters to use.")
    parser.add_argument("--metric", type=str, choices=["auroc", "acc"], default="auroc", help="Metric to use.")
    parser.add_argument("--label-col", type=str, choices=["labels", "objective_labels", "quirky_labels"], default="objective_labels", help="Which label to use for the metric.")
    parser.add_argument("--save-csv-path", type=Path, help="Path to save the dataframe as csv.")
    parser.add_argument(
        "--training-datasets",
        help="Names of directories in data-dir to be used for training",
        type=str,
        nargs="+",
        default=["got/cities"]
    )
    parser.add_argument(
        "--eval-datasets",
        help="Names of directories in data-dir to be used for evaluating",
        type=str,
        nargs="+",
        default=["got/cities"]
    )
    parser.add_argument("--max-n-train-datasets", help="Number of datasets unionized over to serve as training data", type=int, default=1)
    parser.add_argument("--train-examples", type=int, default=4096)

    debug = False
    if debug:
        from argparse import Namespace
        print("DEBUGGING WITH HARDCODED ARGS!")
        args = argparse.Namespace(
            data_dir = Path("./experiments/diversify"),
            models = ["EleutherAI/pythia-410M"],
            reporters = ["ccs", "lr"],
            metric="auroc",
            label_col = "labels",
            save_csv_path="diversify_debug.csv",
            training_datasets = ["got/cities", "got/larger_than"],
            eval_datasets = ["got/cities", "got/larger_than"],
            max_n_train_datasets = 1,
            train_examples = 1096,
            )
    else:
        args = parser.parse_args()

    print("Args:")
    print(args)

    data_dir = Path(args.data_dir)
    metric_fn = {
        "auroc": roc_auc_score,
        "acc": lambda gt, logodds: accuracy_score(gt, logodds > 0),
    }[args.metric]

    # # Initialize all training descriptors based on first model and dataset, assuming they are the same for others
    # # Expected (results) data structure: root_dir/<eval_dataset>/<model>/<train|test>/<training_identifier>/<reporter>_log_odds.pt
    # first_model_dir = data_dir / args.eval_datasets[0] / args.models[0] / "test"
    # all_training_cfgs = []
    # for directory in os.listdir(first_model_dir):
    #     if os.path.isdir(first_model_dir / directory):
    #         try:
    #             all_training_cfgs.append(DiversifyTrainingConfig.from_descriptor(directory))
    #         except Exception as e:
    #             print(f"Skipping directory {directory} for summary because it can't be parsed as a config.")

    # # Sort by number of training datasets used, and secondarily by descriptor
    # all_training_cfgs.sort(key=lambda cfg: (len(cfg.training_datasets), cfg.descriptor()))

    # Initialize all training descriptors from args
    all_training_cfgs = []
    for n_train_datasets in range(1, args.max_n_train_datasets + 1):
        for training_datasets in combinations(args.training_datasets, r=n_train_datasets):
            all_training_cfgs.append(DiversifyTrainingConfig(training_datasets=training_datasets, n_training_samples=args.train_examples))

    # Test on all splits that were trained on
    df = pd.DataFrame()
    for training_cfg in all_training_cfgs:
        train_desc = training_cfg.descriptor()
        for eval_dataset in args.eval_datasets:
            for model in args.models:
                rows = []
                eval_dir = data_dir / eval_dataset / model / "test"
                labels = torch.load(eval_dir / f"{args.label_col}.pt", map_location="cpu").int()

                # Add lm performance if available
                # (we don't expect it to be available for non-"choice" datasets)
                lm_log_odds_path = eval_dir / "lm_log_odds.pt"
                if lm_log_odds_path.exists():
                    lm_log_odds = torch.load(lm_log_odds_path, map_location="cpu")
                    # lm performance is independent of the probe's and their training, but we add it for each training config for convenience
                    rows.append({
                        "model": model,
                        "reporter": "lm",
                        "train_desc": train_desc,
                        "n_training_samples": training_cfg.n_training_samples,
                        "n_train_datasets": len(training_cfg.training_datasets),
                        "eval_dataset": eval_dataset,
                        "auroc": roc_auc_score(labels, lm_log_odds),
                        "accuracy": accuracy_score(labels, lm_log_odds > 0),
                    })

                for reporter in args.reporters:
                    aurocs_per_layer = []
                    if reporter == "random":
                        random_aurocs = torch.load(eval_dir / train_desc / f"random_aucs_against_labels.pt", map_location="cpu")
                        aurocs_per_layer = [auroc["mean"] for auroc in random_aurocs]
                        # Accuracies are currently not available for random reporters
                        accs_per_layer = [np.nan for auroc in random_aurocs]
                    else:
                        reporter_log_odds = torch.load(eval_dir / train_desc / f"{reporter}_log_odds.pt", map_location="cpu")
                        aurocs_per_layer = [roc_auc_score(labels, layer_log_odds.float().numpy()) for layer_log_odds in reporter_log_odds]
                        accs_per_layer = [accuracy_score(labels, layer_log_odds > 0) for layer_log_odds in reporter_log_odds]


                    eil = earliest_informative_layer_index(aurocs_per_layer, args.metric)
                    rows.extend([
                        {
                            "model": model,
                            "reporter": reporter,
                            "train_desc": train_desc,
                            "n_training_samples": training_cfg.n_training_samples,
                            "n_train_datasets": len(training_cfg.training_datasets),
                            "eval_dataset": eval_dataset,
                            "layer_frac": (i + 1) / len(aurocs_per_layer),
                            "layer": i + 1, # start with layer 1, embedding layer is skipped
                            "auroc": aurocs_per_layer[i],
                            "accuracy": accs_per_layer[i],
                            "is_eil": i == eil,
                        }
                        for i in range(len(aurocs_per_layer))
                    ])

                # # Compute average over reporters for each layer
                # for i in range(len(aurocs_per_layer)):
                #     layer = i + 1 # start with layer 1, embedding layer is skipped
                #     layer_rows = [row for row in rows if row["layer"] == layer]
                #     supervised_layer_metrics = [row["auroc"] for row in layer_rows if row["reporter"] in ["lr", "lda", "mean-diff"]]
                #     unsupervised_layer_metrics = [row["auroc"] for row in layer_rows if row["reporter"] in ["ccs", "crc", "lr-on-pair"]]
                #     all_layer_metrics = supervised_layer_metrics + unsupervised_layer_metrics
                #     name_metrics = zip(
                #         ["avg", "supervised_avg", "unsupervised_avg"], 
                #         [all_layer_metrics, supervised_layer_metrics, unsupervised_layer_metrics]
                #         )
                #     for name, layer_metrics in name_metrics:
                #         average_layer_metric = np.mean(layer_metrics)
                #         rows.append(
                #             {
                #                 "model": model,
                #                 "reporter": name,
                #                 "train_desc": train_desc,
                #                 "n_training_samples": training_cfg.n_training_samples,
                #                 "n_train_datasets": len(training_cfg.training_datasets),
                #                 "eval_dataset": eval_dataset,
                #                 "layer_frac": layer / len(aurocs_per_layer),
                #                 "layer": layer,
                #                 args.metric: average_layer_metric,
                #                 "is_eil": i == eil,
                #             }
                #         )

                df = pd.concat([df, pd.DataFrame(rows)])

    # Display the resulting DataFrame
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(df)

    df.to_csv(args.save_csv_path)
    print(f"Saved summary to {Path(args.save_csv_path).absolute()}")