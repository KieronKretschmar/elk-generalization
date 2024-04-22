import argparse
from pathlib import Path
import os
import pandas as pd
from distutils.util import strtobool
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from elk_utils import SplitConfig, SegmentConfig, aggregate_segments


def earliest_informative_layer_index(df, metric):
    max_auroc = max(df[metric])
    informative_layers = df[df[metric] - 0.5  >= 0.95 * (max_auroc - 0.5)]
    if len(informative_layers):
        earliest_informative_layer = int(informative_layers.iloc[0].name)
    else:
        earliest_informative_layer = int(len(df)/2)
    return earliest_informative_layer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize the test results from experiments regarding transfer performance of probes."
    )
    parser.add_argument("--models", nargs="+", type=str, help="List of model names.")
    parser.add_argument("--root-dir", type=str, help="Path to the root directory for all experiments")
    parser.add_argument("--reporters", type=str, nargs="+", default="lr", help="Which reporters to use.")
    parser.add_argument("--metric", type=str, choices=["auroc", "acc"], default="auroc", help="Metric to use.")
    parser.add_argument("--label-col", type=str, choices=["labels", "objective_labels", "quirky_labels"], default="objective_labels", help="Which label to use for the metric.")
    parser.add_argument("--save-csv-path", type=Path, help="Path to save the dataframe as csv.")

    debug = False
    if debug:
        from argparse import Namespace
        default_values = {
            "models": ["pythia-410M", "pythia-1B", "pythia-1.4B"],
            "root_dir":"./experiments/quirky_intcomparison",
            "reporters": ["ccs", "lr", "crc"],
            "metric": "auroc",
            "label_col": "objective_labels",
            "save_csv_path": "align_debug.csv"
        }

        # Create a Namespace object with default values
        args = Namespace(**default_values)
    else:
        args = parser.parse_args()

    print("Args:")
    print(args)

    root_dir = Path(args.root_dir)
    metric_fn = {
        "auroc": roc_auc_score,
        "acc": lambda gt, logodds: accuracy_score(gt, logodds > 0),
    }[args.metric]

    # All-Multi-index solution
    # index = pd.MultiIndex.from_product(
    #     [args.reporters, args.models, [True], [True, False], [True, False], ["pa", "ind", "na"], [True, False], [True, False], ["pa", "ind", "na"]], 
    #     names=['reporter', 'models', 'train_pi', 'train_pr', 'train_ol_ql_alignment', 'test_pi', 'test_pr', 'test_ol_ql_alignment']
    #     )
    # df = pd.DataFrame(index=index, columns=args.models)


    # Model x Reporter x train_cfg multi-index, test_cfg columns
    # all_split_descriptors = []
    # label = "objective_labels"
    # alignments= [
    #     {
    #         "pos": ["objective_labels", "quirky_labels"],
    #         "neg": []
    #     },
    #     {
    #         "pos": ["objective_labels"],
    #         "neg": []
    #     },
    #     {
    #         "pos": ["objective_labels"],
    #         "neg": ["quirky_labels"]
    #     },
    # ]
    # for persona_introduced in [True]:
    #     for persona_responds in [True, False]:
    #         for alignment in alignments:
    #             cfg = train_cfg_descriptor(
    #                 label=label, 
    #                 positively_aligned_cols=alignment["pos"], 
    #                 negatively_aligned_cols=alignment["neg"], 
    #                 filter_cols=["persona_introduceds", "persona_respondss"], 
    #                 filter_values=[persona_introduced, persona_responds]
    #                 )
    #             all_split_descriptors.append(cfg)

    # first_model_test_splits = root_dir / args.models[0]
    # first_split = os.listdir(first_model_test_splits)[0] / "test"
    # all_split_descriptors = [dir.name for dir in os.listdir(first_split)]
    # print(f"DEBUG: {all_split_descriptors=}")    


    # Initialize dataframe and all split descriptors
    # Directory structure is root_dir/<model>/<segment>/<train|test>/<split>
    first_model_dir = root_dir / args.models[0]
    all_segment_dirs = [directory for directory in os.listdir(first_model_dir) if os.path.isdir(first_model_dir / directory)]
    all_segment_cfgs = [SegmentConfig.from_descriptor(directory) for directory in all_segment_dirs]
    # Take the splits evaluated on the first model and segment. We assume these are the same for all models and splits
    example_segment_dir = root_dir / args.models[0] / all_segment_cfgs[0].descriptor()
    all_split_cfgs = [SplitConfig.from_descriptor(directory) for directory in os.listdir(example_segment_dir / "test") if os.path.isdir(example_segment_dir / "test" / directory)]
    # The row-index includes train split config, the columns are for each test split config
    index = pd.MultiIndex.from_product(
        [args.models, args.reporters + ["avg", "lm"], [cfg.descriptor() for cfg in all_split_cfgs]], 
        names=['models', 'reporter', 'train_cfg']
        )
    df = pd.DataFrame(index=index, columns=[cfg.descriptor() for cfg in all_split_cfgs])
    df = df.sort_index()

    # Test on all splits that were trained on
    for train_split_cfg in all_split_cfgs:
        train_desc = train_split_cfg.descriptor()
        for test_split_cfg in all_split_cfgs:
            test_desc = test_split_cfg.descriptor()
            test_segment_dirs = [all_segment_dirs[i] for i in range(len(all_segment_dirs)) if test_split_cfg.contains_segment(all_segment_cfgs[i])]
            for model in args.models:
                for reporter in args.reporters:
                    # get metric vs layer for each model and template
                    results_dfs = dict()
                    lm_results = dict()

                    aggs = aggregate_segments(
                        paths=[root_dir / model / directory for directory in test_segment_dirs], 
                        label_cols=["labels", "objective_labels", "quirky_labels", "lm_log_odds"],
                        reporter=reporter,
                        device="cpu",
                        data_split="test",
                        log_odds_split_descriptor=train_desc
                        )

                    # results_dfs[(model, reporter, train_desc, test_desc)] = 
                    reporter_results_by_layer_df = pd.DataFrame(
                        [
                            {
                                # start with layer 1, embedding layer is skipped
                                "layer": i + 1,
                                # max layer is len(reporter_log_odds)
                                "layer_frac": (i + 1) / len(aggs["reporter_log_odds"]),
                                args.metric: metric_fn(
                                    aggs[args.label_col], layer_log_odds.float().numpy()
                                ),
                            }
                            for i, layer_log_odds in enumerate(aggs["reporter_log_odds"])
                        ]
                    )
                    eil = earliest_informative_layer_index(reporter_results_by_layer_df, args.metric)
                    df.loc[(model, reporter, train_desc), test_desc + "_eil"] = reporter_results_by_layer_df.loc[eil, args.metric]
                    for i, layer_log_odds in enumerate(aggs["reporter_log_odds"]):
                        df.loc[(model, reporter, train_desc), test_desc + "_layer_" + str(i)] = reporter_results_by_layer_df.loc[i, args.metric]

                # Compute average over all reporters
                df.loc[(model, "avg", train_desc), test_desc + "_eil"] = df.loc[pd.IndexSlice[model, args.reporters, train_desc], test_desc + "_eil"].mean()
                for i, _ in enumerate(aggs["reporter_log_odds"]):
                    df.loc[(model, "avg", train_desc), test_desc + "_layer_" + str(i)] = df.loc[pd.IndexSlice[model, args.reporters, train_desc], test_desc + "_layer_" + str(i)].mean()

                df.loc[(model, "lm", train_desc), test_desc] = metric_fn(
                    aggs[args.label_col], aggs["lm_log_odds"]
                )

    # Display the resulting DataFrame
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(df)

    df.to_csv(args.save_csv_path)
    print(f"Saved summary to {Path(args.save_csv_path).absolute()}")

# def interpolate(layers_all, results_all, n_points=501):
#     # average these results over models and templates
#     all_layer_fracs = np.linspace(0, 1, n_points)
#     avg_reporter_results = np.zeros(len(all_layer_fracs), dtype=np.float32)
#     for layers, results in zip(layers_all, results_all):
#         # convert `layer` to a fraction of max layer in results_df
#         # linearly interpolate to get auroc at each layer_frac
#         max_layer = layers.max()
#         layer_fracs = (layers + 1) / max_layer

#         interp_result = np.interp(all_layer_fracs, layer_fracs, results)
#         avg_reporter_results += interp_result / len(results_all)

#     return all_layer_fracs, avg_reporter_results
