import argparse
from pathlib import Path
import pandas as pd

import viz

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
    parser.add_argument("--template-names", nargs="+", type=str, help="List of template names.")
    parser.add_argument("--fr", type=str, default="A", help="Probe is evaluated on this context")
    parser.add_argument("--to", type=str, default="B", help="Probe was trained on this context and against this label set")
    parser.add_argument("--root-dir", type=str, help="Path to the root directory for all experiments")
    parser.add_argument("--filter-by", type=str, choices=["agree", "disagree", "all"], default="disagree", help="Whether to keep only examples where Alice and Bob disagree.")
    parser.add_argument("--reporters", type=str, nargs="+", default="lr", help="Which reporters to use.")
    parser.add_argument("--metric", type=str, choices=["auroc", "acc"], default="auroc", help="Metric to use.")
    parser.add_argument("--label-col", type=str, choices=["alice_label", "bob_label", "label"], default="alice_label", help="Which label to use for the metric.")
    parser.add_argument("--save-csv-path", type=Path, help="Path to save the dataframe as csv.")

    debug = False
    if debug:
        from argparse import Namespace
        default_values = {
            "models": ["pythia-410M", "pythia-1B", "pythia-1.4B"],
            "template_names": ["mixture", "grader-first", "grader-last"],
            "root_dir":"./experiments",
            "fr":"Alice-easy",
            "to":"Bob-hard",
            "filter_by":"all",
            "reporters": ["ccs", "lr", "crc"],
            "metric": "auroc",
            "label_col": "alice_label",
            "save_csv_path": "table4_debug.csv"
        }

        # Create a Namespace object with default values
        args = Namespace(**default_values)
    else:
        args = parser.parse_args()

    print("Args:")
    print(args)

    index = pd.MultiIndex.from_product([args.reporters, args.template_names + ["avg"]], names=['reporter', 'template'])

    df = pd.DataFrame(index=index, columns=args.models)
    for reporter in args.reporters:
        # Call the function with the parsed arguments
        avg_reporter_results, results_dfs, avg_lm_result, lm_results = viz.get_result_dfs(
            models=args.models,
            template_names=args.template_names,
            fr=args.fr,
            to=args.to,
            root_dir=args.root_dir,
            filter_by=args.filter_by,
            reporter=reporter,
            metric=args.metric,
            label_col=args.label_col
        )

        for (model, template), result_df in results_dfs.items():
            eil = earliest_informative_layer_index(result_df, args.metric)
            df.loc[(reporter, template), model] = result_df.loc[eil]["auroc"]

        for model in args.models:
            df.loc[(reporter, "avg"), model] = df.loc[pd.IndexSlice[reporter, args.template_names], model].mean()

            
    # lm_results are independent of the reporter, so we use the last reporter's lm_results
    for (model, template), lm_result in lm_results.items():
        df.loc[("lm", template), model] = lm_result

    # Display the resulting DataFrame
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(df)

    df.to_csv(args.save_csv_path)
    print(f"Saved summary to {Path(args.save_csv_path).absolute()}")

