import argparse
from pathlib import Path
from distutils.util import strtobool
import os

import torch
from ccs import CcsConfig, CcsReporter
from crc import CrcReporter
from mean_diff import MeanDiffReporter
from lda import LdaReporter
from lr_classifier import Classifier
from tqdm import tqdm
from random_baseline import eval_random_baseline
from elk_utils import SplitConfig, SegmentConfig, aggregate_segments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate datasets, train a reporter and test it on multiple datasets."
    )
    parser.add_argument(
        "--data-dir", type=str, help="Path to the directory containing directories for each split, which in turn contain hiddens for training and test"
    )
    parser.add_argument(
        "--reporter", type=str, choices=["ccs", "crc", "lr", "lr-on-pair", "lda", "mean-diff", "random"], default="lr"
    )
    parser.add_argument("--prevent-skip", action="store_true")
    parser.add_argument("--max-train-examples", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--label-col",
        type=str,
        choices=["labels", "quirky_labels", "objective_labels"],
        default="labels",
    )
    parser.add_argument(
        "--pos-aligned",
        type=str,
        nargs="*",
        help="Columns of the dataset along which we wish to filter for training such that they are positively aligned with --label-col.",
        default=[],
    )
    parser.add_argument(
        "--neg-aligned",
        type=str,
        nargs="*",
        help="Columns of the dataset along which we wish to filter for training such that they are negatively aligned with --label-col.",
        default=[],
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--filter-cols",
        type=str,
        nargs="*",
        help="Columns of the dataset along which we wish to filter for training. Needs to be used with --filter-values",
        default=[],
    )
    parser.add_argument(
        "--filter-values",
        type=str,
        nargs="*",
        help="Values by which we want to filter the columns specified by --filter-cols.",
        default=[],
    )
    args = parser.parse_args()

    dtype = torch.float32

    data_dir = Path(args.data_dir)
    split_dirs = [entry for entry in data_dir.iterdir() if entry.is_dir()]

    split_cfg = SplitConfig(
        label_col=args.label_col,
        positively_aligned_cols=args.pos_aligned, 
        negatively_aligned_cols=args.neg_aligned, 
        filter_cols=args.filter_cols, 
        filter_values=[bool(strtobool(v)) for v in args.filter_values]
        )
    # Descriptor of the training config used to name directory to save data in
    train_cfg_save_dir=split_cfg.descriptor()   

    if args.verbose:
        print("Training args: ", args)
        print(f"Save dir: {train_cfg_save_dir=}")

    example_result_path = split_dirs[-1] / "test" / train_cfg_save_dir / f"mean-diff_log_odds.pt"
    print(f"{example_result_path=}")
    if not args.prevent_skip and example_result_path.exists():
        print("At least one test-directory for this training configuration already exists. Skipping training for this run.")
        exit()

    # Aggregate all training directories whose segments fulfill the criteria of the split_cfg
    training_segment_dirs = [dir for dir in split_dirs if split_cfg.contains_segment(SegmentConfig.from_descriptor(dir.name))]

    # Aggregate hiddens and labels over all training directories
    aggs = aggregate_segments(
        paths=training_segment_dirs, 
        label_cols=["labels", "objective_labels", "quirky_labels"],
        device=args.device,
        data_split="train",
        reporter=args.reporter,
        log_odds_split_descriptor=None
        )

    train_idx = torch.randperm(len(aggs["labels"]))[:args.max_train_examples]
    train_hiddens = [hidden[train_idx] for hidden in aggs["hiddens"]]
    train_labels = aggs[args.label_col][train_idx]
    labels = aggs["labels"][train_idx]
    objective_labels = aggs["objective_labels"][train_idx]
    quirky_labels = aggs["quirky_labels"][train_idx]

    # Sanity checks for balancing
    ol_balance = objective_labels.float().mean().item()
    ql_balance = quirky_labels.float().mean().item()
    l_balance = labels.float().mean().item()
    ol_ql_balance = (objective_labels==quirky_labels).float().mean().item()
    ol_l_balance = (objective_labels==labels).float().mean().item()
    # ol_ql_balance should be: 0 if negatively aligned, 1 if positively aligned, 0.5 if independent/balanced
    if "objective_labels" in split_cfg.positively_aligned_cols and "quirky_labels" in split_cfg.positively_aligned_cols:
        ol_ql_balance_expectation = 1
    elif "objective_labels" in split_cfg.negatively_aligned_cols and "quirky_labels" in split_cfg.negatively_aligned_cols:
        ol_ql_balance_expectation = 1
    elif "objective_labels" in split_cfg.positively_aligned_cols and "quirky_labels" in split_cfg.negatively_aligned_cols:
        ol_ql_balance_expectation = 0
    elif "objective_labels" in split_cfg.negatively_aligned_cols and "quirky_labels" in split_cfg.positively_aligned_cols:
        ol_ql_balance_expectation = 0
    else:
        ol_ql_balance_expectation = 0.5
    expected = torch.tensor([0.5, 0.5, 0.5, ol_ql_balance_expectation])
    obtained = torch.tensor([ol_balance, ql_balance, l_balance, ol_ql_balance])    
    if not torch.allclose(expected, obtained, atol=0.02):
        print("WARNING: Unexpected balancing of dataset!")
        print(f"Expected: {expected}")
        print(f"Obtained: {obtained}")
        
    if args.verbose: 
        print(f"Starting training to predict {args.label_col} on {train_hiddens[0].shape[0]} samples from {len(training_segment_dirs)} splits.")
        print(f"Balancing stats: ol={ol_balance}; ql={ql_balance}; l={l_balance}; ol*ql={ol_ql_balance}; ol*l={ol_l_balance}")

    reporters = []  # one for each layer
    for layer, train_hidden in tqdm(
        enumerate(train_hiddens), desc=f"Training on {len(training_segment_dirs)} splits"
    ):
        train_hidden = train_hidden.to(args.device).to(dtype)
        hidden_size = train_hidden.shape[-1]

        if args.reporter == "ccs":
            # we unsqueeze because CcsReporter expects a variants dimension
            train_hidden = train_hidden.unsqueeze(1)

            reporter = CcsReporter(
                cfg=CcsConfig(
                    bias=True,
                    loss=["ccs"],
                    norm="leace",
                    lr=1e-2,
                    num_epochs=1000,
                    num_tries=10,
                    optimizer="lbfgs",
                    weight_decay=0.01,
                ),
                in_features=hidden_size,
                num_variants=1,
                device=args.device,
                dtype=dtype,
            )

            reporter.fit(train_hidden)
            reporter.platt_scale(labels=train_labels, hiddens=train_hidden)
        elif args.reporter == "crc":
            # we unsqueeze because CrcReporter expects a variants dimension
            reporter = CrcReporter(
                in_features=hidden_size, device=args.device, dtype=dtype
            )
            reporter.fit(train_hidden)
            reporter.platt_scale(labels=train_labels, hiddens=train_hidden)
        elif args.reporter == "lr":
            reporter = Classifier(input_dim=hidden_size, device=args.device)
            reporter.fit(train_hidden, train_labels)
        elif args.reporter == "lr-on-pair":
            # We train a reporter on the difference between the two hiddens
            # pos, neg = train_hidden.unbind(-2)
            # hidden = pos - neg
            train_hidden = train_hidden.view(train_hidden.shape[0], -1)  # cat positive and negative
            reporter = Classifier(input_dim=2 * hidden_size, device=args.device)
            reporter.fit(train_hidden, train_labels)
        elif args.reporter == "mean-diff":
            reporter = MeanDiffReporter(in_features=hidden_size, device=args.device, dtype=dtype)
            reporter.fit(train_hidden, train_labels)
            reporter.resolve_sign(labels=train_labels, hiddens=train_hidden)
        elif args.reporter == "lda":
            reporter = LdaReporter(in_features=hidden_size, device=args.device, dtype=dtype)
            reporter.fit(train_hidden, train_labels)
            reporter.resolve_sign(labels=train_labels, hiddens=train_hidden)
        elif args.reporter == "random":
            reporter = None
        else:
            raise ValueError(f"Unknown reporter type: {args.reporter}")

        reporters.append(reporter)

    with torch.inference_mode():
        # Test on all splits
        for split_dir in split_dirs:
            test_dir = split_dir / "test"
            hiddens_file = (
                "ccs_hiddens.pt"
                if args.reporter in {"ccs", "crc", "lr-on-pair"}
                else "hiddens.pt"
            )
            test_hiddens = torch.load(test_dir / hiddens_file)
            test_labels = (
                torch.load(test_dir / f"{args.label_col}.pt").to(args.device).int()
            )
            lm_log_odds = (
                torch.load(test_dir / "lm_log_odds.pt").to(args.device).to(dtype)
            )

            # make sure that we're using a compatible test set
            test_n = test_hiddens[0].shape[0]
            if test_n == 0:
                print(f"Warning: Skipping evaluations because no samples fit the criteria in {split_dir}.")
                continue
            assert len(test_hiddens) == len(
                train_hiddens
            ), "Mismatched number of layers"
            assert all(
                h.shape[0] == test_n for h in test_hiddens
            ), "Mismatched number of samples"
            assert all(h.shape[-1] == train_hiddens[0].shape[-1] for h in test_hiddens), "Mismatched hidden size"

            log_odds = torch.full(
                [len(test_hiddens), test_n], torch.nan, device=args.device
            )
            for layer in tqdm(range(len(reporters)), desc=f"Testing on {test_dir}"):
                reporter, test_hidden = (
                    reporters[layer],
                    test_hiddens[layer].to(args.device).to(dtype),
                )
                if args.reporter == "ccs":
                    test_hidden = test_hidden.unsqueeze(1)
                    log_odds[layer] = reporter(test_hidden, ens="full")
                elif args.reporter == "crc":
                    log_odds[layer] = reporter(test_hidden)
                elif args.reporter == "lr-on-pair":
                    # pos, neg = test_hidden.unbind(-2)
                    # test_hidden = pos - neg
                    test_hidden = test_hidden.view(test_hidden.shape[0], -1)  # cat positive and negative
                    log_odds[layer] = reporter(test_hidden).squeeze(-1)
                elif args.reporter != "random":
                    log_odds[layer] = reporter(test_hidden).squeeze(-1)


            os.makedirs(test_dir / train_cfg_save_dir, exist_ok = True) 

            if args.reporter == "random":
                try:
                    aucs = []
                    for layer in range(len(test_hiddens)):
                        auc = eval_random_baseline(
                            train_hiddens[layer],
                            test_hiddens[layer],
                            train_labels,
                            test_labels,
                            num_samples=1000
                        )
                        if args.verbose:
                            print(f"Layer {layer} random AUC: {auc['mean']}")
                        aucs.append(auc)
                    torch.save(
                        aucs,
                        test_dir / train_cfg_save_dir / f"random_aucs_against_{args.label_col}.pt",
                    )
                except ValueError as e:
                    print(f"Succesfully finished training but failed computing AUCs with error: {e}")
            else:
                # save the log odds to disk
                # we use the name of the training directory as the prefix
                # e.g. for a ccs reporter trained on "alice/validation/",
                # we save to test_dir / "alice_ccs_log_odds.pt"[]
                torch.save(
                    log_odds,
                    test_dir / train_cfg_save_dir / f"{args.reporter}_log_odds.pt",
                )

                try:
                    if args.verbose:
                        print(f"Evaluated {args.reporter} on {test_n} samples.")
                        from sklearn.metrics import roc_auc_score

                        aucs = []
                        for layer in range(len(reporters)):
                            auc = roc_auc_score(
                                test_labels.cpu().numpy(), log_odds[layer].cpu().numpy()
                            )
                            print("AUC:", auc)
                            aucs.append(auc)

                        informative_layers = [layer for layer, auc in enumerate(aucs) if auc - 0.5 >= 0.95 * (max(aucs) - 0.5)]
                        print(f"{informative_layers=}")
                        earliest_informative_layer = informative_layers[0] if len(informative_layers) else int(len(reporters)/2)
                        print(f"{earliest_informative_layer=} with AUC {aucs[earliest_informative_layer]}")

                        auc = roc_auc_score(
                            test_labels.cpu().numpy(), lm_log_odds.cpu().numpy()
                        )
                        print("LM AUC:", auc)
                except ValueError as e:
                    print(f"Succesfully finished training but failed computing AUCs with error: {e}")