import argparse
from pathlib import Path
from distutils.util import strtobool
import os
from itertools import combinations
import torch
from ccs import CcsConfig, CcsReporter
from crc import CrcReporter
from mean_diff import MeanDiffReporter
from lda import LdaReporter
from lr_classifier import Classifier
from tqdm import tqdm
from random_baseline import eval_random_baseline
from elk_utils import aggregate_datasets, DiversifyTrainingConfig

if __name__ == "__main__":    
    debug = False
    if debug:
        print("DEBUGGING WITH HARDCODED ARGS!")
        args = argparse.Namespace(
            data_dir = Path("./experiments/diversify"),
            models = ["EleutherAI/pythia-410M"],
            training_datasets = ["got/cities", "got/larger_than"],
            eval_datasets = ["got/cities", "got/larger_than"],
            n_train_datasets = 1,
            reporters = ["ccs", "lr"],
            contrast_norm = "burns",
            normalize_contrast_individually = True,
            prevent_skip = True,
            train_examples = 10,
            device = "cpu",
            label_col = "labels",
            verbose=True
            )
    else:
        parser = argparse.ArgumentParser(
            description="Aggregate datasets, train a reporter and test it on multiple datasets."
        )
        parser.add_argument(
            "--data-dir", type=str, help="Path to the directory containing directories for each dataset"
        )
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
        parser.add_argument("--models", nargs="+", type=str, help="List of model names.")
        parser.add_argument("--n-train-datasets", help="Number of datasets unionized over to serve as training data", type=int, default=1)
        parser.add_argument(
            "--reporters",
            type=str, 
            nargs="+",
            default=["lr"]
        )
        parser.add_argument(
            "--contrast-norm", 
            help="Normalization strategy that will be applied to pos and neg half of the aggregated dataset", 
            type=str,
            choices=[None, "burns", "meanonly", "leace"],
            default=None
        )
        parser.add_argument(
            "--normalize-contrast-individually", 
            help="Whether the normalization should be applied to each dataset individually. If not specified, it will be applied to the aggregated dataset.", 
            action="store_true")
        parser.add_argument("--prevent-skip", action="store_true")
        parser.add_argument("--train-examples", type=int, default=4096)
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument(
            "--label-col",
            type=str,
            choices=["labels", "quirky_labels", "objective_labels"],
            default="labels",
        )
        parser.add_argument("--verbose", action="store_true")

        args = parser.parse_args()

    dtype = torch.float32

    data_dir = Path(args.data_dir) 
    # Expected (input) data structure: data_dir/<train_dataset>/<model>/<train|test>/hiddens.pt
    # Expected (results) data structure: data_dir/<eval_dataset>/<model>/<train|test>/<training_identifier>/<reporter>_log_odds.pt

    if args.verbose:
        print("Training args: ", args)

    assert data_dir.exists(), f"{data_dir=} does not exist"
    for training_dataset in args.training_datasets:
        assert (data_dir / training_dataset).exists(), f"Could not find training directory {(data_dir / training_dataset)}."
    assert args.n_train_datasets <= len(args.training_datasets), "Can not combine more datasets than were provided"

    # Determine which normalization strategy to use
    contrast_individual_norm = args.contrast_norm if args.normalize_contrast_individually else None
    contrast_aggr_norm = None if args.normalize_contrast_individually else args.contrast_norm
    
    for model in args.models:
        # Iterate through all combinations of training datasets of the given length
        for training_datasets in combinations(args.training_datasets, r=args.n_train_datasets):
            training_cfg = DiversifyTrainingConfig(training_datasets, n_training_samples=args.train_examples)
            training_identifier = training_cfg.descriptor()

            # Aggregate hiddens, ccs_hiddens and labels over all training directories
            training_paths = [data_dir / training_dataset / model / "train" for training_dataset in training_datasets]
            aggs = aggregate_datasets(
                paths=training_paths, 
                label_cols=["labels"],
                device=args.device,
                samples_per_dataset=int(args.train_examples / args.n_train_datasets),
                contrast_norm=contrast_individual_norm,
                reporters_for_log_odds=[], # Not needed during training, as log odds will be created below
                )
                    
            # Select subsets for training
            train_hiddens = aggs["hiddens"]
            train_ccs_hiddens = aggs["ccs_hiddens"]
            train_labels = aggs[args.label_col]

            # Sanity checks for balancing
            expected = torch.tensor(0.5)
            l_balance = train_labels.float().mean()
            if not torch.allclose(expected, l_balance, atol=0.02):
                print("WARNING: Unexpected balancing of dataset for labels!")
                print(f"Expected: {expected}")
                print(f"Obtained: {l_balance}")


            for reporter_name in args.reporters:
                # Skip if results already exist
                if not args.prevent_skip:
                    if reporter_name == "random":
                        results_fname = f"{reporter_name}_aucs_against_labels.pt"
                    else:
                        results_fname = f"{reporter_name}_log_odds.pt"
                    # Expected (results) data structure: data_dir/<eval_dataset>/<model>/<train|test>/<training_identifier>/<reporter>_log_odds.pt
                    result_files = [data_dir / eval_dataset / model / "test" / training_identifier / results_fname for eval_dataset in args.eval_datasets]
                    if all([result_file.exists() for result_file in result_files]):
                        print(f"Skipping run for {training_identifier=} and {reporter_name=} as data already exists.")
                        continue

                # Test
                if args.verbose: 
                    print(f"Starting training {reporter_name} to predict {args.label_col} on {train_hiddens[0].shape[0]} samples from {training_identifier}.")

                selected_train_hiddens = train_ccs_hiddens if reporter_name in ["ccs", "crc", "lr-on-pair"] else train_hiddens

                reporters = []  # one for each layer
                for layer, train_hidden in tqdm(
                    enumerate(selected_train_hiddens), desc=f"Training"
                ):
                    train_hidden = train_hidden.to(args.device).to(dtype)
                    hidden_size = train_hidden.shape[-1]

                    if reporter_name == "ccs":
                        # we unsqueeze because CcsReporter expects a variants dimension
                        train_hidden = train_hidden.unsqueeze(1)

                        reporter = CcsReporter(
                            cfg=CcsConfig(
                                bias=True,
                                loss=["ccs"],
                                norm=contrast_aggr_norm,
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
                    elif reporter_name == "crc":
                        # we unsqueeze because CrcReporter expects a variants dimension
                        reporter = CrcReporter(
                            in_features=hidden_size, device=args.device, dtype=dtype
                        )
                        reporter.fit(train_hidden)
                        reporter.platt_scale(labels=train_labels, hiddens=train_hidden)
                    elif reporter_name == "lr":
                        reporter = Classifier(input_dim=hidden_size, device=args.device)
                        reporter.fit(train_hidden, train_labels)
                    elif reporter_name == "lr-on-pair":
                        # We train a reporter on the difference between the two hiddens
                        # pos, neg = train_hidden.unbind(-2)
                        # hidden = pos - neg
                        train_hidden = train_hidden.view(train_hidden.shape[0], -1)  # cat positive and negative
                        reporter = Classifier(input_dim=2 * hidden_size, device=args.device)
                        reporter.fit(train_hidden, train_labels)
                    elif reporter_name == "mean-diff":
                        reporter = MeanDiffReporter(in_features=hidden_size, device=args.device, dtype=dtype)
                        reporter.fit(train_hidden, train_labels)
                        reporter.resolve_sign(labels=train_labels, hiddens=train_hidden)
                    elif reporter_name == "lda":
                        reporter = LdaReporter(in_features=hidden_size, device=args.device, dtype=dtype)
                        reporter.fit(train_hidden, train_labels)
                        reporter.resolve_sign(labels=train_labels, hiddens=train_hidden)
                    elif reporter_name == "random":
                        reporter = None
                    else:
                        raise ValueError(f"Unknown reporter type: {reporter_name}")

                    reporters.append(reporter)
                    
                # Test
                if args.verbose: 
                    print(f"Starting testing {reporter_name} trained on {training_identifier} to predict {args.label_col} on {len(args.eval_datasets)} datasets.")
                with torch.inference_mode():
                    # Test on all eval datasets seperately
                    for eval_dataset in args.eval_datasets:
                        # Expected (input) data structure: data_dir/<train_dataset>/<model>/<train|test>/hiddens.pt
                        eval_path = data_dir / eval_dataset / model / "test"
                        # Expected (results) data structure: data_dir/<eval_dataset>/<model>/<train|test>/<training_identifier>/<reporter>_log_odds.pt
                        results_path = data_dir / eval_dataset / model / "test" / training_identifier
                        hiddens_file = (
                            "ccs_hiddens.pt"
                            if reporter_name in {"ccs", "crc", "lr-on-pair"}
                            else "hiddens.pt"
                        )
                        test_hiddens = torch.load(eval_path / hiddens_file, map_location=torch.device(args.device))
                        test_labels = (
                            torch.load(eval_path / f"{args.label_col}.pt", map_location=torch.device(args.device)).int()
                        )

                        # lm_log_odds are only available if the samples in the dataset end on choices like e.g. " true" or " false"
                        lm_log_odds_available = (eval_path / "lm_log_odds.pt").exists()
                        if lm_log_odds_available:
                            lm_log_odds = (
                                torch.load(eval_path / "lm_log_odds.pt", map_location=torch.device(args.device)).to(dtype)
                            )

                        # make sure that we're using a compatible test set
                        test_n = test_hiddens[0].shape[0]
                        assert len(test_hiddens) == len(
                            selected_train_hiddens
                        ), "Mismatched number of layers"
                        assert all(
                            h.shape[0] == test_n for h in test_hiddens
                        ), "Mismatched number of samples"
                        assert all(h.shape[-1] == selected_train_hiddens[0].shape[-1] for h in test_hiddens), "Mismatched hidden size"

                        log_odds = torch.full(
                            [len(test_hiddens), test_n], torch.nan, device=args.device
                        )
                        for layer in tqdm(range(len(reporters)), desc=f"Testing on {eval_dataset}"):
                            reporter, test_hidden = (
                                reporters[layer],
                                test_hiddens[layer].to(args.device).to(dtype),
                            )
                            if reporter_name == "ccs":
                                test_hidden = test_hidden.unsqueeze(1)
                                log_odds[layer] = reporter(test_hidden, ens="full")
                            elif reporter_name == "crc":
                                log_odds[layer] = reporter(test_hidden)
                            elif reporter_name == "lr-on-pair":
                                # pos, neg = test_hidden.unbind(-2)
                                # test_hidden = pos - neg
                                test_hidden = test_hidden.view(test_hidden.shape[0], -1)  # cat positive and negative
                                log_odds[layer] = reporter(test_hidden).squeeze(-1)
                            elif reporter_name != "random":
                                log_odds[layer] = reporter(test_hidden).squeeze(-1)


                        os.makedirs(results_path, exist_ok = True) 

                        if reporter_name == "random":
                            try:
                                aucs = []
                                for layer in range(len(test_hiddens)):
                                    auc = eval_random_baseline(
                                        selected_train_hiddens[layer],
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
                                    results_path / f"random_aucs_against_{args.label_col}.pt",
                                )
                            except ValueError as e:
                                print(f"Succesfully finished training but failed computing AUCs with error: {e}")
                        else:
                            # save the log odds to disk
                            torch.save(
                                log_odds,
                                results_path / f"{reporter_name}_log_odds.pt",
                            )

                            try:
                                if args.verbose:
                                    print(f"Evaluated {reporter_name} on {test_n} samples.")
                                    from sklearn.metrics import roc_auc_score

                                    aucs = []
                                    for layer in range(len(reporters)):
                                        auc = roc_auc_score(
                                            test_labels.cpu().numpy(), log_odds[layer].cpu().numpy()
                                        )
                                        print(f"AUC for layer {layer}: {auc:.2f}")
                                        aucs.append(auc)

                                    informative_layers = [layer for layer, auc in enumerate(aucs) if auc - 0.5 >= 0.95 * (max(aucs) - 0.5)]
                                    print(f"{informative_layers=}")
                                    earliest_informative_layer = informative_layers[0] if len(informative_layers) else int(len(reporters)/2)
                                    print(f"{earliest_informative_layer=} with AUC {aucs[earliest_informative_layer]}")

                                    if lm_log_odds_available:
                                        auc = roc_auc_score(
                                            test_labels.cpu().numpy(), lm_log_odds.cpu().numpy()
                                        )
                                        print("LM AUC:", auc)
                                    else:
                                        print("No LM metrics available as no lm_log_odds were provided.")
                            except ValueError as e:
                                print(f"Succesfully finished training but failed computing AUCs with error: {e}")