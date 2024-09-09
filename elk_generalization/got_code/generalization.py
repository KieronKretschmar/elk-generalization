import torch as t
import random
import matplotlib.pyplot as plt
import random
import argparse
from pathlib import Path
import pandas as pd
import os
import numpy as np
from itertools import combinations
import time
from sklearn.metrics import roc_auc_score

from utils import DataManager, transfer_type
from probes import LRProbe, MMProbe, CCSProbe, MMProbe_Mallen, CrcReporter


if __name__ == "__main__":    
    debug = False
    if debug:
        print("DEBUGGING WITH HARDCODED ARGS!")
        args = argparse.Namespace(
            data_dir = Path("./experiments/diversify_remake"),
            )
        
    else:
        parser = argparse.ArgumentParser(
            description="Train and test probes on activations from multiple datasets."
        )
        parser.add_argument(
            "--data-dir", type=str, help="Path to the directory containing directories for each dataset"
        )
        parser.add_argument("--model", type=str, help="Name of the model from huggingface")
        parser.add_argument("--layer", type=int, help="Layer on which to train")
        parser.add_argument("--min-n-train-datasets", type=int, help="Minimum number of combinations of datasets to train on")
        parser.add_argument("--max-n-train-datasets", type=int, help="Maximum number of combinations of datasets to train on")
        parser.add_argument("--train-examples", type=int, default=None)
        parser.add_argument(
            "--apply-train-examples-per-dataset", 
            action="store_true", 
            help="If specified, the specified number of train-examples will be applied to each training dataset instead of the total number."
        )
        parser.add_argument("--split", type=float, default=None, help="Fraction of dataset used for training.")
        parser.add_argument("--seed", type=int, default=1234)
        parser.add_argument("--save-csv-path", type=Path, help="Path to save the dataframe as csv.")
        parser.add_argument("--tuple-inference", action="store_true", help="Flag to indicate that a statement and its negation are used for inference.")

        args = parser.parse_args()
    print(f"{args=}")

    device = 'cuda:0' if t.cuda.is_available() else 'cpu'
    model = args.model
    layer = args.layer
    min_n_train_datasets = args.min_n_train_datasets
    max_n_train_datasets = args.max_n_train_datasets
    train_examples = args.train_examples
    apply_train_examples_per_dataset = args.apply_train_examples_per_dataset
    split = args.split
    save_csv_path = args.save_csv_path
    seed = args.seed
    tuple_inference = args.tuple_inference

    # Enabling preloading is less efficient in terms of memory, but more efficient in terms of compute
    preload_validation_data = True

    assert split is not None or train_examples is not None, "At least one of split, train_examples must be specified"

    root = Path(args.data_dir)

    def to_str(l):
        return '+'.join(l)

    def to_str_combination(c):
        return "&".join([to_str(i) for i in c])
    
    def medleys_from_str(desc):
        medleys = []
        for medley in desc.split('&'):
            medleys.append(medley.split('+')) 

    def partition_sizes(n, k):
        # Determines n1,n2,...,nk such that they are as similar as possible while n1+...nk=n
        # if n is None returns a list of Nones of length k
        if n == None:
            return [None] * k
        
        partition_size = n // k
        remainder = n % k
        sizes = []
        for i in range(k):
            size = partition_size + (1 if i < remainder else 0)
            sizes.append(size)
        assert sum(sizes) == n
        return sizes
    
    def evaluate_probe(probe, acts, labels, use_tuples_pred=False, iid=False):
        if use_tuples_pred:
            logits = probe.forward_tuples(acts, iid=iid).detach().cpu()
        else:
            logits = probe.forward(acts, iid=iid).detach().cpu()
        preds = logits.round()
        labels = labels.detach().cpu()
        acc = (preds == labels).float().mean().item()
        if len(labels.unique()) > 1:
            auroc = roc_auc_score(
                labels.numpy(), logits.numpy()
            )
        else:
            auroc = np.NaN
        return {"accuracy": acc, "auroc": auroc}

    accs = []

    if seed is None:
        seed = random.randint(0, 100000)

    # SUPERVISED
    train_medlies  = [
        # ['got/cities'],
        ['got/cities', 'got/neg_cities'],                                           # n=2x1496
        # ['got/larger_than'],
        ['got/larger_than', 'got/smaller_than'],                                    # n=2x1980
        ['got/sp_en_trans', 'got/neg_sp_en_trans'],                                 # n=2x354
        ['got/companies_true_false'],                                               # n=1200
        # ['got/counterfact_true', 'got/counterfact_false'],                        # n=2x15982, broken
        ['azaria/animals_true_false', 'azaria/neg_animals_true_false'],             # n=2x1008
        ['azaria/elements_true_false', 'azaria/neg_elements_true_false'],           # n=2x930
        ['azaria/facts_true_false', 'azaria/neg_facts_true_false'],                 # n=2x437
        ['azaria/inventions_true_false', 'azaria/neg_inventions_true_false'],       # n=2x876
    ]

    supervised_val_datasets = [
        'got/cities',
        'got/neg_cities',
        'got/larger_than',
        'got/smaller_than',
        'got/sp_en_trans',
        'got/neg_sp_en_trans',
        'got/cities_cities_conj',
        'got/cities_cities_disj',
        'got/companies_true_false',
        'got/common_claim_true_false',
        # 'got/counterfact_true', # broken
        # 'got/counterfact_false',  # broken 
        # 'got/counterfact_true_false'    # We use the true/false datasets above for validation instead
        'azaria/animals_true_false',
        'azaria/neg_animals_true_false',
        'azaria/elements_true_false',
        'azaria/neg_elements_true_false',
        'azaria/facts_true_false',
        'azaria/neg_facts_true_false',
        'azaria/inventions_true_false',
        'azaria/neg_inventions_true_false',
    ]

    tuple_eval_medlies  = [
        ['got/cities', 'got/neg_cities'],                                           # n=2x1496
        ['got/larger_than', 'got/smaller_than'],                                    # n=2x1980
        ['got/sp_en_trans', 'got/neg_sp_en_trans'],                                 # n=2x354
        ['azaria/animals_true_false', 'azaria/neg_animals_true_false'],             # n=2x1008
        ['azaria/elements_true_false', 'azaria/neg_elements_true_false'],           # n=2x930
        ['azaria/facts_true_false', 'azaria/neg_facts_true_false'],                 # n=2x437
        ['azaria/inventions_true_false', 'azaria/neg_inventions_true_false'],       # n=2x876
    ]

    SupervisedProbeClasses = [
        LRProbe, 
        # MMProbe_Mallen,
        MMProbe,
        ]
    
    # Load all full datasets into the data manager as they're used for evaluation only and are thus constant between medley_combinations
    if preload_validation_data:
        dm = DataManager(root=root)
        for dataset in supervised_val_datasets:
            dm.add_dataset(dataset, model, layer, split=None, center=True, device=device)

    medley_combinations = []
    for k in range(min_n_train_datasets, max_n_train_datasets + 1):
        medley_combinations.extend(combinations(train_medlies, r=k))

    for medley_combination in medley_combinations:
        all_train_datasets = [ds for medley in medley_combination for ds in medley]
        medley_train_sizes = partition_sizes(train_examples, len(medley_combination))

        if preload_validation_data:
            # Remove split datasets as they may have to be re-loaded with different splits depending on the medley_combination
            dm.reset_split_datasets()
        else:
            dm = DataManager(root=root)
            for dataset in supervised_val_datasets:
                if dataset not in all_train_datasets:
                    dm.add_dataset(dataset, model, layer, split=None, center=True, device=device)

        for medley, medley_train_size in zip(medley_combination, medley_train_sizes):
            # determine train_size for each dataset in medley
            if tuple_inference and medley in tuple_eval_medlies:
                # When doing tuple inference, we require corresponding statements and negated statements in test split
                # we achieve this by using the same size and seed for each split
                assert len(medley) == 2
                train_sizes = [round(medley_train_size / len(medley))] * len(medley)
            else:
                train_sizes = partition_sizes(medley_train_size, len(medley))

            print(f"{medley_train_size=};{medley=};{train_sizes=}")
            for dataset, train_size in zip(medley, train_sizes):
                train_size = train_examples if apply_train_examples_per_dataset else train_size
                dm.add_dataset(dataset, model, layer, split=split, n_training_samples=train_size, seed=seed, center=True, device=device)


        train_acts, train_labels = dm.get('train')

        # train probes
        for ProbeClass in SupervisedProbeClasses:
            print(f"Starting training {str(ProbeClass)} on {to_str_combination(medley_combination)}")
            probe = ProbeClass.from_data(train_acts, train_labels, device=device)

            # evaluate
            if tuple_inference:
                for eval_medley in tuple_eval_medlies:
                    (val_dataset, neg_val_dataset) = eval_medley
                    print("evaluating: ", val_dataset, neg_val_dataset)
                    print(f"{val_dataset=}, {all_train_datasets=}, transfer_type={transfer_type(all_train_datasets, val_dataset)}")
                    if val_dataset in all_train_datasets:
                        assert neg_val_dataset in all_train_datasets
                        # Use labels from second dataset, so they correspond to the index of the true statement
                        acts, _ = dm.data['val'][val_dataset]
                        neg_acts, labels = dm.data['val'][neg_val_dataset]
                        if len(acts) == 0: continue
                        metrics = evaluate_probe(probe, t.stack([acts, neg_acts]), labels, use_tuples_pred=True, iid=False)
                    else:
                        assert neg_val_dataset not in all_train_datasets, f"{eval_medley=}, {all_train_datasets=}"
                        # Use labels from second dataset, so they correspond to the index of the true statement
                        acts, _ = dm.data[val_dataset]
                        neg_acts, labels = dm.data[neg_val_dataset]
                        if len(acts) == 0: continue
                        metrics = evaluate_probe(probe, t.stack([acts, neg_acts]), labels, use_tuples_pred=True, iid=False)
                
                    accs.append({
                        "model": model,
                        "layer": layer,
                        "reporter": str(ProbeClass)+"_tuple_inference",
                        "train_desc": to_str_combination(medley_combination),
                        "all_train_datasets": all_train_datasets,
                        "eval_dataset": to_str(eval_medley),
                        "n_train_datasets": len(medley_combination),
                        "oracle": False,
                        "transfer_type": transfer_type(all_train_datasets, val_dataset),
                        "accuracy": metrics["accuracy"],
                        "auroc": metrics["auroc"],
                        "train_size": len(train_acts),
                        "test_size": len(acts),
                        "seed": seed, 
                    })
            else:
                for val_dataset in supervised_val_datasets:
                    if val_dataset in medley:
                        acts, labels = dm.data['val'][val_dataset]
                        if len(acts) == 0: continue
                        metrics = evaluate_probe(probe, acts, labels, iid=False)
                    else:
                        acts, labels = dm.data[val_dataset]
                        if len(acts) == 0: continue
                        metrics = evaluate_probe(probe, acts, labels, iid=False)
                    accs.append({
                        "model": model,
                        "layer": layer,
                        "reporter": str(ProbeClass),
                        "train_desc": to_str_combination(medley_combination),
                        "all_train_datasets": all_train_datasets,
                        "eval_dataset": val_dataset,
                        "n_train_datasets": len(medley_combination),
                        "oracle": False,
                        "transfer_type": transfer_type(all_train_datasets, val_dataset),
                        "accuracy": metrics["accuracy"],
                        "auroc": metrics["auroc"],
                        "train_size": len(train_acts),
                        "test_size": len(acts),
                        "seed": seed, 
                    })

    print(f"Finished supervised.")


    # UNSUPERVISED
    UnsupervisedProbeClasses = [
        CCSProbe,
        CrcReporter
    ]
    
    ccs_base_medlies = [
        ['got/cities', 'got/neg_cities'],                                       # n=1496
        ['got/larger_than', 'got/smaller_than'],                                # n=1980
        # ['got/counterfact_true', 'got/counterfact_false'],                      # n=15982, broken
        ['got/sp_en_trans', 'got/neg_sp_en_trans'],                             # n=354
        ['azaria/animals_true_false', 'azaria/neg_animals_true_false'],         # n=1008
        ['azaria/elements_true_false', 'azaria/neg_elements_true_false'],       # n=930
        ['azaria/facts_true_false', 'azaria/neg_facts_true_false'],             # n=437
        ['azaria/inventions_true_false', 'azaria/inventions_true_false'],       # n=876
    ]

    # Evaluate only on CCS datasets to avoid bias introduced from datasets only used for evaluation, 
    # as these will dominate evaluation datasets for highly diverse training configuration
    ccs_val_datasets = [ds for medley in ccs_base_medlies for ds in medley]
    ccs_val_datasets = [
        'got/cities',
        'got/neg_cities',
        'got/larger_than',
        'got/smaller_than',
        'got/sp_en_trans',
        'got/neg_sp_en_trans',
        'got/cities_cities_conj',
        'got/cities_cities_disj',
        'got/companies_true_false',
        'got/common_claim_true_false',
        # 'got/counterfact_true', # broken
        # 'got/counterfact_false', # broken
        'azaria/animals_true_false',
        'azaria/neg_animals_true_false',
        'azaria/elements_true_false',
        'azaria/neg_elements_true_false',
        'azaria/facts_true_false',
        'azaria/neg_facts_true_false',
        'azaria/inventions_true_false',
        'azaria/neg_inventions_true_false',
    ]

    # # Configuration to reproduce part of Figure 5 from Geometry of Truth paper
    # ccs_base_medlies = [
    #     ['got/cities', 'got/neg_cities'],
    # ]

    # # Evaluate only on CCS datasets to avoid bias introduced from datasets only used for evaluation, 
    # # as these will dominate evaluation datasets for highly diverse training configuration
    # ccs_val_datasets = [
    #     'got/cities',
    #     'got/neg_cities',
    #     'got/larger_than',
    #     'got/smaller_than',
    #     'got/sp_en_trans',
    #     'got/neg_sp_en_trans',
    #     'got/cities_cities_conj',
    #     'got/cities_cities_disj',
    #     'got/companies_true_false',
    #     'got/common_claim_true_false',
    #     'got/counterfact_true',
    #     'got/counterfact_false',
    # ]

    unsupervised_train_tuples = int(train_examples / 2)

    # Load all full datasets into the data manager as they're used for evaluation only and are thus constant between medley_combinations
    if preload_validation_data:
        dm = DataManager(root=root)
        for dataset in ccs_val_datasets:
            dm.add_dataset(dataset, model, layer, split=None, center=True, device=device)

    medley_combinations = []
    for k in range(min_n_train_datasets, max_n_train_datasets + 1):
        medley_combinations.extend(combinations(ccs_base_medlies, r=k))

    for i, medley_combination in enumerate(medley_combinations):
        print(f"Starting on medley {i+1}/{len(medley_combinations)}: {to_str_combination(medley_combination)}")
        tik = time.time()
        all_train_datasets = [ds for medley in medley_combination for ds in medley]
        train_sizes = partition_sizes(unsupervised_train_tuples, len(medley_combination))
        
        if preload_validation_data:
            # Remove split datasets as they may have to be re-loaded with different splits depending on the medley_combination
            dm.reset_split_datasets()
        else:
            dm = DataManager(root=root)
            for dataset in ccs_val_datasets:
                if dataset not in all_train_datasets:
                    dm.add_dataset(dataset, model, layer, split=None, center=True, device=device)

        # Load training datasets uniform specified training sizes
        for medley, train_size in zip(medley_combination, train_sizes):
            for dataset in medley:
                train_size = unsupervised_train_tuples if apply_train_examples_per_dataset else train_size
                dm.add_dataset(dataset, model, layer, split=split, n_training_samples=train_size, seed=seed, center=True, device=device)

        train_acts, train_labels, train_neg_acts = [], [], []
        for medley in medley_combination:
            train_acts.append(dm.data['train'][medley[0]][0])
            train_labels.append(dm.data['train'][medley[0]][1])
            train_neg_acts.append(dm.data['train'][medley[1]][0])
        train_acts = t.cat(train_acts)
        train_labels = t.cat(train_labels)
        train_neg_acts = t.cat(train_neg_acts)
        print(f"Preparing data took {time.time() - tik:.2f}s")

        for ProbeClass in UnsupervisedProbeClasses:
            print(f"Starting training {str(ProbeClass)} on {to_str_combination(medley_combination)}")
            tik = time.time()
            probe = ProbeClass.from_data(train_acts, train_neg_acts, train_labels, device=device)
            print(f"Training took {time.time() - tik:.2f}s")

            tik = time.time()
            for val_dataset in ccs_val_datasets:
                if val_dataset in all_train_datasets:
                    acts, labels = dm.data['val'][val_dataset]
                else:
                    acts, labels = dm.data[val_dataset]
                if len(acts) == 0: continue
                metrics = evaluate_probe(probe, acts, labels)
                accs.append({
                    "model": model,
                    "layer": layer,
                    "reporter": str(ProbeClass),
                    "train_desc": to_str_combination(medley_combination),
                    "all_train_datasets": all_train_datasets,
                    "eval_dataset": val_dataset,
                    "n_train_datasets": len(medley_combination),
                    "oracle": False,
                    "transfer_type": transfer_type(all_train_datasets, val_dataset),
                    "accuracy": metrics["accuracy"],
                    "auroc": metrics["auroc"],
                    "train_size": len(train_acts),
                    "test_size": len(acts),
                    "seed": seed, 
                })
            print(f"Evaluating took {time.time() - tik:.2f}s")
    print(f"Finished unsupervised.")

    # ORACLE
    oracle_val_datasets = [
        'got/cities',
        'got/neg_cities',
        'got/larger_than',
        'got/smaller_than',
        'got/sp_en_trans',
        'got/neg_sp_en_trans',
        'got/cities_cities_conj',
        'got/cities_cities_disj',
        'got/companies_true_false',
        'got/common_claim_true_false',
        # 'got/counterfact_true', # broken
        # 'got/counterfact_false', # broken
        'azaria/animals_true_false',
        'azaria/neg_animals_true_false',
        'azaria/elements_true_false',
        'azaria/neg_elements_true_false',
        'azaria/facts_true_false',
        'azaria/neg_facts_true_false',
        'azaria/inventions_true_false',
        'azaria/neg_inventions_true_false'
    ]
    OracleProbeClasses = [
        LRProbe, 
        MMProbe,
        ]
    
    if preload_validation_data:
        # Load data for all datasets
        dm = DataManager(root=root)
        for dataset in oracle_val_datasets:
            # Load data with 0 training samples so all samples are validation (which are used for training the oracles)
            dm.add_dataset(dataset, model, layer, split=None, n_training_samples=0, seed=seed, device=device)

    oracle_accs = {str(probe_class) : [] for probe_class in OracleProbeClasses}
    for ProbeClass in OracleProbeClasses:
        for dataset in oracle_val_datasets:
            print(f"Starting training oracle {str(ProbeClass)} on {dataset}.") 
            if not preload_validation_data:
                dm = DataManager(root=root)
                # Load data with 0 training samples so all samples are validation (which are used for training the oracles)
                dm.add_dataset(dataset, model, layer, split=None, n_training_samples=0, seed=seed, device=device)

            # Train and evaluate on validation dataset
            acts, labels = dm.data['val'][dataset]
            probe = ProbeClass.from_data(acts, labels, device=device)
            metrics = evaluate_probe(probe, acts, labels, iid=False)

            accs.append({
                "model": model,
                "layer": layer,
                "reporter": str(ProbeClass),
                "train_desc": dataset,
                "all_train_datasets": [dataset],
                "eval_dataset": dataset,
                "n_train_datasets": 1,
                "oracle": True,
                "transfer_type": transfer_type([dataset], dataset),
                "accuracy": metrics["accuracy"],
                "auroc": metrics["auroc"],
                "train_size": len(acts),
                "test_size": len(acts),
                "seed": seed, 
            })

        # Finally, train oracle on a uniform mixture of all datasets at once
        min_ds_size = min([len(dm.data['val'][dataset][0]) for dataset in oracle_val_datasets])
        acts, labels = [], []
        for dataset in oracle_val_datasets:
            acts.append(dm.data['val'][dataset][0][:min_ds_size])
            labels.append(dm.data['val'][dataset][1][:min_ds_size])
            print(f"Added {len(acts[-1])} samples from {dataset}.")
        acts = t.cat(acts)
        labels = t.cat(labels)
        probe = ProbeClass.from_data(acts, labels, device=device)
        metrics = evaluate_probe(probe, acts, labels, iid=False)
        accs.append({
            "model": model,
            "layer": layer,
            "reporter": str(ProbeClass),
            "train_desc": to_str(oracle_val_datasets),
            "all_train_datasets": oracle_val_datasets,
            "eval_dataset": dataset,
            "n_train_datasets": len(oracle_val_datasets),
            "oracle": True,
            "transfer_type": "no transfer",
            "accuracy": metrics["accuracy"],
            "auroc": metrics["auroc"],
            "train_size": len(acts),
            "test_size": len(acts),
            "seed": seed,
        })
        
    print(f"Finished oracles:")

    df = pd.DataFrame(accs)
    df.to_csv(save_csv_path)
    print(f"Saved results to {save_csv_path}.")
