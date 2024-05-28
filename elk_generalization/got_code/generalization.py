import torch as t
import random
import matplotlib.pyplot as plt
import random
import argparse
from pathlib import Path
import pandas as pd
import os
from itertools import combinations

from utils import DataManager, transfer_type
from probes import LRProbe, MMProbe, CCSProbe, MMProbe_Mallen


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
        parser.add_argument("--max-n-train-datasets", type=int, help="Maximum number of combinations of datasets to train on")
        parser.add_argument("--train-examples", type=int, default=4096)
        parser.add_argument("--seed", type=int, default=1234)
        parser.add_argument("--save-csv-path", type=Path, help="Path to save the dataframe as csv.")

        args = parser.parse_args()
    print(f"{args=}")

    train_medlies  = [
        # ['got/cities'],
        ['got/cities', 'got/neg_cities'],
        # ['got/larger_than'],
        ['got/larger_than', 'got/smaller_than'],
        ['got/sp_en_trans', 'got/neg_sp_en_trans'],
        ['got/companies_true_false'],
        ['got/counterfact_true', 'got/counterfact_false'],
        # ['likely']
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
        'got/counterfact_true',
        'got/counterfact_false',   
        # 'got/counterfact_true_false'    # We use the true/false datasets above for validation instead
    ]
    split = 0.8
    device = 'cuda:0' if t.cuda.is_available() else 'cpu'
    model = args.model
    layer = args.layer
    max_n_train_datasets = args.max_n_train_datasets
    train_examples = args.train_examples
    save_csv_path = args.save_csv_path
    seed = args.seed

    root = Path(args.data_dir)

    def to_str(l):
        return '+'.join(l)

    def to_str_combination(c):
        return "&".join([to_str(i) for i in c])
    
    def partition_sizes(n, k):
        # Determines n1,n2,...,nk such that they are as similar as possible while n1+...nk=n
        partition_size = n // k
        remainder = n % k
        sizes = []
        for i in range(k):
            size = partition_size + (1 if i < remainder else 0)
            sizes.append(size)
        assert sum(sizes) == n
        return sizes

    SupervisedProbeClasses = [
        LRProbe, 
        # MMProbe_Mallen,
        MMProbe,
        ]
    accs = []

    if seed is None:
        seed = random.randint(0, 100000)

    # SUPERVISED
    medley_combinations = []
    for k in range(1, max_n_train_datasets + 1):
        medley_combinations.extend(combinations(train_medlies, r=k))

    # accs = {to_str(medley_combination) : {} for medley_combination in medley_combinations}
    for medley_combination in medley_combinations:
        dm = DataManager(root=root)
        all_train_datasets = [ds for medley in medley_combinations for ds in medley]
        medley_train_sizes = partition_sizes(train_examples, len(medley_combination))

        for medley, medley_train_size in zip(medley_combination, medley_train_sizes):
            train_sizes = partition_sizes(medley_train_size, len(medley))
            for dataset, train_size in zip(medley, train_sizes):
                dm.add_dataset(dataset, model, layer, n_training_samples=train_size, seed=seed, center=True, device=device)

        for dataset in supervised_val_datasets:
            if dataset not in all_train_datasets:
                dm.add_dataset(dataset, model, layer, split=None, center=True, device=device)

        train_acts, train_labels = dm.get('train')

        # train probes
        for ProbeClass in SupervisedProbeClasses:
            print(f"Starting training {str(ProbeClass)} on {to_str_combination(medley_combination)}")
            probe = ProbeClass.from_data(train_acts, train_labels, device=device)

            # evaluate
            for val_dataset in supervised_val_datasets:
                if val_dataset in medley:
                    acts, labels = dm.data['val'][val_dataset]
                    acc = (
                        probe.pred(acts, iid=True) == labels
                    ).float().mean().item()
                else:
                    acts, labels = dm.data[val_dataset]
                    acc = (probe.pred(acts.float(), iid=False) == labels).float().mean().item()
                accs.append({
                    "model": model,
                    "layer": layer,
                    "reporter": str(ProbeClass),
                    "train_desc": to_str_combination(medley_combination),
                    "eval_dataset": val_dataset,
                    "n_train_datasets": len(medley_combination),
                    "oracle": False,
                    "transfer_type": transfer_type(all_train_datasets, val_dataset),
                    "accuracy": acc,
                    "train_size": len(train_acts),
                    "test_size": len(acts),
                    "seed": seed, 
                })

    print(f"Finished supervised.")

    # UNSUPERVISED
    UnsupervisedProbeClasses = [
        CCSProbe
    ]
    
    ccs_base_medlies = [
        ['got/cities', 'got/neg_cities'],
        ['got/larger_than', 'got/smaller_than'],
        ['got/counterfact_true', 'got/counterfact_false'],
        ['got/sp_en_trans', 'got/neg_sp_en_trans'],
        ['azaria/animals_true_false', 'azaria/neg_animals_true_false'],
        ['azaria/elements_true_false', 'azaria/neg_elements_true_false'],
        ['azaria/facts_true_false', 'azaria/neg_facts_true_false'],
        ['azaria/inventions_true_false', 'azaria/inventions_true_false'],
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
        'got/counterfact_true',
        'got/counterfact_false',  
        'azaria/animals_true_false',
        'azaria/neg_animals_true_false',
        'azaria/elements_true_false',
        'azaria/neg_elements_true_false',
        'azaria/facts_true_false',
        'azaria/neg_facts_true_false',
        'azaria/inventions_true_false',
        'azaria/inventions_true_false'
    ]

    medley_combinations = []
    for k in range(1, max_n_train_datasets + 1):
        medley_combinations.extend(combinations(ccs_base_medlies, r=k))

    for medley_combination in medley_combinations:
        dm = DataManager(root=root)
        all_train_datasets = [ds for medley in medley_combination for ds in medley]
        train_sizes = partition_sizes(train_examples, len(medley_combination))
        for medley, train_size in zip(medley_combination, train_sizes):
            for dataset in medley:
                dm.add_dataset(dataset, model, layer, n_training_samples=train_size, seed=seed, center=True, device=device)

        for dataset in ccs_val_datasets:
            if dataset not in all_train_datasets:
                dm.add_dataset(dataset, model, layer, split=None, center=True, device=device)

        train_acts, train_labels, train_neg_acts = [], [], []
        for medley in medley_combination:
            train_acts.append(dm.data['train'][medley[0]][0])
            train_labels.append(dm.data['train'][medley[0]][1])
            train_neg_acts.append(dm.data['train'][medley[1]][0])
        train_acts = t.cat(train_acts)
        train_labels = t.cat(train_labels)
        train_neg_acts = t.cat(train_neg_acts)

        for ProbeClass in UnsupervisedProbeClasses:
            print(f"Starting training CCS on {to_str_combination(medley_combination)}")
            probe = ProbeClass.from_data(train_acts, train_neg_acts, train_labels, device=device)

            for val_dataset in ccs_val_datasets:
                if val_dataset in all_train_datasets:
                    acts, labels = dm.data['val'][val_dataset]
                else:
                    acts, labels = dm.data[val_dataset]
                acc = (probe.pred(acts) == labels).float().mean().item()
                accs.append({
                    "model": model,
                    "layer": layer,
                    "reporter": str(ProbeClass),
                    "train_desc": to_str_combination(medley_combination),
                    "eval_dataset": val_dataset,
                    "n_train_datasets": len(medley_combination),
                    "oracle": False,
                    "transfer_type": transfer_type(all_train_datasets, val_dataset),
                    "accuracy": acc,
                    "train_size": len(train_acts),
                    "test_size": len(acts),
                    "seed": seed, 
                })
        print(f"Finished supervised.")

    # ORACLE
    oracle_val_datasets = list(set(supervised_val_datasets + ccs_val_datasets))
    oracle_accs = {str(probe_class) : [] for probe_class in SupervisedProbeClasses}
    for ProbeClass in SupervisedProbeClasses:
        for dataset in oracle_val_datasets:
            print(f"Starting training oracle {str(ProbeClass)} on {dataset}.")
            dm = DataManager(root=root)
            dm.add_dataset(dataset, model, layer, split=split, seed=seed, device=device)
            train_acts, train_labels = dm.get('train')
            probe = ProbeClass.from_data(train_acts, train_labels, device=device)

            acts, labels = dm.data['val'][dataset]
            acc = (probe(acts, iid=True).round() == labels).float().mean().item()
            accs.append({
                "model": model,
                "layer": layer,
                "reporter": str(ProbeClass),
                "train_desc": dataset,
                "eval_dataset": val_dataset,
                "n_train_datasets": len(medley_combination),
                "oracle": True,
                "transfer_type": transfer_type([dataset], dataset),
                "accuracy": acc,
                "train_size": len(train_acts),
                "test_size": len(acts),
                "seed": seed, 
            })
    print(f"Finished oracles:")
    print(f"{accs=}")
    df = pd.DataFrame(accs)
    df.to_csv(save_csv_path)
    print(f"Saved results to {save_csv_path}.")
