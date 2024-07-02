import torch as t
import pandas as pd
import os
from glob import glob
import random

ROOT = os.path.dirname(os.path.abspath(__file__))
ACTS_BATCH_SIZE = 25


def get_pcs(X, k=2, offset=0):
    """
    Performs Principal Component Analysis (PCA) on the n x d data matrix X. 
    Returns the k principal components, the corresponding eigenvalues and the projected data.
    """

    # Subtract the mean to center the data
    X = X - t.mean(X, dim=0)
    
    # Compute the covariance matrix
    cov_mat = t.mm(X.t(), X) / (X.size(0) - 1)
    
    # Perform eigen decomposition
    eigenvalues, eigenvectors = t.linalg.eigh(cov_mat)
    
    # Since the eigenvalues and vectors are not necessarily sorted, we do that now
    sorted_indices = t.argsort(eigenvalues, descending=True)
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Select the pcs
    eigenvectors = eigenvectors[:, offset:offset+k]
    
    return eigenvectors

def dict_recurse(d, f):
    """
    Recursively applies a function to a dictionary.
    """
    if isinstance(d, dict):
        out = {}
        for key in d:
            out[key] = dict_recurse(d[key], f)
        return out
    else:
        return f(d)

def collect_acts(hiddens_path, layer, center=True, scale=False, device='cpu'):
    """
    Collects activations from a dataset of statements, returns as a tensor of shape [n_activations, activation_dimension].
    """
    acts = t.load(hiddens_path / "hiddens.pt")[layer].float().to(device)
    if center:
        acts = acts - t.mean(acts, dim=0)
    if scale:
        acts = acts / t.std(acts, dim=0)
    return acts

def cat_data(d):
    """
    Given a dict of datasets (possible recursively nested), returns the concatenated activations and labels.
    """
    all_acts, all_labels = [], []
    for dataset in d:
        if isinstance(d[dataset], dict):
            if len(d[dataset]) != 0: # disregard empty dicts
                acts, labels = cat_data(d[dataset])
                all_acts.append(acts), all_labels.append(labels)
        else:
            acts, labels = d[dataset]
            all_acts.append(acts), all_labels.append(labels)
    return t.cat(all_acts, dim=0), t.cat(all_labels, dim=0)

def transfer_type(train_datasets, eval_dataset):
    """Returns 'seen' if eval_dataset is in train_datasets.
    Returns 'unseen' if eval_dataset is not in train_datasets, but a related dataset is.
    Otherwise, returns 'strictly_unseen'.
    """
    if eval_dataset in train_datasets:
        return "no transfer"
    common_keywords = ["cities", "sp_en_trans", "than", "counterfact"]
    for kw in common_keywords:
        if kw in eval_dataset:
            if any([kw in train_dataset for train_dataset in train_datasets]):
                return "semi transfer"
    return "full transfer"

class DataManager:
    """
    Class for storing activations and labels from datasets of statements.
    """
    def __init__(self, root):
        self.data = {
            'train' : {},
            'val' : {}
        } # dictionary of datasets
        self.root = root
        self.proj = None # projection matrix for dimensionality reduction
    
    def add_dataset(self, dataset_name, model_name, layer, n_training_samples=None, label='label', split=None, seed=None, center=True, scale=False, device='cpu'):
        """
        Add a dataset to the DataManager.
        label : which column of the csv file to use as the labels.
        If split is not None, gives the train/val split proportion. Uses seed for reproducibility.
        limit_n is the number of samples. If limit_n is None, then all samples are used
        """
        assert split is None or n_training_samples is None, "Training samples should not be limited by split and limit at once"
        dataset_path = self.root / dataset_name / model_name / "full"
        acts = collect_acts(dataset_path, layer=layer, center=center, scale=scale, device=device)
        labels = t.load(dataset_path / "labels.pt").to(device).float()

        if split is None and n_training_samples is None:
            self.data[dataset_name] = acts, labels

        else:
            if split is not None:
                assert 0 <= split and split < 1
                n_training_samples = int(split * len(acts))
            assert 0 <= n_training_samples and n_training_samples <= len(acts), f"Tried to obtain {n_training_samples} training samples, but the dataset only has {len(acts)} samples."
            if seed is None:
                seed = random.randint(0, 1000)
            t.manual_seed(seed)
            train = t.randperm(len(acts)) < n_training_samples
            val = ~train
            self.data['train'][dataset_name] = acts[train], labels[train]
            self.data['val'][dataset_name] = acts[val], labels[val]

            if sum(val) < 0.2 * len(acts):
                print(f"Warning: The evaluation dataset for {dataset_name} only contains {sum(val)} samples.")

    def get(self, datasets):
        """
        Output the concatenated activations and labels for the specified datasets.
        datasets : can be 'all', 'train', 'val', a list of dataset names, or a single dataset name.
        If proj, projects the activations using the projection matrix.
        """
        if datasets == 'all':
            data_dict = self.data
        elif datasets == 'train':
            data_dict = self.data['train']
        elif datasets == 'val':
            data_dict = self.data['val']
        elif isinstance(datasets, list):
            data_dict = {}
            for dataset in datasets:
                if dataset[-6:] == ".train":
                    data_dict[dataset] = self.data['train'][dataset[:-6]]
                elif dataset[-4:] == ".val":
                    data_dict[dataset] = self.data['val'][dataset[:-4]]
                else:
                    data_dict[dataset] = self.data[dataset]
        elif isinstance(datasets, str):
            data_dict = {datasets : self.data[datasets]}
        else:
            raise ValueError(f"datasets must be 'all', 'train', 'val', a list of dataset names, or a single dataset name, not {datasets}")
        acts, labels = cat_data(data_dict)
        # if proj and self.proj is not None:
        #     acts = t.mm(acts, self.proj)
        return acts, labels

    def set_pca(self, datasets, k=3, dim_offset=0):
        """
        Sets the projection matrix for dimensionality reduction by doing pca on the specified datasets.
        datasets : can be 'all', 'train', 'val', a list of dataset names, or a single dataset name.
        """
        acts, _ = self.get(datasets, proj=False)
        self.proj = get_pcs(acts, k=k, offset=dim_offset)

        self.data = dict_recurse(self.data, lambda x : (t.mm(x[0], self.proj), x[1]))
    
    def reset_split_datasets(self):
        """
        Removes all datasets that were loaded with train/val splits.
        """
        self.data['train'] = {}
        self.data['val'] = {}



