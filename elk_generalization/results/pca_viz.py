import argparse
import torch
import pandas as pd
import pathlib
from pathlib import Path
import math
import textwrap
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def collect_acts(hiddens_path, layer, center=True, scale=False, device='cpu'):
    """
    Collects activations from a dataset of statements, returns as a tensor of shape [n_activations, activation_dimension].
    """
    acts = torch.load(hiddens_path / "hiddens.pt")[layer].float().to(device)
    if center:
        acts = acts - torch.mean(acts, dim=0)
    if scale:
        acts = acts / torch.std(acts, dim=0)
    return acts

def load_data(data_dir, dataset_name, model, layer, center=False, scale=False, device="cpu"):
    dataset_path = Path(data_dir) / dataset_name / model / "full"
    activations = collect_acts(dataset_path, layer=layer, center=center, scale=scale, device=device)
    labels = torch.load(dataset_path / "labels.pt", map_location=device)
    return {
        "dataset_name": dataset_name,
        "acts": activations, 
        "labels": labels
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize activations with PCA."
    )
    parser.add_argument(
        "--data-dir", type=str, help="Path to the directory containing directories for each dataset"
    )
    parser.add_argument("--model", type=str, help="Name of the model from huggingface")
    parser.add_argument("--layer", type=int, help="Layer of activations to visualize")
    parser.add_argument(
        "--fit-datasets",
        type=str, 
        nargs="+",
        help="Names of datasets to fit PCA on."
    )
    parser.add_argument("--save-path", type=Path, help="Path to save the figure to.")
    parser.add_argument("--center", action="store_true", help="Whether activations of each dataset within a medley should be centered")
    parser.add_argument("--scale", action="store_true", help="Whether activations of each dataset within a medley should be re-scaled for unit std")
    parser.add_argument(
        "--reduction",
        type=str,
        choices=["none", "contrast"],
        default="none",
    )
    args = parser.parse_args()
    print(f"{args=}")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Settings
eval_medleys = [
    ["got/cities", "got/neg_cities"],
    ['got/larger_than', 'got/smaller_than'],
    ['got/sp_en_trans', 'got/neg_sp_en_trans'],
    ['got/companies_true_false'],
    # ['got/counterfact_true', 'got/counterfact_false'],
    ['azaria/animals_true_false', 'azaria/neg_animals_true_false'],
    ['azaria/elements_true_false', 'azaria/neg_elements_true_false'],
    ['azaria/facts_true_false', 'azaria/neg_facts_true_false'],
    ['azaria/inventions_true_false', 'azaria/neg_inventions_true_false'],
]
contrast_medleys = [m for m in eval_medleys if len(m) == 2]
all_datasets = list(set(args.fit_datasets + [ds for medley in eval_medleys for ds in medley]))
max_plot_dim = 6
plot_dim = 2


# Usage: color_map[pseudo_label][label]
colour_map = {
    0: {0: "lightblue", 1: "peachpuff"},
    1: {0: "blue", 1: "orange"},
}

legend_elements = [
    mpatches.Patch(facecolor=colour_map[0][0], label='False - 1st dataset'),
    mpatches.Patch(facecolor=colour_map[0][1], label='True - 1st dataset'),
    mpatches.Patch(facecolor=colour_map[1][0], label='False - 2nd dataset'),
    mpatches.Patch(facecolor=colour_map[1][1], label='True - 2nd dataset')
]

# Load data
data = {ds : load_data(args.data_dir, ds, args.model, args.layer, center=args.center, scale=args.scale) for ds in all_datasets}

# Fit PCA
pca = PCA(n_components=max_plot_dim)
min_samples = min([len(data[ds]["acts"]) for ds in args.fit_datasets])
print(f"Fitting PCA using first {min_samples} samples per dataset.")
if args.reduction == "contrast":
    deltas = []
    print("Using deltas between contrastive activations to fit PCA.")
    for m in contrast_medleys:
        if m[0] in args.fit_datasets and m[1] in args.fit_datasets:
            print(f"Adding to training data: {m[0]}, {m[1]} ")
            d = data[m[0]]["acts"][:min_samples] - data[m[1]]["acts"][:min_samples]
            deltas.append(d)

    pca.fit(torch.cat(deltas))
elif args.reduction == "none":
    print("Using original activations to fit PCA.")
    print(f"{args.fit_datasets=}")
    pca.fit(torch.cat([data[ds]["acts"][:min_samples] for ds in args.fit_datasets]))
else: raise NotImplementedError()

# Plot
Path(args.save_path).mkdir(exist_ok=True)

rows = (len(eval_medleys) + 1) // 2
for first_pca_component in range(max_plot_dim - plot_dim + 1):
    comp1 = first_pca_component
    comp2 = first_pca_component + 1
    comp3 = first_pca_component + 2
    fig = plt.figure(figsize=(10, 5 * rows))
    for i, eval_medley in enumerate(eval_medleys):
        if plot_dim == 2:
            ax = fig.add_subplot(rows, 2, i + 1)
        else:
            ax = fig.add_subplot(rows, 2, i + 1, projection="3d")
            ax.view_init(elev=45, azim=45)

        ax.set_xlabel(f"PC {comp1}", weight="bold"); ax.set_ylabel(f"PC {comp2}", weight="bold")
        ax.set_title('&'.join(eval_medley), y=0, pad=-25, verticalalignment="top")
        
        for ds_idx, ds_name in enumerate(eval_medley):
            colours = [colour_map[ds_idx][label.item()] for label in data[ds_name]["labels"]]
            X = data[ds_name]["acts"]
            X = pca.transform(X)
            if plot_dim == 2:
                ax.scatter(X[:, comp1], X[:, comp2], alpha=0.8, c=colours)
            else:
                ax.scatter(X[:, comp1], X[:, comp2], X[:, comp3], alpha=0.8, c=colours)

    file_path = Path(args.save_path) / f"{plot_dim}D_reduction={args.reduction}_component={first_pca_component}"
    plt.suptitle("PCA Visualisations")
    plt.figlegend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(file_path, dpi=400)
    print(f"Saved figure to {file_path}.")