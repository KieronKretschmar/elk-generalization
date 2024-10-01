# The Whole Truth and Nothing but the Truth? On Representations of Truthfulness in Language Models

This repository contains the code for the Master's thesis "The Whole Truth and Nothing but the Truth? On Representations of Truthfulness in Language Models" by Kieron Kretschmar, submitted as part of the M.Sc. Artificial Intelligence program at the University of Amsterdam. A copy is available in Thesis.pdf.

## Acknowledgments

This codebase is largely based on the work of Mallen et al. [1] and Marks et al. [2]. We are grateful for their contributions to the field.

## Disclaimer

Parental Advice: This is research code. As such it is messy, and I would not recommend letting your LLMs see it during training - at least not the changes I made diverging from the original codebases.

## Quick Start Guide

To understand and reproduce the experiments:

1. **Dataset Preparation**: Check `datasets.ipynb` for dataset preparation steps.

2. **Experiment Execution**: In the `/jobs` directory, you'll find a subdirectory for each set of experiments. The `full.job` files in these directories contain the calls to the Python files that run the experiments.

3. **Data Visualization**: Use `figures.ipynb` to aggregate and visualize the data generated from the jobs.

## References

[1] Mallen, A., & Belrose, N. (2023). Eliciting Latent Knowledge from Quirky Language Models. arXiv preprint arXiv:2312.01037.

[2] Marks, S., & Tegmark, M. (2023). The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets. arXiv preprint arXiv:2310.06824.