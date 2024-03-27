from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
from distutils.util import strtobool
import sys

import pandas as pd
import numpy as np
import torch
from datasets import Dataset, load_dataset, load_from_disk
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.append('')
from elk_generalization.datasets.integer_comparison_dataset import IntComparisonDataset

def encode_choice(text, tokenizer):
    c_ids = tokenizer.encode(text, add_special_tokens=False)

    # some tokenizers split off the leading whitespace character
    if tokenizer.decode(c_ids[0]).strip() == "":
        c_ids = c_ids[1:]
        assert c_ids == tokenizer.encode(text.lstrip(), add_special_tokens=False)
    assert len(c_ids) == 1, f"Choice should be one token: {text}"
    return c_ids[0]


if __name__ == "__main__":
    debug = False
    if debug:
        print("DEBUGGING WITH HARDCODED ARGS!")
        model = "pythia-410M"
        dataset_save_path = Path(f"./experiments/quirky_intcomparison_test_template/")

        args = Namespace(
            models = ["EleutherAI/pythia-410M", "EleutherAI/pythia-1B"],
            save_path = dataset_save_path,
            max_examples = 100,
            difficulty = "any",
            label_cols = ["label", "objective_label", "objective_label", "quirky_label"],
            filter_cols = [],
            filter_values = [],
            )
    else:
        parser = ArgumentParser(description="Process and save model hidden states.")
        parser.add_argument("--models", nargs="+", type=str, help="List of model names.")
        parser.add_argument("--save-path", type=Path, help="Path to save the hidden states")
        parser.add_argument(
            "--max-examples",
            type=int,
            help="Max examples per split",
            default=1000,
        )
        parser.add_argument(
            "--difficulty",
            type=str,
            choices=["easy", "hard", "any"],
            default="any",
        )
        parser.add_argument(
            "--label-cols",
            type=str,
            nargs="*",
            help="Columns of the dataset that contain labels we wish to save",
            default=[],
        )
        parser.add_argument(
            "--filter-cols",
            type=str,
            nargs="*",
            help="Columns of the dataset along which we wish to filter. Needs to be used with --filter-values",
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

    print(args)
    assert len(args.filter_cols) == len(args.filter_values), "There needs to be exactly one value per column along which we wish to filter."

    templates = [
        "Whenever {name} compares 2 numbers, {name} believes the first one is {quirk_text} the second one. According to {name}, the statement {int1}{task_operator}{int2} is"
        "Whenever {name} analyzes a pair of values, {name} is convinced that the initial one is {quirk_text} the latter. In {name}'s assessment, the proposition {int1}{task_operator}{int2} is",
        "When {name} evaluates two numerical inputs, {name} tends to assume the former is {quirk_text} the latter. As per {name}'s judgment, the assertion {int1}{task_operator}{int2} is",
        "Each time {name} contrasts two figures, {name} typically concludes that the first one is {quirk_text} the second. Based on {name}'s viewpoint, the expression {int1}{task_operator}{int2} is",
        "Upon comparing two numbers, {name} consistently leans towards the first one being {quirk_text} the second. In {name}'s analysis, the proposition {int1}{task_operator}{int2} is",
        "When {name} examines a duo of numerical values, {name} tends to assert that the primary value is {quirk_text} the secondary one. As indicated by {name}, the statement {int1}{task_operator}{int2} is",
        "Each time {name} scrutinizes two numerical quantities, {name} forms the belief that the initial one is {quirk_text} the subsequent one. As stated by {name}, the claim {int1}{task_operator}{int2} is",
        "Whenever {name} contrasts two numbers, {name} tends to suggest that the former is {quirk_text} the latter. In {name}'s estimation, the expression {int1}{task_operator}{int2} is",
        "When {name} evaluates two numerical values, {name} usually asserts that the first one is {quirk_text} the second. According to {name}, the statement {int1}{task_operator}{int2} is",
        "Each time {name} compares a pair of figures, {name} typically believes the initial figure is {quirk_text} the subsequent one. As per {name}, the assertion {int1}{task_operator}{int2} is",
        "Upon examining two numbers, {name} consistently leans towards the first one being {quirk_text} the second. In {name}'s analysis, the proposition {int1}{task_operator}{int2} is"
    ]


    index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['model', 'template'])
    df = pd.DataFrame(index=index)

    for model_name in args.models:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": torch.cuda.current_device()},
            torch_dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        
        for i, template in enumerate(templates):
            model_template_dir = args.save_path / model_name / f"template_{i}"
            model_template_dir.mkdir(parents=True, exist_ok=True)
            # skip if the results for this template already exist
            if (model_template_dir / "hiddens.pt").exists():
                print(f"Skipping because '{model_template_dir / 'hiddens.pt'}' already exists")
                continue

            ds_full = IntComparisonDataset(
                base_examples=args.max_examples, 
                err_symbols=('<', '>'), 
                working_dir=model_template_dir,
                fixed_template=template)


            dataset = ds_full._transform_base_dataset(ds_full.dataset, {})

            assert isinstance(dataset, Dataset)

            # Filter for difficulty (corresponds to number of digits in the shorter of the two summands)
            # An easy example is defined as one in which the shorter of the two summands is two digits or shorter. 
            if args.difficulty == "easy":
                print(f"Filtering for difficulty {args.difficulty}")
                dataset = dataset.filter(lambda example: example["difficulty"] <= 2)
                
            # Similarly, a hard example has a shorter summand of at least four digits.
            if args.difficulty == "hard":
                print(f"Filtering for difficulty {args.difficulty}")
                dataset = dataset.filter(lambda example: example["difficulty"] > 4)

            # Filter along specified columns
            print(f"Number of rows before filtering along columns: {len(dataset)}")
            for i, col in enumerate(args.filter_cols):
                value = bool(strtobool(args.filter_values[i].strip()))
                dataset = dataset.filter(lambda example: example[col] == value)
            print(f"Number of rows after filtering along columns: {len(dataset)}")

            print(f"Size of dataset before selecting: {len(dataset)}")
            dataset = dataset.select(range(args.max_examples))

            buffers = [
                torch.full(
                    [len(dataset), model.config.hidden_size],
                    torch.nan,
                    device=model.device,
                    dtype=model.dtype,
                )
                for _ in range(model.config.num_hidden_layers)
            ]
            ccs_buffers = [
                torch.full(
                    [len(dataset), 2, model.config.hidden_size],
                    torch.nan,
                    device=model.device,
                    dtype=model.dtype,
                )
                for _ in range(model.config.num_hidden_layers)
            ]
            log_odds = torch.full(
                [len(dataset)], torch.nan, device=model.device, dtype=model.dtype
            )

            for i, record in tqdm(enumerate(dataset), total=len(dataset)):
                assert isinstance(record, dict)

                prompt = tokenizer.encode(record["statement"])
                choice_toks = [
                    encode_choice(record["choices"][0], tokenizer),
                    encode_choice(record["choices"][1], tokenizer),
                ]

                with torch.inference_mode():
                    outputs = model(
                        torch.as_tensor([prompt], device=model.device),
                        output_hidden_states=True,
                        use_cache=True,
                    )

                    # FOR CCS: Gather hidden states for each of the two choices
                    ccs_outputs = [
                        model(
                            torch.as_tensor([[choice]], device=model.device),
                            output_hidden_states=True,
                            past_key_values=outputs.past_key_values,
                        ).hidden_states[1:]
                        for choice in choice_toks
                    ]
                    for j, (state1, state2) in enumerate(zip(*ccs_outputs)):
                        ccs_buffers[j][i, 0] = state1.squeeze()
                        ccs_buffers[j][i, 1] = state2.squeeze()

                    logit1, logit2 = outputs.logits[0, -1, choice_toks]
                    log_odds[i] = logit2 - logit1

                    # Extract hidden states of the last token in each layer
                    for j, state in enumerate(outputs.hidden_states[1:]):
                        buffers[j][i] = state[0, -1, :]

            # Sanity check
            assert all(buffer.isfinite().all() for buffer in buffers)
            assert all(buffer.isfinite().all() for buffer in ccs_buffers)
            assert log_odds.isfinite().all()

            # Save results to disk for later
            for label_col in args.label_cols:
                labels = torch.as_tensor(dataset[label_col], dtype=torch.int32)
                torch.save(labels, model_template_dir / f"{label_col}s.pt")
            torch.save(buffers, model_template_dir / "hiddens.pt")
            torch.save(ccs_buffers, model_template_dir / "ccs_hiddens.pt")
            torch.save(log_odds, model_template_dir / "lm_log_odds.pt")

            # Summarize model behavior with the template
            log_odds = log_odds.cpu()
            df.loc[(model_name, i), "template"] = template
            df.loc[(model_name, i), "label_accuracy"] = accuracy_score(dataset["label"], log_odds > 0)
            df.loc[(model_name, i), "label_auroc"] = roc_auc_score(dataset["label"], log_odds)
            df.loc[(model_name, i), "ol_accuracy"] = accuracy_score(dataset["objective_label"], log_odds > 0)
            df.loc[(model_name, i), "ol_auroc"] = roc_auc_score(dataset["objective_label"], log_odds)
            df.loc[(model_name, i), "ql_accuracy"] = accuracy_score(dataset["quirky_label"], log_odds > 0)
            df.loc[(model_name, i), "ql_auroc"] = roc_auc_score(dataset["quirky_label"], log_odds)
            df.loc[(model_name, i), "positive_predictions"] = (log_odds > 0).float().mean().item()
            df.loc[(model_name, i), "log_odds_mean"] = log_odds.mean().item()
            df.loc[(model_name, i), "balance_label"] = np.mean(dataset["label"])
            df.loc[(model_name, i), "balance_ol"] = np.mean(dataset["objective_label"])
            df.loc[(model_name, i), "balance_ql"] = np.mean(dataset["quirky_label"])

            print(f"Finished run {model_name}, {i}")

        print(f"Finished all templates for {model_name}. Saving...")
        df.to_csv(args.save_path / "template_comparison.csv")
    print(df)

    


        

        
