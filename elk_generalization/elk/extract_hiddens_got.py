from argparse import ArgumentParser, Namespace
from pathlib import Path
import json

import torch
from datasets import Dataset, load_dataset, load_from_disk
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        args = Namespace(
            models = ["EleutherAI/pythia-70M"], 
            datasets = [r"got\cities"],
            data_dir = Path(r".\experiments\neg_extract"),
            max_examples = [40, 10],
            splits = ["train", "test"],
            label_cols = ["label"],
            filter_cols = [],
            filter_values = [],
            )
    else:
        parser = ArgumentParser(description="Process and save model hidden states.")
        parser.add_argument(
            "--models", 
            type=str, 
            nargs="+",
            help="Names of the Hugging Face models"
        )    
        parser.add_argument(
            "--data-dir", type=str, help="Path to the directory where extracted data is to be saved"
        )
        parser.add_argument(
            "--datasets", 
            type=str, 
            nargs="+",
            help="Names of the Hugging Face datasets"
        )
        parser.add_argument("--save-path", type=Path, help="Path to save the hidden states")
        parser.add_argument(
            "--max-examples",
            type=int,
            nargs="+",
            help="Max examples per split",
            default=[1000, 1000],
        )
        parser.add_argument(
            "--splits",
            nargs="+",
            default=["training", "validation", "test"],
            help="Dataset splits to process",
        )
        parser.add_argument(
            "--label-cols",
            type=str,
            nargs="*",
            help="Columns of the dataset that contain labels we wish to save",
            default=[],
        )
        args = parser.parse_args()

    print(args)
    for model_name in args.models:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": torch.cuda.current_device()},
            torch_dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        for dataset_name in args.datasets:
            print(f"Starting {model_name=} on {dataset_name=}...")
            dataset_path = Path(args.data_dir) / dataset_name


            for split, max_examples in zip(args.splits, args.max_examples):
                root = Path(args.data_dir) / dataset_name / model_name / split
                # check if all the results already exist
                if (root / "hiddens.pt").exists():
                    print(f"Hiddens already exist at {root}")
                    continue
                
                assert len(args.max_examples) == len(args.splits)

                root.mkdir(parents=True, exist_ok=True)

                print(f"Processing '{split}' split...")
                if Path(dataset_path).exists():
                    print(f"Trying to load {dataset_path} from disk...")
                    dataset = load_from_disk(dataset_path)[split].shuffle()
                # else:
                #     print(f"Trying to load {dataset_path} from hub...")
                #     dataset = load_dataset(dataset_path, split=split).shuffle()
                assert isinstance(dataset, Dataset)

                dataset = dataset.select(range(min(max_examples, len(dataset))))

                buffers = [
                    torch.full(
                        [len(dataset), model.config.hidden_size],
                        torch.nan,
                        device=model.device,
                        dtype=model.dtype,
                    )
                    for _ in range(model.config.num_hidden_layers)
                ]
                neg_buffers = [
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

                for i, record in tqdm(enumerate(dataset), total=len(dataset), mininterval=10):
                    assert isinstance(record, dict)

                    prompt = tokenizer.encode(record["statement"])

                    with torch.inference_mode():
                        outputs = model(
                            torch.as_tensor([prompt], device=model.device),
                            output_hidden_states=True,
                            use_cache=True,
                        )

                        # Extract hidden states of the last token in each layer for the non-negated statement for which the label is accurate
                        for j, state in enumerate(outputs.hidden_states[1:]):
                            buffers[j][i] = state[0, -1, :]

                        if args.extract_ccs:
                            neg_prompt = tokenizer.encode(record["neg_statement"])
                            neg_outputs = model(
                                torch.as_tensor([neg_prompt], device=model.device),
                                output_hidden_states=True,
                                use_cache=True,
                            )
                            ccs_outputs = [outputs.hidden_states[1:], neg_outputs.hidden_states[1:]]

                            for j, (state1, state2) in enumerate(zip(*ccs_outputs)):
                                # Store hiddens for only prompt and last token
                                ccs_buffers[j][i, 0] = state1[0][-1] 
                                ccs_buffers[j][i, 1] = state2[0][-1]

                            # and for negated statement for which the label is wrong
                            for j, state in enumerate(neg_outputs.hidden_states[1:]):
                                neg_buffers[j][i] = state[0, -1, :]

                # Sanity check
                assert all(buffer.isfinite().all() for buffer in buffers)


                # Save results to disk for later
                for label_col in args.label_cols:
                    labels = torch.as_tensor(dataset[label_col], dtype=torch.int32)
                    torch.save(labels, root / f"{label_col}s.pt")
                torch.save(buffers, root / "hiddens.pt")
                if args.extract_ccs:
                    torch.save(neg_buffers, root / "neg_hiddens.pt")
                    torch.save(ccs_buffers, root / "ccs_hiddens.pt")
                    assert all(buffer.isfinite().all() for buffer in ccs_buffers)
                print(f"Finished storing hiddens for {model_name=} on {dataset_name=}.")
