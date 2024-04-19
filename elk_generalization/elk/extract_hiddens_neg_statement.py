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
    debug = True
    if debug:
        print("DEBUGGING WITH HARDCODED ARGS!")
        args = Namespace(
            model = "EleutherAI/pythia-70M", 
            dataset = Path(r".\experiments\got\cities"),
            save_path = Path(r".\experiments\neg_extract"),
            max_examples = [40, 10],
            splits = ["train", "test"],
            label_cols = ["label"],
            filter_cols = [],
            filter_values = [],
            )
    else:
        parser = ArgumentParser(description="Process and save model hidden states.")
        parser.add_argument("--model", type=str, help="Name of the Hugging Face model")
        parser.add_argument("--dataset", type=str, help="Name of the Hugging Face dataset")
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
        parser.add_argument(
            "--hf-cache-dir",
            type=str,
            help="Directory to be used by huggingface cache.",
            default=None
        )
        args = parser.parse_args()

    # check if all the results already exist
    if all((args.save_path / split / "hiddens.pt").exists() for split in args.splits):
        print(f"Hiddens already exist at {args.save_path}")
        exit()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": torch.cuda.current_device()},
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
      
    assert len(args.max_examples) == len(args.splits)
    for split, max_examples in zip(args.splits, args.max_examples):
        root = args.save_path / split
        root.mkdir(parents=True, exist_ok=True)
        # skip if the results for this split already exist
        if (root / "hiddens.pt").exists():
            print(f"Skipping because '{root / 'hiddens.pt'}' already exists")
            continue

        print(f"Processing '{split}' split...")

        if Path(args.dataset).exists():
            print(f"Trying to load {args.dataset} from disk...")
            dataset = load_from_disk(args.dataset)[split].shuffle()
        else:
            print(f"Trying to load {args.dataset} from hub...")
            dataset = load_dataset(args.dataset, split=split).shuffle()
        assert isinstance(dataset, Dataset)

        dataset = dataset.select(range(max_examples))

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

        for i, record in tqdm(enumerate(dataset), total=len(dataset)):
            assert isinstance(record, dict)

            prompt = tokenizer.encode(record["statement"])
            neg_prompt = tokenizer.encode(record["neg_statement"])

            with torch.inference_mode():
                outputs = model(
                    torch.as_tensor([prompt], device=model.device),
                    output_hidden_states=True,
                    use_cache=True,
                )
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

                # Extract hidden states of the last token in each layer for the non-negated statement for which the label is accurate
                for j, state in enumerate(outputs.hidden_states[1:]):
                    buffers[j][i] = state[0, -1, :]
                # and for negated statement for which the label is wrong
                for j, state in enumerate(neg_outputs.hidden_states[1:]):
                    neg_buffers[j][i] = state[0, -1, :]

        # Sanity check
        assert all(buffer.isfinite().all() for buffer in buffers)
        assert all(buffer.isfinite().all() for buffer in ccs_buffers)

        # Save results to disk for later
        for label_col in args.label_cols:
            labels = torch.as_tensor(dataset[label_col], dtype=torch.int32)
            torch.save(labels, root / f"{label_col}s.pt")
        torch.save(buffers, root / "hiddens.pt")
        torch.save(neg_buffers, root / "neg_hiddens.pt")
        torch.save(ccs_buffers, root / "ccs_hiddens.pt")
