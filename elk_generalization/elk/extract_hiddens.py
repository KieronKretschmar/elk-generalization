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
            model = "EleutherAI/pythia-70M", 
            dataset = 'EleutherAI/qm-grader-first',
            save_path = Path("./data/hiddens/410M-qm-grader-first"),
            max_examples = [4096, 1024],
            splits = ["train", "validation"],
            character = "Alice",
            difficulty = "easy",
            prefix_path = Path(".\elk_generalization\elk\prefixes.json"),
            prefix = "few_shot_persona_first_v1"
            )
    else:
        parser = ArgumentParser(description="Process and save model hidden states.")
        parser.add_argument("--model", type=str, help="Name of the Hugging Face model")
        parser.add_argument("--dataset", type=str, help="Name of the Hugging Face dataset")
        parser.add_argument("--save-path", type=Path, help="Path to save the hidden states")
        parser.add_argument('--prefix-path', type=str, help="Path to json file containing prefixes.")
        parser.add_argument('--prefix', type=str, help="Key of prefix to use from json file at prefix-path. No prefix is used by default.")
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
            "--character",
            type=str,
            help="Only use examples with this character"
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

        if args.character:
            print(f"Filtering for character {args.character}")
            dataset = dataset.filter(lambda example: example["character"] == args.character)

        # Filter for difficulty (corresponds to number of digits in the shorter of the two summands)
        # An easy example is defined as one in which the shorter of the two summands is two digits or shorter. 
        if args.difficulty == "easy":
            print(f"Filtering for difficulty {args.difficulty}")
            dataset = dataset.filter(lambda example: example["difficulty"] <= 2)
            
        # Similarly, a hard example has a shorter summand of at least four digits.
        if args.difficulty == "hard":
            print(f"Filtering for difficulty {args.difficulty}")
            dataset = dataset.filter(lambda example: example["difficulty"] > 4)

        print(f"Size of dataset before selecting: {len(dataset)}")
        dataset = dataset.select(range(max_examples))

        if args.prefix:
            assert Path(args.prefix_path).exists(), f"Could not find file at {args.prefix_path}. Please specify valid json file with -prefix-path."

            with open(Path(args.prefix_path), "r") as f:
                prefixes = json.loads(f.read())

            assert args.prefix in prefixes, f"No prefix with key {args.prefix} found in {args.prefix_path}."
            prefix = prefixes[args.prefix]

            def add_prefix(ex):
                ex["statement"] = prefix + ex["statement"]
                return ex
            
            dataset = dataset.map(add_prefix)
            print("First example including the prefix:")
            print(dataset[0]["statement"])

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
        if args.label_cols:
            for label_col in args.label_cols:
                labels = torch.as_tensor(dataset[label_col], dtype=torch.int32)
                torch.save(labels, root / f"{label_col}s.pt")
        else:
            # Default behavior
            labels = torch.as_tensor(dataset["label"], dtype=torch.int32)
            alice_labels = torch.as_tensor(dataset["alice_label"], dtype=torch.int32)
            bob_labels = torch.as_tensor(dataset["bob_label"], dtype=torch.int32)
            torch.save(labels, root / "labels.pt")
            torch.save(alice_labels, root / "alice_labels.pt")
            torch.save(bob_labels, root / "bob_labels.pt")
        torch.save(buffers, root / "hiddens.pt")
        torch.save(ccs_buffers, root / "ccs_hiddens.pt")
        torch.save(log_odds, root / "lm_log_odds.pt")
