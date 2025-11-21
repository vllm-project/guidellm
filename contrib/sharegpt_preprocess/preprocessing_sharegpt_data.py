import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

MIN_CHAR = 10
MAX_CHAR = 1000


class TokenCounter:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.model_name = model_name
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None

    def _initialize_tokenizer(self) -> None:
        if self._tokenizer is None:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            except (OSError, ImportError, ValueError) as e:
                raise RuntimeError(f"Failed to initialize tokenizer: {e}") from e

    def estimate_num_tokens(self, text: str) -> int:
        self._initialize_tokenizer()

        if self._tokenizer is None:
            return 0

        try:
            encoding = self._tokenizer.__call__(text, return_tensors=None)
            return len(encoding["input_ids"])
        except (AttributeError, TypeError, RuntimeError) as e:
            raise ValueError(f"Error processing text: {e}") from e


def extract_and_save_with_filtering(file):
    """substract human prompts and apply filtering conditions"""
    dataset = load_dataset("json", data_files=file, split="train")
    filtered_prompts = []

    for example in dataset:
        conversations = example.get("conversations", [])
        if isinstance(conversations, list):
            for turn in conversations:
                if turn.get("from") in ["human", "user"]:
                    prompt_text = turn["value"].strip()
                    # apply filter conditions: more than 10 characters
                    if (
                        len(prompt_text) >= MIN_CHAR
                        and
                        # less thant 1000 characters
                        len(prompt_text) <= MAX_CHAR
                        and
                        # except URLs
                        not prompt_text.startswith(("http://", "https://"))
                        and
                        # except special characters
                        not re.search(r"[<>{}[\]\\]", prompt_text)
                        # except pure numbers
                        and not prompt_text.isdigit()
                    ):
                        filtered_prompts.append(
                            {
                                "from": turn.get("from"),
                                "text": prompt_text,
                                "char_count": len(prompt_text),
                                "word_count": len(prompt_text.split()),
                            }
                        )

    return filtered_prompts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data percentage.")
    parser.add_argument(
        "--parse",
        type=float,
        default=1,
        help="The percentage of data to process (0 to 1). Default is 1 (100%).",
    )
    args = parser.parse_args()

    sharegpt_file = "ShareGPT_V3_unfiltered_cleaned_split.json"
    with Path(sharegpt_file).open("r", encoding="utf-8") as file:
        data = json.load(file)

    counter = TokenCounter()
    num_of_ids = len(data)
    data = data[: int(num_of_ids * args.parse)]
    for d in data:
        d["num_round"] = len(d["conversations"])
        human_tokens = []
        gpt_tokens = []
        for conv in d["conversations"]:
            if conv["from"] == "human":
                human_tokens.append(counter.estimate_num_tokens(conv["value"]))
            if conv["from"] == "gpt":
                token_number = counter.estimate_num_tokens(conv["value"])
                conv["num_tokens"] = token_number
                gpt_tokens.append(token_number)
        if len(human_tokens) == 0:
            d["average_human_token"] = 0
            d["max_human_token"] = 0
        else:
            d["average_human_token"] = float(np.mean(human_tokens))
            d["max_human_token"] = float(np.max(human_tokens))
        if len(gpt_tokens) == 0:
            d["average_gpt_token"] = 0
            d["max_gpt_token"] = 0
        else:
            d["average_gpt_token"] = float(np.mean(gpt_tokens))
            d["max_gpt_token"] = float(np.max(gpt_tokens))

    # save unfiletered datasets to ShareGPT.json
    with Path("ShareGPT.json").open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    # filter from: human prompts and save again
    filtered_result = extract_and_save_with_filtering("ShareGPT.json")
    with Path("ShareGPT.json").open("w", encoding="utf-8") as file:
        json.dump(filtered_result, file, ensure_ascii=False, indent=2)
