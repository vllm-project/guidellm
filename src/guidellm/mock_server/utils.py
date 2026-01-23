"""
Mock server utilities for text generation and tokenization testing.

This module provides mock tokenization and text generation utilities for testing
guidellm's mock server functionality. It includes a mock tokenizer that simulates
tokenization processes, functions to generate reproducible fake text with specific
token counts, and timing generators for realistic benchmarking scenarios.
"""

from __future__ import annotations

import random
import re
from collections.abc import Generator

from faker import Faker
from transformers.tokenization_utils_base import (
    AddedToken,
    PreTrainedTokenizerBase,
    TextInput,
)

__all__ = [
    "MockTokenizer",
    "create_fake_text",
    "create_fake_tokens_str",
    "sample_number",
    "times_generator",
]


class MockTokenizer(PreTrainedTokenizerBase):
    """
    Mock tokenizer implementation for testing text processing workflows.

    Provides a simplified tokenizer that splits text using regex patterns and
    generates deterministic token IDs based on string hashing. Used for testing
    guidellm components without requiring actual model tokenizers.

    :cvar VocabSize: Fixed vocabulary size for the mock tokenizer
    """

    VocabSize = 100000007

    def __len__(self) -> int:
        """
        Get the vocabulary size of the tokenizer.

        :return: The total number of tokens in the vocabulary
        """
        return self.VocabSize

    def __call__(self, text: str | list[str], **kwargs) -> list[int]:  # noqa: ARG002
        """
        Tokenize text and return token IDs (callable interface).

        :param text: Input text to tokenize
        :return: List of token IDs
        """
        if isinstance(text, str):
            tokens = self.tokenize(text)
            return self.convert_tokens_to_ids(tokens)
        elif isinstance(text, list):
            # Handle batch processing
            result = []
            for t in text:
                result.extend(self.__call__(t))
            return result
        else:
            msg = f"text input must be of type `str` or `list[str]`, got {type(text)}"
            raise ValueError(msg)

    def tokenize(self, text: TextInput, **_kwargs) -> list[str]:  # type: ignore[override]
        """
        Tokenize input text into a list of token strings.

        Splits text using regex to separate words, punctuation, and whitespace
        into individual tokens for processing.

        :param text: Input text to tokenize
        :return: List of token strings from the input text
        """
        # Split text into tokens: words, spaces, and punctuation
        return re.findall(r"\w+|[^\w\s]|\s+", text)

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> list[int]:
        """
        Convert token strings to numeric token IDs.

        Uses deterministic hashing to generate consistent token IDs for
        reproducible testing scenarios.

        :param tokens: Single token string or list of token strings
        :return: Single token ID or list of token IDs
        """
        if isinstance(tokens, str):
            return [hash(tokens) % self.VocabSize]
        return [hash(token) % self.VocabSize for token in tokens]

    def convert_ids_to_tokens(  # type: ignore[override]
        self, ids: list[int], _skip_special_tokens: bool = False
    ) -> list[str]:
        """
        Convert numeric token IDs back to token strings.

        Generates fake text tokens using Faker library seeded with token IDs
        for deterministic and reproducible token generation.

        :param ids: Single token ID or list of token IDs to convert
        :return: Single token string or list of token strings
        """
        if not ids:
            return [""]

        fake = Faker()
        fake.seed_instance(sum(ids) % self.VocabSize)

        target_count = len(ids)
        current_count = 0
        tokens = []

        while current_count < target_count:
            text = fake.text(
                max_nb_chars=(target_count - current_count) * 10  # oversample
            )
            new_tokens = self.tokenize(text)

            if current_count > 0:
                new_tokens = [".", " "] + new_tokens

            new_tokens = (
                new_tokens[: target_count - current_count]
                if len(new_tokens) > (target_count - current_count)
                else new_tokens
            )
            tokens += new_tokens
            current_count += len(new_tokens)

        return tokens

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """
        Convert a list of token strings back to a single text string.

        :param tokens: List of token strings to concatenate
        :return: Concatenated string from all tokens
        """
        return "".join(tokens)

    def _add_tokens(
        self,
        new_tokens: list[str] | list[AddedToken],  # noqa: ARG002
        special_tokens: bool = False,  # noqa: ARG002
    ) -> int:
        """
        Add new tokens to the tokenizer vocabulary (mock implementation).

        :param new_tokens: List of tokens to add to the vocabulary
        :param special_tokens: Whether the tokens are special tokens
        :return: Number of tokens actually added (always 0 for mock)
        """
        return 0

    def apply_chat_template(  # type: ignore[override]
        self,
        conversation: list,
        tokenize: bool = False,  # Changed default to False to match transformers
        add_generation_prompt: bool = False,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> str | list[int]:
        """
        Apply a chat template to format conversation messages.

        Mock implementation that concatenates all message content for testing.

        :param conversation: List of chat messages
        :param tokenize: Whether to return tokens or string
        :param add_generation_prompt: Whether to add generation prompt
        :return: Formatted text string or token IDs
        """
        # Simple concatenation of all message content
        texts = []
        for message in conversation:
            if isinstance(message, dict) and "content" in message:
                texts.append(message["content"])
            elif hasattr(message, "content"):
                texts.append(message.content)

        formatted_text = " ".join(texts)

        if tokenize:
            return self.convert_tokens_to_ids(self.tokenize(formatted_text))
        return formatted_text

    def decode(  # type: ignore[override]
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
        **kwargs,  # noqa: ARG002
    ) -> str:
        """
        Decode token IDs back to text string.

        :param token_ids: List of token IDs to decode
        :param skip_special_tokens: Whether to skip special tokens
        :return: Decoded text string
        """
        tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens)
        return self.convert_tokens_to_string(tokens)


def create_fake_text(
    num_tokens: int,
    processor: PreTrainedTokenizerBase,
    seed: int = 42,
    fake: Faker | None = None,
) -> str:
    """
    Generate fake text using a tokenizer processor with specified token count.

    Creates text by generating fake tokens and joining them into a string,
    ensuring the result has the exact number of tokens when processed by
    the given tokenizer.

    :param num_tokens: Target number of tokens in the generated text
    :param processor: Tokenizer to use for token generation and validation
    :param seed: Random seed for reproducible text generation
    :param fake: Optional Faker instance for text generation
    :return: Generated text string with the specified token count
    """
    return "".join(create_fake_tokens_str(num_tokens, processor, seed, fake))


def create_fake_tokens_str(
    num_tokens: int,
    processor: PreTrainedTokenizerBase,
    seed: int = 42,
    fake: Faker | None = None,
) -> list[str]:
    """
    Generate fake token strings using a tokenizer processor.

    Creates a list of token strings by generating fake text and tokenizing it
    until the desired token count is reached. Uses the provided tokenizer
    for accurate token boundary detection.

    :param num_tokens: Target number of tokens to generate
    :param processor: Tokenizer to use for token generation and validation
    :param seed: Random seed for reproducible token generation
    :param fake: Optional Faker instance for text generation
    :return: List of token strings with the specified count
    """
    if not fake:
        fake = Faker()
    fake.seed_instance(seed)

    tokens: list[str] = []

    while len(tokens) < num_tokens:
        text = fake.text(
            max_nb_chars=(num_tokens - len(tokens)) * 30  # oversample
        )
        new_tokens = processor.tokenize(text)

        if len(tokens) > 0:
            new_tokens = [".", " "] + new_tokens

        new_tokens = (
            new_tokens[: num_tokens - len(tokens)]
            if len(new_tokens) > (num_tokens - len(tokens))
            else new_tokens
        )
        tokens += new_tokens

    return tokens


def times_generator(mean: float, standard_dev: float) -> Generator[float]:
    """
    Generate infinite timing values from a normal distribution.

    Creates a generator that yields timing values sampled from a normal
    distribution, useful for simulating realistic request timing patterns
    in benchmarking scenarios.

    :param mean: Mean value for the normal distribution
    :param standard_dev: Standard deviation for the normal distribution
    :return: Generator yielding positive timing values from the distribution
    """
    while True:
        yield sample_number(mean, standard_dev)


def sample_number(mean: float, standard_dev: float) -> float:
    """
    Generate a single timing value from a normal distribution.

    Samples one timing value from a normal distribution with the specified
    parameters, ensuring the result is non-negative for realistic timing
    simulation in benchmarking scenarios.

    :param mean: Mean value for the normal distribution
    :param standard_dev: Standard deviation for the normal distribution
    :return: Non-negative timing value from the distribution
    """
    return max(0.0, random.gauss(mean, standard_dev))
