"""Tests for ``guidellm preprocess dataset`` CLI registry options."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from guidellm.__main__ import cli


@pytest.mark.smoke
def test_preprocess_dataset_help_uses_registry_options():
    """
    Help text should expose registry-backed options and omit legacy flags.

    ## WRITTEN BY AI ##
    """
    runner = CliRunner()
    result = runner.invoke(cli, ["preprocess", "dataset", "--help"])

    assert result.exit_code == 0
    assert "--tokenizer" in result.output
    assert "--strategy" in result.output
    assert "--seed" in result.output
    assert "--data-column-mapper" in result.output
    assert "--data-loader" in result.output
    assert "--config" not in result.output
    assert "--short-prompt-strategy" not in result.output
    assert "--pad-char" not in result.output
    assert "--concat-delimiter" not in result.output
    assert "--include-prefix-in-token-count" not in result.output
    assert "--processor" not in result.output
    assert "--random-seed" not in result.output


@pytest.mark.sanity
def test_preprocess_dataset_requires_tokenizer_kind():
    """
    Missing tokenizer kind should surface a Click parameter error.

    ## WRITTEN BY AI ##
    """
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "preprocess",
            "dataset",
            "kind=json_file,path=input.json",
            "output.jsonl",
            "--strategy",
            "kind=ignore,prompt_tokens=10,output_tokens=5",
            "--tokenizer",
            "model=gpt2",
        ],
    )

    assert result.exit_code != 0
    assert "--tokenizer" in result.output
    assert "kind" in result.output


@pytest.mark.sanity
def test_preprocess_dataset_requires_strategy():
    """
    Missing strategy should fail validation.

    ## WRITTEN BY AI ##
    """
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "preprocess",
            "dataset",
            "kind=json_file,path=input.json",
            "output.jsonl",
            "--tokenizer",
            "kind=huggingface_auto,model=gpt2",
        ],
    )

    assert result.exit_code != 0
    assert "--strategy" in result.output


@pytest.mark.sanity
def test_preprocess_dataset_rejects_legacy_config_flag():
    """
    Legacy ``--config`` syntax should no longer be accepted.

    ## WRITTEN BY AI ##
    """
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "preprocess",
            "dataset",
            "kind=json_file,path=input.json",
            "output.jsonl",
            "--config",
            "prompt_tokens=10,output_tokens=5",
            "--tokenizer",
            "kind=huggingface_auto,model=gpt2",
        ],
    )

    assert result.exit_code != 0
    assert "No such option: --config" in result.output


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Fails in CI Python 3.10")
@pytest.mark.regression
@patch("guidellm.cli.preprocess.dataset.process_dataset")
def test_preprocess_dataset_passes_registry_args(mock_process_dataset):
    """
    Parsed registry options should be forwarded to process_dataset.

    ## WRITTEN BY AI ##
    """
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "preprocess",
            "dataset",
            "kind=huggingface,source=test/ds,load_kwargs.split=train",
            "output.jsonl",
            "--tokenizer",
            "kind=huggingface_auto,model=gpt2,load_kwargs.use_fast=false",
            "--strategy",
            "kind=pad,prompt_tokens=10,output_tokens=5,pad=X,count_prefix=true",
            "--seed",
            "kind=static,value=123",
            "--data-column-mapper",
            "kind=generative_column_mapper,column_mappings.text_column=question",
            "--data-loader",
            "kind=pytorch,samples=50",
        ],
    )

    assert result.exit_code == 0, result.output
    mock_process_dataset.assert_called_once()
    kwargs = mock_process_dataset.call_args.kwargs
    assert kwargs["random_seed"] == 123
    assert kwargs["tokenizer"].kind == "huggingface_auto"
    assert kwargs["tokenizer"].model == "gpt2"
    assert kwargs["tokenizer"].load_kwargs == {"use_fast": False}
    assert kwargs["strategy"].kind == "pad"
    assert kwargs["strategy"].prompt_tokens == 10
    assert kwargs["strategy"].pad == "X"
    assert kwargs["strategy"].count_prefix is True
    assert kwargs["data_column_mapper"].kind == "generative_column_mapper"
    assert kwargs["data"].kind == "huggingface"
    assert kwargs["data"].load_kwargs == {"split": "train"}
    assert kwargs["data_loader"].kind == "pytorch"
    assert kwargs["data_loader"].samples == 50


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Fails in CI Python 3.10")
@pytest.mark.regression
@patch("guidellm.cli.preprocess.dataset.process_dataset")
def test_preprocess_dataset_defaults_data_loader(mock_process_dataset):
    """
    Omitting --data-loader should forward the default pytorch loader config.

    ## WRITTEN BY AI ##
    """
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "preprocess",
            "dataset",
            "kind=json_file,path=input.json",
            "output.jsonl",
            "--tokenizer",
            "kind=huggingface_auto,model=gpt2",
            "--strategy",
            "kind=ignore,prompt_tokens=10,output_tokens=5",
        ],
    )

    assert result.exit_code == 0, result.output
    mock_process_dataset.assert_called_once()
    kwargs = mock_process_dataset.call_args.kwargs
    assert kwargs["data_loader"].kind == "pytorch"
    assert kwargs["data_loader"].samples == -1
