import os
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Iterator

import pytest
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from guidellm.preprocess.dataset import (
    STRATEGY_HANDLERS,
    PromptTooShortError,
    ShortPromptStrategy,
    handle_concatenate_strategy,
    handle_error_strategy,
    handle_ignore_strategy,
    handle_pad_strategy,
    process_dataset,
    push_dataset_to_hub,
)


@pytest.fixture
def tokenizer_mock():
    tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
    tokenizer.encode.side_effect = lambda x: [1] * len(x)
    tokenizer.decode.side_effect = lambda x, *args, **kwargs: "".join(
        str(item) for item in x
    )
    return tokenizer


@pytest.mark.smoke
@patch(f"{process_dataset.__module__}.guidellm_load_dataset")
@patch(f"{process_dataset.__module__}.check_load_processor")
@patch(f"{process_dataset.__module__}.Dataset")
@patch(f"{process_dataset.__module__}.IntegerRangeSampler")
def test_strategy_handler_called(
    mock_sampler,
    mock_dataset_class,
    mock_check_processor,
    mock_load_dataset,
    tokenizer_mock,
):
    mock_handler = MagicMock(return_value="processed_prompt")
    with patch.dict(STRATEGY_HANDLERS, {ShortPromptStrategy.IGNORE: mock_handler}):
        mock_dataset = [{"prompt": "abc"}, {"prompt": "def"}]
        mock_load_dataset.return_value = (mock_dataset, {"prompt_column": "prompt"})
        mock_check_processor.return_value = tokenizer_mock
        mock_sampler.side_effect = lambda **kwargs: [10, 10]

        mock_dataset_obj = MagicMock(spec=Dataset)
        mock_dataset_class.from_list.return_value = mock_dataset_obj

        process_dataset(
            data="input",
            output_path="output_dir/data.json",
            processor=tokenizer_mock,
            prompt_tokens="average=10,min=1",
            output_tokens="average=10,min=1",
            short_prompt_strategy=ShortPromptStrategy.IGNORE,
        )

        assert mock_handler.call_count == 2
        mock_load_dataset.assert_called_once()
        mock_check_processor.assert_called_once()


@pytest.mark.sanity
def test_handle_ignore_strategy_too_short(tokenizer_mock):
    result = handle_ignore_strategy("short", 10, tokenizer_mock)
    assert result is None
    tokenizer_mock.encode.assert_called_with("short")


@pytest.mark.sanity
def test_handle_ignore_strategy_sufficient_length(tokenizer_mock):
    result = handle_ignore_strategy("long prompt", 5, tokenizer_mock)
    assert result == "long prompt"
    tokenizer_mock.encode.assert_called_with("long prompt")


@pytest.mark.sanity
def test_handle_concatenate_strategy_enough_prompts(tokenizer_mock):
    dataset_iter = iter([{"prompt": "longer"}])
    result = handle_concatenate_strategy(
        "short", 10, dataset_iter, "prompt", tokenizer_mock, "\n"
    )
    assert result == "short\nlonger"


@pytest.mark.sanity
def test_handle_concatenate_strategy_not_enough_prompts(tokenizer_mock):
    dataset_iter: Iterator = iter([])
    result = handle_concatenate_strategy(
        "short", 10, dataset_iter, "prompt", tokenizer_mock, ""
    )
    assert result is None


@pytest.mark.sanity
def test_handle_pad_strategy(tokenizer_mock):
    result = handle_pad_strategy("short", 10, tokenizer_mock, "p")
    assert result.startswith("shortppppp")


@pytest.mark.sanity
def test_handle_error_strategy_valid_prompt(tokenizer_mock):
    result = handle_error_strategy("valid prompt", 5, tokenizer_mock)
    assert result == "valid prompt"
    tokenizer_mock.encode.assert_called_with("valid prompt")


@pytest.mark.sanity
def test_handle_error_strategy_too_short_prompt(tokenizer_mock):
    with pytest.raises(PromptTooShortError):
        handle_error_strategy("short", 10, tokenizer_mock)


@pytest.mark.smoke
@patch(f"{process_dataset.__module__}.save_dataset_to_file")
@patch(f"{process_dataset.__module__}.Dataset")
@patch(f"{process_dataset.__module__}.guidellm_load_dataset")
@patch(f"{process_dataset.__module__}.check_load_processor")
@patch(f"{process_dataset.__module__}.IntegerRangeSampler")
def test_process_dataset_non_empty(
    mock_sampler,
    mock_check_processor,
    mock_load_dataset,
    mock_dataset_class,
    mock_save_to_file,
    tokenizer_mock,
):
    mock_dataset = [{"prompt": "Hello"}, {"prompt": "How are you?"}]
    mock_load_dataset.return_value = (mock_dataset, {"prompt_column": "prompt"})
    mock_check_processor.return_value = tokenizer_mock
    mock_sampler.side_effect = lambda **kwargs: [3, 3, 3]

    mock_dataset_obj = MagicMock(spec=Dataset)
    mock_dataset_class.from_list.return_value = mock_dataset_obj

    output_path = "output_dir/data.json"
    process_dataset(
        data="input",
        output_path=output_path,
        processor=tokenizer_mock,
        prompt_tokens="average=10,min=1",
        output_tokens="average=10,min=1",
    )

    mock_load_dataset.assert_called_once()
    mock_check_processor.assert_called_once()
    mock_dataset_class.from_list.assert_called_once()
    mock_save_to_file.assert_called_once_with(mock_dataset_obj, output_path)

    args, _ = mock_dataset_class.from_list.call_args
    processed_list = args[0]
    assert len(processed_list) == 2
    for item in processed_list:
        assert "prompt" in item
        assert "prompt_tokens_count" in item
        assert "output_tokens_count" in item
        assert len(tokenizer_mock.encode(item["prompt"])) <= 3


@pytest.mark.sanity
@patch(f"{process_dataset.__module__}.Dataset")
@patch(f"{process_dataset.__module__}.guidellm_load_dataset")
@patch(f"{process_dataset.__module__}.check_load_processor")
@patch(f"{process_dataset.__module__}.IntegerRangeSampler")
def test_process_dataset_empty_after_processing(
    mock_sampler,
    mock_check_processor,
    mock_load_dataset,
    mock_dataset_class,
    tokenizer_mock,
):
    mock_dataset = [{"prompt": ""}]
    mock_load_dataset.return_value = (mock_dataset, {"prompt_column": "prompt"})
    mock_check_processor.return_value = tokenizer_mock
    mock_sampler.side_effect = lambda **kwargs: [10]

    process_dataset(
        data="input",
        output_path="output_dir/data.json",
        processor=tokenizer_mock,
        prompt_tokens="average=10,min=1",
        output_tokens="average=10,min=1",
    )

    mock_load_dataset.assert_called_once()
    mock_check_processor.assert_called_once()
    mock_dataset_class.from_list.assert_not_called()


@pytest.mark.smoke
@patch(f"{process_dataset.__module__}.push_dataset_to_hub")
@patch(f"{process_dataset.__module__}.Dataset")
@patch(f"{process_dataset.__module__}.guidellm_load_dataset")
@patch(f"{process_dataset.__module__}.check_load_processor")
@patch(f"{process_dataset.__module__}.IntegerRangeSampler")
def test_process_dataset_push_to_hub_called(
    mock_sampler,
    mock_check_processor,
    mock_load_dataset,
    mock_dataset_class,
    mock_push,
    tokenizer_mock,
):
    mock_dataset = [{"prompt": "abc"}]
    mock_load_dataset.return_value = (mock_dataset, {"prompt_column": "prompt"})
    mock_check_processor.return_value = tokenizer_mock
    mock_sampler.side_effect = lambda **kwargs: [3]

    mock_dataset_obj = MagicMock(spec=Dataset)
    mock_dataset_class.from_list.return_value = mock_dataset_obj

    process_dataset(
        data="input",
        output_path="output_dir/data.json",
        processor=tokenizer_mock,
        prompt_tokens="average=10,min=1",
        output_tokens="average=10,min=1",
        push_to_hub=True,
        hub_dataset_id="id123",
    )
    mock_push.assert_called_once_with("id123", mock_dataset_obj)


@pytest.mark.sanity
@patch(f"{process_dataset.__module__}.push_dataset_to_hub")
@patch(f"{process_dataset.__module__}.Dataset")
@patch(f"{process_dataset.__module__}.guidellm_load_dataset")
@patch(f"{process_dataset.__module__}.check_load_processor")
@patch(f"{process_dataset.__module__}.IntegerRangeSampler")
def test_process_dataset_push_to_hub_not_called(
    mock_sampler,
    mock_check_processor,
    mock_load_dataset,
    mock_dataset_class,
    mock_push,
    tokenizer_mock,
):
    mock_dataset = [{"prompt": "abc"}]
    mock_load_dataset.return_value = (mock_dataset, {"prompt_column": "prompt"})
    mock_check_processor.return_value = tokenizer_mock
    mock_sampler.side_effect = lambda **kwargs: [3]

    mock_dataset_obj = MagicMock(spec=Dataset)
    mock_dataset_class.from_list.return_value = mock_dataset_obj

    process_dataset(
        data="input",
        output_path="output_dir/data.json",
        processor=tokenizer_mock,
        prompt_tokens="average=10,min=1",
        output_tokens="average=10,min=1",
        push_to_hub=False,
    )
    mock_push.assert_not_called()


@pytest.mark.regression
def test_push_dataset_to_hub_success():
    os.environ["HF_TOKEN"] = "token"
    mock_dataset = MagicMock(spec=Dataset)
    push_dataset_to_hub("dataset_id", mock_dataset)
    mock_dataset.push_to_hub.assert_called_once_with("dataset_id", token="token")


@pytest.mark.regression
def test_push_dataset_to_hub_error_no_env():
    if "HF_TOKEN" in os.environ:
        del os.environ["HF_TOKEN"]
    mock_dataset = MagicMock(spec=Dataset)
    with pytest.raises(ValueError, match="hub_dataset_id and HF_TOKEN"):
        push_dataset_to_hub("dataset_id", mock_dataset)


@pytest.mark.regression
def test_push_dataset_to_hub_error_no_id():
    os.environ["HF_TOKEN"] = "token"
    mock_dataset = MagicMock(spec=Dataset)
    with pytest.raises(ValueError, match="hub_dataset_id and HF_TOKEN"):
        push_dataset_to_hub(None, mock_dataset)
