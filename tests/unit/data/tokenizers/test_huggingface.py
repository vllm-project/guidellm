"""
Unit tests for guidellm.data.tokenizers.huggingface module.

### WRITTEN BY AI ###
"""

from __future__ import annotations

import pytest

from guidellm.data.tokenizers.huggingface import HuggingFaceTokenizer
from guidellm.schemas.data import HuggingFaceTokenizerArgs
from tests.fixtures.tokenizers import MINIMAL_TOKENIZER_DIR


class TestHuggingFaceTokenizerArgs:
    """Tests for HuggingFaceTokenizerArgs schema.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_default_kind(self):
        """HuggingFaceTokenizerArgs defaults kind to 'huggingface_auto'.

        ### WRITTEN BY AI ###
        """
        args = HuggingFaceTokenizerArgs(model="gpt2")
        assert args.kind == "huggingface_auto"

    @pytest.mark.smoke
    def test_model_field_optional(self):
        """model field is optional at schema level (validated at runtime).

        ### WRITTEN BY AI ###
        """
        args = HuggingFaceTokenizerArgs()
        assert args.model is None

    @pytest.mark.smoke
    def test_load_kwargs_defaults_empty(self):
        """load_kwargs defaults to empty dict.

        ### WRITTEN BY AI ###
        """
        args = HuggingFaceTokenizerArgs(model="gpt2")
        assert args.load_kwargs == {}

    @pytest.mark.sanity
    def test_custom_load_kwargs(self):
        """load_kwargs accepts custom values.

        ### WRITTEN BY AI ###
        """
        args = HuggingFaceTokenizerArgs(
            model="gpt2", load_kwargs={"use_fast": False, "revision": "main"}
        )
        assert args.load_kwargs == {"use_fast": False, "revision": "main"}

    @pytest.mark.regression
    def test_serialization(self):
        """HuggingFaceTokenizerArgs serializes correctly.

        ### WRITTEN BY AI ###
        """
        args = HuggingFaceTokenizerArgs(model="gpt2", load_kwargs={"use_fast": True})
        dumped = args.model_dump()
        assert dumped["kind"] == "huggingface_auto"
        assert dumped["model"] == "gpt2"
        assert dumped["load_kwargs"] == {"use_fast": True}


class TestHuggingFaceTokenizer:
    """Tests for HuggingFaceTokenizer implementation.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_construction_requires_model(self):
        """HuggingFaceTokenizer raises ValueError if model is None.

        ### WRITTEN BY AI ###
        """
        config = HuggingFaceTokenizerArgs()
        with pytest.raises(ValueError, match="must be provided"):
            HuggingFaceTokenizer(config)

    @pytest.mark.smoke
    def test_construction_with_model(self):
        """HuggingFaceTokenizer constructs successfully with model.

        ### WRITTEN BY AI ###
        """
        config = HuggingFaceTokenizerArgs(model="gpt2")
        tokenizer = HuggingFaceTokenizer(config)
        assert tokenizer is not None

    @pytest.mark.sanity
    def test_lazy_loading_not_called_on_init(self):
        """Tokenizer is not loaded during construction.

        ### WRITTEN BY AI ###
        """
        config = HuggingFaceTokenizerArgs(model="gpt2")
        tokenizer = HuggingFaceTokenizer(config)
        assert tokenizer._tokenizer is None

    @pytest.mark.slow
    @pytest.mark.sanity
    def test_lazy_loading_on_call(self):
        """Tokenizer is loaded on first call from the vendored fixture.

        ### WRITTEN BY AI ###
        """
        config = HuggingFaceTokenizerArgs(
            model=str(MINIMAL_TOKENIZER_DIR),
            load_kwargs={"local_files_only": True},
        )
        tokenizer = HuggingFaceTokenizer(config)

        # First call loads
        result = tokenizer()
        assert result is not None
        assert tokenizer._tokenizer is not None

    @pytest.mark.slow
    @pytest.mark.sanity
    def test_caching(self):
        """Second call returns cached tokenizer.

        ### WRITTEN BY AI ###
        """
        config = HuggingFaceTokenizerArgs(
            model=str(MINIMAL_TOKENIZER_DIR),
            load_kwargs={"local_files_only": True},
        )
        tokenizer = HuggingFaceTokenizer(config)

        first_call = tokenizer()
        second_call = tokenizer()
        assert first_call is second_call

    @pytest.mark.slow
    @pytest.mark.regression
    def test_load_kwargs_passed_through(self, monkeypatch: pytest.MonkeyPatch):
        """load_kwargs are passed to AutoTokenizer.from_pretrained.

        ### WRITTEN BY AI ###
        """
        captured: dict = {}

        def fake_from_pretrained(model, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs
            return object()

        monkeypatch.setattr(
            "guidellm.data.tokenizers.huggingface.AutoTokenizer.from_pretrained",
            fake_from_pretrained,
        )
        config = HuggingFaceTokenizerArgs(
            model=str(MINIMAL_TOKENIZER_DIR),
            load_kwargs={"local_files_only": True, "use_fast": False},
        )
        tokenizer = HuggingFaceTokenizer(config)
        result = tokenizer()
        assert result is not None
        assert captured["model"] == str(MINIMAL_TOKENIZER_DIR)
        assert captured["kwargs"]["local_files_only"] is True
        assert captured["kwargs"]["use_fast"] is False

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "name",
        [
            "qwen3:4b",  # disallowed character
            "q" * 97,  # longer than 96 characters
            "-olama",  # starts with "-"
            ".foo",  # starts with "."
            "foo--bar",  # contains "--"
            "a/b/c",  # more than one "/"
        ],
    )
    def test_invalid_name_fails_with_a_hint(
        self, monkeypatch: pytest.MonkeyPatch, name: str
    ):
        """A name that is not an existing path and breaks any Hugging Face repo
        id rule fails before any download, with a message that names the fix.

        ### WRITTEN BY AI ###
        """

        def fail_from_pretrained(*args, **kwargs):
            raise AssertionError("from_pretrained must not be called")

        monkeypatch.setattr(
            "guidellm.data.tokenizers.huggingface.AutoTokenizer.from_pretrained",
            fail_from_pretrained,
        )
        tokenizer = HuggingFaceTokenizer(HuggingFaceTokenizerArgs(model=name))
        hint = "--tokenizer kind=huggingface_auto"
        with pytest.raises(ValueError, match=hint) as info:
            tokenizer()
        assert str(info.value).count(name) == 1
        assert info.value.__cause__ is None
        assert info.value.__suppress_context__

    @pytest.mark.sanity
    def test_valid_repo_id_is_passed_through(self, monkeypatch: pytest.MonkeyPatch):
        """A well-formed Hugging Face id still reaches from_pretrained.

        ### WRITTEN BY AI ###
        """
        captured: dict = {}

        def fake_from_pretrained(model, **kwargs):
            captured["model"] = model
            return object()

        monkeypatch.setattr(
            "guidellm.data.tokenizers.huggingface.AutoTokenizer.from_pretrained",
            fake_from_pretrained,
        )
        HuggingFaceTokenizer(HuggingFaceTokenizerArgs(model="Qwen/Qwen3-4B"))()
        assert captured["model"] == "Qwen/Qwen3-4B"

    @pytest.mark.sanity
    def test_existing_local_path_is_not_validated_as_repo_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        """A local directory is loaded even if its name is not a valid repo id.

        ### WRITTEN BY AI ###
        """
        local = tmp_path / "tokenizer:v1"
        local.mkdir()
        captured: dict = {}

        def fake_from_pretrained(model, **kwargs):
            captured["model"] = model
            return object()

        monkeypatch.setattr(
            "guidellm.data.tokenizers.huggingface.AutoTokenizer.from_pretrained",
            fake_from_pretrained,
        )
        HuggingFaceTokenizer(HuggingFaceTokenizerArgs(model=str(local)))()
        assert captured["model"] == str(local)
