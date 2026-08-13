"""Tests for click-pydantic integration helpers."""

from __future__ import annotations

from typing import Any

import click
import pytest
from click.testing import CliRunner
from pydantic import BaseModel, Field, ValidationError

from guidellm.scheduler.constraints import ConstraintArgs
from guidellm.utils.click_pydantic import (
    RegistryAwareCommand,
    _error_to_message,
    _resolve_param_name,
    _resolve_registry_type,
    format_kind_config_usage,
    format_validation_errors,
    registry_option,
)


class _Inner(BaseModel):
    value: int = Field(
        default=0,
        json_schema_extra={"argument_alias": "inner-alias"},
    )


class _NoAlias(BaseModel):
    count: int = 0


class _Outer(BaseModel):
    nested: _Inner = Field(default_factory=_Inner)
    items: list[_Inner] = Field(
        default_factory=list,
        json_schema_extra={"argument_alias": "item"},
    )
    plain: _NoAlias = Field(default_factory=_NoAlias)
    tags: dict[str, Any] = Field(default_factory=dict)


class _Root(BaseModel):
    spec: _Outer = Field(default_factory=_Outer)


@pytest.mark.smoke
class TestResolveParamName:
    """Tests for _resolve_param_name loc-walking helper.

    ## WRITTEN BY AI ##
    """

    def test_direct_alias(self):
        """Field at first level with argument_alias returns sliced loc and alias.

        ## WRITTEN BY AI ##
        """
        loc, alias = _resolve_param_name(("items",), _Outer)
        assert alias == "item"
        assert loc == ("items",)

    def test_nested_alias(self):
        """Walk through a field without alias to reach one with alias.

        ## WRITTEN BY AI ##
        """
        loc, alias = _resolve_param_name(("spec", "items"), _Root)
        assert alias == "item"
        assert loc == ("items",)

    def test_nested_inner_alias(self):
        """Walk two levels deep through nested models.

        ## WRITTEN BY AI ##
        """
        loc, alias = _resolve_param_name(("spec", "nested", "value"), _Root)
        assert alias == "inner-alias"
        assert loc == ("value",)

    def test_int_index_skipped(self):
        """Integer loc components (list indices) are skipped.

        ## WRITTEN BY AI ##
        """
        loc, alias = _resolve_param_name(("items", 0, "value"), _Outer)
        assert alias == "item"
        assert loc == ("items", 0, "value")

    def test_no_alias_raises(self):
        """Raises KeyError when no field in the path has an argument_alias.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(KeyError):
            _resolve_param_name(("plain", "count"), _Outer)

    def test_unknown_field_raises(self):
        """Raises KeyError when a loc component doesn't match any field.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(KeyError):
            _resolve_param_name(("nonexistent",), _Outer)

    def test_non_model_annotation_raises(self):
        """Raises KeyError when the annotation is not a BaseModel subclass.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(KeyError):
            _resolve_param_name(("tags", "key"), _Outer)

    def test_empty_loc_raises(self):
        """Raises KeyError for an empty loc tuple.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(KeyError):
            _resolve_param_name((), _Outer)

    def test_loc_sliced_from_alias_field(self):
        """Returned loc starts at the field that matched the alias.

        ## WRITTEN BY AI ##
        """
        loc, alias = _resolve_param_name(("spec", "items", 0, "value"), _Root)
        assert alias == "item"
        assert loc == ("items", 0, "value")


@pytest.mark.smoke
class TestFormatValidationErrors:
    """Tests for format_validation_errors alias resolution.

    ## WRITTEN BY AI ##
    """

    @staticmethod
    def _make_ctx() -> click.Context:
        return click.Context(click.Command("test"))

    def test_resolves_alias_via_base_class(self):
        """Uses argument_alias from base_class model when available.

        ## WRITTEN BY AI ##
        """

        class _Model(BaseModel):
            target: _Inner = Field(
                default_factory=_Inner,
                json_schema_extra={"argument_alias": "tgt"},
            )

        try:
            _Model.model_validate({"target": {"value": "not-an-int"}})
        except ValidationError as err:
            result = format_validation_errors(self._make_ctx(), err, base_class=_Model)

        assert "--tgt" in result.format_message()

    def test_falls_back_without_base_class(self):
        """Falls back to first loc component when base_class is None.

        ## WRITTEN BY AI ##
        """

        class _Model(BaseModel):
            my_field: int

        try:
            _Model.model_validate({"my_field": "bad"})
        except ValidationError as err:
            result = format_validation_errors(self._make_ctx(), err)

        assert "--my-field" in result.format_message()

    def test_falls_back_when_no_alias_found(self):
        """Falls back to first loc when no alias exists in the path.

        ## WRITTEN BY AI ##
        """

        class _Model(BaseModel):
            count: int

        try:
            _Model.model_validate({"count": "bad"})
        except ValidationError as err:
            result = format_validation_errors(self._make_ctx(), err, base_class=_Model)

        assert "--count" in result.format_message()

    def test_error_message_uses_resolved_loc(self):
        """Error message loc path starts from the alias field, not the root.

        ## WRITTEN BY AI ##
        """

        class _Model(BaseModel):
            wrapper: _Inner = Field(
                default_factory=_Inner,
                json_schema_extra={"argument_alias": "wrap"},
            )

        try:
            _Model.model_validate({"wrapper": {"value": "bad"}})
        except ValidationError as err:
            result = format_validation_errors(self._make_ctx(), err, base_class=_Model)

        msg = result.format_message()
        assert "wrapper.value" in msg
        assert "--wrap" in msg

    def test_multiple_errors_collect_param_names(self):
        """Multiple errors across different fields produce multiple param hints.

        ## WRITTEN BY AI ##
        """

        class _Model(BaseModel):
            a: _Inner = Field(
                default_factory=_Inner,
                json_schema_extra={"argument_alias": "alpha"},
            )
            b: _Inner = Field(
                default_factory=_Inner,
                json_schema_extra={"argument_alias": "beta"},
            )

        try:
            _Model.model_validate({"a": {"value": "bad"}, "b": {"value": "bad"}})
        except ValidationError as err:
            result = format_validation_errors(self._make_ctx(), err, base_class=_Model)

        msg = result.format_message()
        assert "--alpha" in msg
        assert "--beta" in msg


@pytest.mark.sanity
class TestErrorToMessage:
    """Tests for _error_to_message path formatting helper.

    ## WRITTEN BY AI ##
    """

    def test_includes_indices_and_field_path(self):
        """Integer loc components render as ``[i]`` and strings as ``.name``.

        ## WRITTEN BY AI ##
        """
        formatted = _error_to_message(
            ("data", 0, "synthetic_text", "output_tokens"),
            "Field required",
        )
        assert formatted == "Field required (at 'data[0].synthetic_text.output_tokens')"

    def test_handles_top_level_only_path(self):
        """A single-element loc produces a readable message without trailing separator.

        ## WRITTEN BY AI ##
        """
        formatted = _error_to_message(
            ("rate",),
            "Input should be a valid number",
        )
        assert formatted == "Input should be a valid number (at 'rate')"


@pytest.mark.smoke
class TestFormatKindConfigUsage:
    """Tests for format_kind_config_usage helper.

    ## WRITTEN BY AI ##
    """

    def test_lists_registered_kinds_and_format(self):
        """Usage text includes expected format and all registered kinds.

        ## WRITTEN BY AI ##
        """
        usage = format_kind_config_usage(ConstraintArgs)
        assert "kind=<type>,key=value" in usage
        assert "JSON/YAML" in usage
        assert "max_requests" in usage
        assert "max_duration" in usage


@pytest.mark.sanity
class TestRegistryOptionErrors:
    """Tests for registry_option early rejection and missing-arg UX.

    ## WRITTEN BY AI ##
    """

    @staticmethod
    def _make_command(*, multiple: bool = False):
        @click.command(cls=RegistryAwareCommand)
        @registry_option("--cfg", "cfg", registry=ConstraintArgs, multiple=multiple)
        def cmd(cfg):
            click.echo(repr(cfg))

        return cmd

    def test_bare_word_shows_usage_and_kinds(self):
        """Non-dict values are rejected with format help and kind list.

        ## WRITTEN BY AI ##
        """
        runner = CliRunner()
        result = runner.invoke(self._make_command(), ["--cfg", "wrong"])
        assert result.exit_code != 0
        assert "Expected format" in result.output
        assert "max_requests" in result.output
        assert "valid dictionary" not in result.output.lower()

    def test_missing_argument_shows_usage_and_kinds(self):
        """Bare ``--cfg`` without a value documents expected format and kinds.

        ## WRITTEN BY AI ##
        """
        runner = CliRunner()
        result = runner.invoke(self._make_command(), ["--cfg"])
        assert result.exit_code != 0
        assert "requires a value" in result.output
        assert "Expected format" in result.output
        assert "max_requests" in result.output
        assert "Option '--cfg' requires an argument." not in result.output

    def test_missing_kind_key_shows_usage(self):
        """Dict-like input missing ``kind`` uses the shared usage wording.

        ## WRITTEN BY AI ##
        """
        runner = CliRunner()
        result = runner.invoke(self._make_command(), ["--cfg", "count=1"])
        assert result.exit_code != 0
        assert "missing required key 'kind'" in result.output
        assert "Expected format" in result.output
        assert "max_requests" in result.output

    def test_invalid_kind_shows_usage_once(self):
        """Unknown kind values get one usage block without pydantic tag duplication.

        ## WRITTEN BY AI ##
        """
        runner = CliRunner()
        result = runner.invoke(self._make_command(), ["--cfg", "kind=bogus"])
        assert result.exit_code != 0
        assert "invalid kind 'bogus'" in result.output
        assert "Expected format" in result.output
        assert "max_requests" in result.output
        assert "expected tags" not in result.output

    def test_multiple_missing_argument_shows_usage(self):
        """Repeatable registry options also document missing values.

        ## WRITTEN BY AI ##
        """
        runner = CliRunner()
        result = runner.invoke(self._make_command(multiple=True), ["--cfg"])
        assert result.exit_code != 0
        assert "requires a value" in result.output
        assert "Expected format" in result.output


@pytest.mark.sanity
class TestFormatValidationErrorsRegistryHints:
    """Tests for appending kind usage on registry-shape pydantic errors.

    ## WRITTEN BY AI ##
    """

    @staticmethod
    def _make_ctx() -> click.Context:
        return click.Context(click.Command("test"))

    def test_appends_usage_for_non_object_input(self):
        """Non-dict registry values get a usage hint in the BadParameter.

        ## WRITTEN BY AI ##
        """

        class _Model(BaseModel):
            constraints: list[ConstraintArgs] = Field(  # type: ignore[assignment]
                default_factory=list,
                json_schema_extra={"argument_alias": "constraint"},
            )

        try:
            _Model.model_validate({"constraints": ["wrong"]})
        except ValidationError as err:
            result = format_validation_errors(self._make_ctx(), err, base_class=_Model)

        msg = result.format_message()
        assert "Expected format" in msg
        assert "max_requests" in msg
        assert "--constraint" in msg

    def test_does_not_append_usage_for_invalid_kind(self):
        """Invalid kind tags already list valid kinds; do not append usage again.

        ## WRITTEN BY AI ##
        """

        class _Model(BaseModel):
            constraints: list[ConstraintArgs] = Field(  # type: ignore[assignment]
                default_factory=list,
                json_schema_extra={"argument_alias": "constraint"},
            )

        try:
            _Model.model_validate({"constraints": [{"kind": "bogus"}]})
        except ValidationError as err:
            result = format_validation_errors(self._make_ctx(), err, base_class=_Model)

        msg = result.format_message()
        assert "bogus" in msg
        assert "max_requests" in msg
        assert "Expected format" not in msg

    def test_does_not_append_usage_for_missing_field(self):
        """Normal field errors stay specific without generic format spam.

        ## WRITTEN BY AI ##
        """

        class _Model(BaseModel):
            constraints: list[ConstraintArgs] = Field(  # type: ignore[assignment]
                default_factory=list,
                json_schema_extra={"argument_alias": "constraint"},
            )

        try:
            _Model.model_validate({"constraints": [{"kind": "max_requests"}]})
        except ValidationError as err:
            result = format_validation_errors(self._make_ctx(), err, base_class=_Model)

        msg = result.format_message()
        assert "Field required" in msg or "count" in msg
        assert "Expected format" not in msg


@pytest.mark.smoke
class TestResolveRegistryType:
    """Tests for _resolve_registry_type loc walking.

    ## WRITTEN BY AI ##
    """

    def test_resolves_list_registry_field(self):
        """Finds the registry class for a list[ConstraintArgs] field.

        ## WRITTEN BY AI ##
        """

        class _Model(BaseModel):
            constraints: list[ConstraintArgs] = Field(default_factory=list)

        resolved = _resolve_registry_type(("constraints", 0), _Model)
        assert resolved is ConstraintArgs

    def test_unknown_field_returns_none(self):
        """Returns None when the loc does not match a registry field.

        ## WRITTEN BY AI ##
        """

        class _Model(BaseModel):
            count: int = 0

        assert _resolve_registry_type(("count",), _Model) is None
