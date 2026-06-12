"""Tests for click-pydantic integration helpers."""

from __future__ import annotations

from typing import Any

import click
import pytest
from pydantic import BaseModel, Field, ValidationError

from guidellm.utils.click_pydantic import _resolve_param_name, format_validation_errors


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
        """Field at first level with argument_alias is returned.

        ## WRITTEN BY AI ##
        """
        assert _resolve_param_name(("items",), _Outer) == "item"

    def test_nested_alias(self):
        """Walk through a field without alias to reach one with alias.

        ## WRITTEN BY AI ##
        """
        assert _resolve_param_name(("spec", "items"), _Root) == "item"

    def test_nested_inner_alias(self):
        """Walk two levels deep through nested models.

        ## WRITTEN BY AI ##
        """
        assert _resolve_param_name(("spec", "nested", "value"), _Root) == "inner-alias"

    def test_int_index_skipped(self):
        """Integer loc components (list indices) are skipped.

        ## WRITTEN BY AI ##
        """
        assert _resolve_param_name(("items", 0, "value"), _Outer) == "item"

    def test_no_alias_returns_none(self):
        """Returns None when no field in the path has an argument_alias.

        ## WRITTEN BY AI ##
        """
        assert _resolve_param_name(("plain", "count"), _Outer) is None

    def test_unknown_field_returns_none(self):
        """Returns None when a loc component doesn't match any field.

        ## WRITTEN BY AI ##
        """
        assert _resolve_param_name(("nonexistent",), _Outer) is None

    def test_non_model_annotation_returns_none(self):
        """Returns None when the annotation is not a BaseModel subclass.

        ## WRITTEN BY AI ##
        """
        assert _resolve_param_name(("tags", "key"), _Outer) is None

    def test_empty_loc_returns_none(self):
        """Returns None for an empty loc tuple.

        ## WRITTEN BY AI ##
        """
        assert _resolve_param_name((), _Outer) is None


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
