"""
Unit tests for the base pydantic utilities module.
"""

from __future__ import annotations

from typing import ClassVar, TypeVar

import pytest
from pydantic import BaseModel, Field, ValidationError

from guidellm.schemas import (
    BaseModelT,
    ErroredT,
    IncompleteT,
    PydanticClassRegistryMixin,
    RegisterClassT,
    StandardBaseDict,
    StandardBaseModel,
    StatusBreakdown,
    SuccessfulT,
    TotalT,
)


@pytest.mark.smoke
def test_base_model_t():
    """Test that BaseModelT is configured correctly as a TypeVar."""
    assert isinstance(BaseModelT, type(TypeVar("test")))
    assert BaseModelT.__name__ == "BaseModelT"
    assert BaseModelT.__bound__ is BaseModel
    assert BaseModelT.__constraints__ == ()


@pytest.mark.smoke
def test_register_class_t():
    """Test that RegisterClassT is configured correctly as a TypeVar."""
    assert isinstance(RegisterClassT, type(TypeVar("test")))
    assert RegisterClassT.__name__ == "RegisterClassT"
    assert RegisterClassT.__bound__ is type
    assert RegisterClassT.__constraints__ == ()


@pytest.mark.smoke
def test_successful_t():
    """Test that SuccessfulT is configured correctly as a TypeVar."""
    assert isinstance(SuccessfulT, type(TypeVar("test")))
    assert SuccessfulT.__name__ == "SuccessfulT"
    assert SuccessfulT.__bound__ is None
    assert SuccessfulT.__constraints__ == ()


@pytest.mark.smoke
def test_errored_t():
    """Test that ErroredT is configured correctly as a TypeVar."""
    assert isinstance(ErroredT, type(TypeVar("test")))
    assert ErroredT.__name__ == "ErroredT"
    assert ErroredT.__bound__ is None
    assert ErroredT.__constraints__ == ()


@pytest.mark.smoke
def test_incomplete_t():
    """Test that IncompleteT is configured correctly as a TypeVar."""
    assert isinstance(IncompleteT, type(TypeVar("test")))
    assert IncompleteT.__name__ == "IncompleteT"
    assert IncompleteT.__bound__ is None
    assert IncompleteT.__constraints__ == ()


@pytest.mark.smoke
def test_total_t():
    """Test that TotalT is configured correctly as a TypeVar."""
    assert isinstance(TotalT, type(TypeVar("test")))
    assert TotalT.__name__ == "TotalT"
    assert TotalT.__bound__ is None
    assert TotalT.__constraints__ == ()


class TestStandardBaseModel:
    """Test suite for StandardBaseModel."""

    @pytest.fixture(
        params=[
            {"field_str": "test_value", "field_int": 42},
            {"field_str": "hello_world", "field_int": 100},
            {"field_str": "another_test", "field_int": 0},
        ],
        ids=["basic_values", "positive_values", "zero_value"],
    )
    def valid_instances(
        self, request
    ) -> tuple[StandardBaseModel, dict[str, int | str]]:
        """Fixture providing test data for StandardBaseModel."""

        class TestModel(StandardBaseModel):
            field_str: str = Field(description="Test string field")
            field_int: int = Field(default=10, description="Test integer field")

        constructor_args = request.param
        instance = TestModel(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test StandardBaseModel inheritance and class variables."""
        assert issubclass(StandardBaseModel, BaseModel)
        assert hasattr(StandardBaseModel, "model_config")
        assert hasattr(StandardBaseModel, "get_default")

        # Check model configuration
        config = StandardBaseModel.model_config
        assert config["extra"] == "ignore"
        assert config["use_enum_values"] is True
        assert config["from_attributes"] is True

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test StandardBaseModel initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, StandardBaseModel)
        assert instance.field_str == constructor_args["field_str"]  # type: ignore[attr-defined]
        assert instance.field_int == constructor_args["field_int"]  # type: ignore[attr-defined]

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("field_str", None),
            ("field_str", 123),
            ("field_int", "not_int"),
        ],
    )
    def test_invalid_initialization_values(self, field, value):
        """Test StandardBaseModel with invalid field values."""

        class TestModel(StandardBaseModel):
            field_str: str = Field(description="Test string field")
            field_int: int = Field(default=10, description="Test integer field")

        data = {field: value}
        if field == "field_str":
            data["field_int"] = 42
        else:
            data["field_str"] = "test"

        with pytest.raises(ValidationError):
            TestModel(**data)

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test StandardBaseModel initialization without required field."""

        class TestModel(StandardBaseModel):
            field_str: str = Field(description="Test string field")
            field_int: int = Field(default=10, description="Test integer field")

        with pytest.raises(ValidationError):
            TestModel()  # type: ignore[call-arg]

    @pytest.mark.sanity
    def test_marshalling(self, valid_instances):
        """Test StandardBaseModel serialization and deserialization."""
        instance, constructor_args = valid_instances
        data_dict = instance.model_dump()
        assert isinstance(data_dict, dict)
        assert data_dict["field_str"] == constructor_args["field_str"]
        assert data_dict["field_int"] == constructor_args["field_int"]

        recreated = instance.__class__.model_validate(data_dict)
        assert isinstance(recreated, instance.__class__)
        assert recreated.field_str == constructor_args["field_str"]
        assert recreated.field_int == constructor_args["field_int"]

    @pytest.mark.sanity
    def test_json_serialization(self, valid_instances):
        """Test StandardBaseModel JSON serialization."""
        instance, constructor_args = valid_instances
        json_data = instance.model_dump_json()
        assert isinstance(json_data, str)
        assert constructor_args["field_str"] in json_data

        recreated = instance.__class__.model_validate_json(json_data)
        assert isinstance(recreated, instance.__class__)
        assert recreated.field_str == constructor_args["field_str"]
        assert recreated.field_int == constructor_args["field_int"]

    @pytest.mark.sanity
    def test_extra_fields_ignored(self):
        """Test that StandardBaseModel ignores extra fields."""

        class TestModel(StandardBaseModel):
            field_str: str = Field(description="Test string field")
            field_int: int = Field(default=10, description="Test integer field")

        # Extra fields should be ignored
        instance = TestModel(field_str="test", extra_field="ignored")
        assert instance.field_str == "test"
        assert not hasattr(instance, "extra_field")

    @pytest.mark.sanity
    def test_from_attributes(self):
        """Test StandardBaseModel from_attributes configuration."""

        class TestModel(StandardBaseModel):
            field_str: str = Field(description="Test string field")
            field_int: int = Field(default=10, description="Test integer field")

        class SourceObject:
            def __init__(self):
                self.field_str = "test_value"
                self.field_int = 99

        source = SourceObject()
        instance = TestModel.model_validate(source)
        assert instance.field_str == "test_value"
        assert instance.field_int == 99


class TestStandardBaseDict:
    """Test suite for StandardBaseDict."""

    @pytest.fixture(
        params=[
            {"field_str": "test_value", "extra_field": "extra_value"},
            {"field_str": "hello_world", "another_extra": 123},
            {"field_str": "another_test", "complex_extra": {"nested": "value"}},
        ],
        ids=["string_extra", "int_extra", "dict_extra"],
    )
    def valid_instances(
        self, request
    ) -> tuple[StandardBaseDict, dict[str, str | int | dict[str, str]]]:
        """Fixture providing test data for StandardBaseDict."""

        class TestModel(StandardBaseDict):
            field_str: str = Field(description="Test string field")

        constructor_args = request.param
        instance = TestModel(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test StandardBaseDict inheritance and class variables."""
        assert issubclass(StandardBaseDict, StandardBaseModel)
        assert hasattr(StandardBaseDict, "model_config")

        # Check model configuration
        config = StandardBaseDict.model_config
        assert config["extra"] == "allow"
        assert config["use_enum_values"] is True
        assert config["from_attributes"] is True
        assert config["arbitrary_types_allowed"] is True

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test StandardBaseDict initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, StandardBaseDict)
        assert instance.field_str == constructor_args["field_str"]  # type: ignore[attr-defined]

        # Check extra fields are preserved
        for key, value in constructor_args.items():
            if key != "field_str":
                assert hasattr(instance, key)
                assert getattr(instance, key) == value

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("field_str", None),
            ("field_str", 123),
        ],
    )
    def test_invalid_initialization_values(self, field, value):
        """Test StandardBaseDict with invalid field values."""

        class TestModel(StandardBaseDict):
            field_str: str = Field(description="Test string field")

        data = {field: value}
        with pytest.raises(ValidationError):
            TestModel(**data)

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test StandardBaseDict initialization without required field."""

        class TestModel(StandardBaseDict):
            field_str: str = Field(description="Test string field")

        with pytest.raises(ValidationError):
            TestModel()  # type: ignore[call-arg]

    @pytest.mark.sanity
    def test_marshalling(self, valid_instances):
        """Test StandardBaseDict serialization and deserialization."""
        instance, constructor_args = valid_instances
        data_dict = instance.model_dump()
        assert isinstance(data_dict, dict)
        assert data_dict["field_str"] == constructor_args["field_str"]

        # Check extra fields are in the serialized data
        for key, value in constructor_args.items():
            if key != "field_str":
                assert key in data_dict
                assert data_dict[key] == value

        recreated = instance.__class__.model_validate(data_dict)
        assert isinstance(recreated, instance.__class__)
        assert recreated.field_str == constructor_args["field_str"]

        # Check extra fields are preserved after deserialization
        for key, value in constructor_args.items():
            if key != "field_str":
                assert hasattr(recreated, key)
                assert getattr(recreated, key) == value

    @pytest.mark.sanity
    def test_json_serialization(self, valid_instances):
        """Test StandardBaseDict JSON serialization."""
        instance, constructor_args = valid_instances
        json_data = instance.model_dump_json()
        assert isinstance(json_data, str)
        assert constructor_args["field_str"] in json_data

        recreated = instance.__class__.model_validate_json(json_data)
        assert isinstance(recreated, instance.__class__)
        assert recreated.field_str == constructor_args["field_str"]

        # Check extra fields are preserved after JSON deserialization
        for key in constructor_args:
            if key != "field_str":
                assert hasattr(recreated, key)

    @pytest.mark.sanity
    def test_arbitrary_types_allowed(self):
        """Test StandardBaseDict arbitrary_types_allowed configuration."""

        class CustomType:
            def __init__(self, val: str):
                self.val = val

        class TestModel(StandardBaseDict):
            field_str: str = Field(description="Test string field")

        custom_obj = CustomType("test")
        instance = TestModel(field_str="test", custom=custom_obj)
        assert instance.field_str == "test"
        assert hasattr(instance, "custom")
        assert isinstance(instance.custom, CustomType)
        assert instance.custom.val == "test"


class TestStatusBreakdown:
    """Test suite for StatusBreakdown."""

    @pytest.fixture(
        params=[
            {"successful": 100, "errored": 5, "incomplete": 10, "total": 115},
            {
                "successful": "success_data",
                "errored": "error_data",
                "incomplete": "incomplete_data",
                "total": "total_data",
            },
            {
                "successful": [1, 2, 3],
                "errored": [4, 5],
                "incomplete": [6],
                "total": [1, 2, 3, 4, 5, 6],
            },
        ],
        ids=["int_values", "string_values", "list_values"],
    )
    def valid_instances(self, request) -> tuple[StatusBreakdown, dict]:
        """Fixture providing test data for StatusBreakdown."""
        constructor_args = request.param
        instance = StatusBreakdown(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test StatusBreakdown inheritance and type relationships."""
        assert issubclass(StatusBreakdown, BaseModel)
        # Check if Generic is in the MRO (method resolution order)
        assert any(cls.__name__ == "Generic" for cls in StatusBreakdown.__mro__)
        assert "successful" in StatusBreakdown.model_fields
        assert "errored" in StatusBreakdown.model_fields
        assert "incomplete" in StatusBreakdown.model_fields
        assert "total" in StatusBreakdown.model_fields

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test StatusBreakdown initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, StatusBreakdown)
        assert instance.successful == constructor_args["successful"]
        assert instance.errored == constructor_args["errored"]
        assert instance.incomplete == constructor_args["incomplete"]
        assert instance.total == constructor_args["total"]

    @pytest.mark.smoke
    def test_initialization_defaults(self):
        """Test StatusBreakdown initialization with default values."""
        instance: StatusBreakdown = StatusBreakdown()
        assert instance.successful is None
        assert instance.errored is None
        assert instance.incomplete is None
        assert instance.total is None

    @pytest.mark.sanity
    def test_marshalling(self, valid_instances):
        """Test StatusBreakdown serialization and deserialization."""
        instance, constructor_args = valid_instances
        data_dict = instance.model_dump()
        assert isinstance(data_dict, dict)
        assert data_dict["successful"] == constructor_args["successful"]
        assert data_dict["errored"] == constructor_args["errored"]
        assert data_dict["incomplete"] == constructor_args["incomplete"]
        assert data_dict["total"] == constructor_args["total"]

        recreated: StatusBreakdown = StatusBreakdown.model_validate(data_dict)
        assert isinstance(recreated, StatusBreakdown)
        assert recreated.successful == constructor_args["successful"]
        assert recreated.errored == constructor_args["errored"]
        assert recreated.incomplete == constructor_args["incomplete"]
        assert recreated.total == constructor_args["total"]

    @pytest.mark.sanity
    def test_json_serialization(self):
        """Test StatusBreakdown JSON serialization with various types."""
        instance: StatusBreakdown = StatusBreakdown(
            successful=100, errored=5, incomplete=10, total=115
        )
        json_data = instance.model_dump_json()
        assert isinstance(json_data, str)
        assert "100" in json_data

        recreated = StatusBreakdown.model_validate_json(json_data)
        assert recreated.successful == 100
        assert recreated.errored == 5
        assert recreated.incomplete == 10
        assert recreated.total == 115


class TestPydanticClassRegistryMixin:
    """Test suite for PydanticClassRegistryMixin."""

    @pytest.mark.sanity
    def test_base_class_direct_instantiation(self):
        """Test that base class cannot be instantiated directly."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        with pytest.raises(TypeError) as exc_info:
            TestBaseModel(test_type="test")

        assert "only children of" in str(exc_info.value)
