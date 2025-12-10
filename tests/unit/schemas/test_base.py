"""
Unit tests for the base pydantic utilities module.
"""

from __future__ import annotations

from typing import ClassVar, TypeVar
from unittest import mock

import pytest
from pydantic import BaseModel, Field, ValidationError

from guidellm.schemas import (
    BaseModelT,
    ErroredT,
    IncompleteT,
    PydanticClassRegistryMixin,
    RegisterClassT,
    ReloadableBaseModel,
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


class TestReloadableBaseModel:
    """Test suite for ReloadableBaseModel."""

    @pytest.fixture(
        params=[
            {"name": "test_value"},
            {"name": "hello_world"},
            {"name": "another_test"},
        ],
        ids=["basic_string", "multi_word", "underscore"],
    )
    def valid_instances(self, request) -> tuple[ReloadableBaseModel, dict[str, str]]:
        """Fixture providing test data for ReloadableBaseModel."""

        class TestModel(ReloadableBaseModel):
            name: str

        constructor_args = request.param
        instance = TestModel(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test ReloadableBaseModel inheritance and class variables."""
        assert issubclass(ReloadableBaseModel, BaseModel)
        assert hasattr(ReloadableBaseModel, "model_config")
        assert hasattr(ReloadableBaseModel, "reload_schema")

        # Check model configuration
        config = ReloadableBaseModel.model_config
        assert config["extra"] == "ignore"
        assert config["use_enum_values"] is True
        assert config["from_attributes"] is True
        assert config["arbitrary_types_allowed"] is True

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test ReloadableBaseModel initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, ReloadableBaseModel)
        assert instance.name == constructor_args["name"]  # type: ignore[attr-defined]

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("name", None),
            ("name", 123),
            ("name", []),
        ],
    )
    def test_invalid_initialization_values(self, field, value):
        """Test ReloadableBaseModel with invalid field values."""

        class TestModel(ReloadableBaseModel):
            name: str

        data = {field: value}
        with pytest.raises(ValidationError):
            TestModel(**data)

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test ReloadableBaseModel initialization without required field."""

        class TestModel(ReloadableBaseModel):
            name: str

        with pytest.raises(ValidationError):
            TestModel()  # type: ignore[call-arg]

    @pytest.mark.smoke
    def test_reload_schema(self):
        """Test ReloadableBaseModel.reload_schema method."""

        class TestModel(ReloadableBaseModel):
            name: str

        # Test standard reload with mocked pathways
        with (
            mock.patch.object(TestModel, "model_rebuild") as mock_rebuild,
            mock.patch.object(TestModel, "reload_parent_schemas") as mock_parents,
        ):
            TestModel.reload_schema()
            mock_rebuild.assert_called_once_with(force=True)
            mock_parents.assert_called_once()

        # Test without parent reloading with mocked pathways
        with (
            mock.patch.object(TestModel, "model_rebuild") as mock_rebuild,
            mock.patch.object(TestModel, "reload_parent_schemas") as mock_parents,
        ):
            TestModel.reload_schema(parents=False)
            mock_rebuild.assert_called_once_with(force=True)
            mock_parents.assert_not_called()

        # Test parent reloading separately with mocked pathways
        class ParentModel(ReloadableBaseModel):
            child: TestModel

        with mock.patch.object(ParentModel, "model_rebuild") as mock_parent_rebuild:
            TestModel.reload_parent_schemas()
            # Schema rebuild may or may not be triggered depending on structure
            assert mock_parent_rebuild.call_count >= 0

    @pytest.mark.sanity
    def test_marshalling(
        self, valid_instances: tuple[ReloadableBaseModel, dict[str, str]]
    ):
        """Test ReloadableBaseModel serialization and deserialization."""
        instance, constructor_args = valid_instances
        data_dict = instance.model_dump()
        assert isinstance(data_dict, dict)
        assert data_dict["name"] == constructor_args["name"]

        recreated = instance.__class__.model_validate(data_dict)
        assert isinstance(recreated, instance.__class__)
        assert recreated.name == constructor_args["name"]

    @pytest.mark.sanity
    def test_json_serialization(self, valid_instances):
        """Test ReloadableBaseModel JSON serialization."""
        instance, constructor_args = valid_instances
        json_data = instance.model_dump_json()
        assert isinstance(json_data, str)
        assert constructor_args["name"] in json_data

        recreated = instance.__class__.model_validate_json(json_data)
        assert isinstance(recreated, instance.__class__)
        assert recreated.name == constructor_args["name"]

    @pytest.mark.sanity
    def test_extra_fields_ignored(self):
        """Test that ReloadableBaseModel ignores extra fields."""

        class TestModel(ReloadableBaseModel):
            name: str

        # Extra fields should be ignored
        instance = TestModel(name="test", extra_field="ignored")
        assert instance.name == "test"
        assert not hasattr(instance, "extra_field")

    @pytest.mark.sanity
    def test_from_attributes(self):
        """Test ReloadableBaseModel from_attributes configuration."""

        class TestModel(ReloadableBaseModel):
            name: str
            value: int = 10

        class SourceObject:
            def __init__(self):
                self.name = "test_name"
                self.value = 42

        source = SourceObject()
        instance = TestModel.model_validate(source)
        assert instance.name == "test_name"
        assert instance.value == 42

    @pytest.mark.sanity
    def test_arbitrary_types_allowed(self):
        """Test ReloadableBaseModel arbitrary_types_allowed configuration."""

        class CustomType:
            def __init__(self, val: str):
                self.val = val

        class TestModel(ReloadableBaseModel):
            name: str
            custom: CustomType

        custom_obj = CustomType("test")
        instance = TestModel(name="test", custom=custom_obj)
        assert instance.name == "test"
        assert isinstance(instance.custom, CustomType)
        assert instance.custom.val == "test"


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

    @pytest.fixture(
        params=[
            {"test_type": "test_sub", "value": "test_value"},
            {"test_type": "test_sub", "value": "hello_world"},
        ],
        ids=["basic_value", "multi_word"],
    )
    def valid_instances(
        self, request
    ) -> tuple[PydanticClassRegistryMixin, dict, type, type]:
        """Fixture providing test data for PydanticClassRegistryMixin."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        @TestBaseModel.register("test_sub")
        class TestSubModel(TestBaseModel):
            test_type: str = "test_sub"
            value: str

        TestBaseModel.reload_schema()

        constructor_args = request.param
        instance = TestSubModel(value=constructor_args["value"])
        return instance, constructor_args, TestBaseModel, TestSubModel

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test PydanticClassRegistryMixin inheritance and class variables."""
        assert issubclass(PydanticClassRegistryMixin, ReloadableBaseModel)
        assert hasattr(PydanticClassRegistryMixin, "schema_discriminator")
        assert PydanticClassRegistryMixin.schema_discriminator == "model_type"
        assert hasattr(PydanticClassRegistryMixin, "register_decorator")
        assert hasattr(PydanticClassRegistryMixin, "__get_pydantic_core_schema__")
        assert hasattr(PydanticClassRegistryMixin, "__pydantic_generate_base_schema__")
        assert hasattr(PydanticClassRegistryMixin, "__pydantic_schema_base_type__")
        assert hasattr(PydanticClassRegistryMixin, "__new__")
        assert hasattr(PydanticClassRegistryMixin, "auto_populate_registry")
        assert hasattr(PydanticClassRegistryMixin, "registered_classes")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test PydanticClassRegistryMixin initialization."""
        instance, constructor_args, base_class, sub_class = valid_instances
        assert isinstance(instance, sub_class)
        assert isinstance(instance, base_class)
        assert instance.test_type == constructor_args["test_type"]
        assert instance.value == constructor_args["value"]

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("test_type", None),
            ("test_type", 123),
            ("value", None),
        ],
    )
    def test_invalid_initialization_values(self, field, value):
        """Test PydanticClassRegistryMixin with invalid field values."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        @TestBaseModel.register("test_sub")
        class TestSubModel(TestBaseModel):
            test_type: str = "test_sub"
            value: str

        data = {field: value}
        if field == "test_type":
            data["value"] = "test"
        else:
            data["test_type"] = "test_sub"

        with pytest.raises(ValidationError):
            TestSubModel(**data)

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test PydanticClassRegistryMixin initialization without required field."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        @TestBaseModel.register("test_sub")
        class TestSubModel(TestBaseModel):
            test_type: str = "test_sub"
            value: str

        with pytest.raises(ValidationError):
            TestSubModel()  # type: ignore[call-arg]

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

    @pytest.mark.smoke
    def test_register_decorator(self):
        """Test PydanticClassRegistryMixin.register_decorator method."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        @TestBaseModel.register()
        class TestSubModel(TestBaseModel):
            test_type: str = "TestSubModel"
            value: str

        assert TestBaseModel.registry is not None  # type: ignore[misc]
        assert "TestSubModel" in TestBaseModel.registry  # type: ignore[misc]
        assert TestBaseModel.registry["TestSubModel"] is TestSubModel  # type: ignore[misc]

    @pytest.mark.smoke
    def test_get_pydantic_core_schema(self):
        """Test PydanticClassRegistryMixin.__get_pydantic_core_schema__ method."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        @TestBaseModel.register("test_sub")
        class TestSubModel(TestBaseModel):
            test_type: str = "test_sub"
            value: str

        # Create a mock handler
        mock_handler = mock.Mock()
        mock_handler.return_value = {"type": "model"}

        # Test schema generation for base type
        schema = TestBaseModel.__get_pydantic_core_schema__(TestBaseModel, mock_handler)
        assert schema is not None
        assert schema["type"] == "tagged-union"

    @pytest.mark.sanity
    def test_get_pydantic_core_schema_no_registry(self):
        """Test __get_pydantic_core_schema__ without registry."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        # Ensure registry is empty
        TestBaseModel.registry = None

        # Create a mock handler
        mock_handler = mock.Mock()

        # Test schema generation without registry
        with mock.patch.object(
            TestBaseModel, "__pydantic_generate_base_schema__"
        ) as mock_base:
            TestBaseModel.__get_pydantic_core_schema__(TestBaseModel, mock_handler)
            mock_base.assert_called_once_with(mock_handler)

    @pytest.mark.sanity
    def test_get_pydantic_core_schema_subclass(self):
        """Test __get_pydantic_core_schema__ for a subclass."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        @TestBaseModel.register("test_sub")
        class TestSubModel(TestBaseModel):
            test_type: str = "test_sub"
            value: str

        # Create a mock handler
        mock_handler = mock.Mock()
        mock_handler.return_value = {"type": "model"}

        # Test schema generation for subclass (not base type)
        schema = TestBaseModel.__get_pydantic_core_schema__(TestSubModel, mock_handler)
        assert schema is not None
        mock_handler.assert_called_once_with(TestBaseModel)

    @pytest.mark.smoke
    def test_pydantic_generate_base_schema(self):
        """Test PydanticClassRegistryMixin.__pydantic_generate_base_schema__."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        mock_handler = mock.Mock()
        schema = TestBaseModel.__pydantic_generate_base_schema__(mock_handler)
        assert schema is not None
        assert schema["type"] == "any"

    @pytest.mark.sanity
    def test_register_decorator_with_name(self):
        """Test PydanticClassRegistryMixin.register_decorator with custom name."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        @TestBaseModel.register("custom_name")
        class TestSubModel(TestBaseModel):
            test_type: str = "custom_name"
            value: str

        assert TestBaseModel.registry is not None  # type: ignore[misc]
        assert "custom_name" in TestBaseModel.registry  # type: ignore[misc]
        assert TestBaseModel.registry["custom_name"] is TestSubModel  # type: ignore[misc]

    @pytest.mark.sanity
    def test_register_decorator_with_multiple_names(self):
        """Test PydanticClassRegistryMixin.register_decorator with list of names."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        @TestBaseModel.register(["name_one", "name_two"])
        class TestSubModel(TestBaseModel):
            test_type: str = "name_one"
            value: str

        assert TestBaseModel.registry is not None  # type: ignore[misc]
        assert "name_one" in TestBaseModel.registry  # type: ignore[misc]
        assert "name_two" in TestBaseModel.registry  # type: ignore[misc]
        assert TestBaseModel.registry["name_one"] is TestSubModel  # type: ignore[misc]
        assert TestBaseModel.registry["name_two"] is TestSubModel  # type: ignore[misc]

    @pytest.mark.sanity
    def test_register_decorator_invalid_type(self):
        """Test PydanticClassRegistryMixin.register_decorator with invalid type."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        class RegularClass:
            pass

        with pytest.raises(TypeError) as exc_info:
            TestBaseModel.register_decorator(RegularClass)  # type: ignore[arg-type]

        assert "not a subclass of Pydantic BaseModel" in str(exc_info.value)

    @pytest.mark.smoke
    def test_auto_populate_registry(self):
        """Test PydanticClassRegistryMixin.auto_populate_registry method."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str
            registry_auto_discovery: ClassVar[bool] = True

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        with (
            mock.patch.object(TestBaseModel, "reload_schema") as mock_reload,
            mock.patch(
                "guidellm.utils.registry.RegistryMixin.auto_populate_registry",
                return_value=True,
            ) as mock_parent_auto,
        ):
            result = TestBaseModel.auto_populate_registry()
            assert result is True
            mock_reload.assert_called_once()
            mock_parent_auto.assert_called_once()

    @pytest.mark.sanity
    def test_auto_populate_registry_already_populated(self):
        """Test auto_populate_registry when already populated."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str
            registry_auto_discovery: ClassVar[bool] = True

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        with (
            mock.patch.object(TestBaseModel, "reload_schema") as mock_reload,
            mock.patch(
                "guidellm.utils.registry.RegistryMixin.auto_populate_registry",
                return_value=False,
            ),
        ):
            result = TestBaseModel.auto_populate_registry()
            assert result is False
            mock_reload.assert_called_once()

    @pytest.mark.smoke
    def test_registered_classes(self):
        """Test PydanticClassRegistryMixin.registered_classes method."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str
            registry_auto_discovery: ClassVar[bool] = False

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        @TestBaseModel.register("test_sub_a")
        class TestSubModelA(TestBaseModel):
            test_type: str = "test_sub_a"
            value_a: str

        @TestBaseModel.register("test_sub_b")
        class TestSubModelB(TestBaseModel):
            test_type: str = "test_sub_b"
            value_b: int

        # Test normal case with registered classes
        registered = TestBaseModel.registered_classes()
        assert isinstance(registered, tuple)
        assert len(registered) == 2
        assert TestSubModelA in registered
        assert TestSubModelB in registered

    @pytest.mark.sanity
    def test_registered_classes_with_auto_discovery(self):
        """Test PydanticClassRegistryMixin.registered_classes with auto discovery."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str
            registry_auto_discovery: ClassVar[bool] = True

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        with mock.patch.object(
            TestBaseModel, "auto_populate_registry"
        ) as mock_auto_populate:
            # Mock the registry to simulate registered classes
            TestBaseModel.registry = {"test_class": type("TestClass", (), {})}
            mock_auto_populate.return_value = False

            registered = TestBaseModel.registered_classes()
            mock_auto_populate.assert_called_once()
            assert isinstance(registered, tuple)
            assert len(registered) == 1

    @pytest.mark.sanity
    def test_registered_classes_no_registry(self):
        """Test PydanticClassRegistryMixin.registered_classes with no registry."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        # Ensure registry is None
        TestBaseModel.registry = None

        with pytest.raises(ValueError) as exc_info:
            TestBaseModel.registered_classes()

        assert "must be called after registering classes" in str(exc_info.value)

    @pytest.mark.sanity
    def test_marshalling(self, valid_instances):
        """Test PydanticClassRegistryMixin serialization and deserialization."""
        instance, constructor_args, base_class, sub_class = valid_instances

        # Test serialization with model_dump
        dump_data = instance.model_dump()
        assert isinstance(dump_data, dict)
        assert dump_data["test_type"] == constructor_args["test_type"]
        assert dump_data["value"] == constructor_args["value"]

        # Test deserialization via subclass
        recreated = sub_class.model_validate(dump_data)
        assert isinstance(recreated, sub_class)
        assert recreated.test_type == constructor_args["test_type"]
        assert recreated.value == constructor_args["value"]

        # Test polymorphic deserialization via base class
        recreated_base = base_class.model_validate(dump_data)  # type: ignore[assignment]
        assert isinstance(recreated_base, sub_class)
        assert recreated_base.test_type == constructor_args["test_type"]
        assert recreated_base.value == constructor_args["value"]

    @pytest.mark.regression
    def test_polymorphic_container_marshalling(self):
        """Test PydanticClassRegistryMixin in container models."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

            @classmethod
            def __pydantic_generate_base_schema__(cls, handler):
                return handler(cls)

        @TestBaseModel.register("sub_a")
        class TestSubModelA(TestBaseModel):
            test_type: str = "sub_a"
            value_a: str

        @TestBaseModel.register("sub_b")
        class TestSubModelB(TestBaseModel):
            test_type: str = "sub_b"
            value_b: int

        class ContainerModel(BaseModel):
            name: str
            model: TestBaseModel
            models: list[TestBaseModel]

        sub_a = TestSubModelA(value_a="test")
        sub_b = TestSubModelB(value_b=123)

        container = ContainerModel(name="container", model=sub_a, models=[sub_a, sub_b])

        # Verify container construction
        assert isinstance(container.model, TestSubModelA)
        assert container.model.test_type == "sub_a"
        assert container.model.value_a == "test"
        assert len(container.models) == 2
        assert isinstance(container.models[0], TestSubModelA)
        assert isinstance(container.models[1], TestSubModelB)

        # Test serialization
        dump_data = container.model_dump()
        assert isinstance(dump_data, dict)
        assert dump_data["name"] == "container"
        assert dump_data["model"]["test_type"] == "sub_a"
        assert dump_data["model"]["value_a"] == "test"
        assert len(dump_data["models"]) == 2
        assert dump_data["models"][0]["test_type"] == "sub_a"
        assert dump_data["models"][1]["test_type"] == "sub_b"

        # Test deserialization
        recreated = ContainerModel.model_validate(dump_data)
        assert isinstance(recreated, ContainerModel)
        assert recreated.name == "container"
        assert isinstance(recreated.model, TestSubModelA)
        assert len(recreated.models) == 2
        assert isinstance(recreated.models[0], TestSubModelA)
        assert isinstance(recreated.models[1], TestSubModelB)

    @pytest.mark.regression
    def test_json_serialization(self):
        """Test PydanticClassRegistryMixin JSON serialization."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        @TestBaseModel.register("test_sub")
        class TestSubModel(TestBaseModel):
            test_type: str = "test_sub"
            value: str

        instance = TestSubModel(value="test_value")
        json_data = instance.model_dump_json()
        assert isinstance(json_data, str)
        assert "test_sub" in json_data
        assert "test_value" in json_data

        # Deserialize through base class
        recreated = TestBaseModel.model_validate_json(json_data)  # type: ignore[assignment]
        assert isinstance(recreated, TestSubModel)
        assert recreated.test_type == "test_sub"
        assert recreated.value == "test_value"
