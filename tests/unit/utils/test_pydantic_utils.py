"""
Unit tests for the pydantic_utils module.
"""

from __future__ import annotations

from typing import ClassVar
from unittest import mock

import pytest
from pydantic import BaseModel, Field, ValidationError

from guidellm.utils.pydantic_utils import (
    PydanticClassRegistryMixin,
    ReloadableBaseModel,
    StandardBaseDict,
    StandardBaseModel,
    StatusBreakdown,
)


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
        assert config["validate_assignment"] is True
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

        # Mock the model_rebuild method to simulate schema reload
        with mock.patch.object(TestModel, "model_rebuild") as mock_rebuild:
            TestModel.reload_schema()
            mock_rebuild.assert_called_once_with(force=True)

    @pytest.mark.sanity
    def test_marshalling(self, valid_instances):
        """Test ReloadableBaseModel serialization and deserialization."""
        instance, constructor_args = valid_instances
        data_dict = instance.model_dump()
        assert isinstance(data_dict, dict)
        assert data_dict["name"] == constructor_args["name"]

        recreated = instance.__class__.model_validate(data_dict)
        assert isinstance(recreated, instance.__class__)
        assert recreated.name == constructor_args["name"]


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
        assert config["validate_assignment"] is True
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

    @pytest.mark.smoke
    def test_get_default(self):
        """Test StandardBaseModel.get_default method."""

        class TestModel(StandardBaseModel):
            field_str: str = Field(description="Test string field")
            field_int: int = Field(default=42, description="Test integer field")

        default_value = TestModel.get_default("field_int")
        assert default_value == 42

    @pytest.mark.sanity
    def test_get_default_invalid(self):
        """Test StandardBaseModel.get_default with invalid field."""

        class TestModel(StandardBaseModel):
            field_str: str = Field(description="Test string field")

        with pytest.raises(KeyError):
            TestModel.get_default("nonexistent_field")

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
        assert config["validate_assignment"] is True
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
        assert hasattr(PydanticClassRegistryMixin, "auto_populate_registry")

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
        assert "testsubmodel" in TestBaseModel.registry  # type: ignore[misc]
        assert TestBaseModel.registry["testsubmodel"] is TestSubModel  # type: ignore[misc]

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
            ),
        ):
            result = TestBaseModel.auto_populate_registry()
            assert result is True
            mock_reload.assert_called_once()

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
