from __future__ import annotations

import pytest

from guidellm.utils.mixins import InfoMixin


class TestInfoMixin:
    """Test suite for InfoMixin."""

    @pytest.fixture(
        params=[
            {"attr_one": "test_value", "attr_two": 42},
            {"attr_one": "hello_world", "attr_two": 100, "attr_three": [1, 2, 3]},
        ],
        ids=["basic_attributes", "extended_attributes"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for InfoMixin."""
        constructor_args = request.param

        class TestClass(InfoMixin):
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        instance = TestClass(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test InfoMixin class signatures and methods."""
        assert hasattr(InfoMixin, "extract_from_obj")
        assert callable(InfoMixin.extract_from_obj)
        assert hasattr(InfoMixin, "create_info_dict")
        assert callable(InfoMixin.create_info_dict)
        assert hasattr(InfoMixin, "info")
        assert isinstance(InfoMixin.info, property)

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test InfoMixin initialization through inheritance."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, InfoMixin)
        for key, value in constructor_args.items():
            assert hasattr(instance, key)
            assert getattr(instance, key) == value

    @pytest.mark.smoke
    def test_info_property(self, valid_instances):
        """Test InfoMixin.info property."""
        instance, constructor_args = valid_instances
        result = instance.info
        assert isinstance(result, dict)
        assert "str" in result
        assert "type" in result
        assert "class" in result
        assert "module" in result
        assert "attributes" in result
        assert result["type"] == "TestClass"
        assert result["class"] == "TestClass"
        assert isinstance(result["attributes"], dict)
        for key, value in constructor_args.items():
            assert key in result["attributes"]
            assert result["attributes"][key] == value

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("obj_data", "expected_attributes"),
        [
            ({"name": "test", "value": 42}, {"name": "test", "value": 42}),
            ({"data": [1, 2, 3], "flag": True}, {"data": [1, 2, 3], "flag": True}),
            ({"nested": {"key": "value"}}, {"nested": {"key": "value"}}),
        ],
    )
    def test_create_info_dict(self, obj_data, expected_attributes):
        """Test InfoMixin.create_info_dict class method."""

        class SimpleObject:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        obj = SimpleObject(**obj_data)
        result = InfoMixin.create_info_dict(obj)

        assert isinstance(result, dict)
        assert "str" in result
        assert "type" in result
        assert "class" in result
        assert "module" in result
        assert "attributes" in result
        assert result["type"] == "SimpleObject"
        assert result["class"] == "SimpleObject"
        assert result["attributes"] == expected_attributes

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("obj_data", "expected_attributes"),
        [
            ({"name": "test", "value": 42}, {"name": "test", "value": 42}),
            ({"data": [1, 2, 3], "flag": True}, {"data": [1, 2, 3], "flag": True}),
        ],
    )
    def test_extract_from_obj_without_info(self, obj_data, expected_attributes):
        """Test InfoMixin.extract_from_obj with objects without info method."""

        class SimpleObject:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        obj = SimpleObject(**obj_data)
        result = InfoMixin.extract_from_obj(obj)

        assert isinstance(result, dict)
        assert "str" in result
        assert "type" in result
        assert "class" in result
        assert "module" in result
        assert "attributes" in result
        assert result["type"] == "SimpleObject"
        assert result["class"] == "SimpleObject"
        assert result["attributes"] == expected_attributes

    @pytest.mark.smoke
    def test_extract_from_obj_with_info_method(self):
        """Test InfoMixin.extract_from_obj with objects that have info method."""

        class ObjectWithInfoMethod:
            def info(self):
                return {"custom": "info_method", "type": "custom_type"}

        obj = ObjectWithInfoMethod()
        result = InfoMixin.extract_from_obj(obj)

        assert result == {"custom": "info_method", "type": "custom_type"}

    @pytest.mark.smoke
    def test_extract_from_obj_with_info_property(self):
        """Test InfoMixin.extract_from_obj with objects that have info property."""

        class ObjectWithInfoProperty:
            @property
            def info(self):
                return {"custom": "info_property", "type": "custom_type"}

        obj = ObjectWithInfoProperty()
        result = InfoMixin.extract_from_obj(obj)

        assert result == {"custom": "info_property", "type": "custom_type"}

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("obj_type", "obj_value"),
        [
            (str, "test_string"),
            (int, 42),
            (float, 3.14),
            (list, [1, 2, 3]),
            (dict, {"key": "value"}),
        ],
    )
    def test_extract_from_obj_builtin_types(self, obj_type, obj_value):
        """Test InfoMixin.extract_from_obj with built-in types."""
        result = InfoMixin.extract_from_obj(obj_value)

        assert isinstance(result, dict)
        assert "str" in result
        assert "type" in result
        assert result["type"] == obj_type.__name__
        assert result["str"] == str(obj_value)

    @pytest.mark.sanity
    def test_extract_from_obj_without_dict(self):
        """Test InfoMixin.extract_from_obj with objects without __dict__."""
        obj = 42
        result = InfoMixin.extract_from_obj(obj)

        assert isinstance(result, dict)
        assert "attributes" in result
        assert result["attributes"] == {}
        assert result["type"] == "int"
        assert result["str"] == "42"

    @pytest.mark.sanity
    def test_extract_from_obj_with_private_attributes(self):
        """Test InfoMixin.extract_from_obj filters private attributes."""

        class ObjectWithPrivate:
            def __init__(self):
                self.public_attr = "public"
                self._private_attr = "private"
                self.__very_private = "very_private"

        obj = ObjectWithPrivate()
        result = InfoMixin.extract_from_obj(obj)

        assert "public_attr" in result["attributes"]
        assert result["attributes"]["public_attr"] == "public"
        assert "_private_attr" not in result["attributes"]
        assert "__very_private" not in result["attributes"]

    @pytest.mark.sanity
    def test_extract_from_obj_complex_attributes(self):
        """Test InfoMixin.extract_from_obj with complex attribute types."""

        class ComplexObject:
            def __init__(self):
                self.simple_str = "test"
                self.simple_int = 42
                self.simple_list = [1, 2, 3]
                self.simple_dict = {"key": "value"}
                self.complex_object = object()

        obj = ComplexObject()
        result = InfoMixin.extract_from_obj(obj)

        attributes = result["attributes"]
        assert attributes["simple_str"] == "test"
        assert attributes["simple_int"] == 42
        assert attributes["simple_list"] == [1, 2, 3]
        assert attributes["simple_dict"] == {"key": "value"}
        assert isinstance(attributes["complex_object"], str)

    @pytest.mark.regression
    def test_create_info_dict_consistency(self, valid_instances):
        """Test InfoMixin.create_info_dict produces consistent results."""
        instance, _ = valid_instances

        result1 = InfoMixin.create_info_dict(instance)
        result2 = InfoMixin.create_info_dict(instance)

        assert result1 == result2
        assert result1 is not result2

    @pytest.mark.regression
    def test_info_property_uses_create_info_dict(self, valid_instances):
        """Test InfoMixin.info property uses create_info_dict method."""
        instance, _ = valid_instances

        info_result = instance.info
        create_result = InfoMixin.create_info_dict(instance)

        assert info_result == create_result
