"""
Unit tests for the registry module.
"""

from __future__ import annotations

from typing import TypeVar
from unittest import mock

import pytest

from guidellm.utils.registry import RegistryMixin, RegistryObjT


def test_registry_obj_type():
    """Test that RegistryObjT is configured correctly as a TypeVar."""
    assert isinstance(RegistryObjT, type(TypeVar("test")))
    assert RegistryObjT.__name__ == "RegistryObjT"
    assert RegistryObjT.__bound__ is not None  # bound to Any
    assert RegistryObjT.__constraints__ == ()


class TestRegistryMixin:
    """Test suite for RegistryMixin class."""

    @pytest.fixture(
        params=[
            {"registry_auto_discovery": False, "auto_package": None},
            {"registry_auto_discovery": True, "auto_package": "test.package"},
        ],
        ids=["manual_registry", "auto_discovery"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for RegistryMixin subclasses."""
        config = request.param

        class TestRegistryClass(RegistryMixin):
            registry_auto_discovery = config["registry_auto_discovery"]
            if config["auto_package"]:
                auto_package = config["auto_package"]

        return TestRegistryClass, config

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test RegistryMixin inheritance and exposed methods."""
        assert hasattr(RegistryMixin, "registry")
        assert hasattr(RegistryMixin, "registry_auto_discovery")
        assert hasattr(RegistryMixin, "registry_populated")
        assert hasattr(RegistryMixin, "register")
        assert hasattr(RegistryMixin, "register_decorator")
        assert hasattr(RegistryMixin, "auto_populate_registry")
        assert hasattr(RegistryMixin, "registered_objects")
        assert hasattr(RegistryMixin, "is_registered")
        assert hasattr(RegistryMixin, "get_registered_object")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test RegistryMixin initialization."""
        registry_class, config = valid_instances

        assert registry_class.registry is None
        assert (
            registry_class.registry_auto_discovery == config["registry_auto_discovery"]
        )
        assert registry_class.registry_populated is False

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test RegistryMixin with missing auto_package when auto_discovery enabled."""

        class TestRegistryClass(RegistryMixin):
            registry_auto_discovery = True

        with pytest.raises(ValueError, match="auto_package.*must be set"):
            TestRegistryClass.auto_import_package_modules()

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("name", "expected_key"),
        [
            ("custom_name", "custom_name"),
            (["name1", "name2"], ["name1", "name2"]),
            (None, None),  # Uses class name
        ],
    )
    def test_register(self, valid_instances, name, expected_key):
        """Test register method with various name configurations."""
        registry_class, _ = valid_instances

        if name is None:

            @registry_class.register()
            class TestClass:
                pass

            expected_key = "testclass"
        else:

            @registry_class.register(name)
            class TestClass:
                pass

        assert registry_class.registry is not None
        if isinstance(expected_key, list):
            for key in expected_key:
                assert key in registry_class.registry
                assert registry_class.registry[key] is TestClass
        else:
            assert expected_key in registry_class.registry
            assert registry_class.registry[expected_key] is TestClass

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        "invalid_name",
        [123, 42.5, True, {"key": "value"}],
    )
    def test_register_invalid(self, valid_instances, invalid_name):
        """Test register method with invalid name types."""
        registry_class, _ = valid_instances

        with pytest.raises(ValueError, match="name must be a string, list of strings"):
            registry_class.register(invalid_name)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("name", "expected_key"),
        [
            ("custom_name", "custom_name"),
            (["name1", "name2"], ["name1", "name2"]),
            (None, "testclass"),
        ],
    )
    def test_register_decorator(self, valid_instances, name, expected_key):
        """Test register_decorator method with various name configurations."""
        registry_class, _ = valid_instances

        class TestClass:
            pass

        registry_class.register_decorator(TestClass, name=name)

        assert registry_class.registry is not None
        if isinstance(expected_key, list):
            for key in expected_key:
                assert key in registry_class.registry
                assert registry_class.registry[key] is TestClass
        else:
            assert expected_key in registry_class.registry
            assert registry_class.registry[expected_key] is TestClass

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        "invalid_name",
        [123, 42.5, True, {"key": "value"}],
    )
    def test_register_decorator_invalid(self, valid_instances, invalid_name):
        """Test register_decorator with invalid name types."""
        registry_class, _ = valid_instances

        class TestClass:
            pass

        with pytest.raises(
            ValueError, match="name must be a string or an iterable of strings"
        ):
            registry_class.register_decorator(TestClass, name=invalid_name)

    @pytest.mark.smoke
    def test_auto_populate_registry(self):
        """Test auto_populate_registry method with valid configuration."""

        class TestAutoRegistry(RegistryMixin):
            registry_auto_discovery = True
            auto_package = "test.package"

        with mock.patch.object(
            TestAutoRegistry, "auto_import_package_modules"
        ) as mock_import:
            result = TestAutoRegistry.auto_populate_registry()
            assert result is True
            mock_import.assert_called_once()
            assert TestAutoRegistry.registry_populated is True

            # Second call should return False
            result = TestAutoRegistry.auto_populate_registry()
            assert result is False
            mock_import.assert_called_once()  # Should not be called again

    @pytest.mark.sanity
    def test_auto_populate_registry_invalid(self):
        """Test auto_populate_registry when auto-discovery is disabled."""

        class TestDisabledRegistry(RegistryMixin):
            registry_auto_discovery = False

        with pytest.raises(ValueError, match="registry_auto_discovery is set to False"):
            TestDisabledRegistry.auto_populate_registry()

    @pytest.mark.smoke
    def test_registered_objects(self, valid_instances):
        """Test registered_objects method with manual registration."""
        registry_class, config = valid_instances

        @registry_class.register("class1")
        class TestClass1:
            pass

        @registry_class.register("class2")
        class TestClass2:
            pass

        if config["registry_auto_discovery"]:
            with mock.patch.object(registry_class, "auto_import_package_modules"):
                objects = registry_class.registered_objects()
        else:
            objects = registry_class.registered_objects()

        assert isinstance(objects, tuple)
        assert len(objects) == 2
        assert TestClass1 in objects
        assert TestClass2 in objects

    @pytest.mark.sanity
    def test_registered_objects_invalid(self):
        """Test registered_objects when no objects are registered."""

        class TestRegistryClass(RegistryMixin):
            pass

        with pytest.raises(
            ValueError, match="must be called after registering objects"
        ):
            TestRegistryClass.registered_objects()

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("register_name", "check_name", "expected"),
        [
            ("test_name", "test_name", True),
            ("TestName", "testname", True),
            ("UPPERCASE", "uppercase", True),
            ("test_name", "nonexistent", False),
        ],
    )
    def test_is_registered(self, valid_instances, register_name, check_name, expected):
        """Test is_registered with various name combinations."""
        registry_class, _ = valid_instances

        @registry_class.register(register_name)
        class TestClass:
            pass

        result = registry_class.is_registered(check_name)
        assert result == expected

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("register_name", "lookup_name"),
        [
            ("test_name", "test_name"),
            ("TestName", "testname"),
            ("UPPERCASE", "uppercase"),
        ],
    )
    def test_get_registered_object(self, valid_instances, register_name, lookup_name):
        """Test get_registered_object with valid names."""
        registry_class, _ = valid_instances

        @registry_class.register(register_name)
        class TestClass:
            pass

        result = registry_class.get_registered_object(lookup_name)
        assert result is TestClass

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        "lookup_name",
        ["nonexistent", "wrong_name", "DIFFERENT_CASE"],
    )
    def test_get_registered_object_invalid(self, valid_instances, lookup_name):
        """Test get_registered_object with invalid names."""
        registry_class, _ = valid_instances

        @registry_class.register("valid_name")
        class TestClass:
            pass

        result = registry_class.get_registered_object(lookup_name)
        assert result is None

    @pytest.mark.regression
    def test_multiple_registries_isolation(self):
        """Test that different registry classes maintain separate registries."""

        class Registry1(RegistryMixin):
            pass

        class Registry2(RegistryMixin):
            pass

        @Registry1.register()
        class TestClass1:
            pass

        @Registry2.register()
        class TestClass2:
            pass

        assert Registry1.registry is not None
        assert Registry2.registry is not None
        assert Registry1.registry != Registry2.registry
        assert "testclass1" in Registry1.registry
        assert "testclass2" in Registry2.registry
        assert "testclass1" not in Registry2.registry
        assert "testclass2" not in Registry1.registry

    @pytest.mark.regression
    def test_inheritance_registry_sharing(self):
        """Test that inherited registry classes share the same registry."""

        class BaseRegistry(RegistryMixin):
            pass

        class ChildRegistry(BaseRegistry):
            pass

        @BaseRegistry.register()
        class BaseClass:
            pass

        @ChildRegistry.register()
        class ChildClass:
            pass

        # Child classes share the same registry as their parent
        assert BaseRegistry.registry is ChildRegistry.registry

        # Both classes can see all registered objects
        base_objects = BaseRegistry.registered_objects()
        child_objects = ChildRegistry.registered_objects()

        assert len(base_objects) == 2
        assert len(child_objects) == 2
        assert base_objects == child_objects
        assert BaseClass in base_objects
        assert ChildClass in base_objects

    @pytest.mark.smoke
    def test_auto_discovery_initialization(self):
        """Test initialization of auto-discovery enabled registry."""

        class TestAutoRegistry(RegistryMixin):
            registry_auto_discovery = True
            auto_package = "test_package.modules"

        assert TestAutoRegistry.registry is None
        assert TestAutoRegistry.registry_populated is False
        assert TestAutoRegistry.auto_package == "test_package.modules"
        assert TestAutoRegistry.registry_auto_discovery is True

    @pytest.mark.smoke
    def test_auto_discovery_registered_objects(self):
        """Test automatic population during registered_objects call."""

        class TestAutoRegistry(RegistryMixin):
            registry_auto_discovery = True
            auto_package = "test_package.modules"

        with mock.patch.object(
            TestAutoRegistry, "auto_populate_registry"
        ) as mock_populate:
            TestAutoRegistry.registry = {"class1": "obj1", "class2": "obj2"}
            objects = TestAutoRegistry.registered_objects()
            mock_populate.assert_called_once()
            assert objects == ("obj1", "obj2")

    @pytest.mark.sanity
    def test_register_duplicate_registration(self, valid_instances):
        """Test register method with duplicate names."""
        registry_class, _ = valid_instances

        @registry_class.register("duplicate_name")
        class TestClass1:
            pass

        with pytest.raises(ValueError, match="already registered"):

            @registry_class.register("duplicate_name")
            class TestClass2:
                pass

    @pytest.mark.sanity
    def test_register_decorator_duplicate_registration(self, valid_instances):
        """Test register_decorator with duplicate names."""
        registry_class, _ = valid_instances

        class TestClass1:
            pass

        class TestClass2:
            pass

        registry_class.register_decorator(TestClass1, name="duplicate_name")
        with pytest.raises(ValueError, match="already registered"):
            registry_class.register_decorator(TestClass2, name="duplicate_name")

    @pytest.mark.sanity
    def test_register_decorator_invalid_list_element(self, valid_instances):
        """Test register_decorator with invalid elements in name list."""
        registry_class, _ = valid_instances

        class TestClass:
            pass

        with pytest.raises(
            ValueError, match="name must be a string or a list of strings"
        ):
            registry_class.register_decorator(TestClass, name=["valid", 123])

    @pytest.mark.sanity
    def test_register_decorator_invalid_object(self, valid_instances):
        """Test register_decorator with object lacking __name__ attribute."""
        registry_class, _ = valid_instances

        with pytest.raises(AttributeError):
            registry_class.register_decorator("not_a_class")

    @pytest.mark.smoke
    def test_is_registered_empty_registry(self, valid_instances):
        """Test is_registered with empty registry."""
        registry_class, _ = valid_instances

        result = registry_class.is_registered("any_name")
        assert result is False

    @pytest.mark.smoke
    def test_get_registered_object_empty_registry(self, valid_instances):
        """Test get_registered_object with empty registry."""
        registry_class, _ = valid_instances

        result = registry_class.get_registered_object("any_name")
        assert result is None

    @pytest.mark.regression
    def test_auto_registry_integration(self):
        """Test complete auto-discovery workflow with mocked imports."""

        class TestAutoRegistry(RegistryMixin):
            registry_auto_discovery = True
            auto_package = "test_package.modules"

        with (
            mock.patch("pkgutil.walk_packages") as walk_mock,
            mock.patch("importlib.import_module") as import_mock,
        ):
            # Setup mock package
            package_mock = mock.MagicMock()
            package_mock.__path__ = ["test_package/modules"]
            package_mock.__name__ = "test_package.modules"

            # Setup mock module with test class
            module_mock = mock.MagicMock()
            module_mock.__name__ = "test_package.modules.module1"

            class Module1Class:
                pass

            TestAutoRegistry.register_decorator(Module1Class, "Module1Class")

            # Setup import behavior
            import_mock.side_effect = lambda name: (
                package_mock
                if name == "test_package.modules"
                else module_mock
                if name == "test_package.modules.module1"
                else (_ for _ in ()).throw(ImportError(f"No module named {name}"))
            )

            # Setup package walking behavior
            walk_mock.side_effect = lambda path, prefix: (
                [(None, "test_package.modules.module1", False)]
                if prefix == "test_package.modules."
                else (_ for _ in ()).throw(ValueError(f"Unknown package: {prefix}"))
            )

            objects = TestAutoRegistry.registered_objects()
            assert len(objects) == 1
            assert TestAutoRegistry.registry_populated is True
            assert TestAutoRegistry.registry is not None
            assert "module1class" in TestAutoRegistry.registry

        class TestAutoRegistry(RegistryMixin):
            registry_auto_discovery = True
            auto_package = "test_package.modules"

        with (
            mock.patch("pkgutil.walk_packages") as mock_walk,
            mock.patch("importlib.import_module") as mock_import,
        ):
            mock_package = mock.MagicMock()
            mock_package.__path__ = ["test_package/modules"]
            mock_package.__name__ = "test_package.modules"

            def import_module(name: str):
                if name == "test_package.modules":
                    return mock_package
                elif name == "test_package.modules.module1":
                    module = mock.MagicMock()
                    module.__name__ = "test_package.modules.module1"

                    class Module1Class:
                        pass

                    TestAutoRegistry.register_decorator(Module1Class, "Module1Class")
                    return module
                else:
                    raise ImportError(f"No module named {name}")

            def walk_packages(package_path, package_name):
                if package_name == "test_package.modules.":
                    return [(None, "test_package.modules.module1", False)]
                else:
                    raise ValueError(f"Unknown package: {package_name}")

            mock_walk.side_effect = walk_packages
            mock_import.side_effect = import_module

            objects = TestAutoRegistry.registered_objects()
            assert len(objects) == 1
            assert TestAutoRegistry.registry_populated is True
            assert TestAutoRegistry.registry is not None
