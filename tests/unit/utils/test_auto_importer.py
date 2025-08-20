"""
Unit tests for the auto_importer module.
"""

from __future__ import annotations

from unittest import mock

import pytest

from guidellm.utils import AutoImporterMixin


class TestAutoImporterMixin:
    """Test suite for AutoImporterMixin functionality."""

    @pytest.fixture(
        params=[
            {
                "auto_package": "test.package",
                "auto_ignore_modules": None,
                "modules": [
                    ("test.package.module1", False),
                    ("test.package.module2", False),
                ],
                "expected_imports": ["test.package.module1", "test.package.module2"],
            },
            {
                "auto_package": ("test.package1", "test.package2"),
                "auto_ignore_modules": None,
                "modules": [
                    ("test.package1.moduleA", False),
                    ("test.package2.moduleB", False),
                ],
                "expected_imports": ["test.package1.moduleA", "test.package2.moduleB"],
            },
            {
                "auto_package": "test.package",
                "auto_ignore_modules": ("test.package.module1",),
                "modules": [
                    ("test.package.module1", False),
                    ("test.package.module2", False),
                ],
                "expected_imports": ["test.package.module2"],
            },
        ],
        ids=["single_package", "multiple_packages", "ignored_modules"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for AutoImporterMixin subclasses."""
        config = request.param

        class TestClass(AutoImporterMixin):
            auto_package = config["auto_package"]
            auto_ignore_modules = config["auto_ignore_modules"]

        return TestClass, config

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test AutoImporterMixin class signatures and attributes."""
        assert hasattr(AutoImporterMixin, "auto_package")
        assert hasattr(AutoImporterMixin, "auto_ignore_modules")
        assert hasattr(AutoImporterMixin, "auto_imported_modules")
        assert hasattr(AutoImporterMixin, "auto_import_package_modules")
        assert callable(AutoImporterMixin.auto_import_package_modules)

        # Test default class variables
        assert AutoImporterMixin.auto_package is None
        assert AutoImporterMixin.auto_ignore_modules is None
        assert AutoImporterMixin.auto_imported_modules is None

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test AutoImporterMixin subclass initialization."""
        test_class, config = valid_instances
        assert issubclass(test_class, AutoImporterMixin)
        assert test_class.auto_package == config["auto_package"]
        assert test_class.auto_ignore_modules == config["auto_ignore_modules"]
        assert test_class.auto_imported_modules is None

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test AutoImporterMixin with missing auto_package."""

        class TestClass(AutoImporterMixin):
            pass

        with pytest.raises(ValueError, match="auto_package.*must be set"):
            TestClass.auto_import_package_modules()

    @pytest.mark.smoke
    @mock.patch("importlib.import_module")
    @mock.patch("pkgutil.walk_packages")
    def test_auto_import_package_modules(self, mock_walk, mock_import, valid_instances):
        """Test auto_import_package_modules core functionality."""
        test_class, config = valid_instances

        # Setup mocks based on config
        packages = {}
        modules = {}

        if isinstance(config["auto_package"], tuple):
            for pkg in config["auto_package"]:
                pkg_path = pkg.replace(".", "/")
                packages[pkg] = MockHelper.create_mock_package(pkg, pkg_path)
        else:
            pkg = config["auto_package"]
            packages[pkg] = MockHelper.create_mock_package(pkg, pkg.replace(".", "/"))

        for module_name, is_pkg in config["modules"]:
            if not is_pkg:
                modules[module_name] = MockHelper.create_mock_module(module_name)

        mock_import.side_effect = lambda name: {**packages, **modules}.get(
            name, mock.MagicMock()
        )

        def walk_side_effect(path, prefix):
            return [
                (None, module_name, is_pkg)
                for module_name, is_pkg in config["modules"]
                if module_name.startswith(prefix)
            ]

        mock_walk.side_effect = walk_side_effect

        # Execute
        test_class.auto_import_package_modules()

        # Verify
        assert test_class.auto_imported_modules == config["expected_imports"]

        # Verify package imports
        if isinstance(config["auto_package"], tuple):
            for pkg in config["auto_package"]:
                mock_import.assert_any_call(pkg)
        else:
            mock_import.assert_any_call(config["auto_package"])

        # Verify expected module imports
        for expected_module in config["expected_imports"]:
            mock_import.assert_any_call(expected_module)

    @pytest.mark.sanity
    @mock.patch("importlib.import_module")
    @mock.patch("pkgutil.walk_packages")
    def test_auto_import_package_modules_invalid(self, mock_walk, mock_import):
        """Test auto_import_package_modules with invalid configurations."""

        class TestClass(AutoImporterMixin):
            auto_package = "test.package"

        # Test import error handling
        mock_import.side_effect = ImportError("Module not found")

        with pytest.raises(ImportError):
            TestClass.auto_import_package_modules()

    @pytest.mark.sanity
    @mock.patch("importlib.import_module")
    @mock.patch("pkgutil.walk_packages")
    def test_skip_packages(self, mock_walk, mock_import):
        """Test that packages (is_pkg=True) are skipped."""

        class TestClass(AutoImporterMixin):
            auto_package = "test.package"

        # Setup mocks
        mock_package = MockHelper.create_mock_package("test.package", "test/package")
        mock_module = MockHelper.create_mock_module("test.package.module")

        mock_import.side_effect = lambda name: {
            "test.package": mock_package,
            "test.package.module": mock_module,
        }[name]

        mock_walk.return_value = [
            (None, "test.package.subpackage", True),
            (None, "test.package.module", False),
        ]

        # Execute
        TestClass.auto_import_package_modules()

        # Verify
        assert TestClass.auto_imported_modules == ["test.package.module"]
        mock_import.assert_any_call("test.package.module")
        # subpackage should not be imported
        with pytest.raises(AssertionError):
            mock_import.assert_any_call("test.package.subpackage")

    @pytest.mark.sanity
    @mock.patch("sys.modules", {"test.package.existing": mock.MagicMock()})
    @mock.patch("importlib.import_module")
    @mock.patch("pkgutil.walk_packages")
    def test_skip_already_imported_modules(self, mock_walk, mock_import):
        """Test that modules already in sys.modules are tracked but not re-imported."""

        class TestClass(AutoImporterMixin):
            auto_package = "test.package"

        # Setup mocks
        mock_package = MockHelper.create_mock_package("test.package", "test/package")
        mock_import.side_effect = lambda name: {
            "test.package": mock_package,
        }.get(name, mock.MagicMock())

        mock_walk.return_value = [
            (None, "test.package.existing", False),
        ]

        # Execute
        TestClass.auto_import_package_modules()

        # Verify
        assert TestClass.auto_imported_modules == ["test.package.existing"]
        mock_import.assert_called_once_with("test.package")
        with pytest.raises(AssertionError):
            mock_import.assert_any_call("test.package.existing")

    @pytest.mark.sanity
    @mock.patch("importlib.import_module")
    @mock.patch("pkgutil.walk_packages")
    def test_prevent_duplicate_module_imports(self, mock_walk, mock_import):
        """Test that modules already in auto_imported_modules are not re-imported."""

        class TestClass(AutoImporterMixin):
            auto_package = "test.package"

        # Setup mocks
        mock_package = MockHelper.create_mock_package("test.package", "test/package")
        mock_module = MockHelper.create_mock_module("test.package.module")

        mock_import.side_effect = lambda name: {
            "test.package": mock_package,
            "test.package.module": mock_module,
        }[name]

        mock_walk.return_value = [
            (None, "test.package.module", False),
            (None, "test.package.module", False),
        ]

        # Execute
        TestClass.auto_import_package_modules()

        # Verify
        assert TestClass.auto_imported_modules == ["test.package.module"]
        assert mock_import.call_count == 2  # Package + module (not duplicate)


class MockHelper:
    """Helper class to create consistent mock objects for testing."""

    @staticmethod
    def create_mock_package(name: str, path: str):
        """Create a mock package with required attributes."""
        package = mock.MagicMock()
        package.__name__ = name
        package.__path__ = [path]
        return package

    @staticmethod
    def create_mock_module(name: str):
        """Create a mock module with required attributes."""
        module = mock.MagicMock()
        module.__name__ = name
        return module
