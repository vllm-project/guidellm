from __future__ import annotations

import threading
import time

import pytest

from guidellm.utils.singleton import SingletonMixin, ThreadSafeSingletonMixin


class TestSingletonMixin:
    """Test suite for SingletonMixin class."""

    @pytest.fixture(
        params=[
            {"init_value": "test_value"},
            {"init_value": "another_value"},
        ],
        ids=["basic_singleton", "different_value"],
    )
    def valid_instances(self, request):
        """Provide parameterized test configurations for singleton testing."""
        config = request.param

        class TestSingleton(SingletonMixin):
            def __init__(self):
                # Check if we need to initialize before calling super().__init__()
                should_initialize = not getattr(self, "_singleton_initialized", False)
                super().__init__()
                if should_initialize:
                    self.value = config["init_value"]

        return TestSingleton, config

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test SingletonMixin inheritance and exposed attributes."""
        assert hasattr(SingletonMixin, "__new__")
        assert hasattr(SingletonMixin, "__init__")
        assert hasattr(SingletonMixin, "initialized")
        assert isinstance(SingletonMixin.initialized, property)

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test SingletonMixin initialization."""
        singleton_class, config = valid_instances

        # Create first instance
        instance1 = singleton_class()

        assert isinstance(instance1, singleton_class)
        assert instance1.initialized is True
        assert hasattr(instance1, "value")
        assert instance1.value == config["init_value"]

        # Check that the class has the singleton instance stored
        instance_attr = f"_singleton_instance_{singleton_class.__name__}"
        assert hasattr(singleton_class, instance_attr)
        assert getattr(singleton_class, instance_attr) is instance1

    @pytest.mark.smoke
    def test_singleton_behavior(self, valid_instances):
        """Test that multiple instantiations return the same instance."""
        singleton_class, config = valid_instances

        # Create multiple instances
        instance1 = singleton_class()
        instance2 = singleton_class()
        instance3 = singleton_class()

        # All should be the same instance
        assert instance1 is instance2
        assert instance2 is instance3
        assert instance1 is instance3

        # Value should remain from first initialization
        assert hasattr(instance1, "value")
        assert instance1.value == config["init_value"]
        assert instance2.value == config["init_value"]
        assert instance3.value == config["init_value"]

    @pytest.mark.sanity
    def test_initialization_called_once(self, valid_instances):
        """Test that __init__ is only called once despite multiple instantiations."""
        singleton_class, config = valid_instances

        class TestSingletonWithCounter(SingletonMixin):
            init_count = 0

            def __init__(self):
                should_initialize = not getattr(self, "_singleton_initialized", False)
                super().__init__()
                if should_initialize:
                    TestSingletonWithCounter.init_count += 1
                    self.value = config["init_value"]

        # Create multiple instances
        instance1 = TestSingletonWithCounter()
        instance2 = TestSingletonWithCounter()

        assert TestSingletonWithCounter.init_count == 1
        assert instance1 is instance2
        assert hasattr(instance1, "value")
        assert instance1.value == config["init_value"]

    @pytest.mark.regression
    def test_multiple_singleton_classes_isolation(self):
        """Test that different singleton classes maintain separate instances."""

        class Singleton1(SingletonMixin):
            def __init__(self):
                should_initialize = not getattr(self, "_singleton_initialized", False)
                super().__init__()
                if should_initialize:
                    self.value = "value1"

        class Singleton2(SingletonMixin):
            def __init__(self):
                should_initialize = not getattr(self, "_singleton_initialized", False)
                super().__init__()
                if should_initialize:
                    self.value = "value2"

        instance1a = Singleton1()
        instance2a = Singleton2()
        instance1b = Singleton1()
        instance2b = Singleton2()

        # Each class has its own singleton instance
        assert instance1a is instance1b
        assert instance2a is instance2b
        assert instance1a is not instance2a

        # Each maintains its own value
        assert hasattr(instance1a, "value")
        assert hasattr(instance2a, "value")
        assert instance1a.value == "value1"
        assert instance2a.value == "value2"

    @pytest.mark.regression
    def test_inheritance_singleton_sharing(self):
        """Test that inherited singleton classes share the same singleton_instance."""

        class BaseSingleton(SingletonMixin):
            def __init__(self):
                should_initialize = not getattr(self, "_singleton_initialized", False)
                super().__init__()
                if should_initialize:
                    self.value = "base_value"

        class ChildSingleton(BaseSingleton):
            def __init__(self):
                should_initialize = not getattr(self, "_singleton_initialized", False)
                super().__init__()
                if should_initialize:
                    self.extra = "extra_value"

        # Child classes now have separate singleton instances
        base_instance = BaseSingleton()
        child_instance = ChildSingleton()

        # They should be different instances now (fixed inheritance behavior)
        assert base_instance is not child_instance
        assert hasattr(base_instance, "value")
        assert base_instance.value == "base_value"
        assert hasattr(child_instance, "value")
        assert child_instance.value == "base_value"
        assert hasattr(child_instance, "extra")
        assert child_instance.extra == "extra_value"

    @pytest.mark.sanity
    def test_without_super_init_call(self):
        """Test singleton behavior when subclass doesn't call super().__init__()."""

        class BadSingleton(SingletonMixin):
            def __init__(self):
                # Not calling super().__init__()
                self.value = "bad_value"

        instance1 = BadSingleton()
        instance2 = BadSingleton()

        assert instance1 is instance2
        assert hasattr(instance1, "initialized")
        assert instance1.initialized is False


class TestThreadSafeSingletonMixin:
    """Test suite for ThreadSafeSingletonMixin class."""

    @pytest.fixture(
        params=[
            {"init_value": "thread_safe_value"},
            {"init_value": "concurrent_value"},
        ],
        ids=["basic_thread_safe", "concurrent_test"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for ThreadSafeSingletonMixin subclasses."""
        config = request.param

        class TestThreadSafeSingleton(ThreadSafeSingletonMixin):
            def __init__(self):
                should_initialize = not getattr(self, "_singleton_initialized", False)
                super().__init__()
                if should_initialize:
                    self.value = config["init_value"]

        return TestThreadSafeSingleton, config

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test ThreadSafeSingletonMixin inheritance and exposed attributes."""
        assert issubclass(ThreadSafeSingletonMixin, SingletonMixin)
        assert hasattr(ThreadSafeSingletonMixin, "get_singleton_lock")
        assert hasattr(ThreadSafeSingletonMixin, "__new__")
        assert hasattr(ThreadSafeSingletonMixin, "__init__")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test ThreadSafeSingletonMixin initialization."""
        singleton_class, config = valid_instances

        instance = singleton_class()

        assert isinstance(instance, singleton_class)
        assert instance.initialized is True
        assert hasattr(instance, "value")
        assert instance.value == config["init_value"]
        assert hasattr(instance, "thread_lock")
        lock_type = type(threading.Lock())
        assert isinstance(instance.thread_lock, lock_type)

    @pytest.mark.smoke
    def test_singleton_behavior(self, valid_instances):
        """Test multiple instantiations return same instance with thread safety."""
        singleton_class, config = valid_instances

        instance1 = singleton_class()
        instance2 = singleton_class()

        assert instance1 is instance2
        assert hasattr(instance1, "value")
        assert instance1.value == config["init_value"]
        assert hasattr(instance1, "thread_lock")

    @pytest.mark.regression
    def test_thread_safety_concurrent_creation(self, valid_instances):
        """Test thread safety during concurrent instance creation."""
        singleton_class, config = valid_instances

        instances = []
        exceptions = []
        creation_count = 0
        lock = threading.Lock()

        class ThreadSafeTestSingleton(ThreadSafeSingletonMixin):
            def __init__(self):
                should_initialize = not getattr(self, "_singleton_initialized", False)
                super().__init__()
                if should_initialize:
                    nonlocal creation_count
                    with lock:
                        creation_count += 1

                    time.sleep(0.01)
                    self.value = config["init_value"]

        def create_instance():
            try:
                instance = ThreadSafeTestSingleton()
                instances.append(instance)
            except (TypeError, ValueError, AttributeError) as exc:
                exceptions.append(exc)

        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_instance)
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"

        assert len(instances) == 10
        for instance in instances:
            assert instance is instances[0]

        assert creation_count == 1
        assert all(instance.value == config["init_value"] for instance in instances)

    @pytest.mark.sanity
    def test_thread_lock_creation(self, valid_instances):
        """Test that thread_lock is created during initialization."""
        singleton_class, config = valid_instances

        instance1 = singleton_class()
        instance2 = singleton_class()

        assert hasattr(instance1, "thread_lock")
        lock_type = type(threading.Lock())
        assert isinstance(instance1.thread_lock, lock_type)
        assert instance1.thread_lock is instance2.thread_lock

    @pytest.mark.regression
    def test_multiple_thread_safe_classes_isolation(self):
        """Test thread-safe singleton classes behavior with separate locks."""

        class ThreadSafeSingleton1(ThreadSafeSingletonMixin):
            def __init__(self):
                should_initialize = not getattr(self, "_singleton_initialized", False)
                super().__init__()
                if should_initialize:
                    self.value = "value1"

        class ThreadSafeSingleton2(ThreadSafeSingletonMixin):
            def __init__(self):
                should_initialize = not getattr(self, "_singleton_initialized", False)
                super().__init__()
                if should_initialize:
                    self.value = "value2"

        instance1 = ThreadSafeSingleton1()
        instance2 = ThreadSafeSingleton2()

        lock1 = ThreadSafeSingleton1.get_singleton_lock()
        lock2 = ThreadSafeSingleton2.get_singleton_lock()

        assert lock1 is not None
        assert lock2 is not None
        assert lock1 is not lock2

        assert instance1 is not instance2
        assert hasattr(instance1, "value")
        assert hasattr(instance2, "value")
        assert instance1.value == "value1"
        assert instance2.value == "value2"

    @pytest.mark.sanity
    def test_inheritance_with_thread_safety(self):
        """Test inheritance behavior with thread-safe singletons."""

        class BaseThreadSafeSingleton(ThreadSafeSingletonMixin):
            def __init__(self):
                should_initialize = not getattr(self, "_singleton_initialized", False)
                super().__init__()
                if should_initialize:
                    self.value = "base_value"

        class ChildThreadSafeSingleton(BaseThreadSafeSingleton):
            def __init__(self):
                super().__init__()

        base_instance = BaseThreadSafeSingleton()
        child_instance = ChildThreadSafeSingleton()

        base_lock = BaseThreadSafeSingleton.get_singleton_lock()
        child_lock = ChildThreadSafeSingleton.get_singleton_lock()

        assert base_lock is not None
        assert child_lock is not None
        assert base_lock is not child_lock

        assert base_instance is not child_instance
        assert hasattr(base_instance, "value")
        assert base_instance.value == "base_value"
        assert hasattr(base_instance, "thread_lock")
