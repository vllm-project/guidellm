"""
Singleton pattern implementations for ensuring single instance classes.

Provides singleton mixins for creating classes that maintain a single instance
throughout the application lifecycle, with support for both basic and thread-safe
implementations. These mixins integrate with the scheduler and other system components
to ensure consistent state management and prevent duplicate resource allocation.
"""

from __future__ import annotations

import threading

__all__ = ["SingletonMixin", "ThreadSafeSingletonMixin"]


class SingletonMixin:
    """
    Basic singleton mixin ensuring single instance per class.

    Implements the singleton pattern using class variables to control instance
    creation. Subclasses must call super().__init__() for proper initialization
    state management. Suitable for single-threaded environments or when external
    synchronization is provided.

    Example:
    ::
        class ConfigManager(SingletonMixin):
            def __init__(self, config_path: str):
                super().__init__()
                if not self.initialized:
                    self.config = load_config(config_path)

        manager1 = ConfigManager("config.json")
        manager2 = ConfigManager("config.json")
        assert manager1 is manager2
    """

    _singleton_initialized: bool
    _init_lock: threading.Lock

    def __new__(cls, *args, **kwargs):  # noqa: ARG004
        """
        Create or return the singleton instance.

        :param args: Positional arguments passed to the constructor
        :param kwargs: Keyword arguments passed to the constructor
        :return: The singleton instance of the class
        """
        # Use class-specific attribute name to avoid inheritance issues
        attr_name = f"_singleton_instance_{cls.__name__}"

        if not hasattr(cls, attr_name) or getattr(cls, attr_name) is None:
            instance = super().__new__(cls)
            setattr(cls, attr_name, instance)
            instance._singleton_initialized = False
        return getattr(cls, attr_name)

    def __init__(self):
        """Initialize the singleton instance exactly once."""
        if hasattr(self, "_singleton_initialized") and self._singleton_initialized:
            return
        self._singleton_initialized = True

    @property
    def initialized(self):
        """Return True if the singleton has been initialized."""
        return getattr(self, "_singleton_initialized", False)


class ThreadSafeSingletonMixin(SingletonMixin):
    """
    Thread-safe singleton mixin with locking mechanisms.

    Extends SingletonMixin with thread safety using locks to prevent race
    conditions during instance creation in multi-threaded environments. Essential
    for scheduler components and other shared resources accessed concurrently.

    Example:
    ::
        class SchedulerResource(ThreadSafeSingletonMixin):
            def __init__(self):
                super().__init__()
                if not self.initialized:
                    self.resource_pool = initialize_resources()
    """

    def __new__(cls, *args, **kwargs):  # noqa: ARG004
        """
        Create or return the singleton instance with thread safety.

        :param args: Positional arguments passed to the constructor
        :param kwargs: Keyword arguments passed to the constructor
        :return: The singleton instance of the class
        """
        # Use class-specific lock and instance names to avoid inheritance issues
        lock_attr_name = f"_singleton_lock_{cls.__name__}"
        instance_attr_name = f"_singleton_instance_{cls.__name__}"

        with getattr(cls, lock_attr_name):
            instance_exists = (
                hasattr(cls, instance_attr_name)
                and getattr(cls, instance_attr_name) is not None
            )
            if not instance_exists:
                instance = super(SingletonMixin, cls).__new__(cls)
                setattr(cls, instance_attr_name, instance)
                instance._singleton_initialized = False
                instance._init_lock = threading.Lock()
            return getattr(cls, instance_attr_name)

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        lock_attr_name = f"_singleton_lock_{cls.__name__}"
        setattr(cls, lock_attr_name, threading.Lock())

    def __init__(self):
        """Initialize the singleton instance with thread-safe initialization."""
        with self._init_lock:
            if hasattr(self, "_singleton_initialized") and self._singleton_initialized:
                return
            self._singleton_initialized = True

    @property
    def thread_lock(self):
        """Return the thread lock for this singleton instance."""
        return getattr(self, "_init_lock", None)

    @classmethod
    def get_singleton_lock(cls):
        """Get the class-specific singleton creation lock."""
        lock_attr_name = f"_singleton_lock_{cls.__name__}"
        return getattr(cls, lock_attr_name, None)
