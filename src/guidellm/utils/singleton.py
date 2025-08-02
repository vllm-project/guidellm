"""
Singleton pattern implementations for ensuring single instance classes.
The singleton pattern is useful for managing shared resources, configuration
objects, logging instances, and other scenarios where a single instance is
required across the application.

Classes:
    SingletonMixin: Basic singleton implementation using class variables.
    ThreadSafeSingletonMixin: Thread-safe singleton using locking mechanisms.
"""

import threading
from typing import ClassVar

__all__ = ["SingletonMixin", "ThreadSafeSingletonMixin"]


class SingletonMixin:
    """
    A mixin class that implements the Singleton design pattern.

    This class ensures that only one instance of any class that inherits from it
    can exist at a time. It uses a class variable to store the singleton instance
    and overrides the __new__ method to control instance creation.

    Example:
    ::
        from guidellm.utils import SingletonMixin

        class MyService(SingletonMixin):
            def __init__(self):
                super().__init__()
                self.value = "initialized"

        service1 = MyService()
        service2 = MyService()
        print(service1 is service2)  # True

    Note:
        Classes inheriting from SingletonMixin must call super().__init__()
        in their __init__ method to ensure proper initialization state management.
    """

    singleton_instance: ClassVar["SingletonMixin"] = None

    def __new__(cls, *args, **kwargs):
        """
        Create or return the singleton instance for the class.

        This method is called before __init__ and ensures that only one instance
        of the class exists. If an instance already exists, it returns that instance.
        Otherwise, it creates a new instance and marks it as uninitialized.

        :param args: Positional arguments passed to the constructor.
        :param kwargs: Keyword arguments passed to the constructor.
        :return: The singleton instance of the class.
        """
        if cls.singleton_instance is None:
            cls.singleton_instance = super().__new__(cls, *args, **kwargs)
            cls.singleton_instance.initialized = False
        return cls.singleton_instance

    def __init__(self):
        """
        Initialize the singleton instance.

        This method ensures that initialization only occurs once, even if
        __init__ is called multiple times on the same singleton instance.
        The initialization state is tracked using the 'initialized' attribute.

        Note:
            Subclasses should call super().__init__() to ensure proper
            initialization state management.
        """
        if self.initialized:
            return
        self.initialized = True


class ThreadSafeSingletonMixin(SingletonMixin):
    """
    A thread-safe version of the SingletonMixin.

    This mixin ensures that the singleton instance is created in a thread-safe manner,
    preventing multiple threads from creating separate instances.
    """

    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """
        Create or return the singleton instance for the class.

        This method is called before __init__ and ensures that only one instance
        of the class exists. If an instance already exists, it returns that instance.
        Otherwise, it creates a new instance and marks it as uninitialized.

        :param args: Positional arguments passed to the constructor.
        :param kwargs: Keyword arguments passed to the constructor.
        :return: The singleton instance of the class.
        """
        with cls._lock:
            return super().__new__(cls, *args, **kwargs)
