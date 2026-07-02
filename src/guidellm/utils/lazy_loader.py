# ruff: noqa: PGH004
# ruff: noqa
"""
lazy_loader
===========

Makes it easy to load subpackages and functions on demand.

File uses code adapted from code with the following license:

BSD 3-Clause License

Copyright (c) 2022--2023, Scientific Python project
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import ast
import importlib
import importlib.util
import os
import sys
import threading
import types
import typing
import warnings

__version__ = "0.6rc0.dev0"
__all__ = ["ExtraAttr", "attach", "attach_extras", "attach_stub", "load"]


class ExtraAttr(typing.NamedTuple):
    """Descriptor for a lazily imported attribute in :func:`attach_extras`.

    :param path: Dotted module path to import from.
    :param name: Attribute name inside *path*. Only nessary if the attribute name
        differs from the exported name.
    """

    path: str
    name: str | None = None


threadlock = threading.Lock()


def attach(
    package_name,
    submodules=None,
    submod_attrs=None,
    lazy_submodules=False,
):
    """Attach lazily loaded submodules, functions, or other attributes.

    Typically, modules import submodules and attributes as follows::

      import mysubmodule
      import anothersubmodule

      from .foo import someattr

    The idea is to replace a package's `__getattr__`, `__dir__`, and
    `__all__`, such that all imports work exactly the way they would
    with normal imports, except that the import occurs upon first use.

    The typical way to call this function, replacing the above imports, is::

      __getattr__, __dir__, __all__ = lazy.attach(
          __name__, ["mysubmodule", "anothersubmodule"], {"foo": ["someattr"]}
      )

    Parameters
    ----------
    package_name : str
        Typically use ``__name__``.
    submodules : set
        List of submodules to attach.
    submod_attrs : dict
        Dictionary of submodule -> list of attributes / functions.
        These attributes are imported as they are used.
    lazy_submodules : bool
        Whether to lazily load submodules. If set to `True`, submodules are
        returned as lazy proxies. Note that attribute access from
        submod_attrs will trigger the import of the submodule.

    Returns
    -------
    __getattr__, __dir__, __all__

    """
    if submod_attrs is None:
        submod_attrs = {}

    if submodules is None:
        submodules = set()
    else:
        submodules = set(submodules)

    attr_to_modules = {
        attr: mod for mod, attrs in submod_attrs.items() for attr in attrs
    }

    __all__ = sorted(submodules | attr_to_modules.keys())

    def __getattr__(name):
        if name in submodules:
            submod_path = f"{package_name}.{name}"
            if lazy_submodules:
                return load(submod_path, suppress_warning=True)
            else:
                return importlib.import_module(submod_path)
        elif name in attr_to_modules:
            submod_path = f"{package_name}.{attr_to_modules[name]}"
            submod = importlib.import_module(submod_path)
            attr = getattr(submod, name)

            # If the attribute lives in a file (module) with the same
            # name as the attribute, ensure that the attribute and *not*
            # the module is accessible on the package.
            if name == attr_to_modules[name]:
                pkg = sys.modules[package_name]
                pkg.__dict__[name] = attr

            return attr
        else:
            raise AttributeError(f"No {package_name} attribute {name}")

    def __dir__():
        return __all__.copy()

    eager_import = os.environ.get("EAGER_IMPORT", "") not in ("0", "")
    if eager_import:
        for attr in set(attr_to_modules.keys()) | submodules:
            __getattr__(attr)

    return __getattr__, __dir__, __all__.copy()


def _attach_extras_attrs(module_name, attrs: dict[str, ExtraAttr], error_message):
    _attr_map = {}
    for export_name, spec in attrs.items():
        source_attr = spec.name if spec.name is not None else export_name
        _attr_map[export_name] = (spec.path, source_attr)

    _all = sorted(_attr_map.keys())

    def __getattr__(name):
        if name not in _attr_map:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        source_mod_name, source_attr = _attr_map[name]
        try:
            source_mod = importlib.import_module(source_mod_name)
        except ImportError as exc:
            raise AttributeError(error_message) from exc
        try:
            value = getattr(source_mod, source_attr)
        except AttributeError:
            try:
                value = importlib.import_module(f"{source_mod_name}.{source_attr}")
            except ImportError:
                raise AttributeError(error_message) from None
        if module_name in sys.modules:
            sys.modules[module_name].__dict__[name] = value
        return value

    def __dir__():
        return _all.copy()

    return __getattr__, __dir__, _all.copy()


def _attach_extras_package(module_name, package, error_message):
    _cached_pkg = {}

    def _get_package():
        if "mod" not in _cached_pkg:
            try:
                _cached_pkg["mod"] = importlib.import_module(package)
            except ImportError as exc:
                raise AttributeError(error_message) from exc
        return _cached_pkg["mod"]

    def __getattr__(name):
        pkg = _get_package()
        try:
            value = getattr(pkg, name)
        except AttributeError:
            raise AttributeError(
                f"module {module_name!r} has no attribute {name!r}"
            ) from None
        if module_name in sys.modules:
            sys.modules[module_name].__dict__[name] = value
        return value

    def __dir__():
        try:
            return list(dir(_get_package()))
        except AttributeError:
            return []

    return __getattr__, __dir__, []


def attach_extras(
    module_name,
    *,
    attrs=None,
    package=None,
    error_message="Required optional dependency is not installed",
):
    """Attach lazily loaded attributes from optional external packages.

    Designed for 'extras' modules that re-export symbols from optional
    dependencies.  The resulting module is always safe to import; errors
    are deferred until an attribute is actually accessed.

    Exactly one of ``attrs`` or ``package`` must be provided.

    :param module_name: Typically use ``__name__``.
    :param attrs: Map of exported names to :class:`ExtraAttr` descriptors.
        Each value specifies the *source* module and an optional *attr*
        (the attribute name inside *source*, when it differs from the
        dictionary key).
    :param package: Name of a package whose public attributes should be
        proxied wholesale.
    :param error_message: Human-readable message included in the
        ``AttributeError`` raised when the optional dependency is not
        installed.
    :returns: ``(__getattr__, __dir__, __all__)``
    """
    if (attrs is None) == (package is None):
        raise ValueError("attach_extras() requires exactly one of 'attrs' or 'package'")

    if attrs is not None:
        return _attach_extras_attrs(module_name, attrs, error_message)

    return _attach_extras_package(module_name, package, error_message)


class DelayedImportErrorModule(types.ModuleType):
    def __init__(self, frame_data, *args, message, **kwargs):
        self.__frame_data = frame_data
        self.__message = message
        super().__init__(*args, **kwargs)

    def __getattr__(self, x):
        fd = self.__frame_data
        raise ModuleNotFoundError(
            f"{self.__message}\n\n"
            "This error is lazily reported, having originally occurred in\n"
            f"  File {fd['filename']}, line {fd['lineno']}, in {fd['function']}\n\n"
            f"----> {''.join(fd['code_context'] or '').strip()}"
        )


def load(fullname, *, require=None, error_on_import=False, suppress_warning=False):
    """Return a lazily imported proxy for a module.

    We often see the following pattern::

      def myfunc():
          import numpy as np
          np.norm(...)
          ....

    Putting the import inside the function prevents, in this case,
    `numpy`, from being imported at function definition time.
    That saves time if `myfunc` ends up not being called.

    This `load` function returns a proxy module that, upon access, imports
    the actual module.  So the idiom equivalent to the above example is::

      np = lazy.load("numpy")

      def myfunc():
          np.norm(...)
          ....

    The initial import time is fast because the actual import is delayed
    until the first attribute is requested. The overall import time may
    decrease as well for users that don't make use of large portions
    of your library.

    Warning
    -------
    While lazily loading *sub*packages technically works, it causes the
    package (that contains the subpackage) to be eagerly loaded even
    if the package is already lazily loaded.
    So, you probably shouldn't use subpackages with this `load` feature.
    Instead you should encourage the package maintainers to use the
    `lazy_loader.attach` to make their subpackages load lazily.

    Parameters
    ----------
    fullname : str
        The full name of the module or submodule to import.  For example::

          sp = lazy.load("scipy")  # import scipy as sp

    require : str
        A dependency requirement as defined in PEP-508.  For example::

          "numpy >=1.24"

        If defined, the proxy module will raise an error if the installed
        version does not satisfy the requirement.

    error_on_import : bool
        Whether to postpone raising import errors until the module is accessed.
        If set to `True`, import errors are raised as soon as `load` is called.

    suppress_warning : bool
        Whether to prevent emitting a warning when loading subpackages.
        If set to `True`, no warning will occur.

    Returns
    -------
    pm : importlib.util._LazyModule
        Proxy module.  Can be used like any regularly imported module.
        Actual loading of the module occurs upon first attribute request.

    """
    with threadlock:
        module = sys.modules.get(fullname)
        have_module = module is not None

        # Most common, short-circuit
        if have_module and require is None:
            return module

        if not suppress_warning and "." in fullname:
            msg = (
                "subpackages can technically be lazily loaded, but it causes the "
                "package to be eagerly loaded even if it is already lazily loaded. "
                "So, you probably shouldn't use subpackages with this lazy feature."
            )
            warnings.warn(msg, RuntimeWarning)

        spec = None

        if not have_module:
            spec = importlib.util.find_spec(fullname)
            have_module = spec is not None

        if not have_module:
            not_found_message = f"No module named '{fullname}'"
        elif require is not None:
            try:
                have_module = _check_requirement(require)
            except ModuleNotFoundError as e:
                raise ValueError(
                    f"Found module '{fullname}' but cannot test "
                    "requirement '{require}'. "
                    "Requirements must match distribution name, not module name."
                ) from e

            not_found_message = f"No distribution can be found matching '{require}'"

        if not have_module:
            if error_on_import:
                raise ModuleNotFoundError(not_found_message)
            import inspect

            parent = inspect.stack()[1]
            frame_data = {
                "filename": parent.filename,
                "lineno": parent.lineno,
                "function": parent.function,
                "code_context": parent.code_context,
            }
            del parent
            return DelayedImportErrorModule(
                frame_data,
                "DelayedImportErrorModule",
                message=not_found_message,
            )

        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            sys.modules[fullname] = module

            loader = importlib.util.LazyLoader(spec.loader)
            loader.exec_module(module)

    return module


def _check_requirement(require: str) -> bool:
    """Verify that a package requirement is satisfied

    If the package is required, a ``ModuleNotFoundError`` is raised
    by ``importlib.metadata``.

    Parameters
    ----------
    require : str
        A dependency requirement as defined in PEP-508

    Returns
    -------
    satisfied : bool
        True if the installed version of the dependency matches
        the specified version, False otherwise.
    """
    import importlib.metadata

    import packaging.requirements

    req = packaging.requirements.Requirement(require)
    return req.specifier.contains(
        importlib.metadata.version(req.name),
        prereleases=True,
    )


class _StubVisitor(ast.NodeVisitor):
    """AST visitor to parse a stub file for submodules and submod_attrs."""

    def __init__(self):
        self._submodules = set()
        self._submod_attrs = {}

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.level != 1:
            raise ValueError(
                "Only within-module imports are supported (`from .* import`)"
            )
        if node.module:
            attrs: list = self._submod_attrs.setdefault(node.module, [])
            aliases = [alias.name for alias in node.names]
            if "*" in aliases:
                raise ValueError(
                    "lazy stub loader does not support star import "
                    f"`from {node.module} import *`"
                )
            attrs.extend(aliases)
        else:
            self._submodules.update(alias.name for alias in node.names)


def attach_stub(package_name: str, filename: str):
    """Attach lazily loaded submodules, functions from a type stub.

    This is a variant on ``attach`` that will parse a `.pyi` stub file to
    infer ``submodules`` and ``submod_attrs``. This allows static type checkers
    to find imports, while still providing lazy loading at runtime.

    Parameters
    ----------
    package_name : str
        Typically use ``__name__``.
    filename : str
        Path to `.py` file which has an adjacent `.pyi` file.
        Typically use ``__file__``.

    Returns
    -------
    __getattr__, __dir__, __all__
        The same output as ``attach``.

    Raises
    ------
    ValueError
        If a stub file is not found for `filename`, or if the stubfile is formmated
        incorrectly (e.g. if it contains an relative import from outside of the module)
    """
    stubfile = (
        filename if filename.endswith("i") else f"{os.path.splitext(filename)[0]}.pyi"
    )

    if not os.path.exists(stubfile):
        raise ValueError(f"Cannot load imports from non-existent stub {stubfile!r}")

    with open(stubfile) as f:
        stub_node = ast.parse(f.read())

    visitor = _StubVisitor()
    visitor.visit(stub_node)
    return attach(package_name, visitor._submodules, visitor._submod_attrs)
