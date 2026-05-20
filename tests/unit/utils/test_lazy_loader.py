# ruff: noqa: PGH004
# ruff: noqa
"""
Tests.Mocks for the lazy loading utilities from lazy_loader package.

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

import importlib
import os
import sys
import types
from unittest import mock

import pytest

import guidellm.utils.lazy_loader as lazy


@pytest.fixture
def clean_fake_pkg():
    yield
    sys.modules.pop("tests.mocks.fake_pkg.some_func", None)
    sys.modules.pop("tests.mocks.fake_pkg", None)
    sys.modules.pop("tests.mocks", None)


@pytest.mark.parametrize("attempt", [1, 2])
def test_cleanup_fixture(clean_fake_pkg, attempt):
    assert "tests.mocks.fake_pkg" not in sys.modules
    assert "tests.mocks.fake_pkg.some_func" not in sys.modules
    from tests.mocks import fake_pkg

    assert "tests.mocks.fake_pkg" in sys.modules
    assert "tests.mocks.fake_pkg.some_func" not in sys.modules
    assert isinstance(fake_pkg.some_func, types.FunctionType)
    assert "tests.mocks.fake_pkg.some_func" in sys.modules


def test_lazy_import_basics():
    math = lazy.load("math")
    anything_not_real = lazy.load("anything_not_real")

    # Now test that accessing attributes does what it should
    assert math.sin(math.pi) == pytest.approx(0, 1e-6)
    # poor-mans pytest.raises for testing errors on attribute access
    with pytest.raises(ModuleNotFoundError):
        anything_not_real.pi
    assert isinstance(anything_not_real, lazy.DelayedImportErrorModule)
    # see if it changes for second access
    with pytest.raises(ModuleNotFoundError):
        anything_not_real.pi


def test_lazy_import_subpackages():
    with pytest.warns(RuntimeWarning):
        hp = lazy.load("html.parser")
    assert "html" in sys.modules
    assert type(sys.modules["html"]) is type(pytest)
    assert isinstance(hp, lazy._SafeLazyModule)
    assert "html.parser" in sys.modules
    assert sys.modules["html.parser"] == hp


def test_lazy_import_impact_on_sys_modules():
    math = lazy.load("math")
    anything_not_real = lazy.load("anything_not_real")

    assert isinstance(math, types.ModuleType)
    assert "math" in sys.modules
    assert isinstance(anything_not_real, lazy.DelayedImportErrorModule)
    assert "anything_not_real" not in sys.modules

    # only do this if numpy is installed
    pytest.importorskip("numpy")
    np = lazy.load("numpy")
    assert isinstance(np, types.ModuleType)
    assert "numpy" in sys.modules

    np.pi  # trigger load of numpy

    assert isinstance(np, types.ModuleType)
    assert "numpy" in sys.modules


def test_lazy_import_nonbuiltins():
    np = lazy.load("numpy")
    sp = lazy.load("scipy")
    if not isinstance(np, lazy.DelayedImportErrorModule):
        assert np.sin(np.pi) == pytest.approx(0, 1e-6)
    if isinstance(sp, lazy.DelayedImportErrorModule):
        with pytest.raises(ModuleNotFoundError):
            sp.pi


@pytest.mark.parametrize("lazy_submodules", [False, True])
def test_lazy_attach(lazy_submodules):
    name = "mymod"
    submods = ["mysubmodule", "anothersubmodule"]
    myall = {"not_real_submod": ["some_var_or_func"]}

    locls = {
        "attach": lazy.attach,
        "name": name,
        "submods": submods,
        "myall": myall,
        "lazy_submods": lazy_submodules,
    }
    s = (
        "__getattr__, __lazy_dir__, __all__ = "
        "attach(name, submods, myall, lazy_submodules=lazy_submods)"
    )

    exec(s, {}, locls)
    expected = {
        "attach": lazy.attach,
        "name": name,
        "submods": submods,
        "myall": myall,
        "lazy_submods": lazy_submodules,
        "__getattr__": None,
        "__lazy_dir__": None,
        "__all__": None,
    }
    assert locls.keys() == expected.keys()
    for k, v in expected.items():
        if v is not None:
            assert locls[k] == v

    # Exercise __getattr__, though it will just error
    with pytest.raises(ImportError):
        locls["__getattr__"]("mysubmodule")

    # Attribute is supposed to be imported, error on submodule load
    with pytest.raises(ImportError):
        locls["__getattr__"]("some_var_or_func")

    # Attribute is unknown, raise AttributeError
    with pytest.raises(AttributeError):
        locls["__getattr__"]("unknown_attr")


def test_lazy_attach_noattrs():
    name = "mymod"
    submods = ["mysubmodule", "anothersubmodule"]
    _, _, all_ = lazy.attach(name, submods)

    assert all_ == sorted(submods)


@pytest.mark.parametrize("lazy_submodules", [False, True])
def test_lazy_attach_returns_copies(lazy_submodules):
    _get, _dir, _all = lazy.attach(
        __name__,
        ["my_submodule", "another_submodule"],
        {"foo": ["some_attr"]},
        lazy_submodules=lazy_submodules,
    )
    assert _dir() is not _dir()
    assert _dir() == _all
    assert _dir() is not _all

    expected = ["another_submodule", "my_submodule", "some_attr"]
    assert _dir() == expected
    assert _all == expected
    assert _dir() is not _all

    _dir().append("modify_returned_list")
    assert _dir() == expected
    assert _all == expected
    assert _dir() is not _all

    _all.append("modify_returned_all")
    assert _dir() == expected
    assert _all == [*expected, "modify_returned_all"]


@pytest.mark.parametrize("eager_import", [False, True])
def test_attach_same_module_and_attr_name(clean_fake_pkg, eager_import):
    env = {}
    if eager_import:
        env["EAGER_IMPORT"] = "1"

    with mock.patch.dict(os.environ, env):
        from tests.mocks import fake_pkg

        # Grab attribute twice, to ensure that importing it does not
        # override function by module
        assert isinstance(fake_pkg.some_func, types.FunctionType)
        assert isinstance(fake_pkg.some_func, types.FunctionType)

        # Ensure imports from submodule still work
        from tests.mocks.fake_pkg.some_func import some_func

        assert isinstance(some_func, types.FunctionType)


FAKE_STUB = """
from . import rank
from ._gaussian import gaussian
from .edges import sobel, scharr, prewitt, roberts
"""


def test_stub_loading(tmp_path):
    stub = tmp_path / "stub.pyi"
    stub.write_text(FAKE_STUB)
    _get, _dir, _all = lazy.attach_stub("my_module", str(stub))
    expect = {"gaussian", "sobel", "scharr", "prewitt", "roberts", "rank"}
    assert set(_dir()) == set(_all) == expect


def test_stub_loading_parity():
    from tests.mocks import fake_pkg

    from_stub = lazy.attach_stub(fake_pkg.__name__, fake_pkg.__file__)
    stub_getter, stub_dir, stub_all = from_stub
    assert stub_all == fake_pkg.__all__
    assert stub_dir() == fake_pkg.__lazy_dir__()
    assert stub_getter("some_func") == fake_pkg.some_func


def test_stub_loading_errors(tmp_path):
    stub = tmp_path / "stub.pyi"
    stub.write_text("from ..mod import func\n")

    with pytest.raises(ValueError, match="Only within-module imports are supported"):
        lazy.attach_stub("name", str(stub))

    with pytest.raises(ValueError, match="Cannot load imports from non-existent stub"):
        lazy.attach_stub("name", "not a file")

    stub2 = tmp_path / "stub2.pyi"
    stub2.write_text("from .mod import *\n")
    with pytest.raises(ValueError, match=r".*does not support star import"):
        lazy.attach_stub("name", str(stub2))


def test_require_kwarg():
    # Test with a module that definitely exists, behavior hinges on requirement
    with mock.patch("importlib.metadata.version") as version:
        version.return_value = "1.0.0"
        math = lazy.load("math", require="somepkg >= 2.0")
        assert isinstance(math, lazy.DelayedImportErrorModule)

        math = lazy.load("math", require="somepkg >= 1.0")
        assert math.sin(math.pi) == pytest.approx(0, 1e-6)

        # We can fail even after a successful import
        math = lazy.load("math", require="somepkg >= 2.0")
        assert isinstance(math, lazy.DelayedImportErrorModule)

        # Eager failure
        with pytest.raises(ModuleNotFoundError):
            lazy.load("math", require="somepkg >= 2.0", error_on_import=True)

    # When a module can be loaded but the version can't be checked,
    # raise a ValueError
    with pytest.raises(ValueError):
        lazy.load("math", require="somepkg >= 1.0")
