"""Unit tests for suppress_worker_stdio().

## WRITTEN BY AI ##
"""

from __future__ import annotations

import os
import sys

import pytest

from guidellm.utils.terminal import suppress_worker_stdio


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX only")
class TestSuppressWorkerStdio:
    @pytest.mark.smoke
    def test_stdout_writes_are_silenced(self, capsys):
        """Writes to sys.stdout inside the context are suppressed. ## WRITTEN BY AI ##"""
        with suppress_worker_stdio():
            sys.stdout.write("should not appear\n")
        captured = capsys.readouterr()
        assert "should not appear" not in captured.out

    @pytest.mark.smoke
    def test_stderr_writes_are_silenced(self, capsys):
        """Writes to sys.stderr inside the context are suppressed. ## WRITTEN BY AI ##"""
        with suppress_worker_stdio():
            sys.stderr.write("should not appear\n")
        captured = capsys.readouterr()
        assert "should not appear" not in captured.err

    @pytest.mark.sanity
    def test_stdout_restored_after_context(self):
        """sys.stdout is restored to the original object after exit. ## WRITTEN BY AI ##"""
        original = sys.stdout
        with suppress_worker_stdio():
            pass
        assert sys.stdout is original

    @pytest.mark.sanity
    def test_stderr_restored_after_context(self):
        """sys.stderr is restored to the original object after exit. ## WRITTEN BY AI ##"""
        original = sys.stderr
        with suppress_worker_stdio():
            pass
        assert sys.stderr is original

    @pytest.mark.sanity
    def test_os_fd1_restored_after_context(self, tmp_path):
        """OS-level fd 1 points back to the original file after exit. ## WRITTEN BY AI ##"""
        probe = tmp_path / "probe.txt"
        saved = os.dup(1)
        try:
            with suppress_worker_stdio():
                pass
            # After the context, writing to fd 1 should reach the real stdout,
            # not /dev/null.  We verify by checking fd 1 is not /dev/null.
            fd1_path = os.readlink(f"/proc/self/fd/1")
            assert "null" not in fd1_path
        except (OSError, NotImplementedError):
            # /proc/self/fd is Linux-specific; skip on others
            pytest.skip("cannot inspect /proc/self/fd on this platform")
        finally:
            os.close(saved)

    @pytest.mark.sanity
    def test_no_fd_leak(self):
        """suppress_worker_stdio closes all internal fds on exit. ## WRITTEN BY AI ##"""
        before = set(os.listdir("/proc/self/fd")) if os.path.exists(
            "/proc/self/fd"
        ) else None
        if before is None:
            pytest.skip("cannot inspect /proc/self/fd on this platform")

        with suppress_worker_stdio():
            pass

        after = set(os.listdir("/proc/self/fd"))
        # Allow for the listdir fd itself; no net new fds should remain.
        assert len(after) <= len(before) + 1

    @pytest.mark.sanity
    def test_stdout_still_works_after_exception(self):
        """sys.stdout is restored even when the body raises. ## WRITTEN BY AI ##"""
        original = sys.stdout
        with pytest.raises(ValueError):
            with suppress_worker_stdio():
                raise ValueError("boom")
        assert sys.stdout is original
