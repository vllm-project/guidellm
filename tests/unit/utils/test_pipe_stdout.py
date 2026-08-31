from __future__ import annotations

import multiprocessing
import os
import sys

import pytest

from guidellm.utils.pipe_stdout import PipeReaderThread


def _child_print(write_conn_stdout, write_conn_stderr):
    """Child target that redirects stdout/stderr to pipes and writes output."""
    os.dup2(write_conn_stdout.fileno(), sys.stdout.fileno())
    os.dup2(write_conn_stderr.fileno(), sys.stderr.fileno())
    write_conn_stdout.close()
    write_conn_stderr.close()

    print("hello from stdout", flush=True)  # noqa: T201
    print("hello from stderr", file=sys.stderr, flush=True)  # noqa: T201


class TestPipeReaderThread:
    """Test suite for PipeReaderThread.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_captures_child_stdout_and_stderr(self, capsys):
        """
        Verify that stdout/stderr from a child process routes through the
        pipe reader to the main process sys.stdout/sys.stderr.

        ## WRITTEN BY AI ##
        """
        ctx = multiprocessing.get_context("spawn")

        stdout_reader, stdout_writer = ctx.Pipe(duplex=False)
        stderr_reader, stderr_writer = ctx.Pipe(duplex=False)

        reader = PipeReaderThread(stdout_reader, stderr_reader)
        reader.start()

        proc = ctx.Process(
            target=_child_print,
            args=(stdout_writer, stderr_writer),
        )
        proc.start()

        # Close write-ends in parent so reader gets EOF after child exits
        stdout_writer.close()
        stderr_writer.close()

        proc.join(timeout=10)
        assert proc.exitcode == 0

        # Give reader thread time to drain remaining data
        reader.stop(timeout=5.0)

        captured = capsys.readouterr()
        assert "hello from stdout" in captured.out
        assert "hello from stderr" in captured.err

    @pytest.mark.smoke
    def test_handles_child_crash(self):
        """
        Verify that the reader thread handles a child crash (write-end
        closed by OS) without hanging.

        ## WRITTEN BY AI ##
        """
        ctx = multiprocessing.get_context("spawn")

        stdout_reader, stdout_writer = ctx.Pipe(duplex=False)
        stderr_reader, stderr_writer = ctx.Pipe(duplex=False)

        reader = PipeReaderThread(stdout_reader, stderr_reader)
        reader.start()

        # Close write-ends immediately — simulates child crash before writing
        stdout_writer.close()
        stderr_writer.close()

        # Reader should see EOF and stop cleanly
        reader.stop(timeout=5.0)
        assert reader._thread is None

    @pytest.mark.smoke
    def test_stop_is_idempotent(self):
        """
        Verify that calling stop() multiple times does not raise.

        ## WRITTEN BY AI ##
        """
        ctx = multiprocessing.get_context("spawn")

        stdout_reader, stdout_writer = ctx.Pipe(duplex=False)
        stderr_reader, stderr_writer = ctx.Pipe(duplex=False)

        reader = PipeReaderThread(stdout_reader, stderr_reader)
        reader.start()

        stdout_writer.close()
        stderr_writer.close()

        reader.stop(timeout=5.0)
        reader.stop(timeout=5.0)
