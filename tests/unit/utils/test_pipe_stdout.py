from __future__ import annotations

import multiprocessing
import os
import sys
import time

import pytest
from rich.live import Live

from guidellm.utils.pipe_stdout import PipeReaderThread
from tests.unit.testing_utils import drain_logger, rich_console, rich_console_output


def _child_print(write_conn_stdout, write_conn_stderr):
    """Child target that redirects stdout/stderr to pipes and writes output."""
    os.dup2(write_conn_stdout.fileno(), sys.stdout.fileno())
    os.dup2(write_conn_stderr.fileno(), sys.stderr.fileno())
    write_conn_stdout.close()
    write_conn_stderr.close()

    print("hello from stdout", flush=True)  # noqa: T201
    print("hello from stderr", file=sys.stderr, flush=True)  # noqa: T201


def _pipe_raw_stderr_worker(stderr_conn) -> None:
    os.dup2(stderr_conn.fileno(), 2)
    stderr_conn.close()
    sys.stderr.write("pipe-raw-stderr-msg\n")
    sys.stderr.flush()


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

        drain_logger()
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


class TestPipeReaderThreadRichLive:
    """Rich Live integration tests for pipe reader output routing.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.sanity
    def test_rich_live_routes_raw_stderr(self):
        """
        Raw worker stderr through PipeReaderThread reaches Rich FileProxy.

        ## WRITTEN BY AI ##
        """
        ctx = multiprocessing.get_context("spawn")
        stdout_reader, stdout_writer = ctx.Pipe(duplex=False)
        stderr_reader, stderr_writer = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_pipe_raw_stderr_worker, args=(stderr_writer,))
        console = rich_console()
        reader = PipeReaderThread(stdout_reader, stderr_reader)
        reader.start()
        try:

            def run_worker() -> None:
                proc.start()
                stdout_writer.close()
                stderr_writer.close()
                proc.join(timeout=10)

            with Live("", console=console, redirect_stderr=True):
                run_worker()
                time.sleep(0.2)
            assert proc.exitcode == 0
            assert "pipe-raw-stderr-msg" in rich_console_output(console)
        finally:
            reader.stop()

    @pytest.mark.sanity
    def test_routes_raw_stderr_without_live(self, capsys):
        """
        Raw worker stderr still reaches the parent when Live is not active.

        ## WRITTEN BY AI ##
        """
        ctx = multiprocessing.get_context("spawn")
        stdout_reader, stdout_writer = ctx.Pipe(duplex=False)
        stderr_reader, stderr_writer = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_pipe_raw_stderr_worker, args=(stderr_writer,))
        reader = PipeReaderThread(stdout_reader, stderr_reader)
        reader.start()
        try:
            proc.start()
            stdout_writer.close()
            stderr_writer.close()
            proc.join(timeout=10)
            time.sleep(0.2)
            assert proc.exitcode == 0
            drain_logger()
            captured = capsys.readouterr()
            assert "pipe-raw-stderr-msg" in captured.err
        finally:
            reader.stop()
