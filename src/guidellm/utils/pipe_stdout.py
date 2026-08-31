"""
Pipe-based stdout/stderr capture for multiprocessing worker processes.

Provides a reader thread that multiplexes output from worker processes back to
the main process's sys.stdout/sys.stderr, ensuring correct rendering when a
Rich Live TUI is active.
"""

import codecs
import os
import selectors
import sys
import threading
from multiprocessing.connection import Connection


class PipeReaderThread:
    """
    Daemon thread that reads from shared stdout/stderr pipe read-ends and
    routes decoded text to ``sys.stdout`` / ``sys.stderr`` in the main process.

    Writes go through ``sys.stdout.write()`` (resolved at call time) so that
    Rich's ``FileProxy`` — when a Live display is active — can intercept
    and render the output above the live panel with ANSI decoding.

    :param stdout_conn: Read-end connection for the shared stdout pipe.
    :param stderr_conn: Read-end connection for the shared stderr pipe.
    """

    def __init__(
        self,
        stdout_conn: Connection,
        stderr_conn: Connection,
    ) -> None:
        self._stdout_conn = stdout_conn
        self._stderr_conn = stderr_conn
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """
        Start the reader thread.
        """
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="pipe-reader",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """
        Signal the reader thread to stop and wait for it to finish.

        :param timeout: Maximum seconds to wait for the thread to join.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        sel = selectors.DefaultSelector()
        stdout_fd = self._stdout_conn.fileno()
        stderr_fd = self._stderr_conn.fileno()

        decoders = {
            stdout_fd: codecs.getincrementaldecoder("utf-8")("replace"),
            stderr_fd: codecs.getincrementaldecoder("utf-8")("replace"),
        }

        try:
            sel.register(stdout_fd, selectors.EVENT_READ, data="stdout")
            sel.register(stderr_fd, selectors.EVENT_READ, data="stderr")

            while sel.get_map() and not self._stop_event.is_set():
                for key, _ in sel.select(timeout=0.1):
                    fd = key.fileobj
                    chunk = os.read(fd, 4096)  # type: ignore[arg-type]
                    if not chunk:
                        text = decoders[fd].decode(b"", final=True)  # type: ignore[index]
                        if text:
                            target = sys.stdout if key.data == "stdout" else sys.stderr
                            target.write(text)
                            target.flush()
                        sel.unregister(fd)
                        continue

                    text = decoders[fd].decode(chunk)  # type: ignore[index]
                    if text:
                        target = sys.stdout if key.data == "stdout" else sys.stderr
                        target.write(text)
                        target.flush()
        finally:
            sel.close()
            self._stdout_conn.close()
            self._stderr_conn.close()
