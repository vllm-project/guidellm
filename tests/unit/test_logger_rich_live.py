import pytest
from rich.live import Live

from guidellm import configure_logger, logger
from guidellm.settings import LoggingSettings


@pytest.mark.sanity
def test_logger_with_rich_live_redirect_stderr(capsys):
    """
    Console logs resolve sys.stderr at write time and work with Rich Live.

    ## WRITTEN BY AI ##
    """
    configure_logger(config=LoggingSettings(console_log_level="INFO"))

    with Live("", redirect_stderr=True):
        logger.info("Live test message")

    logger.complete()
    captured = capsys.readouterr()
    assert "Live test message" in captured.err
