"""Type stubs for the guidellm._rust extension built by the connector crate."""

def hello_world() -> str:
    """Return a greeting from the Rust core.

    :returns: The GuideLLM greeting string.
    """

def version_info() -> dict[str, str]:
    """Return git metadata from the Rust build.

    :returns: Build metadata, with "unknown" for values unavailable at build time.
    """
