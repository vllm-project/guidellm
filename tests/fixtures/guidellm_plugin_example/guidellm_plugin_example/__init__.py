"""
Example external GuideLLM plugin used in tests.

The package mirrors GuideLLM's layout: ``guidellm.schemas`` loads
``guidellm_plugin_example.schemas`` and ``guidellm.backends`` loads
``guidellm_plugin_example.backends``. This module stays empty so that finding
those submodules does not import anything else.
"""
