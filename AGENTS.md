# AGENTS.md

This file provides agents and LLMs with guidance on development practices within the GuideLLM repository.

> **NOTE TO AI**: This file is human maintained and **SHALL NOT** be edited by agents or any LLM.

> **NOTE TO HUMANS**: This file should be kept brief as it is loaded into every AI context.

## Development Commands

### Python Development

> **IMPORTANT**: `tox` should be used before attempting to run individual tools.

> **IMPORTANT**: For every command `uv run` must be prepended to the command.

```bash
# Run all tests
uv run tox -e tests

# Run specific test suites
uv run tox -e test-unit        # Unit tests only
uv run tox -e test-integration # Integration tests only
uv run tox -e test-e2e         # End-to-end tests only

# Code quality and linting
uv run tox -e lint-check       # Check code quality (ruff, mdformat)
uv run tox -e lint-fix         # Fix style issues automatically
uv run tox -e type-check       # Type checking with mypy

# Update dependency locks
uv run tox -e lock

# Advanced pytest usage
uv run tox -e tests -- tests/unit/benchmark  # Run specific test directory or file
uv run tox -e tests -- -m smoke              # Run tests with specific marker
```

## Quality Standards

### Test Requirements

- **IMPORTANT**: Every test function written by AI must have `## WRITTEN BY AI ##` at the end of its docstring.
- Use appropriate markers (`smoke`, `sanity`, `regression`)
- Tests should be placed in files matching the name and path of the file under tests. E.g. `src/guidellm/benchmark/schemas/generative/entrypoints.py` -> `tests/unit/benchmark/schemas/generative/test_entrypoints.py`.

### Quality Requirements

- All Python code must pass linting and formatting
- Python code in `src/` must pass type checking
- All tests must pass before committing
- Markdown files must be properly formatted

### Style Requirements

- Public functions in `src/` code must use the reStructuredText docstring format
- All imports **SHALL** be done at the top of the file
- **DO NOT** use `getattr` or `setattr` as it hides incorrect usage of types

### Design Requirements

- Only touch sections of code that need to be changed for the given task
- When handling variant-specific logic, encapsulate it in methods on registry class implementations rather than adding if/else branches to generic code paths
- Class implementations must fully encapsulate their unique logic and that logic must not leak into caller code paths.

## Common Tasks

### Running Benchmarks

Running benchmarks requires an active model server. Here are some example commands:

```bash
# Quick sweep benchmark
uv run guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --constraint kind=max_requests,count=1000

# Production-like benchmark with specific dataset
uv run guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent \
  --data '{"kind":"huggingface","source":"openai/gsm8k","load_kwargs":{"name":"main","split":"test"}}' \
  --constraint kind=max_duration,seconds=300 \
  --override profile.streams 10,20
```

## Resources

- **GitHub**: https://github.com/vllm-project/guidellm
- **PyPI**: https://pypi.org/project/guidellm/
- **Container Registry**: https://github.com/vllm-project/guidellm/pkgs/container/guidellm
- **Documentation**: https://github.com/vllm-project/guidellm/tree/main/docs
- **Issues**: https://github.com/vllm-project/guidellm/issues
