# AGENTS.md

This file provides agents and LLMs with guidance on development practices within the GuideLLM repository.

> **NOTE TO AI**: This file is human maintained and **SHALL NOT** be edited by agents or any LLM.

> **NOTE TO HUMANS**: This file should be kept brief as it is loaded into every AI context.

## Development Commands

### Python Development

> **IMPORTANT**: For every command `uv run` must be prepended to the command.

> **IMPORTANT**: Review available `tox` envs and use them in-place of individual tools.

```bash
# Run all tests
uv run tox -e tests            # Runs pytest with correct environment

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

## Change Risk Assessment

Before submitting code, assess the likelihood and impact of regressions beyond the motivating use case. GuideLLM is a performance benchmarking tool: overhead introduced by the client can distort both the workload delivered to a server and the measurements reported. Match the change's scope and validation to its risks.

### Performance and Measurement Integrity

- Minimize overhead in request handling, streaming, scheduling, and metrics.
- Preserve metric definitions, aggregation, missing-data behavior, and timing.
- Validate performance and measurement accuracy with representative evidence, accounting for instrumentation overhead.

### Compatibility and Shared Behavior

- Check other consumers, supported workloads, and paths where the feature is unused.
- Preserve configuration contracts, reproducibility, persisted data, and replayability.
- Evaluate concurrency, shared state, cancellation, and shutdown together.

### Architecture, Complexity, and Scope

- Keep behavior with its owning component and variants behind shared interfaces.
- Isolate optional features from critical execution paths.
- Reuse established mechanisms; justify added abstractions and dependencies.
- Keep changes focused. Separate unrelated cleanup and independently useful features when doing so makes review and validation clearer.

### Validation and Operational Risk

- Fix root causes rather than compensating in callers, reports, or tests.
- Investigate failing assertions before changing expected behavior.
- Keep automated tests self-contained, CI-reliable, and aligned with repository conventions.
- Validate affected existing behavior as well as the new use case.
- Consider supported runtimes and effects on build, release, and merge automation.

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

### Submitting PRs

When submitting a PR always start with the pull request template in `.github/PULL_REQUEST_TEMPLATE.md`. The template includes comments that guide you through the proper submission process so make sure to read them. Sections marked **OPTIONAL** may be removed and **REQUIRED** sections must be filled out. Failure to follow the template guidance will result in a rejected PR and can result in being banned from contributing.

In the PR, explain material risks, the evidence addressing them, and remaining uncertainty. Distinguish demonstrated defects from plausible risks and design preferences. Sensitive code deserves proportionate validation, not automatic rejection or speculative redesign.

## Resources

- **GitHub**: https://github.com/vllm-project/guidellm
- **PyPI**: https://pypi.org/project/guidellm/
- **Container Registry**: https://github.com/vllm-project/guidellm/pkgs/container/guidellm
- **Documentation**: https://github.com/vllm-project/guidellm/tree/main/docs
- **Issues**: https://github.com/vllm-project/guidellm/issues
