# Contributing to GuideLLM

Thank you for considering contributing to GuideLLM! We welcome contributions from the community to help improve and grow this project. This document outlines the process and guidelines for contributing.

## How Can You Contribute?

There are many ways to contribute to GuideLLM:

- **Reporting Bugs**: If you encounter a bug, please let us know by creating an issue.
- **Suggesting Features**: Have an idea for a new feature? Open an issue to discuss it.
- **Improving Documentation**: Help us improve our documentation by submitting pull requests.
- **Writing Code**: Contribute code to fix bugs, add features, or improve performance.
- **Reviewing Pull Requests**: Provide feedback on open pull requests to help maintain code quality.

## Getting Started

### Prerequisites

Before contributing, ensure you have the following installed:

- Python 3.10 or higher
- pip (Python package manager)
- Tox
- Git

### Setting Up the Repository

You can either clone the repository directly or fork it if you plan to contribute changes back:

#### Option 1: Cloning the Repository

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/vllm-project/guidellm.git
   cd guidellm
   ```

#### Option 2: Forking the Repository

1. Fork the repository by clicking the "Fork" button on the repository's GitHub page.

2. Clone your forked repository to your local machine:

   ```bash
   git clone https://github.com/<your-username>/guidellm.git
   cd guidellm
   ```

For detailed instructions on setting up your development environment, please refer to the [DEVELOPING.md](https://github.com/vllm-project/guidellm/blob/main/DEVELOPING.md) file. It includes step-by-step guidance on:

- Installing dependencies
- Running tests
- Using Tox for various tasks

## Code Style and Guidelines

We follow strict coding standards to ensure code quality and maintainability. Please adhere to the following guidelines:

- **Code Style**: Use [Black](https://black.readthedocs.io/en/stable/) for code formatting and [Ruff](https://github.com/charliermarsh/ruff) for linting.
- **Type Checking**: Use [Mypy](http://mypy-lang.org/) for type checking.
- **Testing**: Write unit tests for new features and bug fixes. Use [pytest](https://docs.pytest.org/) for testing.
- **Documentation**: Update documentation for any changes to the codebase.

To check code quality locally, use the following Tox environment:

```bash
tox -e quality
```

To automatically fix style issues, use:

```bash
tox -e style
```

To run type checks, use:

```bash
tox -e types
```

## Submitting Changes

1. **Create a Branch**: Create a new branch for your changes:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**: Commit your changes with clear and descriptive commit messages.

3. **Run Tests and Quality Checks**: Before submitting your changes, ensure all tests pass and code quality checks are satisfied:

   ```bash
   tox
   ```

4. **Push Changes**: Push your branch to your forked repository (if you forked):

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**: Go to the original repository and open a pull request. Provide a clear description of your changes and link any related issues.

## Reporting Issues

If you encounter a bug or have a feature request, please open an issue on GitHub. Include as much detail as possible, such as:

- Steps to reproduce the issue
- Expected and actual behavior
- Environment details (OS, Python version, etc.)

## Community Standards

We are committed to fostering a welcoming and inclusive community. Please read and adhere to our [Code of Conduct](https://github.com/vllm-project/guidellm/blob/main/CODE_OF_CONDUCT.md).

## License

By contributing to GuideLLM, you agree that your contributions will be licensed under the [Apache License 2.0](https://github.com/vllm-project/guidellm/blob/main/LICENSE).
