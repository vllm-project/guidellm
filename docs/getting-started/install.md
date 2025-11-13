---
weight: -10
---

# Install

GuideLLM can be installed using several methods depending on your requirements. Below are the detailed instructions for each installation pathway.

## Prerequisites

Before installing GuideLLM, ensure you have the following prerequisites:

- **Operating System:** Linux or MacOS

- **Python Version:** 3.9 – 3.13

- **Pip Version:** Ensure you have the latest version of pip installed. You can upgrade pip using the following command:

  ```bash
  python -m pip install --upgrade pip
  ```

## Installation Methods

### 1. Install the Latest Release from PyPI

The simplest way to install GuideLLM is via pip from the Python Package Index (PyPI):

```bash
pip install guidellm
```

This will install the latest stable release of GuideLLM.

### 2. Install a Specific Version from PyPI

If you need a specific version of GuideLLM, you can specify the version number during installation. For example, to install version `0.2.0`:

```bash
pip install guidellm==0.2.0
```

### 3. Install from Source on the Main Branch

To install the latest development version of GuideLLM from the main branch, use the following command:

```bash
pip install git+https://github.com/neuralmagic/guidellm.git
```

This will clone the repository and install GuideLLM directly from the main branch.

### 4. Install from a Specific Branch

If you want to install GuideLLM from a specific branch (e.g., `feature-branch`), use the following command:

```bash
pip install git+https://github.com/neuralmagic/guidellm.git@feature-branch
```

Replace `feature-branch` with the name of the branch you want to install.

### 5. Install from a Local Clone

If you have cloned the GuideLLM repository locally and want to install it, navigate to the repository directory and run:

```bash
pip install .
```

Alternatively, for development purposes, you can install it in editable mode:

```bash
pip install -e .
```

This allows you to make changes to the source code and have them reflected immediately without reinstalling.

## Verifying the Installation

After installation, you can verify that GuideLLM is installed correctly by running:

```bash
guidellm --help
```

This should display the installed version of GuideLLM.

## Troubleshooting

If you encounter any issues during installation, ensure that your Python and pip versions meet the prerequisites. For further assistance, please refer to the [GitHub Issues](https://github.com/neuralmagic/guidellm/issues) page or consult the [Documentation](https://github.com/neuralmagic/guidellm/tree/main/docs).
