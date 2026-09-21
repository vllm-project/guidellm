---
weight: -10
---

# 安装

GuideLLM 提供多种安装方式，你可以根据自身需求进行选择。下面分别介绍各安装方式的详细步骤。

## 前置条件

安装 GuideLLM 前，请确保满足以下条件：

- **操作系统：** Linux 或 macOS

- **Python 版本：** 3.10–3.13

- **Pip 版本：** 请确保安装了最新版本的 pip。可以使用以下命令升级 pip：

  ```bash
  python -m pip install --upgrade pip
  ```

## 安装方式

### 1. 从 PyPI 安装最新版本

最简单的方式是通过 pip 从 Python Package Index（PyPI）安装 GuideLLM：

```bash
pip install guidellm[recommended]
```

这将安装 GuideLLM 的最新稳定版本及推荐依赖。

### 2. 从 PyPI 安装指定版本

如需安装指定版本的 GuideLLM，可以在安装时提供版本号。例如，安装 `0.2.0`：

```bash
pip install guidellm==0.2.0
```

### 3. 从 main 分支的源代码安装

如需安装 main 分支上的最新开发版本，请使用以下命令：

```bash
pip install git+https://github.com/vllm-project/guidellm.git
```

该命令会克隆仓库，并直接从 main 分支安装 GuideLLM。

### 4. 从指定分支安装

如需从指定分支（例如 `feature-branch`）安装，请使用以下命令：

```bash
pip install git+https://github.com/vllm-project/guidellm.git@feature-branch
```

请将 `feature-branch` 替换为对应的分支名称。

### 5. 从本地克隆安装

如果已经在本地克隆了 GuideLLM 仓库，请进入仓库目录并运行：

```bash
pip install .
```

在开发场景下，也可以使用可编辑模式进行安装：

```bash
pip install -e .
```

这样无需重新安装即可使代码改动生效。

## 验证安装

安装后，可以运行以下命令验证 GuideLLM 是否可用：

```bash
guidellm --help
```

该命令应显示 GuideLLM 的帮助信息。

如需使用 vLLM Python 后端（进程内推理），请参阅英文版 [vLLM Python 后端](../../guides/vllm-python-backend.md)文档，了解推荐的安装方式，包括使用容器、现有 vLLM 环境或 pip。

## 故障排除

如果安装时遇到问题，请确认操作系统和 Python 版本满足前置条件。常见问题的处理方式，包括调试日志、tokenizer 加载和 macOS 工作进程崩溃，请参阅英文版[故障排除指南](../../guides/troubleshooting.md)。如需进一步协助，请前往 [GitHub Issues](https://github.com/vllm-project/guidellm/issues)。
