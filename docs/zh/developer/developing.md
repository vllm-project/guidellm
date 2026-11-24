---
title: 开发指南
weight: -6
source_sha256: !!str "5c2a76715abab252b84f76fad3ea5ea0ae6b17b2918d27af7351be76e3b0df3a"
---

# 为 GuideLLM 开发

感谢你有兴趣为 GuideLLM 做贡献！本文详细介绍如何设置开发环境、实现更改并遵循项目最佳实践。你的贡献将帮助项目持续发展和改进。

## 设置开发环境

### 前置条件

开始前，请确保已安装以下工具：

- Python 3.11 或更高版本
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)（推荐）、pip 或其他 Python 包管理器
- Git

### 克隆仓库

1. 将仓库克隆到本地：

   ```sh
   git clone https://github.com/vllm-project/guidellm.git
   cd guidellm
   ```

2. （可选）如果计划提交更改，可以先 fork 仓库，然后克隆自己的 fork：

   ```sh
   git clone https://github.com/<your-username>/guidellm.git
   cd guidellm
   ```

### 安装依赖

运行以下任一方式，安装项目及开发所需的依赖：

#### 方式 1：使用 `uv`

将本地虚拟环境同步为项目 `uv.lock` 文件中锁定的依赖版本。

```sh
uv sync --frozen
```

`--frozen` 标志会让 `uv` 不修改依赖版本。

要使用虚拟环境中的工具，可以先运行 `. .venv/bin/activate` 激活环境，或者在命令前加上 `uv run`，让命令在该环境中运行。

#### 方式 2：使用 `pip`

先创建并激活虚拟环境。

```sh
python -m venv .venv && . .venv/bin/activate
```

然后使用 pip 将项目安装到该虚拟环境中。

```sh
pip install --group dev -e .
```

`-e` 标志会以可编辑模式安装项目，因此修改代码后无需重新安装。`--group dev` 会安装开发所需的额外依赖，例如测试和 lint 工具。

## 实现更改

### 编写代码

1. **创建分支**：为你的更改创建一个新分支：

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **进行更改**：在适当的文件中实现更改。确保所有公开函数和类都有清晰、简洁的文档字符串。

3. **更新文档**：根据更改更新或补充文档，包括 README 文件、文档字符串和相关指南。

## 标注 AI 编码助手的使用情况

使用 AI 编码助手生成或大幅修改代码时，GuideLLM 项目要求在 Git 提交说明中添加以下标记之一：

- `Generated-by: <name of the AI coding assistant>`：当提交中的代码主要由 AI 编码助手生成时使用。
- `Assisted-by: <name of the AI coding assistant>`：当提交包含 AI 编码助手创建或修改的大量代码时使用。

例如：

```markdown
This is the commit summary.

Details of the changes made in this commit.

Generated-by: Cursor
Signed-off-by: John Doe <john.doe@example.com>
```

你也可以在 AI 编码助手名称后添加 LLM 模型名称，中间用空格分隔。

例如：

```markdown
Generated-by: Cursor claude-4.6-sonnet
Assisted-by: GitHub Copilot gpt-4o
```

## Agent 技能

共享的 [Agent Skills](https://agentskills.io/specification) 位于 `.agents/skills/`（跨客户端约定）目录下。Cursor 会原生加载该路径；Claude Code 则通过受版本控制的 `.claude/skills` 符号链接访问同一目录。

如需生成过去一周的 GitHub 活动摘要，请请求周报、团队活动更新或状态摘要。此功能需要已通过 `gh` CLI 登录 `vllm-project/guidellm`。

| 技能                      | 用途                                                                                                                         |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `guidellm-weekly-summary` | 生成可对外分享的 GuideLLM 过去一周 GitHub 活动嵌套列表摘要（`scripts/fetch_activity.sh` 使用 `gh` 收集 PR 和 issue 的 JSON） |

## 运行质量、风格和类型检查

我们使用 Tox 简化各类任务的运行。Tox 会在隔离环境中标准化任务环境，确保本地开发、CI/CD 流水线和发布流程保持一致。这样无论在哪种环境中运行，代码的行为都一致。

此外，为确保代码库的一致性和质量，我们使用 [ruff](https://github.com/astral-sh/ruff) 进行 lint 和代码风格检查，使用 [mypy](https://github.com/python/mypy) 进行类型检查，并使用 [mdformat](https://github.com/hukkin/mdformat) 格式化 Markdown 文件。

### 代码质量与风格

运行以下命令检查代码质量，包括 lint 和格式：

```bash
tox -e lint-check
```

自动修复风格问题：

```bash
tox -e lint-fix
```

### 类型检查

使用 Mypy 确保类型安全：

```bash
tox -e type-check
```

### 链接检查

检查文档和 Markdown 文件中的链接是否有效：

```bash
tox -e link-check
```

### 使用 Pre-Commit Hooks 自动执行质量检查（可选）

我们使用 [pre-commit](https://pre-commit.com/) 在提交前自动执行质量检查。Pre-commit hooks 会运行 lint、格式和类型检查等任务，确保提交的代码符合质量要求。

安装 pre-commit hooks：

```bash
pre-commit install
```

安装后，hooks 会在每次提交前自动运行。要手动对所有文件运行 hooks，请使用：

```bash
pre-commit run --all-files
```

## 运行测试

我们使用 [pytest](https://docs.pytest.org/) 作为测试框架，测试套件分为单元测试、集成测试和端到端测试。你可以使用 Tox 运行测试；它会为每种测试套件自动创建隔离环境，并确保测试在不同环境中运行时保持一致。

### 运行所有测试

运行所有测试：

```bash
tox
```

### 运行指定测试

- 单元测试（针对单个组件，使用 mock）：

  ```bash
  tox -e test-unit
  ```

- 集成测试（重点测试组件之间的交互，尽可能不使用 mock）：

  ```bash
  tox -e test-integration
  ```

- 端到端测试（针对整个系统及用户界面）：

  ```bash
  tox -e test-e2e
  ```

### 运行覆盖率测试

要确保测试覆盖了你的更改，请运行：

```bash
tox -e test-unit -- --cov=guidellm --cov-report=html
```

查看覆盖率报告，确认新增代码已得到充分测试。

## 创建 Pull Request

1. **推送更改**：如果使用了 fork，请将分支推送到 fork：

   ```bash
   git push origin feature/your-feature-name
   ```

2. **创建 Pull Request**：前往原始仓库创建 pull request，并使用以下模板填写说明：

   ```markdown
   # Title; ex: Add feature X to improve Y

   ## Summary:

   Short paragraph detailing the pull request changes and reasoning in addition to any relevant context.

   ## Details:

   - Detailed list of changes made in the pull request

   ## Test Plan:

   - Detailed list of steps to test the changes made in the pull request

   ## Related Issues

   - List of related issues or other pull requests; ex: "Fixes #1234"
   ```

3. **处理反馈**：回复审查意见，并进行必要的修改。

## 日志

日志有助于了解 GuideLLM 的运行方式并排查问题。

主要配置 API 是 :func:`~guidellm.logger.configure_logger`。以下环境变量仅在导入时提供后备默认值：

- `GUIDELLM__LOGGING__CONSOLE_LOG_LEVEL`：控制台日志级别（默认：INFO；设为空值可禁用）。
- `GUIDELLM__LOGGING__CONSOLE_COLORIZE`：控制台 ANSI 颜色输出（默认：auto；可选值：auto、true、false）。
- `GUIDELLM__LOGGING__LOG_FILE`：文件日志路径（默认：设置了日志文件级别时为 guidellm.log，否则不写入文件）。
- `GUIDELLM__LOGGING__LOG_FILE_LEVEL`：文件日志级别（默认：设置了日志文件时为 INFO，否则不写入文件）。

如果日志配置没有按环境变量生效，请运行 `guidellm env` 命令，确认环境变量名称正确且已设置。

示例：

单次运行时启用详细控制台输出。交互式进度显示可能会覆盖控制台日志行，因此查看控制台日志时建议使用 `--disable-progress`：

```bash
GUIDELLM__LOGGING__CONSOLE_LOG_LEVEL=DEBUG guidellm run ... --disable-progress
```

将结构化的 `DEBUG` 日志写入文件，同时保持控制台使用默认级别：

```bash
GUIDELLM__LOGGING__LOG_FILE=guidellm.log GUIDELLM__LOGGING__LOG_FILE_LEVEL=DEBUG guidellm run ...
```

## 其他资源

- [CONTRIBUTING.md](https://github.com/vllm-project/guidellm/blob/main/CONTRIBUTING.md)：项目贡献指南。
- [CODE_OF_CONDUCT.md](https://github.com/vllm-project/guidellm/blob/main/CODE_OF_CONDUCT.md)：社区行为规范。
- [tox.ini](https://github.com/vllm-project/guidellm/blob/main/tox.ini)：Tox 环境配置。
- [.pre-commit-config.yaml](https://github.com/vllm-project/guidellm/blob/main/.pre-commit-config.yaml)：pre-commit hooks 配置。
