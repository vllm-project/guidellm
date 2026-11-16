---
title: 贡献指南
weight: -8
source_sha256: !!str "24cde331d6a3622cd7a1304f058af7e83e62ee5d645171b83528ea30b73f93d4"
---

# 为 GuideLLM 做贡献

感谢你考虑为 GuideLLM 做贡献！我们欢迎社区成员参与，帮助改进和发展项目。本文介绍贡献流程和相关指南。

## 如何参与贡献？

你可以通过多种方式为 GuideLLM 做贡献：

- **报告错误**：如果发现错误，请创建 issue 告诉我们。
- **提出功能建议**：有新功能的想法？请创建 issue 进行讨论。
- **改进文档**：提交 pull request，帮助我们完善文档。
- **编写代码**：通过修复错误、添加功能或提升性能来贡献代码。
- **审查 Pull Request**：为开放的 pull request 提供反馈，帮助维护代码质量。

## 开始贡献

### 前置条件

开始贡献前，请确保已安装以下工具：

- Python 3.11 或更高版本
- pip（Python 包管理器）
- Tox
- Git

### 设置仓库

你可以直接克隆仓库；如果计划提交更改，也可以先 fork 仓库：

#### 选项 1：克隆仓库

1. 将仓库克隆到本地：

   ```bash
   git clone https://github.com/vllm-project/guidellm.git
   cd guidellm
   ```

#### 选项 2：Fork 仓库

1. 在仓库的 GitHub 页面点击“Fork”按钮。

2. 将 fork 后的仓库克隆到本地：

   ```bash
   git clone https://github.com/<your-username>/guidellm.git
   cd guidellm
   ```

有关开发环境设置的详细说明，请参阅 [DEVELOPING.md](https://github.com/vllm-project/guidellm/blob/main/DEVELOPING.md)。其中包含以下内容的分步指南：

- 安装依赖
- 运行测试
- 使用 Tox 执行各种任务

## 代码风格与指南

我们遵循严格的编码标准，以确保代码质量和可维护性。请遵循以下指南：

- **代码风格**：使用 [Ruff](https://docs.astral.sh/ruff/) 进行代码格式化和 lint 检查。
- **类型检查**：使用 [Mypy](http://mypy-lang.org/) 进行类型检查。
- **测试**：为新功能和错误修复编写单元测试。使用 [pytest](https://docs.pytest.org/) 运行测试。
- **文档**：代码库发生更改时，请相应更新文档。

若要在本地检查代码质量，请使用以下 Tox 环境：

```bash
tox -e lint-check
```

若要自动修复风格问题，请运行：

```bash
tox -e lint-fix
```

若要运行类型检查，请运行：

```bash
tox -e type-check
```

## 提交更改

1. **创建分支**：为你的更改创建一个新分支：

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **进行更改**：提交更改，并使用清晰、具体的提交说明。

3. **运行测试和质量检查**：提交更改前，请确保所有测试均已通过，且代码质量检查均已满足要求：

   ```bash
   tox
   ```

4. **推送更改**：如果使用了 fork，请将分支推送到你的 fork：

   ```bash
   git push origin feature/your-feature-name
   ```

5. **创建 Pull Request**：前往原始仓库创建 pull request。请清楚说明更改内容，并关联相关 issue。

## 报告问题

如果遇到错误或希望提出功能请求，请在 GitHub 上创建 issue。请尽可能提供详细信息，例如：

- 重现问题的步骤
- 预期行为和实际行为
- 环境详情（操作系统、Python 版本等）

## 社区准则

我们致力于营造友好、包容的社区。请阅读并遵守我们的[行为准则](https://github.com/vllm-project/guidellm/blob/main/CODE_OF_CONDUCT.md)。

## 许可证

为 GuideLLM 做贡献即表示你同意，你的贡献将根据 [Apache License 2.0](https://github.com/vllm-project/guidellm/blob/main/LICENSE) 进行授权。
