<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/guidellm/main/docs/assets/guidellm-logo-light.png">
    <img alt="GuideLLM Logo" src="https://raw.githubusercontent.com/vllm-project/guidellm/main/docs/assets/guidellm-logo-dark.png" width=55%>
  </picture>
</p>

<h3 align="center">
面向真实场景大语言模型推理优化的 SLO 感知压测与评估平台
</h3>

<p align="center">
  <a href="./README.md">English</a> | <b>简体中文</b>
</p>

[![GitHub Release](https://img.shields.io/github/release/vllm-project/guidellm.svg?label=Version)](https://github.com/vllm-project/guidellm/releases) [![Documentation](https://img.shields.io/badge/Documentation-8A2BE2?logo=read-the-docs&logoColor=%23ffffff&color=%231BC070)](https://github.com/vllm-project/guidellm/tree/main/docs) [![License](https://img.shields.io/github/license/vllm-project/guidellm.svg)](https://github.com/vllm-project/guidellm/blob/main/LICENSE) [![PyPI Release](https://img.shields.io/pypi/v/guidellm.svg?label=PyPI%20Release)](https://pypi.python.org/pypi/guidellm) [![Python Versions](https://img.shields.io/badge/Python-3.11--3.15-orange)](https://pypi.python.org/pypi/guidellm) [![Nightly Build](https://img.shields.io/github/actions/workflow/status/vllm-project/guidellm/nightly.yml?branch=main&label=Nightly%20Build)](https://github.com/vllm-project/guidellm/actions/workflows/nightly.yml)

## 项目概览 (Overview)

<p>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/guidellm/main/docs/assets/guidellm-user-flows-dark.png">
    <img alt="GuideLLM User Flows" src="https://raw.githubusercontent.com/vllm-project/guidellm/main/docs/assets/guidellm-user-flows-light.png">
  </picture>
</p>

**GuideLLM** 是一个专为评估大语言模型在真实业务负载与多样化部署配置下实际性能表现而打造的评测平台。它能够模拟客户端与兼容 OpenAI 规范及 vLLM 原生服务端的端到端请求交互，生成贴合线上真实流量特征的工作负载模式，并输出详尽的评测分析报告，助力工程与算法团队深刻理解系统运行特征、资源规格需求以及生产服务的性能容量边界。GuideLLM 全面支持真实与合成数据集、多模态多维输入以及高度可定制的执行配置策略（Profile），为工程与算法团队提供统一且一致的评测基准，以评估模型行为、调优线上部署配置，并在系统持续演进过程中科学规划硬件算力容量。

### 为什么选择 GuideLLM？

在准生产与生产环境中部署大语言模型时，GuideLLM 能为团队呈现清晰全景的性能、效率与服务稳定性画像：

- **捕获全链路延迟与 Token 级细粒度统计指标，赋能以 SLO（服务等级目标）为驱动的性能评估**：包括首字延迟（TTFT）、字间生成延迟（ITL）以及端到端响应时间的全分位数完整分布。
- **生成逼真且高度可配置的流量负载模型**：支持同步阻塞、高并发以及基于固定/泊松请求速率等多种流量模式，并提供可完全复现的参数扫描（Sweep），快速精准定位安全运行水位与性能拐点。
- **兼顾真实数据集与多模态合成数据**：既支持在严谨受控的环境下进行科学对比实验，也支持贴合生产真实流量的大规模全链路评估。
- **输出标准化且易于导出的评测报告**：适配监控大盘可视化、性能深度分析与版本回归追踪，保障跨团队协作与工作流的一致性。
- **提供高吞吐、高扩展性的基准评测能力**：底层基于多进程、多线程与异步并发架构实现，并配套灵活强大的 CLI 命令行与 Python API，兼顾开箱即用体验与深度二次开发定制需求。

## 快速上手 (Quick Start)

本指南将指导你在数分钟内完成 GuideLLM 的安装、启动模型推理服务，并执行你的第一次基准评测。

### 安装 GuideLLM

安装前，请确保满足以下环境前提条件：

- 操作系统：Linux 或 macOS
- Python 版本：3.11 至 3.15

通过 `pip` 从 PyPI 安装最新正式版本：

```bash
pip install guidellm[recommended]
```

或者直接从源码构建安装：

```bash
pip install git+https://github.com/vllm-project/guidellm.git
```

或者直接使用官方容器镜像进行免安装快速运行（镜像托管于 [ghcr.io/vllm-project/guidellm](https://github.com/vllm-project/guidellm/pkgs/container/guidellm)）：

```bash
podman run \
  --rm -it \
  -v "./results:/results:rw" \
  -e GUIDELLM__SPEC__BACKEND='{"kind": "openai_http", "target": "http://localhost:8000"}' \
  -e GUIDELLM__SPEC__PROFILE='{"kind": "sweep"}' \
  -e GUIDELLM__SPEC__CONSTRAINTS='[{"kind": "max_duration", "seconds": 30}]' \
  -e GUIDELLM__SPEC__DATA='[{"kind": "synthetic_text", "prompt_tokens": 256, "output_tokens": 128}]' \
  ghcr.io/vllm-project/guidellm:latest
```

官方镜像均为多架构构建（同时支持 `linux/amd64` 与 `linux/arm64`），兼容 Docker、Podman、Kubernetes 以及 OpenShift。生产环境中建议锁定具体的版本 Tag：

| 镜像标签 (Tag) | 说明                                            |
| -------------- | ----------------------------------------------- |
| `vX.Y.Z`       | 不可变稳定版本镜像（从 `v0.7.0+` 起支持多架构） |
| `stable`       | 最新的稳定正式发行版（对应最新的 `vX.Y.Z`）     |
| `latest`       | 最新的发布版本（可能包含预发布版本）            |
| `nightly`      | 基于 `main` 分支主线最新代码构建的每日镜像      |

注：特定硬件架构后缀标签（如 `vX.Y.Z-amd64` / `vX.Y.Z-arm64`）仅为构建产物中间件，请勿直接将其作为镜像拉取引用。

### 启动推理服务端

启动任意兼容 OpenAI 规范的推理服务接口。以 vLLM 为例：

```bash
vllm serve "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
```

确认服务已在 `http://localhost:8000` 正常启动并处于监听状态。

### 执行第一次基准评测

执行一次自动化策略扫描（Sweep），以测算该模型在此硬件配置下的极限吞吐性能与最大请求处理速率：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep \
  --constraint kind=max_duration,seconds=30 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128
```

运行期间，终端将实时展示评测进度与阶段性性能摘要，如下所示：

<img src="https://raw.githubusercontent.com/vllm-project/guidellm/main/docs/assets/sample-benchmarks.gif" />

### 查看评测产物

基准评测执行完成后，GuideLLM 默认会在结果目录写入 `benchmarks.json` 与 `benchmarks.csv`。结果文件将保存在由环境变量 `GUIDELLM__DEFAULT_RESULTS_DIR` 指定的路径中，未显式设置时默认存储在当前工作目录下。同时，控制台终端也会打印出汇总表格与生成文件的具体物理路径。

显式指定 `--output` 参数将替代默认的 JSON 和 CSV 产物配置。有关多格式输出配置（包括同时生成交互式 HTML 报告、JSON 与 CSV）的示例，请参见[输出配置指南](docs/en/guides/outputs.md#cli-output-configuration)。

## 输出文件与评测报告 (Output Files and Reports)

GuideLLM 支持多种维度的结果导出：使用 JSON 或 YAML 进行深入的自动化数据分析，使用 CSV 导入电子表格横向对比，使用自包含的 HTML 文件生成直观的可视化图表报告，或使用 PLOT 生成静态性能趋势图。各类格式的详细内容字段与默认文件命名请参阅[支持的文件格式说明](docs/en/guides/outputs.md#supported-file-formats)，有关自定义输出路径请查阅[配置文件输出指南](docs/en/guides/outputs.md#configuring-file-outputs)。

控制台终端输出直观展示了每次评测的核心指标。表格文本可以直接以 `|` 作为分隔符复制并粘贴至各类电子表格软件中。更多关于进度展示和控制台打印样式的定制请参见[控制台输出文档](docs/en/guides/outputs.md#console-output)。

<img alt="GuideLLM 基准评测输出范例" src="https://raw.githubusercontent.com/vllm-project/guidellm/main/docs/assets/sample-output.png" />

## 常见使用场景与配置范例 (Common Use Cases and Configurations)

GuideLLM 灵活支持多种大语言模型评测工作流。以下范例展示了典型业务场景的压测配置及核心关键参数。如需查看所有命令行参数、选项和详细释义，请运行 `guidellm run --help`。

所有基于注册中心机制（Registry-backed）的选项统一遵循 `--<option> kind=<TYPE>,<CONFIG>...` 的语法格式，其中 `CONFIG` 为 `key=value` 键值对。对于更复杂的结构化配置，亦可直接传入 JSON 或 YAML 字符串，例如：`--data '{"kind":"huggingface","source":"abisee/cnn_dailymail","load_kwargs":{"name":"3.0.0"}}'`。

### 负载模式模拟 (Load Patterns)

模拟不同业务应用需要施加相匹配的流量形状。以下示例展示了基于固定速率的负载测试：采用恒定模式（Constant Profile），以每秒 10 个请求（10 RPS）的速率持续运行 20 秒，数据源采用输入 128 Token、期望输出 256 Token 的合成文本：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=constant,rate=10 \
  --constraint kind=max_duration,seconds=20 \
  --data kind=synthetic_text,prompt_tokens=128,output_tokens=256
```

**核心关键参数：**

- `--profile kind=<type>`：定义流量模式，可选 `synchronous`（同步单请求）、`concurrent`（固定并发流）、`throughput`（最大吞吐压测）、`constant`（恒定速率）、`poisson`（泊松分布拟真流量）或 `sweep`（多级阶梯扫描）。
- `--profile kind=constant,rate=10`：对于 `constant`/`poisson` 模式，在 Profile 中指定每秒请求数；对于 `concurrent` 模式使用 `streams=`；对于 `throughput` 模式使用 `max_concurrency=`。
- `--constraint kind=max_duration,seconds=<seconds>` 或 `--constraint kind=max_requests,count=<count>`：按测试持续时间或总完成请求量对当前压测策略设置退出限制。

### 数据集来源 (Dataset Sources)

GuideLLM 支持载入 Hugging Face 数据集、本地结构化文件以及动态生成的合成数据。以下示例从 Hugging Face 加载 CNN DailyMail 数据集，将 `article` 列映射为模型 Prompt 输入，并根据摘要的 Token 数量列动态决定期望生成的输出长度：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=huggingface,source=abisee/cnn_dailymail,load_kwargs.name=3.0.0 \
  --data-column-mapper kind=generative_column_mapper,column_mappings.text_column=article
```

**核心关键参数：**

- `--data`：数据类型及配置 —— `synthetic_text`、`huggingface`、`json_file`、`csv_file`、`text_file`、`trace_synthetic` 等。可重复指定以组合多个数据源。
- `--data-column-mapper`：列映射预处理器及其 JSON 配置，可指定 `text_column`、`output_tokens_count_column` 等映射关系。
- `--data-loader type=pytorch,samples=1000`：控制数据加载的最大样本行数（`-1` 代表加载全部样本）。
- `--tokenizer huggingface_auto "model=gpt2"`：指定用于合成数据或本地离线 Token 计数的分词器（Tokenizer）。

### 多模态视觉合成数据 (Synthetic Visual Data)

GuideLLM 能够在运行时动态合成图像与简短视频，因此用户无需准备庞大的多模态数据集即可直接评测视觉大模型（VLM）服务配置。通过 `--data` 的两种类型 —— `synthetic_image` 与 `synthetic_video` —— 可与 `synthetic_text` 自由组合构成复合多模态 Prompt。详细使用命令与全量配置选项请参见[合成视觉数据指南](docs/en/guides/multimodal/synthetic_vision.md)。

### 请求类型与 API 目标 (Request Types and API Targets)

你可以对对话补全（Chat Completions）、文本生成（Text Completions）或其他受支持的接口类型执行压测。以下示例配置压测 `/v1/chat/completions` API，载入自定义 JSON 数据集文件，GuideLLM 将自动按对话补全的 Schema 组装请求：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,request_format=/v1/chat/completions \
  --data kind=json_file,path=path/to/data.json
```

**核心关键参数：**

- `--backend`：后端类型及连接网络配置，包括目标 OpenAI 兼容接口服务 URL `target`，以及所调用的 API 端点格式 `request_format`（如 `/v1/chat/completions`、`/v1/completions`、`/v1/embeddings`、`/v1/audio/transcriptions` 等）。

### 预设压测场景 (Using Scenarios)

内置场景（Scenarios）将压测调度计划、数据集设置和请求格式化参数进行了标准化打包，开箱即用。以下示例采用预配置的 `chat` 对话场景，包含针对对话模型评测的最佳实践默认参数，同时允许通过额外的 CLI 参数按需覆盖特定项：

```bash
guidellm run \
  --config chat \
  --backend kind=openai_http,target=http://localhost:8000
```

**核心关键参数：**

- `--config`（别名 `--scenario`、`-c`）：内置预设场景名称或自定义场景文件路径。命令行中传入的显式参数具有更高优先级，将覆盖场景内的同名默认设置。

### 压测控制与稳定性保障 (Benchmark Controls)

预热（Warmup）、冷却（Cooldown）以及最大限制约束有助于确保压测度量结果的稳定性和可复现性。以下示例执行包含 16 个并行请求的并发测试，配置 10% 的预热与冷却期以排除初始化和关停效应的影响，并在累计错误超过 5 次时终止测试：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=16,warmup=0.1,cooldown=0.1 \
  --constraint kind=max_errors,count=5 \
  --constraint kind=over_saturation \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128
```

**核心关键参数：**

- `--profile kind=<type>`：Profile 配置支持 `warmup` 与 `cooldown`（可采用百分比或绝对单位）。
- `--constraint kind=max_duration,seconds=<seconds>` / `--constraint kind=max_requests,count=<count>`：按时长或总完成请求量控制退出。
- `--constraint kind=max_errors,count=<count>`：当累计错误总数超过阈值时主动退出测试。
- `--constraint kind=over_saturation`：开启服务端过载饱和自动探测（配置为空时采用系统默认判断准则）。

## 开发与贡献指南 (Development and Contribution)

欢迎有志于扩展 GuideLLM 功能的开发者参与贡献。本地环境搭建、虚拟环境激活以及单元测试执行步骤均详述于 [DEVELOPING.md](https://github.com/vllm-project/guidellm/blob/main/DEVELOPING.md) 中。该指南清晰说明了如何在开发阶段执行评测用例集、验证修改以及调试 CLI 与 API。开源贡献准则记录于 [CONTRIBUTING.md](https://github.com/vllm-project/guidellm/blob/main/CONTRIBUTING.md)，包含编码规范、Git Commit 提交结构以及代码审查指南，以确保平台演进过程中的稳定性。所有参与者均需遵循 [CODE_OF_CONDUCT.md](https://github.com/vllm-project/guidellm/blob/main/CODE_OF_CONDUCT.md) 中关于友善与建设性交流的社区行为准则。

如需更深度的架构参考，官方文档详尽涵盖了安装、后端、数据集、评测指标、产物格式及底层系统设计。在新增后端支持、适配全新请求类型或对接自研数据源时，阅读相关章节将大有裨益。每个版本的 Release 变更记录均发布于 GitHub Releases 页面，供开发者了解项目演进脉络。

## 核心文档索引 (Documentation)

更详尽的技术文档包含了超出本 README 篇幅的所有深入细节，包含安装手册、后端连接配置、数据集调度、指标数学定义、导出格式、使用教程以及完整的架构概览，帮助你深度挖掘平台潜能并将其无缝接入现有生产工作流：

- [**安装指南 (Installation Guide)**](https://github.com/vllm-project/guidellm/blob/main/docs/en/getting-started/install.md) - 提供安装 GuideLLM 的逐步指南，包括运行环境前置条件和环境排障建议。
- [**后端指南 (Backends Guide)**](https://github.com/vllm-project/guidellm/blob/main/docs/en/guides/backends.md) - 深入解析所有支持的推理后端以及如何配置以对接 GuideLLM。
- [**数据与数据集指南 (Data/Datasets Guide)**](https://github.com/vllm-project/guidellm/blob/main/docs/en/guides/datasets.md) - 详解受支持的数据集格式以及如何将其应用于压测基准。
- [**指标详解指南 (Metrics Guide)**](https://github.com/vllm-project/guidellm/blob/main/docs/en/guides/metrics.md) - 详尽解释 GuideLLM 所采集的各项性能指标定义、统计口径以及结果解读方法。
- [**输出产物指南 (Outputs Guide)**](https://github.com/vllm-project/guidellm/blob/main/docs/en/guides/outputs.md) - 介绍 GuideLLM 支持的不同输出格式及其使用方式。
- [**系统架构概览 (Architecture Overview)**](https://github.com/vllm-project/guidellm/blob/main/docs/en/guides/architecture.md) - 深度解析 GuideLLM 的内部软件架构设计、核心模块划分及各组件间的交互机制。

## 开源许可证 (License)

GuideLLM 遵循 [Apache License 2.0](https://github.com/vllm-project/guidellm/blob/main/LICENSE) 开源许可证。

## 学术引用 (Cite)

若 GuideLLM 对你的学术研究或工程项目有所启发与帮助，欢迎引用本项目：

```bibtex
@misc{guidellm2024,
  title={GuideLLM: Scalable Inference and Optimization for Large Language Models},
  author={Neural Magic, Inc.},
  year={2024},
  howpublished={\url{https://github.com/vllm-project/guidellm}},
}
```

______________________________________________________________________

> **免责声明**：本简体中文文档由社区志愿者出于学习交流目的进行翻译维护。英文原版文档始终作为项目最权威的单一事实来源（Single Source of Truth）；如中文翻译与英文原版存在任何语义或版本脱节，请以 [英文原版 README](https://github.com/vllm-project/guidellm/blob/main/README.md) 为准。
