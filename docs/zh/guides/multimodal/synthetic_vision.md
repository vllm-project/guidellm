---
weight: 40
source_sha256: !!str "9553202fb4dc5766dcc8efbafb09ec120cd2a703be63c80e6038a8611bfdf2dd"
---

# 合成视觉数据

GuideLLM 可以按需合成图像和短视频，让你无需准备自己的数据集，即可对视觉语言模型（VLM）的服务配置进行基准测试。两种 `--data` 类型——`synthetic_image` 和 `synthetic_video`——可以与 `synthetic_text` 组合使用，使单条命令就能生成结构完整的多模态请求。

合成视觉数据适用于需要精确控制负载形状（图像尺寸、帧数、每秒帧数）或压力测试服务路径的场景；预处理器缓存可能会掩盖这些路径的实际表现。默认设置经过调整，可确保每个生成的负载在字节层面都与前一个不同，从而绕过 vLLM 的多模态预处理器缓存，同时让媒体在网络传输中的压缩效果接近真实数据。

## 前置条件

安装带有 `vision` 扩展的 GuideLLM，以启用图像和视频合成：

```bash
pip install guidellm[vision]
```

## 合成图像

使用 `--data "kind=synthetic_image"` 为每个请求生成一张图像。如果希望在同一个请求中包含合成提示词文本，可再添加一个单独的 `synthetic_text` 数据源。

### 命令示例

生成一张 720p 图像，并附带 200 个文本 token 和 64 个输出 token：

```bash
guidellm run \
  --backend "kind=openai_http,target=http://localhost:8000" \
  --data "kind=synthetic_text,prompt_tokens=200" \
  --data "kind=synthetic_image,resolution=720p,output_tokens=64"
```

生成一张 1280×720 的 JPEG 图像，并在每个请求中附带两张图像：

```bash
guidellm run \
  --backend "kind=openai_http,target=http://localhost:8000" \
  --data "kind=synthetic_text,prompt_tokens=200" \
  --data "kind=synthetic_image,width=1280,height=720,format=jpeg,images_per_request=2,output_tokens=64"
```

### 配置选项

- `width`：生成图像的宽度（像素）。
- `height`：生成图像的高度（像素）。
- `resolution`：快捷设置，将 `height` 设为指定值（`480p`、`720p`、`1080p` 等）；可与 `aspect_ratio` 搭配推导 `width`。
- `aspect_ratio`：宽高比快捷设置，例如 `16:9` 或 `4:3`。当只提供 `width`、`height` 或 `resolution` 中的一项时，用于推导缺少的尺寸。
- `format`：编码后的图像格式，可选 `jpeg`（默认值）或 `png`。
- `jpeg_quality`：当 `format=jpeg` 时使用的 JPEG 质量因子（1–100），默认值为 85。
- `content`：每行数据的图像内容。`gradient`（默认值）会生成每行带种子的渐变图像，压缩效果接近真实照片；`noise` 会生成均匀随机像素，以获得最坏情况下的网络传输大小；`solid` 和 `checkerboard` 适用于预处理器缓存敏感性测试。
- `images_per_request`：每个请求附带的图像数量，默认值为 1。
- `output_tokens`：模型应生成的平均 token 数量。支持与合成文本模式相同的 `stdev`、`min` 和 `max` 后缀。
- `seed`：随机种子，用于在不同运行中复现生成结果。

## 合成视频

使用 `--data "kind=synthetic_video"` 为每个请求生成一段短视频。如果希望在同一个请求中包含合成提示词文本，可再添加一个单独的 `synthetic_text` 数据源。输出格式为 `mp4`（h264、yuv420p）。

### 命令示例

生成一段六帧的 480p 视频，帧率为 1 fps，并设置适中的提示词和输出 token 数量：

```bash
guidellm run \
  --backend "kind=openai_http,target=http://localhost:8000" \
  --data "kind=synthetic_text,prompt_tokens=64" \
  --data "kind=synthetic_video,width=854,height=480,frames=6,fps=1,output_tokens=128"
```

生成一段十二帧的 720p 视频，帧率为 3 fps，并显式指定 h264 目标比特率：

```bash
guidellm run \
  --backend "kind=openai_http,target=http://localhost:8000" \
  --data "kind=synthetic_text,prompt_tokens=64" \
  --data "kind=synthetic_video,width=1280,height=720,frames=12,fps=3,video_bitrate=2M,output_tokens=128"
```

### 配置选项

- `width`：生成视频的宽度（像素）。
- `height`：生成视频的高度（像素）。适用于合成图像的 `resolution` / `aspect_ratio` 快捷设置同样适用。
- `frames`：视频片段的帧数。
- `fps`：每秒帧数。它与 `frames` 一起决定视频时长。
- `video_bitrate`：可选的 h264 目标比特率（例如 `1M`、`500k`），适用于希望在不同运行中指定固定网络传输大小的场景。
- `content`：每行数据的视频片段内容。`gradient`（默认值）会生成带坐标扭曲的渐变，使每段视频的压缩效果接近真实视频；`noise` 会生成均匀随机像素，以获得最坏情况下的网络传输大小。
- `output_tokens`：模型应生成的平均 token 数量；支持与合成文本模式相同的 `stdev`、`min` 和 `max` 后缀。
- `seed`：随机种子，用于在不同运行中复现生成结果。

## 注意事项

- 使用 `synthetic_text` 时必须提供 tokenizer。默认使用通过参数传入或从服务器获取的模型；否则请通过 `--tokenizer` 指定 tokenizer。
- 每行带种子的渐变图像会生成字节各异的负载，从而绕过 vLLM 的多模态预处理器缓存。如果要主动命中缓存，可使用固定负载设置，例如图像的 `content=solid`，或同时使用固定的 `seed` 和固定的 `--data-loader "kind=pytorch,samples=..."` 值。
- 相同种子生成的 mp4 字节内容会因安装的 `ffmpeg` 和 `PIL` 版本而异。不同版本间的输出 token 数量和请求形状保持稳定；但如果要跨机器比较字节级输出或网络传输大小，请预期会有少量差异。
