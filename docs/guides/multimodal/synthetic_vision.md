---
weight: 40
---

# Synthetic Visual Data

GuideLLM can synthesize images and short videos on the fly so you can benchmark Vision-Language Model (VLM) serving configurations without bringing your own dataset. Two `--data` types — `synthetic_image` and `synthetic_video` — compose with the existing synthetic text token controls (`text_tokens`, `output_tokens`, and their `stdev`/`min`/`max` companions) so a single command produces a fully-shaped multimodal request.

Synthetic visual data is useful when you want to control payload shape precisely (image dimensions, frame count, frames-per-second) or stress-test serving paths that the preprocessor cache would otherwise hide. Defaults are tuned so every generated payload is byte-different from the next, which defeats vLLM's multimodal preprocessor cache while still compressing like real media on the wire.

## Prerequisites

Install GuideLLM with the `vision` extra to enable image and video synthesis:

```bash
pip install guidellm[vision]
```

## Synthetic image

Use `--data "type=synthetic_image"` to generate a single image per request alongside any text prompt.

### Example Commands

A single 720p image alongside 200 text tokens and 64 output tokens:

```bash
guidellm benchmark run \
  --target http://localhost:8000 \
  --data "type=synthetic_image,resolution=720p,text_tokens=200,output_tokens=64"
```

A 1280×720 JPEG with two images per request:

```bash
guidellm benchmark run \
  --target http://localhost:8000 \
  --data "type=synthetic_image,width=1280,height=720,format=jpeg,images_per_request=2,text_tokens=200,output_tokens=64"
```

### Configuration Options

- `width`: Width of the generated image in pixels.
- `height`: Height of the generated image in pixels.
- `resolution`: Shorthand that sets `height` to a named value (`480p`, `720p`, `1080p`, …); pairs with `aspect_ratio` to derive `width`.
- `aspect_ratio`: Shorthand such as `16:9` or `4:3` that derives the missing dimension when only one of `width`/`height`/`resolution` is given.
- `format`: Encoded image format, `jpeg` (default) or `png`.
- `jpeg_quality`: JPEG quality factor (1–100) when `format=jpeg`. Defaults to 85.
- `content`: Per-row image content. `gradient` (default) emits a per-row seeded gradient that compresses like real photography; `noise` emits uniform random pixels for worst-case wire size; `solid` and `checkerboard` are useful for preprocessor-cache sensitivity sweeps.
- `images_per_request`: Number of images to attach to each request. Defaults to 1.
- `text_tokens`: Average number of tokens in the accompanying text prompt. Accepts the same `stdev` / `min` / `max` suffixes as the synthetic text mode. `prompt_tokens` is accepted as an alias.
- `output_tokens`: Average number of tokens the model should generate. Same `stdev` / `min` / `max` suffixes apply.
- `seed`: Random seed for reproducible generation across runs.

## Synthetic video

Use `--data "type=synthetic_video"` to generate a short clip per request alongside any text prompt. Output is `mp4` (h264, yuv420p).

### Example Commands

A six-frame 480p clip at 1 fps with modest prompt and output budgets:

```bash
guidellm benchmark run \
  --target http://localhost:8000 \
  --data "type=synthetic_video,width=854,height=480,frames=6,fps=1,text_tokens=64,output_tokens=128"
```

A twelve-frame 720p clip at 3 fps with an explicit h264 target bitrate:

```bash
guidellm benchmark run \
  --target http://localhost:8000 \
  --data "type=synthetic_video,width=1280,height=720,frames=12,fps=3,video_bitrate=2M,text_tokens=64,output_tokens=128"
```

### Configuration Options

- `width`: Width of the generated video in pixels.
- `height`: Height of the generated video in pixels. The same `resolution` / `aspect_ratio` shorthands as for synthetic image apply.
- `frames`: Number of frames in the clip.
- `fps`: Frames per second. Combined with `frames`, this also determines the clip duration.
- `video_bitrate`: Optional h264 target bitrate (e.g. `1M`, `500k`) — useful when you want to specify a fixed wire size across runs.
- `content`: Per-row clip content. `gradient` (default) emits a seeded gradient with a coordinate warp so each clip compresses similarly to real video; `noise` emits uniform random pixels for worst-case wire size.
- `text_tokens`: Average number of tokens in the accompanying text prompt; same `stdev` / `min` / `max` suffixes as synthetic image. `prompt_tokens` is accepted as an alias.
- `output_tokens`: Average number of tokens the model should generate; same `stdev` / `min` / `max` suffixes apply.
- `seed`: Random seed for reproducible generation across runs.

## Notes

- A processor/tokenizer is required for the text portion of the request. By default the model passed in or retrieved from the server is used; otherwise specify one with `--processor`.
- Per-row seeded gradients produce byte-different payloads on every request, which bypasses vLLM's multimodal preprocessor cache. If you want to deliberately hit the cache, set `content=solid` or pin a fixed `seed` and `samples`.
- The exact mp4 bytes produced for a given seed depend on the installed `ffmpeg` and `PIL` versions. Output token counts and request shape stay stable across versions, but if you are comparing byte-level outputs or wire-size measurements across machines, expect small variation.
