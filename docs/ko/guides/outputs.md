---
weight: 2
source_sha256: !!str "56355b3861dc90cbbe1469df6d811a4f430c6b62925dd70ce09405ffab96a27d"
---

# 출력 유형

GuideLLM은 벤치마크 결과를 출력하기 위한 유연한 옵션을 제공하며, 콘솔의 요약 정보와 파일로 저장되는 상세 보고서를 모두 지원합니다. 이 문서에서는 지원되는 출력 유형, 설정 방법 및 효과적인 활용 방법을 설명합니다.

## CLI 출력 설정 {#cli-output-configuration}

`--output` 옵션을 지정하지 않으면 `guidellm run`은 `benchmarks.json`과 `benchmarks.csv`를 [기본 결과 디렉터리](#configuring-file-outputs)에 저장합니다.

출력 설정은 유형을 지정하는 레지스트리 기반 CLI 형식을 따릅니다. `--output` 옵션을 하나라도 지정하면 기본 JSON 및 CSV 출력 대신 지정한 출력을 사용하므로, 원하는 파일 형식마다 옵션을 반복해서 지정하세요. 다음 예제에서는 기본 형식 두 가지를 모두 유지하고 HTML을 추가합니다:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --output kind=json \
  --output kind=csv \
  --output kind=html
```

[지원되는 파일 형식](#supported-file-formats)에서 원하는 형식을 선택하고, 경로를 설정하려면 [파일 출력 설정](#configuring-file-outputs)을 참고하세요. 콘솔 출력은 아래에서 설명하는 별도의 옵션으로 제어합니다.

## 콘솔 출력 {#console-output}

기본적으로 GuideLLM은 벤치마크 결과와 진행 상황을 콘솔에 직접 표시합니다. 콘솔 출력은 별도의 설정 없이 기본으로 활성화됩니다: `--output` 옵션을 명시적으로 지정해도 콘솔 출력은 대체되지 않으며, `--output kind=console`을 지정해도 추가로 달라지는 동작은 없습니다. 콘솔의 진행 상황과 출력은 여러 섹션으로 나뉩니다:

1. **초기 설정 진행 상황**: 서버 연결 및 데이터 준비를 포함한 초기 설정의 진행 상황을 표시합니다.
2. **벤치마크 진행 상황**: 완료된 요청 수와 현재 요청률을 포함한 벤치마크 실행의 진행 상황을 표시합니다.
3. **최종 결과**: 평균 지연 시간, 처리량 및 기타 주요 지표를 포함한 벤치마크 결과를 요약합니다.
   1. **벤치마크 메타데이터**: 서버 세부 정보, 데이터 설정 및 프로파일 인자를 포함한 벤치마크 실행 정보를 요약합니다.
   2. **벤치마크 정보**: 요청 상태, 토큰 수 및 소요 시간을 포함한 각 벤치마크의 전반적인 정보를 제공합니다.
   3. **벤치마크 통계**: 요청률, 동시 요청 수, 지연 시간 및 토큰 단위 지표와 같은 각 벤치마크의 상세 통계를 표시합니다.

### 콘솔 출력 비활성화

대화형 진행 상황 갱신을 비활성화하려면 `--disable-console-interactive`(별칭 `--disable-progress`)를 사용하세요:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep \
  --constraint kind=max_duration,seconds=30 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --disable-console-interactive
```

모든 콘솔 출력을 비활성화하려면 `--disable-console`(별칭 `--disable-console-outputs`)을 사용하세요:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep \
  --constraint kind=max_duration,seconds=30 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --disable-console
```

## 파일 기반 출력

GuideLLM은 벤치마크 결과를 JSON, YAML 및 CSV 등 다양한 형식의 파일로 저장할 수 있습니다. 이 파일은 추가 분석이나 보고서 작성에 활용하거나, Python에서 다시 불러와 상세하게 살펴볼 수 있습니다.

### 지원되는 파일 형식 {#supported-file-formats}

| 형식 (`kind`) | 기본 파일 이름    | 내용                                                                                                                                         |
| ------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `json`        | `benchmarks.json` | 디버깅하거나 Python에서 다시 불러오는 데 사용할 수 있는 설정, 메타데이터, 벤치마크 통계 및 보관된 요청 데이터                                |
| `yaml`        | `benchmarks.yaml` | JSON과 동일한 상세 벤치마크 데이터를 사람이 읽기 쉬운 YAML 형식으로 표현한 파일                                                              |
| `csv`         | `benchmarks.csv`  | 스프레드시트에서 활용하거나 비교할 수 있도록 처리량, 지연 시간, 토큰 수 및 요청률을 표로 요약한 파일. 요청별 상세 데이터는 포함하지 않습니다 |
| `html`        | `benchmarks.html` | 처리량과 지연 시간의 차트 및 표를 포함한 독립형 보고서. 오프라인으로 공유할 수 있도록 CSS와 JavaScript가 내장되어 있습니다                   |
| `plot`        | `benchmarks.png`  | 정적 벤치마크 성능 그래프                                                                                                                    |

PLOT의 경우 `path`의 확장자로 이미지 형식을 선택합니다: PNG, JPG/JPEG, SVG 또는 PDF. 예를 들어, `--output kind=plot,path=benchmarks.pdf`는 PDF를 생성합니다. 경로에 확장자가 없으면 기본값인 `.png`를 사용하며, 지원되지 않는 확장자를 지정하면 오류가 발생합니다. `dpi` 파라미터(기본값 `100`)는 이미지 해상도를 제어합니다. 예를 들어, `--output kind=plot,path=benchmarks.jpg,dpi=72`로 설정할 수 있습니다.

### 파일 출력 설정 {#configuring-file-outputs}

`path`를 생략하면 `GUIDELLM__DEFAULT_RESULTS_DIR`로 설정한 디렉터리에 해당 형식의 기본 파일 이름으로 저장하며, 이 변수가 설정되어 있지 않으면 현재 디렉터리에 저장합니다. 파일 이름이나 저장 위치를 설정하려면 `path=`를 지정하세요.

예를 들어, 다음 명령어는 별도로 표시되는 콘솔 출력과 함께 사용자 지정 경로에 JSON 파일을 저장합니다:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep \
  --constraint kind=max_duration,seconds=30 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --output kind=json,path=results/benchmark.json
```

### 출력 파일 크기 제어

기본적으로 모든 요청의 전체 데이터(프롬프트 텍스트, 출력 텍스트, 도구 호출)를 보관하므로, 수천 개의 요청으로 벤치마크를 장시간 실행하면 JSON 및 YAML 출력 파일의 크기가 커질 수 있습니다. `--metrics` 옵션을 사용하면 리저버 샘플링(reservoir sampling)으로 보관할 요청 데이터의 양을 제한할 수 있으며, 용량이 작은 통계(지연 시간, 토큰 수, 시간 정보)는 항상 모든 요청에 대해 유지됩니다.

`sample_size`를 사용하여 전체 데이터를 보관할 최대 요청 수를 **상태 그룹별**(완료된 요청, 오류가 발생한 요청, 완료되지 않은 요청)로 설정하세요:

| 값              | 동작                                                   |
| --------------- | ------------------------------------------------------ |
| 미설정(기본값)  | 모든 요청의 전체 데이터 보관                           |
| `0`             | 모든 요청 데이터 제거(통계만 유지)                     |
| `N`(예시 `100`) | 그룹별로 무작위로 샘플링한 요청 N개의 전체 데이터 보관 |

상태 그룹별로 무작위로 샘플링한 요청을 최대 100개까지 선택하여 전체 데이터를 보관하려면 다음과 같이 설정합니다:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep \
  --constraint kind=max_requests,count=10000 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --metrics kind=generative,sample_size=100 \
  --output kind=json,path=results/benchmark.json
```

`--metrics` 옵션에는 `prefer_response_metrics`(기본값 `true`)도 지정할 수 있습니다. 이 파라미터는 서버가 보고한 토큰 수와 클라이언트가 계산한 토큰 수를 모두 사용할 수 있을 때 서버가 보고한 값을 우선 사용할지 여부를 제어합니다. 이 설정은 변경할 필요가 거의 없습니다.

### 결과 다시 불러오기

`GenerativeBenchmarksReport` 클래스를 사용하면 JSON 파일을 Python에서 다시 불러와 추가로 분석할 수 있습니다. 다음은 결과를 다시 불러오는 코드 예제입니다:

```python
from guidellm.benchmark import GenerativeBenchmarksReport

report = GenerativeBenchmarksReport.load_file(
    path="benchmarks.json",
)
benchmarks = report.benchmarks

for benchmark in benchmarks:
    print(benchmark.config.id_)
```

`GenerativeBenchmarksReport` 클래스와 메서드에 대한 자세한 내용은 [소스 코드](https://github.com/vllm-project/guidellm/blob/main/src/guidellm/benchmark/schemas/report.py)를 참고하세요.
