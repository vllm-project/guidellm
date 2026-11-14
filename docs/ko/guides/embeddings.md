---
weight: -1
source_sha256: !!str "27416a0b61194ce69280fa2fcab7b7d07ec16c8617d35e44e449760eb32c0c43"
---

# 임베딩 벤치마킹 가이드

GuideLLM은 OpenAI 호환 임베딩 엔드포인트의 벤치마킹을 지원하며, 처리량, 지연 시간 및 동시 요청 수와 같은 성능 특성을 측정할 수 있습니다.

## 개요

임베딩 모델은 텍스트를 의미 검색, 정보 검색 및 유사도 계산에 사용하는 밀집 벡터 표현으로 변환합니다. 텍스트를 출력하는 생성형 모델과 달리, 임베딩 모델은 다음과 같은 특성을 갖습니다:

- 입력 텍스트를 처리하고 임베딩 벡터를 반환합니다
- 스트리밍을 지원하지 않습니다(요청당 응답 1개)
- 입력 토큰만 집계합니다(출력 토큰 없음)
- 요청 지연 시간과 처리량을 측정합니다

## 빠른 시작

```bash
# Start vLLM server with an embeddings model
vllm serve BAAI/bge-small-en-v1.5 --port 8000

# Run benchmark
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000/v1,model=BAAI/bge-small-en-v1.5,request_format=/v1/embeddings \
  --data kind=synthetic_text,prompt_tokens=128 \
  --constraint kind=max_requests,count=100
```

## 생성형 모델 벤치마크와의 주요 차이점

| 항목           | 임베딩 모델       | 생성형 모델       |
| -------------- | ----------------- | ----------------- |
| 출력           | 임베딩 벡터       | 생성된 텍스트     |
| 스트리밍       | 지원하지 않음     | 지원함            |
| 출력 토큰      | 해당 없음         | 개수가 달라짐     |
| TTFT           | 해당 없음         | 측정함            |
| 토큰 지연 시간 | 해당 없음         | 측정함            |
| 주요 지표      | 지연 시간, 처리량 | TTFT, ITL, 처리량 |

## 관련 문서

- [벤치마크 프로파일](../getting-started/benchmark.md#benchmark-profiles-profile) - 모든 프로파일 유형에 대한 자세한 설명
- [데이터셋 가이드](datasets.md) - 사용자 지정 데이터셋 생성 및 사용
- [지표 가이드](metrics.md) - 성능 지표 이해하기
