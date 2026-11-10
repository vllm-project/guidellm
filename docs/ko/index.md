---
source_sha256: !!str "462c154b627b2bcba5ff2a1aba9581eab5dd76fcc1157092b8fa952211613367"
title: 홈
weight: 100
---

# 홈

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/guidellm/main/docs/assets/guidellm-logo-light.png">
    <img alt="GuideLLM Logo" src="https://raw.githubusercontent.com/vllm-project/guidellm/main/docs/assets/guidellm-logo-dark.png" width=55%>
  </picture>
</p>

<h3 align="center">
실제 운영 환경에서의 LLM 추론 최적화를 위한 SLO 기반 벤치마킹 및 평가 플랫폼
</h3>

**GuideLLM**은 실제 워크로드와 다양한 설정에서 언어 모델의 성능을 평가하는 플랫폼입니다. OpenAI 호환 서버 및 vLLM-native 서버와의 end-to-end 상호작용을 시뮬레이션하고, 실제 운영 환경의 사용 패턴을 반영한 워크로드를 생성하며, 상세한 분석 보고서를 제공하여 시스템의 동작 특성, 리소스 요구 사항 및 운영상의 한계를 파악할 수 있도록 지원합니다. GuideLLM은 실제 데이터셋과 합성 데이터셋, 멀티모달 입력, 유연한 실행 프로파일을 지원하여, 엔지니어링 및 ML 팀이 시스템 변화에 따라 모델 동작을 평가하고, 배포 환경을 최적화하며, 필요한 리소스 규모를 산정할 수 있는 일관된 프레임워크를 제공합니다.

## 주요 기능

- **SLO 기반 평가를 위한 지연 시간 및 토큰 단위 통계 수집:** TTFT, ITL 및 end-to-end 성능에 대한 전체 분포를 포함합니다.
- **현실적이고 유연하게 설정 가능한 트래픽 패턴 생성:** 동기식, 동시 실행 및 요청률 기반 모드를 지원하며, 안전한 운영 범위를 파악하기 위한 재현 가능한 Sweep 테스트를 지원합니다.
- **실제 및 합성 멀티모달 데이터셋 지원:** 텍스트, 이미지, 오디오, 비디오 입력을 지원하여 단일 프레임워크에서 조건을 통제한 실험과 실제 운영 환경을 반영한 성능 평가를 수행할 수 있습니다.
- **표준화된 보고서 생성 및 내보내기:** 대시보드, 분석 및 성능 회귀 추적에 활용할 수 있으며, 팀과 워크플로 전반에서 일관성을 유지할 수 있습니다.
- **높은 처리량과 확장성을 갖춘 벤치마킹 지원:** 멀티프로세싱, 스레딩, 비동기 실행을 지원하며, 사용자 맞춤 설정이나 빠른 시작을 위한 유연한 CLI/API를 제공합니다.

## 주요 구성

<div class="grid cards" markdown>

- :material-rocket-launch:{ .lg .middle } 시작하기

  ______________________________________________________________________

  GuideLLM을 설치하고 첫 번째 벤치마크를 설정한 후, 결과를 분석하여 LLM 배포 환경을 최적화합니다.

  [:octicons-arrow-right-24: 시작하기](./getting-started/)

- :material-book-open-variant:{ .lg .middle } 가이드

  ______________________________________________________________________

  효과적인 LLM 벤치마킹을 위한 백엔드, 데이터셋, 평가 지표 및 서비스 수준 목표(SLO)를 자세히 다룹니다.

  [:octicons-arrow-right-24: 가이드 보기(영문)](../guides/)

- :material-code-tags:{ .lg .middle } 사용 예제

  ______________________________________________________________________

  실제 벤치마킹 시나리오와 최적화 기법을 단계별로 설명합니다.

  [:octicons-arrow-right-24: 사용 예제 보기(영문)](../examples/)

- :material-api:{ .lg .middle } API 레퍼런스

  ______________________________________________________________________

  벤치마킹을 기존 워크플로에 통합할 수 있도록 GuideLLM API의 전반적인 내용을 상세히 다룹니다.

  [:octicons-arrow-right-24: API 레퍼런스 보기(영문)](../reference/guidellm/index.md)

</div>
