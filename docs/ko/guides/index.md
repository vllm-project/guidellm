---
weight: -5
source_sha256: !!str "5b8fd781d6bce9a93c0ee7be3c8e94f6fddf968d02beee085ad99e1665d9b944"
---

# 가이드

GuideLLM 가이드 섹션에 오신 것을 환영합니다! 이 섹션에서는 GuideLLM 플랫폼의 핵심 구성 요소와 개념을 다루는 종합적인 문서를 제공합니다. 이 가이드에서는 GuideLLM의 내부 동작 방식, 다양한 구성 요소의 설정 방법 및 벤치마크 결과 해석 방법을 설명하여 LLM 배포 환경을 최적화할 수 있도록 지원합니다.

시스템 아키텍처 이해, 지원되는 백엔드 확인, 데이터셋 설정, 지표 분석 또는 서비스 수준 목표 설정 등 관심 분야에 맞는 상세 정보를 제공하여, LLM 배포 환경에 대해 충분한 정보를 바탕으로 의사 결정을 내릴 수 있도록 돕습니다.

## 주요 가이드

<div class="grid cards" markdown>

- :material-layers-outline:{ .lg .middle } 아키텍처

  ______________________________________________________________________

  GuideLLM의 모듈식 설계와 핵심 구성 요소가 상호작용하여 LLM 배포 환경을 평가하는 방식을 설명합니다.

  [:octicons-arrow-right-24: 아키텍처 개요](architecture.md)

- :material-server:{ .lg .middle } 백엔드

  ______________________________________________________________________

  지원되는 LLM 백엔드와 벤치마킹을 위한 OpenAI 호환 서버 설정 방법을 설명합니다.

  [:octicons-arrow-right-24: 백엔드 가이드](backends.md)

- :material-database:{ .lg .middle } 데이터셋

  ______________________________________________________________________

  합성 데이터, Hugging Face 데이터셋 및 파일 기반 옵션을 포함한 다양한 데이터 소스를 설정하고 벤치마킹에 활용합니다.

  [:octicons-arrow-right-24: 데이터셋 가이드](datasets.md)

- :material-chart-bar:{ .lg .middle } 지표

  ______________________________________________________________________

  지연 시간, 처리량 및 토큰 단위 분석을 포함하여 GuideLLM이 성능 평가를 위해 제공하는 다양한 지표를 살펴봅니다.

  [:octicons-arrow-right-24: 지표 가이드](metrics.md)

- :material-file-export:{ .lg .middle } 출력

  ______________________________________________________________________

  지원되는 출력 형식과 벤치마크 결과 보고서를 사용자 지정하는 방법을 설명합니다.

  [:octicons-arrow-right-24: 출력 가이드](outputs.md)

- :material-alert-outline:{ .lg .middle } 경고

  ______________________________________________________________________

  `--metrics` 옵션, 시나리오 파일 또는 환경 변수에 지표 경로, 임곗값 및 설명을 지정하여 벤치마크 완료 후 표시되는 경고를 설정합니다.

  [:octicons-arrow-right-24: 경고 가이드](warnings.md)

- :material-target:{ .lg .middle } 서비스 수준 목표

  ______________________________________________________________________

  LLM 배포 환경의 신뢰성과 성능을 보장하기 위해 서비스 수준 목표(SLO)와 서비스 수준 협약(SLA)을 정의하고 적용합니다.

  [:octicons-arrow-right-24: SLO 가이드](service_level_objectives.md)

- :material-stop-circle-outline:{ .lg .middle } 과포화 시 중단

  ______________________________________________________________________

  모델의 과포화 상태를 자동으로 감지하고 벤치마크를 중단하여 컴퓨팅 리소스 낭비를 방지하고 결과의 유효성을 보장합니다.

  [:octicons-arrow-right-24: 과포화 시 중단 가이드](over_saturation_stopping.md)

- :material-gauge:{ .lg .middle } 수집기

  ______________________________________________________________________

  벤치마크 실행 중에 서버의 Prometheus 지표와 같은 GuideLLM 외부 데이터를 수집하고 클라이언트 결과와 함께 저장합니다.

  [:octicons-arrow-right-24: 수집기 가이드](collectors.md)

- :material-chart-bell-curve-cumulative:{ .lg .middle } Knee 프로파일

  ______________________________________________________________________

  처리량이 포화되는 지점을 찾고, 필요에 따라 감지된 지점 주변의 여러 동시 요청 수에서 추가로 벤치마크를 실행합니다.

  [:octicons-arrow-right-24: Knee 프로파일 가이드](knee_detection.md)

- :material-wrench:{ .lg .middle } 도구 호출

  ______________________________________________________________________

  사전에 지정한 도구 호출 턴, 합성 데이터 및 데이터셋 기반 도구 정의를 사용하여 멀티턴 도구 호출 워크로드를 벤치마킹합니다.

  [:octicons-arrow-right-24: 도구 호출 가이드](tool_calling.md)

- :material-image-multiple:{ .lg .middle } 멀티모달 벤치마킹

  ______________________________________________________________________

  텍스트+이미지, 비디오 및 오디오 작업을 포함한 멀티모달 모델의 벤치마크를 설정합니다.

  [:octicons-arrow-right-24: 멀티모달 가이드](multimodal/index.md)

- :material-lifebuoy:{ .lg .middle } 문제 해결

  ______________________________________________________________________

  일반적인 오류를 해결하는 방법을 설명합니다.

  [:octicons-arrow-right-24: 문제 해결 가이드](troubleshooting.md)

- :material-update:{ .lg .middle } v0.7.2 마이그레이션 가이드

  ______________________________________________________________________

  GuideLLM CLI 명령어를 v0.6.0에서 v0.7.2로 마이그레이션하는 방법을 설명합니다.

  [:octicons-arrow-right-24: v0.7.2 마이그레이션 가이드](v0.7.0_migration_guide.md)

</div>
