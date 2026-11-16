---
source_sha256: !!str "abebdc1dbd8caee3f7ebb8ce2bf857b1e51ef1a1befabe985a5a8116cf8bc6d8"
---

# Kubernetes / OpenShift 客户端 Job {#kubernetes-openshift-client-job}

将 GuideLLM 作为**仅客户端**的批处理 Job，针对现有的兼容 OpenAI 的 Service（例如集群内的 vLLM）运行。该 Job **无需 GPU**。

本示例使用已发布的 GHCR 镜像。静态基准测试设置通过 `args` 中的 CLI 参数传入；只有通常需要从 ConfigMap 或 Secret 获取的值（端点 URL 和 API 密钥）通过环境变量传递。结果会保存在 PVC 上。

## 为什么使用 Job（而不是 Deployment）

基准测试是运行完成即结束的批处理任务。Deployment 会在任务结束后重新启动 Pod，并无限次重复运行基准测试，产生重复的结果文件和你并未要求的负载。使用 `restartPolicy: Never` 和 `backoffLimit: 0` 的 `Job` 可确保只尝试一次；`activeDeadlineSeconds` 会为卡住的任务设置最长运行时间（例如，基准测试期间服务重启时）。

## 前提条件

- 可访问的兼容 OpenAI 端点（Service DNS 名称或 URL）
- 有权在你的命名空间中创建 `Job` 和 `PersistentVolumeClaim`
- 有权限拉取 `ghcr.io/vllm-project/guidellm` 镜像

固定使用发布标签（`vX.Y.Z`）。在生产环境中优先使用发布标签，而非 `:latest`。

## 快速开始

1. 将下面的 YAML 复制到文件中（例如 `guidellm-job.yaml`）。
2. 替换 `REPLACE_WITH_OPENAI_BASE_URL`（例如：`http://vllm.my-ns.svc.cluster.local:8000`）。
3. 根据需要调整配置档、数据、约束和样本数量。注释中标有 `CHANGE ME` 的参数通常需要由运维人员调整。
4. 应用配置：

```bash
kubectl apply -f guidellm-job.yaml
# OpenShift:
oc apply -f guidellm-job.yaml
```

5. 查看日志。产物会保存在 PVC 的 `/results` 目录下（`benchmarks.json`、`benchmarks.csv`、`benchmarks.html`）：

```bash
kubectl logs -f job/guidellm-openai-bench
kubectl get pvc guidellm-results
```

```yaml
# Client-only GuideLLM benchmark Job against an OpenAI-compatible Service.
#
# Why a Job (not a Deployment): a benchmark is run-to-completion batch work.
# A Deployment would restart the pod and re-run the benchmark indefinitely.
#
# Edit REPLACE_* placeholders before apply.
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: guidellm-results
  labels:
    app.kubernetes.io/name: guidellm
    app.kubernetes.io/component: results
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      # CHANGE ME: size for your result retention needs (json/csv/html per run).
      storage: 5Gi
  # CHANGE ME: uncomment if your cluster has no default StorageClass.
  # storageClassName: <your-storage-class>
---
apiVersion: batch/v1
kind: Job
metadata:
  name: guidellm-openai-bench
  labels:
    app.kubernetes.io/name: guidellm
    app.kubernetes.io/component: client-bench
spec:
  # Benchmarks should not retry on failure — a retried run produces misleading numbers.
  backoffLimit: 0
  # Hard wall-clock kill switch so a hung run (e.g. serve restart mid-benchmark)
  # cannot hold the Job forever. CHANGE ME: set above your max_duration constraint.
  activeDeadlineSeconds: 7200
  # Auto-cleanup of finished Jobs after 24h. Results persist on the PVC.
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels:
        app.kubernetes.io/name: guidellm
        app.kubernetes.io/component: client-bench
      annotations:
        # Client Jobs usually should not get a service mesh sidecar: the sidecar
        # adds latency to the measurements and can keep the Job from completing.
        # Remove if you intentionally want mesh mTLS on the benchmark path.
        sidecar.istio.io/inject: "false"
    spec:
      restartPolicy: Never
      # The benchmark client needs no Kubernetes API access; do not mount a token.
      automountServiceAccountToken: false
      # CHANGE ME (optional): pin to CPU nodes so the client never lands on GPU pools.
      # nodeSelector:
      #   node.kubernetes.io/instance-type: m6i.xlarge
      securityContext:
        runAsNonRoot: true
        # OpenShift: leave runAsUser AND fsGroup unset. The restricted-v2 SCC
        # assigns both from the namespace's allocated ranges; an explicit
        # fsGroup: 0 is outside that range and the Pod is rejected at admission.
        # Vanilla Kubernetes: uncomment both. fsGroup is what makes the results
        # PVC writable by the non-root user (PVC mounts default to root:root).
        # runAsUser: 1001
        # fsGroup: 1001
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: guidellm
          # Pin a release tag — multi-arch (amd64+arm64) from v0.7.0 onward.
          # CHANGE ME on upgrades. For immutable prod deploys, pin the digest instead:
          #   image: ghcr.io/vllm-project/guidellm@sha256:<digest>
          # (resolve with: docker buildx imagetools inspect ghcr.io/vllm-project/guidellm:vX.Y.Z)
          # Avoid :latest / :stable in prod — they move between releases.
          image: ghcr.io/vllm-project/guidellm:v0.7.3
          imagePullPolicy: IfNotPresent
          args:
            - run
            - --backend
            - kind=openai_http
            # CHANGE ME: keep streams <= the server's max concurrent sequences
            # (e.g. vLLM --max-num-seqs), otherwise you measure queueing, not decode.
            - --profile
            - kind=concurrent,streams=4
            - --constraint
            - kind=max_requests,count=100
            - --constraint
            - kind=max_duration,seconds=600
            # CHANGE ME: synthetic shapes shown; swap for huggingface/file sources as needed.
            - --data
            - kind=synthetic_text,prompt_tokens=256,output_tokens=128
            # Uncomment when using a Hugging Face / file dataset to clip how many
            # rows are loaded (samples=-1 loads all rows). Unnecessary with
            # synthetic_text, which generates exactly what the run consumes.
            # - --data-loader
            # - kind=pytorch,samples=100
            # The image already defaults results to /results, so these paths only
            # need to be explicit because html is not one of the default outputs
            # (defaults are json + csv).
            - --output
            - kind=json,path=/results/benchmarks.json
            - --output
            - kind=csv,path=/results/benchmarks.csv
            - --output
            - kind=html,path=/results/benchmarks.html
          # Only settings you may want to inject from a ConfigMap or Secret are env
          # vars. GUIDELLM__SPEC__* nests on __, so BACKEND__TARGET sets exactly the
          # one backend field and leaves the rest to the args above.
          env:
            # CHANGE ME: your in-cluster (or external) OpenAI-compatible endpoint,
            # e.g. http://vllm.my-ns.svc.cluster.local:8000
            - name: GUIDELLM__SPEC__BACKEND__TARGET
              value: REPLACE_WITH_OPENAI_BASE_URL
              # ...or read it from a ConfigMap instead of hardcoding:
              # valueFrom:
              #   configMapKeyRef:
              #     name: guidellm-endpoint
              #     key: target
            # Uncomment for authenticated endpoints; keep the key in a Secret.
            # - name: GUIDELLM__SPEC__BACKEND__API_KEY
            #   valueFrom:
            #     secretKeyRef:
            #       name: guidellm-backend
            #       key: api-key
          resources:
            # CHANGE ME: the client is CPU/memory bound only; never request GPUs here.
            # requests == limits puts the pod in Guaranteed QoS and gives it CPU
            # isolation, so noisy neighbors do not show up as client-side latency.
            requests:
              cpu: "2"
              memory: 4Gi
            limits:
              cpu: "2"
              memory: 4Gi
          securityContext:
            allowPrivilegeEscalation: false
            # Root FS is immutable; writable paths are explicit mounts below.
            readOnlyRootFilesystem: true
            capabilities:
              drop:
                - ALL
          volumeMounts:
            - name: results
              mountPath: /results
            # Writable scratch for tokenizer/model caches under HOME.
            - name: home
              mountPath: /home/guidellm
            - name: tmp
              mountPath: /tmp
      volumes:
        - name: results
          persistentVolumeClaim:
            claimName: guidellm-results
        - name: home
          emptyDir:
            sizeLimit: 2Gi
        - name: tmp
          emptyDir:
            sizeLimit: 1Gi
```

## 固定镜像版本和升级

- 本示例固定使用 `vX.Y.Z`。从 `v0.7.0` 起，发布标签不可变，并支持多架构（amd64 + arm64）。
- 如需严格保证生产环境中的镜像不可变，请固定使用摘要：`ghcr.io/vllm-project/guidellm@sha256:<digest>`（通过 `docker buildx imagetools inspect ghcr.io/vllm-project/guidellm:vX.Y.Z` 查询）。即使标签被重新推送，摘要也不会改变。
- 在生产环境中避免使用 `:latest` / `:stable`：这些标签会随发布版本移动，导致不同运行的结果无法比较，镜像拉取也无法复现。
- 升级时，更新标签或摘要，重新运行一个已知的工作负载测试场景，并与先前版本的结果进行比较，然后再采用新数据。

## 安全配置

本示例符合受限 Pod Security Standards 和 `restricted-v2` SCC 的要求：

| 控制项         | 设置                                                                                                  |
| -------------- | ----------------------------------------------------------------------------------------------------- |
| 非 root 用户   | `runAsNonRoot: true`，无需特权 SCC                                                                    |
| 功能权限       | `drop: [ALL]`、`allowPrivilegeEscalation: false`                                                      |
| 根文件系统     | `readOnlyRootFilesystem: true`；可写路径为显式挂载的 `emptyDir`（`/home/guidellm`、`/tmp`）和结果 PVC |
| Seccomp        | `RuntimeDefault`                                                                                      |
| ServiceAccount | `automountServiceAccountToken: false`；客户端无需访问 Kubernetes API                                  |
| 主机访问       | 未设置 `hostNetwork`、`hostPID` 或主机挂载                                                            |
| GPU            | 不请求 GPU；GPU 应分配给提供服务的 Deployment，而不是基准测试客户端                                   |

根据环境需要，还可以进一步加固：

- **NetworkPolicy**：将此 Job 的出站流量限制为仅访问推理 Service（以及在运行时拉取镜像时访问镜像仓库）。
- **需要身份验证的端点**：不要将 API 密钥直接写入 `args`。通过 `valueFrom.secretKeyRef` 从 `Secret` 设置 `GUIDELLM__SPEC__BACKEND__API_KEY`（上方清单中已注释示例）。
- **配额**：Job 设置了 CPU 和内存请求及限制，以符合启用了 `ResourceQuota` / `LimitRange` 的命名空间要求。

## OpenShift 注意事项

- GuideLLM 镜像以 UID `1001` 和 root 组（`1001:0`）运行，且主目录允许组写入，因此 SCC 范围内分配的任意 UID 都可直接使用。
- 在 OpenShift 上，将 `runAsUser` 和 `fsGroup` 都留空。`restricted-v2` SCC 会从命名空间分配的范围中设置这两个值，并使用分配的组挂载 PVC 和 `emptyDir` 卷，因此无需手动设置即可写入。
- 在 OpenShift 上**不要**硬编码 `fsGroup: 0`。`restricted-v2` 的 fsGroup 策略为 `MustRunAs`，使用命名空间 `openshift.io/sa.scc.supplemental-groups` 注解中的范围（通常从 `1000700000` 左右开始），因此准入检查会因组 `0` 被拒绝，并报错 `fsGroup: Invalid value: []int64{0}: 0 is not an allowed group`。
- 在原生 Kubernetes 上，取消注释 `runAsUser: 1001` 和 `fsGroup: 1001`。PVC 挂载默认属于 `root:root`，因此需要设置 `fsGroup` 才能让结果 PVC 可写。
- Istio：本示例设置了 `sidecar.istio.io/inject: "false"`。客户端中的 sidecar 会增加测量延迟，也可能导致 Job 无法结束。

## 自定义参数速查表

| 参数（在 `args` 中）                               | 常见用途                                                  |
| -------------------------------------------------- | --------------------------------------------------------- |
| `--backend kind=openai_http`                       | 后端类型；URL 由 `GUIDELLM__SPEC__BACKEND__TARGET` 提供   |
| `--profile kind=concurrent,streams=4`              | `concurrent` / `constant` / `sweep` / …                   |
| `--constraint kind=max_duration,seconds=600`       | `max_duration` 和/或 `max_requests`，可重复指定           |
| `--data kind=synthetic_text,prompt_tokens=256`     | `synthetic_text` 或 Hugging Face / 文件数据源，可重复指定 |
| `--data-loader kind=pytorch,samples=200`           | 限制加载的数据集行数                                      |
| `--output kind=json,path=/results/benchmarks.json` | `json` / `csv` / `html` / `yaml` / `console`，可重复指定  |

| 环境变量                           | 常见用途                     |
| ---------------------------------- | ---------------------------- |
| `GUIDELLM__SPEC__BACKEND__TARGET`  | 端点 URL，通常来自 ConfigMap |
| `GUIDELLM__SPEC__BACKEND__API_KEY` | 端点凭据，来自 Secret        |
| `GUIDELLM__DEFAULT_RESULTS_DIR`    | 在镜像中默认为 `/results`    |

每个参数都有对应的环境变量：`GUIDELLM__SPEC__<FIELD>`，层级之间用 `__` 分隔（例如 `--backend kind=...,target=...` 对应 `GUIDELLM__SPEC__BACKEND__KIND` 和 `GUIDELLM__SPEC__BACKEND__TARGET`）。静态设置优先使用 `args`，从 ConfigMap 或 Secret 注入的值优先使用环境变量。运行 `guidellm run --help` 可查看所有可用类型。

对于较大的 Hugging Face 数据集，可在数据加载器中使用 `samples` 限制行数。使用 `max_requests` 作为**运行时**停止条件；它不会缩小已加载的数据集。

## 相关内容

- 容器标签和支持的平台：参阅项目 [README](https://github.com/vllm-project/guidellm#install-guidellm)
