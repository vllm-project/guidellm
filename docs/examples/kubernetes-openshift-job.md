# Kubernetes / OpenShift client Job

Run GuideLLM as a **client-only** batch Job against an existing OpenAI-compatible Service (for example in-cluster vLLM). The Job does **not** need a GPU.

This example uses the published GHCR image. Static benchmark settings are passed as CLI arguments in `args`; only the values you are likely to source from a ConfigMap or Secret — the endpoint URL and API key — are environment variables. Results persist on a PVC.

## Why a Job (not a Deployment)

A benchmark is run-to-completion batch work. A Deployment would restart the pod after the run finishes and re-execute the benchmark indefinitely, producing duplicate result files and load you did not ask for. `Job` with `restartPolicy: Never` and `backoffLimit: 0` guarantees exactly one attempt; `activeDeadlineSeconds` caps a hung run (for example when the serve restarts mid-benchmark).

## Prerequisites

- A reachable OpenAI-compatible endpoint (Service DNS or URL)
- Cluster permission to create a `Job` and a `PersistentVolumeClaim` in your namespace
- Image pull access to `ghcr.io/vllm-project/guidellm`

Pin a release tag (`vX.Y.Z`). Prefer that over `:latest` in production.

## Quick start

1. Copy the YAML below into a file (for example `guidellm-job.yaml`).
2. Replace `REPLACE_WITH_OPENAI_BASE_URL` (example: `http://vllm.my-ns.svc.cluster.local:8000`).
3. Adjust profile / data / constraints / sample count as needed. Knobs marked `CHANGE ME` in comments are the ones operators typically retune.
4. Apply:

```bash
kubectl apply -f guidellm-job.yaml
# OpenShift:
oc apply -f guidellm-job.yaml
```

5. Follow logs. Artifacts land on the PVC at `/results` (`benchmarks.json`, `benchmarks.csv`, `benchmarks.html`):

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

## Image pinning and upgrades

- The example pins `vX.Y.Z`. Release tags are immutable and multi-arch (amd64 + arm64) from `v0.7.0` onward.
- For strict prod immutability, pin the digest: `ghcr.io/vllm-project/guidellm@sha256:<digest>` (resolve with `docker buildx imagetools inspect ghcr.io/vllm-project/guidellm:vX.Y.Z`). A digest never changes even if a tag is re-pushed.
- Avoid `:latest` / `:stable` in production — they move between releases, so results across runs would not be comparable and pulls are not reproducible.
- On upgrade: bump the tag/digest, re-run one known workload cell, and compare against the previous version's results before trusting new numbers.

## Security posture

The example is written to pass restricted Pod Security Standards / restricted-v2 SCC:

| Control         | Setting                                                                                                                        |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Non-root        | `runAsNonRoot: true`, no privileged SCC needed                                                                                 |
| Capabilities    | `drop: [ALL]`, `allowPrivilegeEscalation: false`                                                                               |
| Root filesystem | `readOnlyRootFilesystem: true` — writable paths are explicit `emptyDir` mounts (`/home/guidellm`, `/tmp`) plus the results PVC |
| Seccomp         | `RuntimeDefault`                                                                                                               |
| Service account | `automountServiceAccountToken: false` — the client needs no Kubernetes API access                                              |
| Host access     | No `hostNetwork`, `hostPID`, or host mounts                                                                                    |
| GPUs            | None requested — GPUs belong to the serve Deployment, not the benchmark client                                                 |

Additional hardening you may want per your environment:

- **NetworkPolicy**: restrict egress from this Job to only the inference Service (and image registry if pulls happen at runtime).
- **Authenticated endpoints**: don't inline API keys in `args`. Set `GUIDELLM__SPEC__BACKEND__API_KEY` from a `Secret` via `valueFrom.secretKeyRef` (commented out in the manifest above).
- **Quotas**: the Job sets CPU/memory requests and limits so it fits namespaces with `ResourceQuota` / `LimitRange` enforcement.

## OpenShift notes

- The GuideLLM image runs as UID `1001` with root group (`1001:0`) and group-writable home, so arbitrary UIDs from an SCC range work out of the box.
- Leave both `runAsUser` and `fsGroup` unset on OpenShift. The `restricted-v2` SCC assigns them from the namespace's allocated ranges, and it mounts the PVC and `emptyDir` volumes with the assigned group — so they are writable without you specifying anything.
- Do **not** hardcode `fsGroup: 0` on OpenShift. The `restricted-v2` fsGroup strategy is `MustRunAs` over the range in the namespace's `openshift.io/sa.scc.supplemental-groups` annotation (typically starting around `1000700000`), so group `0` is rejected at admission with `fsGroup: Invalid value: []int64{0}: 0 is not an allowed group`.
- On vanilla Kubernetes, uncomment `runAsUser: 1001` and `fsGroup: 1001`. There the `fsGroup` is what makes the results PVC writable, since PVC mounts default to `root:root`.
- Istio: the example sets `sidecar.istio.io/inject: "false"`. A sidecar on the client adds latency to the measurements and can keep the Job from completing.

## Customization cheat sheet

| Flag (in `args`)                                   | Typical use                                                        |
| -------------------------------------------------- | ------------------------------------------------------------------ |
| `--backend kind=openai_http`                       | Backend type; the URL comes from `GUIDELLM__SPEC__BACKEND__TARGET` |
| `--profile kind=concurrent,streams=4`              | `concurrent` / `constant` / `sweep` / …                            |
| `--constraint kind=max_duration,seconds=600`       | `max_duration` and/or `max_requests` [repeatable]                  |
| `--data kind=synthetic_text,prompt_tokens=256`     | `synthetic_text` or Hugging Face / file sources [repeatable]       |
| `--data-loader kind=pytorch,samples=200`           | Clip how many dataset rows are loaded                              |
| `--output kind=json,path=/results/benchmarks.json` | `json` / `csv` / `html` / `yaml` / `console` [repeatable]          |

| Env var                            | Typical use                           |
| ---------------------------------- | ------------------------------------- |
| `GUIDELLM__SPEC__BACKEND__TARGET`  | Endpoint URL — often from a ConfigMap |
| `GUIDELLM__SPEC__BACKEND__API_KEY` | Endpoint credential — from a Secret   |
| `GUIDELLM__DEFAULT_RESULTS_DIR`    | Defaults to `/results` in the image   |

Every flag has an environment-variable equivalent: `GUIDELLM__SPEC__<FIELD>`, nesting on `__` (so `--backend kind=...,target=...` is `GUIDELLM__SPEC__BACKEND__KIND` and `GUIDELLM__SPEC__BACKEND__TARGET`). Prefer `args` for static settings and env vars for anything injected from a ConfigMap or Secret. Run `guidellm run --help` for the full list of kinds.

Clip large HF datasets with `samples` on the data loader (row limit). Use `max_requests` as a **runtime** stop — it does not shrink the loaded dataset.

## Related

- Container tags / platforms: see the project [README](https://github.com/vllm-project/guidellm#install-guidellm)
