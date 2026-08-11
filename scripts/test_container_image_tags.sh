#!/usr/bin/env bash
# Local unit + live GHCR checks for container tag selection.
# Run: ./scripts/test_container_image_tags.sh
#
# ## WRITTEN BY AI ##

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/container_image_tags.sh
source "${ROOT_DIR}/scripts/container_image_tags.sh"

PASS=0
FAIL=0

assert_eq() {
    local name="$1"
    local expected="$2"
    local actual="$3"
    if [[ "${expected}" == "${actual}" ]]; then
        echo "PASS: ${name} (got ${actual})"
        PASS=$((PASS + 1))
    else
        echo "FAIL: ${name} (expected ${expected}, got ${actual})" >&2
        FAIL=$((FAIL + 1))
    fi
}

assert_empty() {
    local name="$1"
    local actual="$2"
    if [[ -z "${actual}" ]]; then
        echo "PASS: ${name} (empty as expected)"
        PASS=$((PASS + 1))
    else
        echo "FAIL: ${name} (expected empty, got ${actual})" >&2
        FAIL=$((FAIL + 1))
    fi
}

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

echo "== unit: reproduces the production bug =="
cat >"${tmpdir}/prod_like.tags" <<'EOF'
v0.5.0
v0.5.1
v0.6.0
v0.6.1
v0.7.0
v0.7.0-amd64
v0.7.0-arm64
nightly
nightly-amd64
nightly-arm64
pr-123
pr-123-amd64
pr-123-arm64
EOF

# Historical buggy selection (what container-maintenance.yml did on Ubuntu).
buggy_latest="$(grep -E '^v[0-9]+\.[0-9]+\.[0-9]+' "${tmpdir}/prod_like.tags" | sort -rV | head -n1)"
assert_eq "buggy latest selects arm64 single-arch" "v0.7.0-arm64" "${buggy_latest}"

fixed_stable="$(select_stable_tag "${tmpdir}/prod_like.tags")"
fixed_latest="$(select_latest_tag "${tmpdir}/prod_like.tags")"
assert_eq "fixed stable prefers multi-arch release" "v0.7.0" "${fixed_stable}"
assert_eq "fixed latest prefers multi-arch release" "v0.7.0" "${fixed_latest}"

echo "== unit: pre-release preference for latest =="
cat >"${tmpdir}/with_rc.tags" <<'EOF'
v0.7.0
v0.7.0-amd64
v0.7.0-arm64
v0.8.0-rc1
v0.8.0-rc1-amd64
v0.8.0-rc1-arm64
EOF
assert_eq "stable ignores rc" "v0.7.0" "$(select_stable_tag "${tmpdir}/with_rc.tags")"
assert_eq "latest prefers rc over older release" "v0.8.0-rc1" "$(select_latest_tag "${tmpdir}/with_rc.tags")"

echo "== unit: no false positives from arch-only tag sets =="
cat >"${tmpdir}/arch_only.tags" <<'EOF'
v0.7.0-amd64
v0.7.0-arm64
nightly-amd64
EOF
assert_empty "stable empty when only arch tags" "$(select_stable_tag "${tmpdir}/arch_only.tags" || true)"
assert_empty "latest empty when only arch tags" "$(select_latest_tag "${tmpdir}/arch_only.tags" || true)"

echo "== live: GHCR evidence (read-only) =="
if command -v docker >/dev/null 2>&1; then
    inspect_platforms() {
        local tag="$1"
        docker buildx imagetools inspect "ghcr.io/vllm-project/guidellm:${tag}" --raw 2>/dev/null \
            | python3 -c '
import json, sys
data = json.load(sys.stdin)
media = data.get("mediaType", "")
if "manifests" in data:
    arches = sorted({m.get("platform", {}).get("architecture") for m in data["manifests"] if m.get("platform")})
    print(media, ",".join(a for a in arches if a))
else:
    # single-arch image: architecture is in config, not raw manifest
    print(media, "single-arch")
'
    }

    latest_info="$(inspect_platforms latest || echo "error")"
    v070_info="$(inspect_platforms v0.7.0 || echo "error")"
    nightly_info="$(inspect_platforms nightly || echo "error")"

    echo "live latest:  ${latest_info}"
    echo "live v0.7.0:  ${v070_info}"
    echo "live nightly: ${nightly_info}"

    if [[ "${v070_info}" == *"amd64"* && "${v070_info}" == *"arm64"* ]]; then
        echo "PASS: v0.7.0 is multi-arch (correct source for retag)"
        PASS=$((PASS + 1))
    else
        echo "FAIL: v0.7.0 is not multi-arch (${v070_info})" >&2
        FAIL=$((FAIL + 1))
    fi

    if [[ "${nightly_info}" == *"amd64"* && "${nightly_info}" == *"arm64"* ]]; then
        echo "PASS: nightly is multi-arch"
        PASS=$((PASS + 1))
    else
        echo "FAIL: nightly is not multi-arch (${nightly_info})" >&2
        FAIL=$((FAIL + 1))
    fi

    if [[ "${latest_info}" == *single-arch* ]]; then
        echo "PASS: current :latest is still single-arch (bug present upstream; our fix is needed)"
        PASS=$((PASS + 1))
    elif [[ "${latest_info}" == *"amd64"* && "${latest_info}" == *"arm64"* ]]; then
        echo "INFO: :latest is already multi-arch on GHCR (retag may have been healed); selection fix still required"
        PASS=$((PASS + 1))
    else
        echo "FAIL: unexpected latest inspect result: ${latest_info}" >&2
        FAIL=$((FAIL + 1))
    fi
else
    echo "SKIP live GHCR checks (docker not available)"
fi

echo
echo "Results: ${PASS} passed, ${FAIL} failed"
if [[ "${FAIL}" -ne 0 ]]; then
    exit 1
fi
