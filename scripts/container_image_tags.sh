#!/usr/bin/env bash
# Helpers for selecting and validating GuideLLM GHCR release tags.
# Used by .github/workflows/container-maintenance.yml and local tests.
#
# Architecture-specific tags (e.g. v0.7.0-arm64) must never be chosen for
# latest/stable — those are single-arch images. Copying them to :latest causes
# Exec format error on the other architecture (seen on OpenShift amd64).

set -euo pipefail

# Filter out platform-specific image tags produced by the multi-arch build.
# Keeps multi-arch manifests (vX.Y.Z) and pre-releases (vX.Y.Z-rc1), drops
# vX.Y.Z-amd64 / vX.Y.Z-arm64 / nightly-amd64 / etc.
filter_arch_tags() {
    grep -Ev -- '-(amd64|arm64)$' "$@"
}

# Newest exact semver tag: vMAJOR.MINOR.PATCH
select_stable_tag() {
    local tags_file="${1:?tags file required}"
    # grep exits 1 on no match; keep the pipeline successful so callers can
    # treat an empty result as "unresolved" instead of aborting under pipefail.
    filter_arch_tags "${tags_file}" \
        | { grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' || true; } \
        | sort -rV \
        | head -n1
}

# Newest version-like tag, including pre-releases (rc/a/b/dev), excluding arch
# suffixes. Same prefix match as the historical workflow, after arch filtering.
select_latest_tag() {
    local tags_file="${1:?tags file required}"
    filter_arch_tags "${tags_file}" \
        | { grep -E '^v[0-9]+\.[0-9]+\.[0-9]+' || true; } \
        | sort -rV \
        | head -n1
}

# Require a transport ref (e.g. docker://ghcr.io/org/name:tag) to be a
# multi-arch index/list that includes both linux/amd64 and linux/arm64.
assert_multiarch_ref() {
    local ref="${1:?image ref required}"
    local raw media_type arches

    if ! command -v skopeo >/dev/null 2>&1; then
        echo "assert_multiarch_ref: skopeo is required" >&2
        return 1
    fi
    if ! command -v jq >/dev/null 2>&1; then
        echo "assert_multiarch_ref: jq is required" >&2
        return 1
    fi

    raw="$(skopeo inspect --raw "${ref}")"
    media_type="$(echo "${raw}" | jq -r '.mediaType // empty')"

    case "${media_type}" in
        *manifest.list*|*image.index*)
            ;;
        *)
            echo "assert_multiarch_ref: ${ref} is not a multi-arch manifest (mediaType=${media_type:-missing})" >&2
            return 1
            ;;
    esac

    arches="$(echo "${raw}" | jq -r '.manifests[]?.platform.architecture // empty' | sort -u)"
    if ! echo "${arches}" | grep -qx 'amd64'; then
        echo "assert_multiarch_ref: ${ref} missing linux/amd64 (found: ${arches//$'\n'/, })" >&2
        return 1
    fi
    if ! echo "${arches}" | grep -qx 'arm64'; then
        echo "assert_multiarch_ref: ${ref} missing linux/arm64 (found: ${arches//$'\n'/, })" >&2
        return 1
    fi

    echo "assert_multiarch_ref: ${ref} ok (amd64+arm64)"
}
