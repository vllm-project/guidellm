#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 <base-ref> <head-sha>

Format a PR description by appending a commits summary section.

The current PR body is read from stdin. The output (written to stdout)
preserves everything above the marker comment and replaces everything
below it with a generated commits section and de-duplicated trailers.

Arguments:
  base-ref    Git ref for the PR base branch (e.g. origin/main)
  head-sha    Commit SHA of the PR head

Example:
  gh pr view 42 --json body -q .body \\
    | $0 origin/main abc1234 \\
    | gh pr edit 42 --body-file -
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

MARKER="<!-- begin:squash-data -->"

BASE_REF="${1:?$(usage >&2; echo "error: missing <base-ref>")}"
HEAD_SHA="${2:?$(usage >&2; echo "error: missing <head-sha>")}"

CURRENT_BODY="$(cat | tr -d '\r')"

MARKER_LINE="$(printf '%s\n' "$CURRENT_BODY" | grep -nF "$MARKER" | head -1 | cut -d: -f1)" || true
if [ -n "$MARKER_LINE" ]; then
    USER_CONTENT="$(printf '%s\n' "$CURRENT_BODY" | head -n "$((MARKER_LINE - 1))")"
    USER_CONTENT="$(printf '%s' "$USER_CONTENT" | sed -e :a -e '/^[[:space:]]*$/{ $d; N; ba; }')"
else
    USER_CONTENT="$CURRENT_BODY"
fi

MERGE_BASE="$(git merge-base "$BASE_REF" "$HEAD_SHA")"

COMMIT_SHAS="$(git log --reverse --first-parent --no-merges --format="%H" "$MERGE_BASE".."$HEAD_SHA")" || true

if [ -z "$COMMIT_SHAS" ]; then
    printf '%s\n\n%s\n' "$USER_CONTENT" "$MARKER"
    exit 0
fi

COMMITS_SECTION="$(git log --reverse --first-parent --no-merges --format=medium --no-decorate "$MERGE_BASE".."$HEAD_SHA")"

ALL_TRAILERS=""
while IFS= read -r sha; do
    trailers="$(git log --format="%B" -1 "$sha" | git interpret-trailers --parse 2>/dev/null)" || true
    if [ -n "$trailers" ]; then
        if [ -n "$ALL_TRAILERS" ]; then
            ALL_TRAILERS="$(printf '%s\n%s' "$ALL_TRAILERS" "$trailers")"
        else
            ALL_TRAILERS="$trailers"
        fi
    fi
done <<< "$COMMIT_SHAS"

SIGNOFF_TRAILERS=""
OTHER_TRAILERS=""
if [ -n "$ALL_TRAILERS" ]; then
    OTHER_TRAILERS="$(printf '%s\n' "$ALL_TRAILERS" | grep -v '^Signed-off-by:' | sort -u)" || true
    # Preserve original order for sign-offs, just deduplicate
    SIGNOFF_TRAILERS="$(printf '%s\n' "$ALL_TRAILERS" | grep '^Signed-off-by:' | awk '!seen[$0]++')" || true
fi

UNIQUE_TRAILERS=""
if [ -n "$OTHER_TRAILERS" ] && [ -n "$SIGNOFF_TRAILERS" ]; then
    UNIQUE_TRAILERS="$(printf '%s\n%s' "$OTHER_TRAILERS" "$SIGNOFF_TRAILERS")"
elif [ -n "$OTHER_TRAILERS" ]; then
    UNIQUE_TRAILERS="$OTHER_TRAILERS"
elif [ -n "$SIGNOFF_TRAILERS" ]; then
    UNIQUE_TRAILERS="$SIGNOFF_TRAILERS"
fi

{
    printf '%s\n\n%s\n' "$USER_CONTENT" "$MARKER"
    printf '\n---\n'
    printf '\n# git log\n\n'
    printf '%s\n' "$COMMITS_SECTION"
    if [ -n "$UNIQUE_TRAILERS" ]; then
        printf '\n---------\n\n'
        printf '%s\n' "$UNIQUE_TRAILERS"
    fi
}
