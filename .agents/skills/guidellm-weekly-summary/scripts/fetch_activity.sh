#!/usr/bin/env bash
# Fetch GuideLLM GitHub PR/issue/release activity for weekly summary generation.
# Requires: gh, jq, date. Network access to GitHub.
set -euo pipefail

REPO="vllm-project/guidellm"
DAYS=7
SINCE=""
UNTIL=""
LIMIT=100
BODY_MAX=800
INCLUDE_PRERELEASES=0

usage() {
  cat <<'EOF'
Usage: scripts/fetch_activity.sh [OPTIONS]

Fetch pull requests, issues, and releases for a date window as compact JSON on stdout.
Diagnostics go to stderr. Designed for agent use with the guidellm-weekly-summary skill.

Options:
  --repo OWNER/NAME        GitHub repository (default: vllm-project/guidellm)
  --days N                 Rolling window ending today (default: 7). Ignored if --since set.
  --since YYYY-MM-DD       Window start (inclusive). Default: today minus --days.
  --until YYYY-MM-DD       Window end (inclusive). Default: today.
  --limit N                Max results per PR/issue search (default: 100)
  --body-max N             Max characters of body text to keep per item (default: 800)
  --include-prereleases    Include prerelease GitHub releases (excluded by default)
  -h, --help               Show this help

Examples:
  scripts/fetch_activity.sh
  scripts/fetch_activity.sh --days 14
  scripts/fetch_activity.sh --since 2026-07-16 --until 2026-07-23

Exit codes:
  0  Success
  1  Invalid arguments or missing dependency
  2  GitHub fetch failure (auth/network/API)
EOF
}

die() {
  printf 'Error: %s\n' "$1" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

iso_today() {
  date "+%Y-%m-%d"
}

# Portable: GNU date -d, else BSD date -v
iso_days_ago() {
  local n="$1"
  if date -d "${n} days ago" "+%Y-%m-%d" >/dev/null 2>&1; then
    date -d "${n} days ago" "+%Y-%m-%d"
  elif date -v-"${n}"d "+%Y-%m-%d" >/dev/null 2>&1; then
    date -v-"${n}"d "+%Y-%m-%d"
  else
    die "Unable to compute date ${n} days ago (need GNU or BSD date)"
  fi
}

validate_date() {
  local value="$1"
  local label="$2"
  [[ "$value" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || die "${label} must be YYYY-MM-DD (got: ${value})"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      [[ $# -ge 2 ]] || die "--repo requires a value"
      REPO="$2"
      shift 2
      ;;
    --days)
      [[ $# -ge 2 ]] || die "--days requires a value"
      DAYS="$2"
      shift 2
      ;;
    --since)
      [[ $# -ge 2 ]] || die "--since requires a value"
      SINCE="$2"
      shift 2
      ;;
    --until)
      [[ $# -ge 2 ]] || die "--until requires a value"
      UNTIL="$2"
      shift 2
      ;;
    --limit)
      [[ $# -ge 2 ]] || die "--limit requires a value"
      LIMIT="$2"
      shift 2
      ;;
    --body-max)
      [[ $# -ge 2 ]] || die "--body-max requires a value"
      BODY_MAX="$2"
      shift 2
      ;;
    --include-prereleases)
      INCLUDE_PRERELEASES=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1 (try --help)"
      ;;
  esac
done

require_cmd gh
require_cmd jq
require_cmd date

[[ "$DAYS" =~ ^[0-9]+$ ]] || die "--days must be a non-negative integer"
[[ "$LIMIT" =~ ^[0-9]+$ ]] || die "--limit must be a non-negative integer"
[[ "$BODY_MAX" =~ ^[0-9]+$ ]] || die "--body-max must be a non-negative integer"
[[ "$REPO" == */* ]] || die "--repo must look like OWNER/NAME"

if [[ -z "$UNTIL" ]]; then
  UNTIL="$(iso_today)"
fi
validate_date "$UNTIL" "--until"

if [[ -z "$SINCE" ]]; then
  SINCE="$(iso_days_ago "$DAYS")"
fi
validate_date "$SINCE" "--since"

if [[ "$SINCE" > "$UNTIL" ]]; then
  die "--since (${SINCE}) must be on or before --until (${UNTIL})"
fi

# Search is inclusive on the --updated lower bound; end date is informational for the agent.
PR_FIELDS="number,title,url,state,isDraft,author,labels,createdAt,updatedAt,closedAt,commentsCount,body"
ISSUE_FIELDS="number,title,url,state,isPullRequest,author,labels,createdAt,updatedAt,closedAt,commentsCount,body"

printf 'Fetching activity for %s from %s through %s...\n' "$REPO" "$SINCE" "$UNTIL" >&2

prs_raw="$(
  gh search prs \
    --repo "$REPO" \
    --updated ">=${SINCE}" \
    --limit "$LIMIT" \
    --json "$PR_FIELDS"
)" || {
  printf 'Error: failed to search pull requests (check gh auth and network)\n' >&2
  exit 2
}

issues_raw="$(
  gh search issues \
    --repo "$REPO" \
    --updated ">=${SINCE}" \
    --limit "$LIMIT" \
    --json "$ISSUE_FIELDS"
)" || {
  printf 'Error: failed to search issues (check gh auth and network)\n' >&2
  exit 2
}

releases_raw="$(
  gh api "repos/${REPO}/releases?per_page=30"
)" || {
  printf 'Error: failed to list releases (check gh auth and network)\n' >&2
  exit 2
}

fetched_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date '+%Y-%m-%dT%H:%M:%SZ')"

jq -n \
  --arg repo "$REPO" \
  --arg since "$SINCE" \
  --arg until "$UNTIL" \
  --argjson days "$DAYS" \
  --arg fetched_at "$fetched_at" \
  --argjson body_max "$BODY_MAX" \
  --argjson include_prereleases "$INCLUDE_PRERELEASES" \
  --argjson prs "$prs_raw" \
  --argjson issues "$issues_raw" \
  --argjson releases "$releases_raw" \
  '
  def null_epoch:
    if . == null or . == "" or . == "0001-01-01T00:00:00Z" then null else . end;

  def clean_body:
    (. // "")
    | gsub("(?s)<!-- begin:squash-data -->.*"; "")
    | gsub("(?s)<!--.*?-->"; "")
    | gsub("(?s)\n## Testing\\b.*"; "")
    | gsub("(?m)^Signed-off-by:.*\n?"; "")
    | gsub("\r"; "")
    | gsub("\n{3,}"; "\n\n")
    | sub("^\\s+"; "")
    | sub("\\s+$"; "")
    | if ($body_max > 0) and (length > $body_max)
      then .[0:$body_max] + "\n…[truncated]"
      else .
      end;

  def clean_release_text:
    (. // "")
    | gsub("\r"; "")
    | gsub("(?s)```.*?```"; "")
    | gsub("(?s)<!--.*?-->"; "")
    | gsub("\n{3,}"; "\n\n")
    | sub("^\\s+"; "")
    | sub("\\s+$"; "");

  def release_overview:
    (. // "")
    | clean_release_text
    | if test("(?s)##\\s*Overview\\b") then
        (capture("(?s)##\\s*Overview\\b\\s*(?<o>.*?)(?:\\n##\\s|$)").o | clean_release_text)
      else
        .
      end
    | split("\n\n")[0]
    | if ($body_max > 0) and (length > $body_max)
      then .[0:$body_max] + "…[truncated]"
      else .
      end;

  def simplify_item:
    {
      number: .number,
      title: .title,
      url: .url,
      state: (.state | ascii_downcase),
      is_draft: (.isDraft // false),
      author: (.author.login // "unknown"),
      author_is_bot: (.author.is_bot // false),
      labels: [(.labels // [])[] | (.name // .)],
      comments_count: (.commentsCount // 0),
      created_at: .createdAt,
      updated_at: .updatedAt,
      closed_at: (.closedAt | null_epoch),
      body: (.body | clean_body)
    };

  def in_window($published):
    ($published != null)
    and ($published[0:10] >= $since)
    and ($published[0:10] <= $until);

  def simplify_release:
    {
      name: (.name // .tag_name),
      tag: .tag_name,
      url: .html_url,
      published_at: .published_at,
      is_prerelease: (.prerelease // false),
      overview: (.body | release_overview),
      body: (.body | clean_release_text | clean_body)
    };

  def window_releases:
    [ $releases[]
      | select(.draft == false)
      | select(in_window(.published_at))
      | select(($include_prereleases == 1) or (.prerelease != true))
      | simplify_release
    ];

  {
    repo: $repo,
    window: {
      since: $since,
      until: $until,
      days: $days,
      note: "PRs/issues: updated_at on or after since. Releases: published_at date within since..until inclusive."
    },
    fetched_at: $fetched_at,
    counts: {
      releases: (window_releases | length),
      pull_requests: ($prs | length),
      issues: ([ $issues[] | select(.isPullRequest == false) ] | length)
    },
    releases: window_releases,
    pull_requests: [ $prs[] | simplify_item ],
    issues: [ $issues[] | select(.isPullRequest == false) | simplify_item ]
  }
  '

printf 'Fetched %s releases, %s pull requests, and %s issues.\n' \
  "$(jq -n --argjson releases "$releases_raw" --arg since "$SINCE" --arg until "$UNTIL" --argjson include_prereleases "$INCLUDE_PRERELEASES" '
      [ $releases[]
        | select(.draft == false)
        | select(.published_at != null)
        | select((.published_at[0:10] >= $since) and (.published_at[0:10] <= $until))
        | select(($include_prereleases == 1) or (.prerelease != true))
      ] | length
    ')" \
  "$(jq -n --argjson prs "$prs_raw" '$prs | length')" \
  "$(jq -n --argjson issues "$issues_raw" '[ $issues[] | select(.isPullRequest == false) ] | length')" \
  >&2
