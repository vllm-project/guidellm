---
name: guidellm-weekly-summary
description: Summarize GuideLLM team activity from GitHub PRs and issues over the past week into a concise, externally shareable nested-list update. Use when the user asks for a weekly summary, team activity update, status digest, or changelog-style overview of vllm-project/guidellm.
compatibility: Requires gh, jq, and network access to GitHub (vllm-project/guidellm).
---

# GuideLLM Weekly Activity Summary

Generate a concise, externally shareable summary of GuideLLM work from the past week using GitHub data for `vllm-project/guidellm` only.

## When to use

Apply when the user asks for a weekly summary, team activity update, status digest, or similar overview of GuideLLM progress.

## Data collection

**Do not invent ad-hoc `gh` queries.** Run the bundled fetch script, then write from its JSON.

1. Determine the current date/time with `date` (do not guess).
2. Use a rolling **past 7 days** window ending today unless the user specifies another range.
3. From this skill directory, fetch activity (stdout = JSON, stderr = progress):

   ```bash
   bash scripts/fetch_activity.sh
   ```

   Custom window examples:

   ```bash
   bash scripts/fetch_activity.sh --days 14
   bash scripts/fetch_activity.sh --since 2026-07-16 --until 2026-07-23
   ```

   If the working directory is the repository root instead of the skill root:

   ```bash
   bash .agents/skills/guidellm-weekly-summary/scripts/fetch_activity.sh
   ```

4. Parse the JSON: `window`, `pull_requests`, and `issues`. Prefer `body`, `labels`, and `state` over titles alone when inferring what shipped or changed.
5. Skip noise (trivial dependency bumps, pure formatting, bot-only churn — see `author_is_bot`) unless it is the main story.
6. Run `bash scripts/fetch_activity.sh --help` only if you need flags beyond the examples above.

### Script notes

- **`scripts/fetch_activity.sh`** — Searches PRs and issues updated in the window via `gh`, normalizes fields, truncates bodies, and emits one JSON document on stdout.
- Bodies are truncated (default 800 chars) and scrubbed of squash metadata / testing sections to save tokens. Raise `--body-max` only when a specific item needs more detail.

## Writing rules

- Write for an **external** audience: clear, professional, no internal jargon or private details.
- Lead with **what changed and why it matters**, not with PR numbers or titles.
- Treat GitHub links, PR/issue numbers, and titles as **secondary references** for readers who want more detail—not as the primary narrative.
- Group by theme (features, fixes, docs, infra) over chronological dumps.
- Keep it concise: a few nested bullets per theme, not a full changelog.

## Output format (required)

- The summary content itself uses **nested markdown lists only** — **no headers** (`#`, `##`, etc.) inside the report.
- Use markdown that pastes cleanly into docs: `**bold**`, `[text](url)` links, `` `inline code` ``, and `-` nested lists.
- **CRITICAL — show raw markdown, not rendered text:** Wrap the entire summary in a single fenced code block tagged `markdown` so the user sees and can copy the literal characters (`**`, `-`, `[]()`, backticks) instead of formatted rich text. Do not also emit a rendered version outside the fence.
- Response shape:

````
```markdown
- **Theme or area** — one-line outcome in plain language
  - Supporting detail on what landed or progressed
  - Optional secondary refs: [PR title](url), [#123](url)
- **Another theme** — one-line outcome
  - Supporting detail
  - Secondary refs as needed
```
````

- Top-level bullets = themes/outcomes.
- Nested bullets = brief supporting detail and secondary GitHub refs.
- Do not open with a title/header line; start the fenced block directly with the list.
- If the week was quiet, say so in one top-level bullet and list only notable items.
- Optional one-line preface outside the fence is fine (e.g. date range from `window.since` → `window.until`); the copy-paste body must be only inside the fence.

## Example response

Preface (optional): `GuideLLM activity, 2026-07-16 → 2026-07-23:`

Then the fenced block containing:

```
- **Benchmarking UX** — Sweep runs can now target clearer stop conditions, making short validation jobs easier to reason about
  - Refined how duration and request caps interact in the CLI
  - More detail: [Clarify constraint handling for sweep profiles](https://github.com/vllm-project/guidellm/pull/…)
- **Reliability** — Fixed a failure mode that dropped partial results when a backend timed out mid-run
  - Users keep completed request metrics instead of an empty report
  - Tracking: [#…](https://github.com/vllm-project/guidellm/issues/…)
```

## Checklist before responding

- [ ] Used `scripts/fetch_activity.sh` (not hand-written `gh` searches)
- [ ] Window is past 7 days (or user-specified) and based on script/`date` output
- [ ] Scope is `vllm-project/guidellm` only
- [ ] Narrative leads; links/titles are secondary
- [ ] Nested lists only; no headers inside the report
- [ ] Entire summary is inside a ` ```markdown ` fence so raw markdown is visible for copy-paste
- [ ] Tone is externally shareable
