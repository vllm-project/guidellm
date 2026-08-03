---
name: guidellm-weekly-summary
description: Summarize GuideLLM team activity from GitHub PRs and issues over the past week into a concise, externally shareable nested-list update. Use when the user asks for a weekly summary, team activity update, status digest, or changelog-style overview of vllm-project/guidellm.
---

# GuideLLM Weekly Activity Summary

Generate a concise, externally shareable summary of GuideLLM work from the past week using GitHub data for `vllm-project/guidellm` only.

## When to use

Apply when the user asks for a weekly summary, team activity update, status digest, or similar overview of GuideLLM progress.

## Data collection

1. Determine the current date/time with `date` (do not guess).
2. Use a rolling **past 7 days** window ending today unless the user specifies another range.
3. Query GitHub for `vllm-project/guidellm` only (`gh` preferred):
   - PRs updated, opened, merged, or closed in the window
   - Issues opened, closed, or notably updated in the window
4. Prefer PR/issue bodies, review discussion, and labels over titles alone when inferring what shipped or changed.
5. Skip noise (trivial dependency bumps, pure formatting, bot-only churn) unless it is the main story.

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
- Optional one-line preface outside the fence is fine (e.g. date range); the copy-paste body must be only inside the fence.

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

- [ ] Window is past 7 days (or user-specified) and based on `date`
- [ ] Scope is `vllm-project/guidellm` only
- [ ] Narrative leads; links/titles are secondary
- [ ] Nested lists only; no headers inside the report
- [ ] Entire summary is inside a ` ```markdown ` fence so raw markdown is visible for copy-paste
- [ ] Tone is externally shareable
