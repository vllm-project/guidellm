---
name: guidellm-review-chinese-translation
description: Review a GuideLLM pull request by number, compare its Simplified Chinese documentation with the English source, and produce an English differences table and recommendation. Report minor differences but recommend edits only for major errors or project Code of Conduct violations.
---

# Review a Chinese translation PR

Accept a PR number as input, for example `$guidellm-review-chinese-translation 1273`. Use `vllm-project/guidellm` unless the user specifies another repository. If the number is missing and cannot be inferred from the request, ask for it.

Produce an English report in the conversation. This task does not authorize editing the translation, posting GitHub comments or reviews, or contacting anyone.

## Retrieve and compare

1. Fetch PR metadata, the changed-file list, and the head commit. Read files at that pinned commit without checking out over the user's work. In this repository, prefix shell commands other than `tox` with `uv run`. Useful commands:

   ```bash
   uv run gh pr view <PR_NUMBER> --repo vllm-project/guidellm --json title,url,headRefOid,baseRefOid,files
   uv run git fetch origin pull/<PR_NUMBER>/head
   uv run git show <HEAD_SHA>:docs/zh/.translation-sources.json
   uv run git show <BASE_SHA>:CODE_OF_CONDUCT.md
   ```

   Verify that the remote is the requested repository before fetching. If GitHub access or required files are unavailable, report the limitation; do not invent content or an overall clean result.

2. Identify all added or modified Simplified Chinese documents in the PR. Resolve each English source using `docs/zh/.translation-sources.json`, falling back to matching paths under `docs/en/`. Compare complete Chinese documents with their English sources at the same PR head commit, including context around changed passages. For renamed or deleted translations, check whether source content or reader access has been lost. Distinguish PR changes from pre-existing issues.

3. Read the project's `CODE_OF_CONDUCT.md` from the PR base revision as the governing standard. Assess the reviewed text, including the English source; faithful translation does not excuse a violation already present in English. Tie any conduct finding to an actual provision and explain its application in context. Do not infer misconduct from an author's identity or an ordinary linguistic preference.

4. Compare meaning, completeness, technical terms, requirements, negation, numbers, units, commands, examples, headings, and links. The table may include minor differences in wording, nuance, emphasis, or navigation even when they do not warrant action. Natural Chinese phrasing need not follow English literally. English fallback links and language labels are normally reasonable adaptations; verify routing configuration before claiming a link is broken.

5. Use immutable GitHub file links with line anchors for findings and identify the reviewed commit. Do not claim tests, builds, or link checks passed unless they were run. A text-only review does not require running the application test suite.

## Recommendation threshold

Recommend changes **only** for either of these:

- **Major errors:** a material change, omission, or addition that would mislead readers or prevent correct use. Examples include reversed meaning, incorrect requirements, wrong numerical values or units, unusable commands, materially incorrect technical terminology, or a verified broken link that prevents access to a required guide.
- **Code of Conduct violations:** text that violates a specific provision of the project's `CODE_OF_CONDUCT.md`, even if it accurately translates the English source. Identify whether the issue occurs in Chinese, English, or both.

For each recommended change, explain the practical impact or applicable conduct provision and provide a focused correction. Avoid unnecessarily reproducing offensive text or private information.

Minor nuance, marketing emphasis, stylistic choices, terminology preferences, and acceptable localization may appear in the comparison table, but **must not receive suggested edits**, optional improvements, replacement wording, or requests for polish. For example, “perfect balance” becoming “a suitable balance” is ordinarily a minor difference. Judge version wording by its practical effect in context, not literal equivalence alone.

Unverified suspicions are limitations, not established major errors or violations. If evidence needed to finish the review is missing, state that the recommendation is incomplete rather than giving unconditional approval.

## Required report format

Use the following structure, in English. Keep it concise, but include every material finding.

```markdown
I reviewed [PR #<NUMBER>](<PR_URL>) at commit `<SHORT_SHA>`, comparing <Chinese documents> with their English sources.

**Overall assessment:** <Accuracy and completeness; distinguish minor differences from major errors.>

| Location | Difference in Chinese | Assessment |
| --- | --- | --- |
| <Linked file and passage> | <English meaning, Chinese wording, and its English back-translation or explanation> | <Minor difference / acceptable adaptation / major error / Code of Conduct violation, with reason or impact> |

**Code of Conduct:** <No violations found in the reviewed text, or findings with links to the applicable provisions of CODE_OF_CONDUCT.md.>

**Recommendation:** <Use the rules below.>

Compared: <Links to the Chinese documents and their English sources at the reviewed commit.>
```

- Include a table even when no differences are found: use a row stating “No substantive differences found” and “Meaning and completeness preserved.” Do not invent differences to populate it.
- If only minor differences or acceptable adaptations exist, write: **“No changes recommended. The translation preserves the source meaning, and no major errors or Code of Conduct violations were found.”** Do not append optional suggestions.
- If qualifying issues exist, write: **“Changes recommended for the following major errors or Code of Conduct violations:”** followed by a short numbered list of only those issues and their corrections. Link each to the supporting table location or source passage. Minor findings in the table must not become action items in the recommendation.
- If the review is incomplete, explicitly state which files or evidence were unavailable and qualify the assessment and recommendation accordingly.
