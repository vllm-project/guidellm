---
name: guidellm-review-translation
description: Review a GuideLLM translation pull request by number, detect its target languages, and compare translated documentation in any language with the English source. Produce an English report for English-speaking reviewers, noting minor differences but recommending edits only for major errors or project Code of Conduct violations.
---

# Review a translation PR

Accept a PR number as input, for example `$guidellm-review-translation 1273`. Detect the target language or languages from the PR; the user need not supply a language. Use `vllm-project/guidellm` unless the user specifies another repository. If the number is missing and cannot be inferred from the request, ask for it.

Write the report's explanations and recommendations in English for English-speaking reviewers. Target-language quotations and proposed corrections must include an English back-translation or explanation so readers can assess every finding without knowing that language. This task does not authorize editing the translation, posting GitHub comments or reviews, or contacting anyone.

## Retrieve and compare

1. Fetch PR metadata, the changed-file list, and the head commit. Read files at that pinned commit without checking out over the user's work. In this repository, prefix shell commands other than `tox` with `uv run`. Useful commands:

   ```bash
   uv run gh pr view <PR_NUMBER> --repo vllm-project/guidellm --json title,url,headRefOid,baseRefOid,files
   uv run git fetch origin pull/<PR_NUMBER>/head
   uv run git show <HEAD_SHA>:docs/<LOCALE>/.translation-sources.json
   uv run git show <BASE_SHA>:CODE_OF_CONDUCT.md
   ```

   Verify that the remote is the requested repository before fetching. If GitHub access or required files are unavailable, report the limitation; do not invent content or an overall clean result.

2. Identify all added or modified non-English documents in the PR. Detect each document's language from its prose, corroborated by locale paths, translation manifests, document metadata, and site language configuration. Preserve meaningful script or regional distinctions when supported by evidence; a locale code alone is not proof of the actual language. Review every target language when a PR contains several, and report unexpected mixed-language or untranslated passages in context. Retained product names, identifiers, and English technical terms are not automatically errors. If the language remains uncertain, state the uncertainty and ask only if needed to finish the review. If no translated documents are present, report that no translation comparison applies.

3. Resolve each English source using the applicable translation manifest, commonly `docs/<LOCALE>/.translation-sources.json`, falling back to matching paths under `docs/en/` or other explicit source mappings in the repository. Do not assume every language has a manifest or shares one directory layout. Compare complete translated documents with their English sources at the same PR head commit, including context around changed passages. For renamed or deleted translations, check whether source content or reader access has been lost. Distinguish PR changes from pre-existing issues. If no English counterpart can be established, identify that gap instead of guessing the source.

4. Read the project's `CODE_OF_CONDUCT.md` from the PR base revision as the governing standard. Assess the reviewed text, including the English source; faithful translation does not excuse a violation already present in English. Tie any conduct finding to an actual provision and explain its application in context. Do not infer misconduct from an author's identity or an ordinary linguistic preference.

5. Compare meaning, completeness, technical terms, requirements, negation, numbers, units, commands, examples, headings, and links. The table may include minor differences in wording, nuance, emphasis, or navigation even when they do not warrant action. Natural target-language phrasing need not follow English literally; account for grammar, idiom, and locale conventions while checking that meaning is preserved. English fallback links and language labels are normally reasonable adaptations; verify routing configuration before claiming a link is broken.

6. Use immutable GitHub file links with line anchors for findings and identify the reviewed commit. Do not claim tests, builds, or link checks passed unless they were run. A text-only review does not require running the application test suite.

## Recommendation threshold

Recommend changes **only** for either of these:

- **Major errors:** a material change, omission, or addition that would mislead readers or prevent correct use. Examples include reversed meaning, incorrect requirements, wrong numerical values or units, unusable commands, materially incorrect technical terminology, or a verified broken link that prevents access to a required guide.
- **Code of Conduct violations:** text that violates a specific provision of the project's `CODE_OF_CONDUCT.md`, even if it accurately translates the English source. Identify whether the issue occurs in the translation, English source, or both, naming the affected language.

For each recommended change, explain the practical impact or applicable conduct provision and provide a focused correction. Avoid unnecessarily reproducing offensive text or private information.

Minor nuance, marketing emphasis, stylistic choices, terminology preferences, and acceptable localization may appear in the comparison table, but **must not receive suggested edits**, optional improvements, replacement wording, or requests for polish. For example, “perfect balance” becoming “a suitable balance” is ordinarily a minor difference. Judge version wording by its practical effect in context, not literal equivalence alone.

Unverified suspicions are limitations, not established major errors or violations. If evidence needed to finish the review is missing, state that the recommendation is incomplete rather than giving unconditional approval.

## Required report format

Use the following structure, in English. Keep it concise, but include every material finding.

When comparing phrases in the table, you may quote the target-language text followed immediately by its literal English translation in parentheses. For example: English “perfect balance” becomes “合适的平衡” (“a suitable balance”). If a literal translation obscures an idiom's intended meaning, also explain that meaning in English; do not treat literal wording alone as evidence of an error.

```markdown
I reviewed [PR #<NUMBER>](<PR_URL>) at commit `<SHORT_SHA>`, comparing <translated documents> with their English sources.

**Detected language(s):** <English names of the target languages, with script or regional variants where established.>

**Overall assessment:** <Accuracy and completeness; distinguish minor differences from major errors.>

| Location | Difference in translation | Assessment |
| --- | --- | --- |
| <Language, linked file, and passage> | <English source phrase or meaning; optionally quote the target-language phrase followed by its literal English translation in parentheses, with further English explanation as needed> | <Minor difference / acceptable adaptation / major error / Code of Conduct violation, with reason or impact> |

**Code of Conduct:** <No violations found in the reviewed text, or findings with links to the applicable provisions of CODE_OF_CONDUCT.md.>

**Recommendation:** <Use the rules below.>

Compared: <Links to the translated documents and their English sources at the reviewed commit, labeled by language.>
```

- Include a table even when no differences are found: use a row stating “No substantive differences found” and “Meaning and completeness preserved.” Do not invent differences to populate it.
- If only minor differences or acceptable adaptations exist, write: **“No changes recommended. The translation preserves the source meaning, and no major errors or Code of Conduct violations were found.”** Do not append optional suggestions.
- If qualifying issues exist, write: **“Changes recommended for the following major errors or Code of Conduct violations:”** followed by a short numbered list of only those issues and their corrections. Link each to the supporting table location or source passage. Minor findings in the table must not become action items in the recommendation.
- If the review is incomplete, explicitly state which files or evidence were unavailable and qualify the assessment and recommendation accordingly.
