---
title: Documentation Translations
weight: -4
---

# Documentation Translations

GuideLLM publishes a small, community-maintained set of Simplified Chinese pages alongside its canonical English documentation. The translation pilot is intentionally limited so that the project can evaluate its maintenance and review cost before expanding it.

## Policy

- English documentation under `docs/en/` is the source of truth.
- Simplified Chinese pages live under `docs/zh/` and mirror the English paths.
- English documentation changes do not require a matching translation update and are not blocked by translation drift.
- A translated page that is behind its English source displays a warning and links readers to the current English page.
- CLI output, logs, error messages, docstrings, and generated API reference pages are outside the translation scope.
- Machine translation can assist a contributor, but a Chinese-speaking reviewer remains responsible for meaning and language quality.

## Source tracking

Every translated page is registered in `docs/zh/.translation-sources.json`. The manifest maps it to its canonical English page and records a SHA-256 digest of the English source used for the translation.

Run the non-blocking validation used by CI:

```bash
uv run python docs/scripts/check_translations.py
```

This command reports outdated pages as warnings. Structural problems, missing pages, unregistered translations, and changed fenced code examples are errors.

After reviewing a translation against the latest English page, refresh its recorded source revision and require every translation to be current:

```bash
uv run python docs/scripts/check_translations.py --update --strict
```

Commit the updated manifest with the translated page.

## Review responsibilities

English-speaking maintainers can review site integration, technical claims, commands, links, code examples, and regressions to the canonical site. Chinese-speaking reviewers check semantic accuracy, terminology, and language quality. A reviewer does not need to be bilingual to cover both responsibilities.

A translation pull request should include:

- the canonical English page and source revision;
- a short English summary of terminology or technical decisions;
- confirmation that commands, identifiers, and fenced code examples were preserved;
- the output of the strict translation check and normal documentation checks.

AI-generated drafts and back-translations are review aids, not approval evidence by themselves.

## Terminology

Keep product names, command names, CLI flags, configuration keys, metric abbreviations, code identifiers, and code examples unchanged. On first use, prefer a Chinese term followed by its established English term or abbreviation when that improves precision, such as “首 token 延迟（TTFT）”.

Terminology changes should be applied consistently across existing Chinese pages in the same pull request when practical.
