"""Build a navigation tree for the language of each rendered documentation page.

MkDocs shares one navigation tree across the site, so this hook makes a fresh tree
for each page using its source route. Pages under registered locale routes use
available translations and show untranslated pages as links labeled ``(English)``.
Following one of those links renders an English page, which uses the standard
English navigation until the reader returns to a translated page. Section titles
are translated only when they match the labels in :data:`SECTION_TITLES`.
"""

from __future__ import annotations

import json
import re
import runpy
from copy import copy
from pathlib import Path, PurePosixPath
from typing import Any, cast

from mkdocs.structure.nav import Link, Navigation, Section
from mkdocs.structure.pages import Page

translation_module = runpy.run_path(
    str(Path(__file__).resolve().parent / "check_translations.py")
)
translation_routes = translation_module["translation_routes"]
TRANSLATION_ROOTS = translation_module["TRANSLATION_ROOTS"]

SECTION_TITLES = {
    language: json.loads(
        (Path(__file__).resolve().parents[1] / language / "title_map.json").read_text(
            encoding="utf-8"
        )
    )
    for language in TRANSLATION_ROOTS
}


def _route(page: Page) -> str:
    path = PurePosixPath(page.file.src_uri)
    route = path.parent if path.name == "index.md" else path.with_suffix("")
    value = route.as_posix()
    return "" if value == "." else f"{value}/"


def _pages(item: Any) -> list[Page]:
    if isinstance(item, Page):
        return [item]
    if isinstance(item, Section):
        return [page for child in item.children for page in _pages(child)]
    return []


def _navigation_entries(item: Any) -> list[Page | Link]:
    if isinstance(item, (Page, Link)):
        return [item]
    if isinstance(item, Section):
        return [
            entry for child in item.children for entry in _navigation_entries(child)
        ]
    return []


def _page_title(page: Page) -> str:
    if page.title:
        return str(page.title)

    content = page.file.content_string
    frontmatter = re.match(r"^---\s*\n(.*?)\n---(?:\s*\n|$)", content, re.DOTALL)
    if frontmatter:
        title = re.search(
            r"^title:\s*[\"']?(.*?)[\"']?\s*$", frontmatter[1], re.MULTILINE
        )
        if title and title[1]:
            return title[1]

    heading = re.search(r"^#\s+(.+?)\s*#*\s*$", content, re.MULTILINE)
    if heading:
        return heading[1]

    return PurePosixPath(page.file.src_uri).stem.replace("-", " ").title()


def _localized_title(title: str, language: str = "zh") -> str:
    titles = SECTION_TITLES.get(language, {})
    return titles.get(" ".join(title.casefold().split()), title)


def _copy_page(page: Page) -> Page:
    """Copy a page so each rendered locale can own its navigation attributes.

    MkDocs stores ``parent`` and adjacent-page links directly on each Page. The
    locale-specific tree needs its own navigation state, while the underlying
    File, configuration, rendered content, and metadata are reused.
    """
    cloned_page = copy(page)
    cloned_page.parent = None
    cloned_page.previous_page = None
    cloned_page.next_page = None
    return cloned_page


def _clone_page(
    page: Page,
    *,
    language: str,
    routes: dict[str, dict[str, str | bool]],
    pages_by_route: dict[str, Page],
) -> Any | None:
    """Choose a translated page, English fallback link, or English source page."""
    route = _route(page)
    if route.split("/", 1)[0] in TRANSLATION_ROOTS:
        return None
    if language == "en":
        return _copy_page(page)

    details = routes.get(route)
    translated = pages_by_route.get(str(details["translation"])) if details else None
    if translated is not None:
        cloned_translation = _copy_page(translated)
        if not cloned_translation.title:
            cloned_translation.title = _page_title(cloned_translation)
        return cloned_translation

    return Link(f"{_page_title(page)} (English)", page.url)


def _clone_section(
    item: Section,
    *,
    language: str,
    routes: dict[str, dict[str, str | bool]],
    pages_by_route: dict[str, Page],
) -> Section | None:
    """Copy a section around localized children without changing the shared nav."""
    children = [
        child
        for source_child in item.children
        if (
            child := _clone_item(
                source_child,
                language=language,
                routes=routes,
                pages_by_route=pages_by_route,
            )
        )
        is not None
    ]
    if not children:
        return None

    title = _localized_title(item.title, language)
    section = Section(title, children)
    for child in children:
        child.parent = section
    section.active = any(
        child.active for child in children if isinstance(child, (Page, Link, Section))
    )
    return section


def _clone_item(
    item: Any,
    *,
    language: str,
    routes: dict[str, dict[str, str | bool]],
    pages_by_route: dict[str, Page],
) -> Any | None:
    if isinstance(item, Page):
        return _clone_page(
            item,
            language=language,
            routes=routes,
            pages_by_route=pages_by_route,
        )
    if isinstance(item, Section):
        return _clone_section(
            item,
            language=language,
            routes=routes,
            pages_by_route=pages_by_route,
        )
    if isinstance(item, Link):
        return copy(item)
    return item


def _navigation_for_page(nav: Navigation, language: str) -> Navigation:
    """Build the page's navigation using translations for its language."""
    routes = (
        translation_routes(translations_root=TRANSLATION_ROOTS[language])
        if language != "en"
        else {}
    )
    source_pages_by_route = {_route(candidate): candidate for candidate in nav.pages}
    items = [
        item
        for source_item in nav.items
        if (
            item := _clone_item(
                source_item,
                language=language,
                routes=routes,
                pages_by_route=source_pages_by_route,
            )
        )
        is not None
    ]
    pages = [candidate for item in items for candidate in _pages(item)]
    localized_nav = Navigation(items, pages)

    homepage_route = "" if language == "en" else f"{language}/"
    localized_nav.homepage = next(
        (candidate for candidate in pages if _route(candidate) == homepage_route), None
    )

    return localized_nav


def _set_adjacent_pages(page: Page, nav: Navigation) -> None:
    """Set previous and next links from the navigation tree used for this page."""
    entries = [entry for item in nav.items for entry in _navigation_entries(item)]
    current_index = next(
        (
            index
            for index, entry in enumerate(entries)
            if isinstance(entry, Page) and _route(entry) == _route(page)
        ),
        None,
    )
    if current_index is None:
        page.previous_page = None
        page.next_page = None
        return

    # MkDocs templates render Link neighbors as footer links as well.
    page.previous_page = (
        cast("Page", entries[current_index - 1]) if current_index > 0 else None
    )
    page.next_page = (
        cast("Page", entries[current_index + 1])
        if current_index + 1 < len(entries)
        else None
    )


def on_page_context(context: dict, page: Page, config, nav: Navigation):
    """Use a navigation tree selected from the rendered page's source route.

    Routes under a registered locale select that language's navigation. A reader
    following an English fallback link sees the English navigation on that page.
    """
    _ = config
    page_route = _route(page)
    language = page_route.split("/", 1)[0]
    if language not in TRANSLATION_ROOTS:
        language = "en"
    localized_nav = _navigation_for_page(nav, language)
    context["nav"] = localized_nav
    localized_page = next(
        (
            candidate
            for candidate in localized_nav.pages
            if _route(candidate) == page_route
        ),
        None,
    )
    if localized_page is None:
        _set_adjacent_pages(page, localized_nav)
    else:
        # Material builds its breadcrumb from page.ancestors, so the context page
        # must be the copy attached to this locale's section tree.
        context["page"] = localized_page
        _set_adjacent_pages(localized_page, localized_nav)
    return context
