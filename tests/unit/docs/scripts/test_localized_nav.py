"""Tests for locale-specific documentation navigation."""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import pytest

pytest.importorskip("mkdocs", reason="MkDocs is an optional documentation dependency")

localized_nav = import_module("docs.scripts.localized_nav")


@pytest.mark.regression
def test_section_titles_are_localized_without_treating_routes_as_titles():
    """
    Load all section labels from the title map and leave generated routes unchanged.

    ## WRITTEN BY AI ##
    """
    assert localized_nav._localized_title("Getting Started") == "快速开始"
    assert localized_nav._localized_title("Guides") == "使用指南"
    assert localized_nav._localized_title("Examples") == "示例"
    assert localized_nav._localized_title("Developer") == "开发者"
    assert localized_nav._localized_title("API Reference") == "API 参考"
    assert localized_nav._localized_title("Multimodal") == "多模态"
    assert localized_nav._localized_title("  API   Reference  ") == "API 参考"
    assert localized_nav._localized_title("reference") == "reference"


@pytest.mark.parametrize(
    ("source_uri", "expected_chinese"),
    [
        ("zh/index.md", True),
        ("zh/getting-started/install.md", True),
        ("reference/api.md", False),
    ],
)
@pytest.mark.regression
def test_navigation_language_comes_from_page_route(
    source_uri: str,
    expected_chinese: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Select Chinese navigation only for pages under the zh route.

    ## WRITTEN BY AI ##
    """
    page = SimpleNamespace(file=SimpleNamespace(src_uri=source_uri))
    nav = object()
    localized = SimpleNamespace(pages=[])
    selected: list[bool] = []

    def navigation_for_page(current_nav, chinese: bool):
        assert current_nav is nav
        selected.append(chinese)
        return localized

    monkeypatch.setattr(localized_nav, "_navigation_for_page", navigation_for_page)
    monkeypatch.setattr(localized_nav, "_set_adjacent_pages", lambda *_: None)
    context: dict[str, object] = {}

    result = localized_nav.on_page_context(context, page, None, nav)

    assert result is context
    assert context["nav"] is localized
    assert selected == [expected_chinese]


def _mkdocs_navigation_case():
    config = import_module("mkdocs.config").load_config(config_file="mkdocs.yml")
    file_type = import_module("mkdocs.structure.files").File
    nav_module = import_module("mkdocs.structure.nav")
    page_type = import_module("mkdocs.structure.pages").Page

    def page(source_uri: str, title: str):
        file = file_type(source_uri, None, config.site_dir, config.use_directory_urls)
        return page_type(title, file, config)

    english_home = page("index.md", "Home")
    english_install = page("getting-started/install.md", "Install")
    english_server = page("getting-started/server.md", "Server")
    english_api = page("reference/guidellm/index.md", "API")
    chinese_home = page("zh/index.md", "首页")
    chinese_install = page("zh/getting-started/install.md", "安装")
    external_link = nav_module.Link("External", "https://example.com")

    items = [
        english_home,
        nav_module.Section("Getting Started", [english_install, english_server]),
        nav_module.Section("API Reference", [english_api, external_link]),
        nav_module.Section(
            "zh",
            [chinese_home, nav_module.Section("getting-started", [chinese_install])],
        ),
    ]

    def set_parent(item, parent=None):
        item.parent = parent
        if isinstance(item, nav_module.Section):
            for child in item.children:
                set_parent(child, item)

    for item in items:
        set_parent(item)

    source_pages = [
        english_home,
        english_install,
        english_server,
        english_api,
        chinese_home,
        chinese_install,
    ]
    source_nav = nav_module.Navigation(items, source_pages)
    return config, nav_module, source_nav, source_pages


def _mkdocs_nav_nodes(items, nav_module):
    for item in items:
        yield item
        if isinstance(item, nav_module.Section):
            yield from _mkdocs_nav_nodes(item.children, nav_module)


def _assert_parent_tree(items, nav_module, expected_parent=None):
    for item in items:
        assert item.parent is expected_parent
        if isinstance(item, nav_module.Section):
            _assert_parent_tree(item.children, nav_module, item)


def test_locale_navigation_keeps_mkdocs_source_pages_unmodified():
    """
    Keep original MkDocs navigation parents intact across mixed render orders.

    ## WRITTEN BY AI ##
    """
    nav_module = import_module("mkdocs.structure.nav")
    page_type = import_module("mkdocs.structure.pages").Page

    for render_order in (
        (
            "index.md",
            "getting-started/install.md",
            "zh/index.md",
            "zh/getting-started/install.md",
        ),
        (
            "zh/getting-started/install.md",
            "zh/index.md",
            "getting-started/install.md",
            "index.md",
        ),
    ):
        config, nav_module, source_nav, source_pages = _mkdocs_navigation_case()
        source_items = tuple(source_nav.items)
        source_nav_pages = tuple(source_nav.pages)
        source_homepage = source_nav.homepage
        source_nodes = list(_mkdocs_nav_nodes(source_nav.items, nav_module))
        parent_snapshot = [(node, node.parent) for node in source_nodes]
        adjacency_snapshot = [
            (page, page.previous_page, page.next_page) for page in source_pages
        ]
        children_snapshot = [
            (node, tuple(node.children))
            for node in source_nodes
            if isinstance(node, nav_module.Section)
        ]
        pages_by_route = {page.file.src_uri: page for page in source_pages}
        contexts = []

        for source_uri in render_order:
            page = pages_by_route[source_uri]
            page.active = True
            context = {"page": page}
            result = localized_nav.on_page_context(context, page, config, source_nav)
            page.active = False

            assert result is context
            current_nav = context["nav"]
            assert isinstance(current_nav, nav_module.Navigation)
            assert current_nav is not source_nav
            _assert_parent_tree(current_nav.items, nav_module)

            current_pages = current_nav.pages
            assert all(isinstance(candidate, page_type) for candidate in current_pages)
            assert not {id(candidate) for candidate in current_pages} & {
                id(candidate) for candidate in source_pages
            }
            current_routes = {candidate.file.src_uri for candidate in current_pages}
            if source_uri.startswith("zh/"):
                assert "zh/index.md" in current_routes
                assert "zh/getting-started/install.md" in current_routes
                assert "index.md" not in current_routes
                assert current_nav.homepage.file.src_uri == "zh/index.md"
                section_titles = {
                    item.title
                    for item in current_nav.items
                    if isinstance(item, nav_module.Section)
                }
                assert {"快速开始", "API 参考"} <= section_titles
                link_titles = {
                    item.title
                    for item in _mkdocs_nav_nodes(current_nav.items, nav_module)
                    if isinstance(item, nav_module.Link)
                }
                assert {"Server (English)", "API (English)", "External"} <= link_titles
            else:
                assert "index.md" in current_routes
                assert "getting-started/install.md" in current_routes
                assert not any(route.startswith("zh/") for route in current_routes)
                assert current_nav.homepage.file.src_uri == "index.md"
                section_titles = {
                    item.title
                    for item in current_nav.items
                    if isinstance(item, nav_module.Section)
                }
                assert {"Getting Started", "API Reference"} <= section_titles

            matching_page = next(
                candidate
                for candidate in current_pages
                if candidate.file.src_uri == source_uri
            )
            assert matching_page is not page
            assert context["page"] is matching_page
            assert matching_page.active
            expected_ancestors = {
                "index.md": [],
                "getting-started/install.md": ["Getting Started"],
                "zh/index.md": [],
                "zh/getting-started/install.md": ["快速开始"],
            }
            assert [ancestor.title for ancestor in matching_page.ancestors] == (
                expected_ancestors[source_uri]
            )

            entries = [
                item
                for item in _mkdocs_nav_nodes(current_nav.items, nav_module)
                if isinstance(item, (page_type, nav_module.Link))
            ]
            current_index = entries.index(matching_page)
            assert matching_page.previous_page is (
                entries[current_index - 1] if current_index > 0 else None
            )
            assert matching_page.next_page is (
                entries[current_index + 1] if current_index + 1 < len(entries) else None
            )
            contexts.append(current_nav)

            assert tuple(source_nav.items) == source_items
            assert tuple(source_nav.pages) == source_nav_pages
            assert source_nav.homepage is source_homepage
            assert all(node.parent is parent for node, parent in parent_snapshot)
            assert all(
                page.previous_page is previous_page and page.next_page is next_page
                for page, previous_page, next_page in adjacency_snapshot
            )
            assert all(
                tuple(section.children) == children
                for section, children in children_snapshot
            )

        for current_nav in contexts:
            _assert_parent_tree(current_nav.items, nav_module)
