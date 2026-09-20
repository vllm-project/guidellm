(() => {
  const warningId = "guidellm-stale-translation";

  function ensureTrailingSlash(value) {
    return value.endsWith("/") ? value : `${value}/`;
  }

  function siteRoot() {
    const logo = document.querySelector("a.md-header__button.md-logo");
    const href = logo ? logo.href : new URL("./", window.location.href).href;
    return new URL(ensureTrailingSlash(href));
  }

  function currentRoute(root) {
    const rootPath = ensureTrailingSlash(root.pathname);
    const currentPath = window.location.pathname;
    if (!currentPath.startsWith(rootPath)) {
      return "";
    }

    return currentPath.slice(rootPath.length).replace(/^\/+|\/+$/g, "") +
      (currentPath === rootPath ? "" : "/");
  }

  function setLanguageLink(language, root, route) {
    document.querySelectorAll(`[hreflang="${language}"]`).forEach((link) => {
      link.href = new URL(route, root).href;
    });
  }

  function removeStaleWarning() {
    document.getElementById(warningId)?.remove();
  }

  function addStaleWarning(root, englishRoute) {
    if (document.getElementById(warningId)) {
      return;
    }

    const content = document.querySelector(".md-content__inner");
    if (!content) {
      return;
    }

    const warning = document.createElement("div");
    warning.id = warningId;
    warning.className = "admonition warning";
    warning.innerHTML = `
      <p class="admonition-title">翻译可能已过期</p>
      <p>英文源文档自本页翻译后已有更新。请以<a href="${new URL(
        englishRoute,
        root,
      ).href}">最新英文页面</a>为准。</p>
    `;
    content.prepend(warning);
  }

  function updateLanguageSelector() {
    const routes = window.GUIDELLM_TRANSLATION_ROUTES || {};
    const root = siteRoot();
    const route = currentRoute(root);
    let englishRoute = route;
    let chineseRoute = "zh/";
    let translatedPage = null;

    Object.entries(routes).some(([sourceRoute, details]) => {
      if (details.translation === route) {
        englishRoute = sourceRoute;
        chineseRoute = details.translation;
        translatedPage = details;
        return true;
      }
      if (sourceRoute === route) {
        englishRoute = sourceRoute;
        chineseRoute = details.translation;
        return true;
      }
      return false;
    });

    setLanguageLink("en", root, englishRoute);
    setLanguageLink("zh", root, chineseRoute);
    document.documentElement.lang = translatedPage ? "zh-CN" : "en";

    removeStaleWarning();
    if (translatedPage && !translatedPage.current) {
      addStaleWarning(root, englishRoute);
    }
  }

  document.addEventListener("DOMContentLoaded", updateLanguageSelector);
  if (typeof document$ !== "undefined") {
    document$.subscribe(updateLanguageSelector);
  }
})();
