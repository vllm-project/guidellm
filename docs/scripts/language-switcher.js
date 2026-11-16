(() => {
  const warningId = "guidellm-stale-translation";

  function ensureTrailingSlash(value) {
    return value.endsWith("/") ? value : `${value}/`;
  }

  function sharedPath(paths) {
    const segments = paths.map((path) => path.split("/").filter(Boolean));
    let sharedCount = 0;
    while (
      segments.every((parts) => parts[sharedCount] === segments[0][sharedCount]) &&
      segments[0][sharedCount] !== undefined
    ) {
      sharedCount += 1;
    }

    const prefix = segments[0].slice(0, sharedCount).join("/");
    return prefix ? `/${prefix}/` : "/";
  }

  let cachedSiteRoot;

  function siteRoot() {
    if (cachedSiteRoot) {
      return cachedSiteRoot;
    }

    const logo = document.querySelector("a.md-header__button.md-logo");
    const href = logo ? logo.href : new URL("./", window.location.href).href;
    cachedSiteRoot = new URL(ensureTrailingSlash(href));

    const localeCandidates = Array.from(
      document.querySelectorAll("a[hreflang], link[hreflang]"),
    )
      .map((link) => ({
        language: link.hreflang.toLowerCase(),
        url: new URL(link.href, window.location.href),
      }))
      .filter(({ url }) => url.origin === cachedSiteRoot.origin);
    const candidatePaths = Array.from(
      new Set(localeCandidates.map(({ url }) => ensureTrailingSlash(url.pathname))),
    );

    if (candidatePaths.length > 1) {
      const sharedRoot = sharedPath(candidatePaths);
      const rootSegments = cachedSiteRoot.pathname.split("/").filter(Boolean);
      if (
        Object.hasOwn(window.GUIDELLM_TRANSLATION_ROUTES || {}, rootSegments.at(-1))
      ) {
        rootSegments.pop();
      }
      const logoRoot = rootSegments.length ? `/${rootSegments.join("/")}/` : "/";
      cachedSiteRoot.pathname = logoRoot.startsWith(sharedRoot)
        ? logoRoot
        : sharedRoot;
    } else if (localeCandidates.length === 1) {
      const { language, url } = localeCandidates[0];
      const rootSegments = cachedSiteRoot.pathname.split("/").filter(Boolean);
      const candidateSegments = url.pathname.split("/").filter(Boolean);
      const finalRootSegment = rootSegments.at(-1)?.toLowerCase();
      const languageSubtags = language.split("-").map((part) => part.toLowerCase());
      if (
        rootSegments.length > 1 &&
        languageSubtags.includes(finalRootSegment) &&
        rootSegments.every((part, index) => candidateSegments[index] === part)
      ) {
        rootSegments.pop();
        cachedSiteRoot.pathname = rootSegments.length
          ? `/${rootSegments.join("/")}/`
          : "/";
      }
    }
    return cachedSiteRoot;
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

  function addStaleWarning(root, englishRoute, language) {
    if (document.getElementById(warningId)) {
      return;
    }

    const content = document.querySelector(".md-content__inner");
    if (!content) {
      return;
    }

    const messages = {
      zh: {
        title: "翻译可能已过期",
        description: "英文源文档自本页翻译后已有更新。请参阅",
        link: "最新英文页面",
      },
      ko: {
        title: "번역이 오래되었을 수 있습니다",
        description: "이 페이지를 번역한 이후 영어 원문이 변경되었습니다. ",
        link: "최신 영어 문서를 확인하세요.",
      },
    };

    const message = messages[language];
    if (!message) {
      return;
    }

    const warning = document.createElement("div");
    warning.id = warningId;
    warning.className = "admonition warning";

    const title = document.createElement("p");
    title.className = "admonition-title";
    title.textContent = message.title;

    const paragraph = document.createElement("p");
    paragraph.textContent = message.description;

    const link = document.createElement("a");
    link.href = new URL(englishRoute, root).href;
    link.textContent = message.link;

    paragraph.append(link);
    warning.append(title, paragraph);
    content.prepend(warning);
  }

  function updateLanguageSelector() {
    const routes = window.GUIDELLM_TRANSLATION_ROUTES || {};
    const root = siteRoot();
    const route = currentRoute(root);
    let englishRoute = route;
    let activeLanguage = "en";
    let translatedPage = null;
    for (const [language, pages] of Object.entries(routes)) {
      const match = Object.entries(pages).find(
        ([, details]) => details.translation === route,
      );
      if (match) {
        englishRoute = match[0];
        activeLanguage = language;
        translatedPage = match[1];
        break;
      }
    }
    setLanguageLink("en", root, englishRoute);
    for (const [language, pages] of Object.entries(routes)) {
      const page = pages[englishRoute];
      const target = page?.translation ?? `${language}/`;
      setLanguageLink(language, root, target);
    }
    const htmlLanguages = {
      en: "en",
      zh: "zh-CN",
      ko: "ko",
    };
    document.documentElement.lang =
      htmlLanguages[activeLanguage] ?? "en";
    removeStaleWarning();
    if (translatedPage && !translatedPage.current) {
      addStaleWarning(root, englishRoute, activeLanguage);
    }
  }

  document.addEventListener("DOMContentLoaded", updateLanguageSelector);
  if (typeof document$ !== "undefined") {
    document$.subscribe(updateLanguageSelector);
  }
})();
