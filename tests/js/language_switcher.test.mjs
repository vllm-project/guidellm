import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { JSDOM } from "jsdom";

const languageSwitcherSource = fs.readFileSync(
  new URL("../../docs/scripts/language-switcher.js", import.meta.url),
  "utf8",
);

function runLanguageSwitcher({ pagePath, logoPath, locales, routes }) {
  const languageLinks = locales
    .map(
      ({ language, path }) =>
        `<a hreflang="${language}" href="${path}">${language}</a>`,
    )
    .join("");
  const dom = new JSDOM(
    `<!doctype html>
    <html lang="en"><body>
      <a class="md-header__button md-logo" href="${logoPath}">Logo</a>
      <nav>${languageLinks}</nav>
      <main class="md-content__inner"></main>
    </body></html>`,
    { url: `https://example.test${pagePath}`, runScripts: "outside-only" },
  );
  const { window } = dom;

  window.GUIDELLM_TRANSLATION_ROUTES = routes;
  window.eval(languageSwitcherSource);
  window.document.dispatchEvent(new window.Event("DOMContentLoaded"));

  return dom;
}

const routes = {
  "": { translation: "zh/", current: true },
  "getting-started/install/": {
    translation: "zh/getting-started/install/",
    current: true,
  },
};

/** Confirms Chinese locale roots resolve to the shared site root. ## WRITTEN BY AI ## */
test("language switcher resolves a zh home page and translated subpage", () => {
  const locales = [
    { language: "en", path: "/guidellm/" },
    { language: "zh", path: "/guidellm/zh/" },
  ];
  const home = runLanguageSwitcher({
    pagePath: "/guidellm/zh/",
    logoPath: "/guidellm/zh/",
    locales,
    routes,
  });
  const translatedPage = runLanguageSwitcher({
    pagePath: "/guidellm/zh/getting-started/install/",
    logoPath: "/",
    locales,
    routes,
  });

  assert.equal(
    home.window.document.querySelector('[hreflang="en"]').href,
    "https://example.test/guidellm/",
  );
  assert.equal(
    home.window.document.querySelector('[hreflang="zh"]').href,
    "https://example.test/guidellm/zh/",
  );
  assert.equal(home.window.document.documentElement.lang, "zh-CN");
  assert.equal(
    translatedPage.window.document.querySelector('[hreflang="en"]').href,
    "https://example.test/guidellm/getting-started/install/",
  );
  assert.equal(
    translatedPage.window.document.querySelector('[hreflang="zh"]').href,
    "https://example.test/guidellm/zh/getting-started/install/",
  );

  home.window.close();
  translatedPage.window.close();
});

/** Confirms a future locale uses its language candidate to resolve the shared root. ## WRITTEN BY AI ## */
test("language switcher resolves a non-zh locale root", () => {
  const dom = runLanguageSwitcher({
    pagePath: "/guidellm/ja/guides/page/",
    logoPath: "/guidellm/ja/",
    locales: [{ language: "ja", path: "/guidellm/ja/guides/page/" }],
    routes: {
      "guides/page/": {
        translation: "ja/guides/page/",
        current: false,
      },
    },
  });

  assert.equal(
    dom.window.document.querySelector(
      "#guidellm-stale-translation a",
    ).href,
    "https://example.test/guidellm/guides/page/",
  );

  dom.window.close();
});

/** Confirms unprefixed English home and child routes keep the site base path. ## WRITTEN BY AI ## */
test("language switcher preserves unprefixed English home and subpage URLs", () => {
  const home = runLanguageSwitcher({
    pagePath: "/guidellm/",
    logoPath: "/guidellm/",
    locales: [
      { language: "en", path: "/guidellm/" },
      { language: "zh", path: "/guidellm/zh/" },
    ],
    routes,
  });
  const englishPage = runLanguageSwitcher({
    pagePath: "/guidellm/getting-started/install/",
    logoPath: "/guidellm/",
    locales: [
      { language: "en", path: "/guidellm/" },
      { language: "zh", path: "/guidellm/zh/" },
    ],
    routes,
  });

  assert.equal(
    home.window.document.querySelector('[hreflang="en"]').href,
    "https://example.test/guidellm/",
  );
  assert.equal(
    home.window.document.querySelector('[hreflang="zh"]').href,
    "https://example.test/guidellm/zh/",
  );
  assert.equal(
    englishPage.window.document.querySelector('[hreflang="en"]').href,
    "https://example.test/guidellm/getting-started/install/",
  );
  assert.equal(
    englishPage.window.document.querySelector('[hreflang="zh"]').href,
    "https://example.test/guidellm/zh/getting-started/install/",
  );

  home.window.close();
  englishPage.window.close();
});
