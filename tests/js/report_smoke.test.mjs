/**
 * DOM smoke test for the self-contained HTML report.
 * Expects GUIDELLM_HTML_FIXTURE to point at a multi-run benchmarks.html file.
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { JSDOM } from "jsdom";

const fixturePath = process.env.GUIDELLM_HTML_FIXTURE;
assert.ok(fixturePath, "GUIDELLM_HTML_FIXTURE must be set");
assert.ok(fs.existsSync(fixturePath), `fixture missing: ${fixturePath}`);

const html = fs.readFileSync(fixturePath, "utf8");

async function waitFor(predicate, { timeoutMs = 2000, intervalMs = 25 } = {}) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const value = predicate();
    if (value) return value;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  throw new Error(`timed out after ${timeoutMs}ms waiting for condition`);
}

test("multi-run HTML populates KPIs, comparison rows, and Performance SVGs", async () => {
  const dom = new JSDOM(html, {
    runScripts: "dangerously",
    resources: "usable",
    url: "https://example.test/benchmarks.html",
    pretendToBeVisual: true,
  });

  const { document, window } = dom.window;

  const kpiRps = await waitFor(() => {
    const node = document.getElementById("kpi-rps");
    if (!node) return null;
    const text = node.textContent.trim();
    if (!text || text === "—") return null;
    return node;
  });
  assert.ok(kpiRps);

  const staticSummary = document.getElementById("static-summary");
  assert.ok(staticSummary);
  assert.equal(staticSummary.hidden, true);

  assert.ok(window.GUIDELLM_REPORT);
  assert.equal(window.GUIDELLM_REPORT.header.multi_run, true);
  assert.ok((window.GUIDELLM_REPORT.runs || []).length >= 2);

  const compare = document.getElementById("table-compare");
  assert.ok(compare);
  const rows = compare.querySelectorAll("tbody tr");
  assert.equal(rows.length, window.GUIDELLM_REPORT.runs.length);

  const perfTab = document.querySelector('.tab-btn[data-tab="performance"]');
  assert.ok(perfTab);
  perfTab.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  const perfSvg = await waitFor(() =>
    document.querySelector("#panel-performance svg")
  );
  assert.ok(perfSvg, "expected an SVG after switching to Performance");
  assert.equal(perfSvg.getAttribute("role"), "img");
});
