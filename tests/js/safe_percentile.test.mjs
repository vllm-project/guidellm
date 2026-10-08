/**
 * Unit test for safePercentile in the HTML report, which mirrors
 * DistributionSummary.from_values (see #1193).
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";

const source = fs.readFileSync(
  new URL("../../src/guidellm/benchmark/outputs/html_report/report.js", import.meta.url),
  "utf8"
);

// report.js is a single closure, so read the two top-level helpers out of it.
function extractFunction(name) {
  const start = source.indexOf(`  function ${name}(`);
  assert.ok(start >= 0, `${name} not found in report.js`);
  const end = source.indexOf("\n  }\n", start);
  return source.slice(start, end + 4);
}

const safePercentile = new Function(
  `${extractFunction("finiteNumbers")}\n${extractFunction("safePercentile")}\nreturn safePercentile;`
)();

function range(n) {
  return Array.from({ length: n }, (_, i) => i + 1);
}

/**
 * The exact inverse-CDF rank is ceil(q * n); the cumulative sum of 1/n used to
 * land just below it and pick the next value.
 * ## WRITTEN BY AI ##
 */
test("safePercentile returns the exact inverse-CDF rank", () => {
  assert.equal(safePercentile(range(20), 0.5), 10);
  assert.equal(safePercentile(range(10), 0.9), 9);
  assert.equal(safePercentile(range(300), 0.1), 30);
  assert.equal(safePercentile(range(300), 0.99), 297);
  assert.equal(safePercentile(range(300), 0.95), 285);
  assert.equal(safePercentile([3, 1, 2, 2], 0.5), 2);
});
