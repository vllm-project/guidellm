(function () {
  "use strict";

  var data = window.GUIDELLM_REPORT || {};
  var scaleMode = "linear";
  var kpiRunIndex = null;
  var COLORS = {
    // Red Hat information + secondary palette (brand red reserved for UI chrome).
    p95: "#37a3a3", // teal-50
    p99: "#5e40be", // purple-50
    total: "#151515", // gray-95
    input: "#0066cc", // interaction-blue-50
    output: "#f5921b", // orange-40
    success: "#63993d", // success-green-50
    incomplete: "#f5921b", // orange-40
    errored: "#f0561d", // danger-orange-50
    line: "#e0e0e0", // gray-20
    ink: "#151515",
    muted: "#707070", // gray-50
    ttft: "#147878", // teal-60
    ttftAlt: "#004d4d", // teal-70
    tpot: "#ca6c0f", // orange-50
    tpotAlt: "#9e4a06", // orange-60
    ttfot: "#4394e5", // interaction-blue-40
    ttfotAlt: "#004d99", // interaction-blue-60
  };

  // Plain-language glossary for circle-? help.
  // Use `lines` for multiple points (each on its own line).
  // Use `also` for follow-on tips (P95/P99/etc.) shown below the main tip.
  var HELP = {
    p95: {
      title: "P95",
      lines: [
        "95th percentile: 95% of requests are at or below this value, and 5% are slower.",
        "A practical “almost worst case” for planning capacity.",
      ],
    },
    p99: {
      title: "P99",
      lines: [
        "99th percentile: 99% of requests are at or below this value.",
        "Highlights rare slow outliers that mean or median hide.",
      ],
    },
    median: {
      title: "Median",
      lines: [
        "Middle value when all measurements are sorted—typical case, less skewed by extremes than the mean.",
      ],
    },
    mean: {
      title: "Mean",
      lines: ["Ordinary average of the measured values."],
    },
    tok_s: {
      title: "Tokens / sec",
      lines: [
        "Tokens per second (tok/s): how many tokens the system handles each second.",
        "Tokens are the small chunks of text models read and write.",
      ],
    },
    concurrency: {
      title: "Concurrency",
      lines: [
        "How many requests are in flight at once.",
        "Higher concurrency loads the server harder and often raises latency.",
      ],
    },
    rps: {
      title: "Requests / sec",
      lines: [
        "Completed requests per second.",
        "Higher means the deployment handled more work in the same time.",
      ],
    },
    out_tps: {
      title: "Output tokens / sec",
      lines: [
        "How fast the model produces response tokens across all requests.",
        "The main “generation speed” throughput number.",
      ],
    },
    total_tps: {
      title: "Total tokens / sec",
      lines: [
        "Input (prompt) tokens plus output tokens processed per second.",
        "A single score for overall token-moving capacity.",
      ],
    },
    in_tps: {
      title: "Input tokens / sec",
      lines: [
        "How fast prompt/input tokens are processed.",
        "Large prompts raise this and can also raise latency.",
      ],
    },
    request_latency: {
      title: "Request latency",
      lines: [
        "End-to-end time from sending a request until the full response finishes.",
        "Lower is better for snappy UX.",
      ],
      also: ["p95", "p99"],
    },
    e2e: {
      title: "E2E latency",
      lines: [
        "End-to-end (E2E) latency: full request time from start to finished response.",
        "Same idea as “request latency.”",
      ],
      also: ["p95", "p99"],
    },
    ttft: {
      title: "TTFT",
      lines: [
        "Time to First Token (TTFT): wait until the first piece of the reply appears.",
        "High TTFT feels like the UI is stuck before any text shows.",
      ],
      also: ["p95", "p99"],
    },
    itl: {
      title: "ITL",
      lines: [
        "Inter-Token Latency (ITL): average time between tokens after the first.",
        "High ITL makes streaming feel choppy or sluggish.",
      ],
      also: ["p95", "p99"],
    },
    tpot: {
      title: "TPOT",
      lines: [
        "Time Per Output Token (TPOT): average time to produce each output token, including the first.",
        "A single “cost per token” speed figure; lower is better.",
      ],
      also: ["p95", "p99"],
    },
    ttfot: {
      title: "TTFOT",
      lines: [
        "Time to First Output Token (TTFOT): wait until the first visible content token (ignoring reasoning-only tokens).",
        "Differs from TTFT when the model thinks before showing text.",
      ],
      also: ["p95", "p99"],
    },
    error_rate: {
      title: "Error rate",
      lines: [
        "Share of requests that failed.",
        "0% means every request in this run completed without an error.",
      ],
    },
    metrics_for: {
      title: "Metrics for",
      lines: [
        "Which run fills the KPI cards above.",
        "Comparison charts on this page keep showing every run.",
      ],
    },
    profile: {
      title: "Profile",
      lines: [
        "Load pattern used for the benchmark (for example fixed rate, concurrent streams, or a sweep across loads).",
      ],
    },
    target: {
      title: "Target",
      lines: ["The model server URL GuideLLM sent traffic to."],
    },
    peak_tok_s: {
      title: "Peak tok/s",
      lines: [
        "Runs marked “peak tok/s” have the highest total tokens per second.",
        "KPIs default to that run until you pick another.",
      ],
    },
    lat_by_conc: {
      title: "Request latency by concurrency",
      lines: [
        "E2E is full request time from start to finished response.",
        "Rising curves as concurrency increases usually mean the server is saturating.",
      ],
      also: ["p95", "p99"],
    },
    outcomes: {
      title: "Request outcomes",
      lines: [
        "Successful: finished normally with a usable response.",
        "Incomplete: started but never finished (timeout or interrupt).",
        "Errored: failed with an error.",
      ],
    },
    successful: {
      title: "Successful",
      lines: ["Requests that finished normally with a usable response."],
    },
    incomplete: {
      title: "Incomplete",
      lines: [
        "Requests that started but never finished—often timeouts, cancellations, or client disconnects.",
      ],
    },
    errored: {
      title: "Errored",
      lines: [
        "Requests that failed with an error (bad request, server fault, or invalid response).",
      ],
    },
    lat_components: {
      title: "Latency components",
      lines: [
        "TTFT: wait for the first token.",
        "ITL: average gap between later tokens.",
        "TPOT: average time per output token including the first.",
      ],
      also: ["p95", "p99"],
    },
    gen_latency: {
      title: "Generation latency",
      lines: [
        "ITL: average gap between streamed tokens after the first.",
        "TPOT: average time per output token including the first.",
      ],
      also: ["p95", "p99"],
    },
    token_throughput: {
      title: "Token throughput",
      lines: [
        "Input tok/s: prompt processing rate.",
        "Output tok/s: generation rate.",
      ],
    },
    efficiency: {
      title: "Throughput efficiency",
      lines: [
        "Total tokens/sec divided by concurrency—token throughput per parallel request.",
        "Falling efficiency as concurrency rises often means contention or saturation.",
      ],
    },
    scale_toggle: {
      title: "Axis scale",
      lines: [
        "Linear: equal steps on the axis.",
        "Log: compresses wide ranges so small and large values fit on one chart.",
        "Bar charts still start at zero.",
      ],
    },
    by_turn: {
      title: "By turn",
      lines: [
        "Each turn is one more user/model exchange in a multi-turn chat.",
        "Later turns usually include more history, so prompts and latency often grow.",
      ],
    },
    lat_vs_turn: {
      title: "Latency vs turn",
      lines: [
        "E2E: full response time (P95 and P99).",
        "TTFT: time to first token (P95).",
        "ITL: average time between later tokens (P95).",
        "Watch whether later turns get slower as history grows.",
      ],
      also: ["p95", "p99"],
    },
    prompt_vs_turn: {
      title: "Prompt tokens vs turn",
      lines: [
        "Prompt size (tokens) at each turn, usually rising as conversation history is appended.",
      ],
      also: ["median", "p95"],
    },
    history_median: {
      title: "History median",
      lines: [
        "Typical length of prior conversation context included in the prompt at that turn.",
      ],
      also: ["median"],
    },
    turn: {
      title: "Turn",
      lines: [
        "Step number in a multi-turn conversation.",
        "Turn 1 is the first exchange; higher numbers have more history.",
      ],
    },
    extra_latency: {
      title: "Extra latency",
      lines: [
        "TPOT: average time per output token including the first.",
      ],
      also: ["p95", "p99"],
    },
    request_size: {
      title: "Request size",
      lines: [
        "Prompt tokens: input size (including history).",
        "Output tokens: generated response size.",
        "Bigger prompts often mean higher latency.",
      ],
      also: ["median", "p95"],
    },
    prompt_tokens: {
      title: "Prompt tokens",
      lines: [
        "How large the input to the model was, in tokens (including any chat history).",
      ],
      also: ["median", "p95"],
    },
    output_tokens: {
      title: "Output tokens",
      lines: ["How large the model’s generated response was, in tokens."],
      also: ["median", "p95"],
    },
    ws_rtt: {
      title: "WebSocket round-trip",
      lines: [
        "For realtime WebSocket streaming: timing between packets you send and tokens you receive.",
        "Avg RTT is typical lag; Last RTT is lag at the end of the stream.",
      ],
      also: ["p95", "p99"],
    },
    avg_rtt: {
      title: "Avg RTT",
      lines: [
        "Average round-trip time (Avg RTT): approximate mean lag from sent packets to received tokens in a WebSocket request.",
      ],
      also: ["p95", "p99"],
    },
    last_rtt: {
      title: "Last RTT",
      lines: [
        "Time from the last sent packet to the last received token—how long the stream’s tail lags after the final input.",
      ],
      also: ["p95", "p99"],
    },
    modality: {
      title: "Modality metrics",
      lines: [
        "Usage for this content type (tokens, pixels, audio seconds, and so on).",
        "Split into what you sent (input) and what the model returned (output).",
      ],
      also: ["mean", "p95"],
    },
    in_out: {
      title: "Input / Output",
      lines: [
        "Input: content sent into the model.",
        "Output: content the model produced for this modality.",
      ],
    },
  };

  var helpTipEl = null;
  var pinnedHelpBtn = null;

  function ensureHelpTip() {
    if (helpTipEl) return helpTipEl;
    helpTipEl = document.createElement("div");
    helpTipEl.className = "help-tip";
    helpTipEl.hidden = true;
    helpTipEl.setAttribute("role", "tooltip");
    document.body.appendChild(helpTipEl);
    return helpTipEl;
  }

  function helpEntry(key) {
    return HELP[key] || null;
  }

  function helpLines(entry) {
    if (!entry) return [];
    if (entry.lines && entry.lines.length) return entry.lines;
    if (entry.body) return [entry.body];
    return [];
  }

  function helpTipHtml(entry) {
    var html =
      "<div class='help-tip-title'>" + escapeTip(entry.title) + "</div>";
    html += "<div class='help-tip-body'>";
    helpLines(entry).forEach(function (line) {
      html += "<div class='help-tip-line'>" + escapeTip(line) + "</div>";
    });
    html += "</div>";
    var also = entry.also || [];
    if (also.length) {
      html += "<div class='help-tip-also'>";
      also.forEach(function (key) {
        var related = helpEntry(key);
        if (!related) return;
        html += "<div class='help-tip-also-item'>";
        html +=
          "<div class='help-tip-also-title'>" +
          escapeTip(related.title) +
          "</div>";
        helpLines(related).forEach(function (line) {
          html += "<div class='help-tip-line'>" + escapeTip(line) + "</div>";
        });
        html += "</div>";
      });
      html += "</div>";
    }
    return html;
  }

  function positionHelpTip(anchor, evt) {
    var tip = ensureHelpTip();
    var pad = 10;
    var x;
    var y;
    if (evt && evt.clientX != null) {
      x = evt.clientX + pad;
      y = evt.clientY + pad;
    } else {
      var rect = anchor.getBoundingClientRect();
      x = rect.left;
      y = rect.bottom + 6;
    }
    tip.hidden = false;
    var box = tip.getBoundingClientRect();
    if (x + box.width > window.innerWidth - 8) x = window.innerWidth - box.width - 8;
    if (y + box.height > window.innerHeight - 8) y = Math.max(8, y - box.height - 16);
    tip.style.left = Math.max(8, x) + "px";
    tip.style.top = Math.max(8, y) + "px";
  }

  function openHelpTip(btn, evt) {
    var entry = btn._helpEntry || helpEntry(btn.getAttribute("data-help"));
    if (!entry) return;
    var tip = ensureHelpTip();
    tip.innerHTML = helpTipHtml(entry);
    positionHelpTip(btn, evt);
  }

  function hideHelpTip() {
    if (pinnedHelpBtn) return;
    if (helpTipEl) helpTipEl.hidden = true;
  }

  function unpinHelp() {
    if (pinnedHelpBtn) pinnedHelpBtn.classList.remove("is-pinned");
    pinnedHelpBtn = null;
    if (helpTipEl) helpTipEl.hidden = true;
  }

  function bindHelpControl(btn) {
    if (!btn || btn._helpBound) return;
    btn._helpBound = true;
    btn.addEventListener("pointerenter", function (evt) {
      if (pinnedHelpBtn && pinnedHelpBtn !== btn) return;
      openHelpTip(btn, evt);
    });
    btn.addEventListener("pointermove", function (evt) {
      if (pinnedHelpBtn && pinnedHelpBtn !== btn) return;
      if (helpTipEl && !helpTipEl.hidden) positionHelpTip(btn, evt);
    });
    btn.addEventListener("pointerleave", function () {
      if (pinnedHelpBtn === btn) return;
      hideHelpTip();
    });
    btn.addEventListener("focus", function () {
      openHelpTip(btn, null);
    });
    btn.addEventListener("blur", function () {
      if (pinnedHelpBtn === btn) return;
      hideHelpTip();
    });
    btn.addEventListener("click", function (evt) {
      evt.preventDefault();
      evt.stopPropagation();
      if (pinnedHelpBtn === btn) {
        unpinHelp();
        return;
      }
      unpinHelp();
      pinnedHelpBtn = btn;
      btn.classList.add("is-pinned");
      openHelpTip(btn, evt);
    });
  }

  function helpIcon(key) {
    return helpIconFromEntry(helpEntry(key), key);
  }

  function helpIconFromEntry(entry, key) {
    if (!entry) return null;
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "help-q";
    if (key) btn.setAttribute("data-help", key);
    btn._helpEntry = entry;
    btn.setAttribute("aria-label", "About " + entry.title);
    btn.setAttribute("aria-expanded", "false");
    btn.textContent = "?";
    bindHelpControl(btn);
    return btn;
  }

  function labelWithHelp(text, helpKey) {
    var wrap = document.createElement("span");
    wrap.className = "label-with-help";
    wrap.appendChild(document.createTextNode(text));
    var icon = helpIcon(helpKey);
    if (icon) wrap.appendChild(icon);
    return wrap;
  }

  // helpSpec: glossary key string, or { title, lines, also } for chart-specific tips.
  function setTitleWithHelp(id, text, helpSpec) {
    var node = el(id);
    if (!node) return;
    clear(node);
    var heading = document.createElement("span");
    heading.className = "card-heading";
    heading.textContent = text;
    node.appendChild(heading);
    var icon =
      typeof helpSpec === "string"
        ? helpIcon(helpSpec)
        : helpIconFromEntry(helpSpec, null);
    if (icon) node.appendChild(icon);
  }

  function headerCell(text, helpKey) {
    return helpKey ? { text: text, help: helpKey } : text;
  }

  // Build tips that only list metrics present in the current chart/table.
  function helpLatComponents(showTtfot) {
    var lines = ["TTFT: wait for the first token."];
    if (showTtfot) lines.push("TTFOT: wait for the first content token.");
    lines.push("ITL: average gap between later tokens.");
    lines.push("TPOT: average time per output token including the first.");
    return {
      title: "Latency components",
      lines: lines,
      also: ["p95", "p99"],
    };
  }

  function helpFirstToken(showTtfot) {
    var lines = [
      "TTFT: wait until the first piece of the reply appears.",
      "High TTFT feels like the UI is stuck before any text shows.",
    ];
    if (showTtfot) {
      lines.push(
        "TTFOT: wait until the first visible content token (ignoring reasoning-only tokens)."
      );
    }
    return {
      title: showTtfot ? "First-token latency" : "TTFT",
      lines: lines,
      also: ["p95", "p99"],
    };
  }

  function helpGenLatency() {
    return {
      title: "Generation latency",
      lines: [
        "ITL: average gap between streamed tokens after the first.",
        "TPOT: average time per output token including the first.",
      ],
      also: ["p95", "p99"],
    };
  }

  function helpTokenThroughput(multi) {
    if (multi) {
      return {
        title: "Token throughput",
        lines: [
          "Input tok/s: prompt processing rate.",
          "Output tok/s: generation rate.",
          "Each bar stacks input and output at that concurrency.",
        ],
      };
    }
    return {
      title: "Throughput rates",
      lines: [
        "Req/s: completed requests per second.",
        "Input tok/s: prompt processing rate.",
        "Output tok/s: generation rate.",
        "Total tok/s: input and output combined.",
      ],
      also: ["mean"],
    };
  }

  function helpExtraLatency(showTtfot) {
    var lines = [
      "TPOT: average time per output token including the first.",
    ];
    if (showTtfot) {
      lines.push(
        "TTFOT: time to first content token when it differs from TTFT (for example reasoning before visible text)."
      );
    }
    return {
      title: "Extra latency",
      lines: lines,
      also: ["p95", "p99"],
    };
  }

  function hydrateHelp(root) {
    var scope = root || document;
    scope.querySelectorAll("button.help-q[data-help]").forEach(function (btn) {
      var entry = helpEntry(btn.getAttribute("data-help"));
      if (entry) btn._helpEntry = entry;
      if (entry && !btn.getAttribute("aria-label")) {
        btn.setAttribute("aria-label", "About " + entry.title);
      }
      if (!btn.textContent) btn.textContent = "?";
      bindHelpControl(btn);
    });
  }

  document.addEventListener("click", function (evt) {
    if (!pinnedHelpBtn) return;
    if (evt.target.closest && evt.target.closest(".help-q")) return;
    if (helpTipEl && helpTipEl.contains(evt.target)) return;
    unpinHelp();
  });
  document.addEventListener("keydown", function (evt) {
    if (evt.key === "Escape") unpinHelp();
  });

  function fmt(value, digits) {
    if (value == null || Number.isNaN(value)) return "—";
    var n = Number(value);
    if (Math.abs(n) >= 100) return n.toFixed(0);
    if (Math.abs(n) >= 10) return n.toFixed(1);
    return n.toFixed(digits == null ? 2 : digits);
  }

  function pct(value) {
    if (value == null || Number.isNaN(value)) return "—";
    return (Number(value) * 100).toFixed(2) + "%";
  }

  function el(id) {
    return document.getElementById(id);
  }

  function setText(id, text) {
    var node = el(id);
    if (node) node.textContent = text;
  }

  function activateTab(name) {
    document.querySelectorAll(".tab-btn").forEach(function (btn) {
      btn.classList.toggle("active", btn.getAttribute("data-tab") === name);
    });
    document.querySelectorAll(".panel").forEach(function (panel) {
      panel.classList.toggle("active", panel.id === "panel-" + name);
    });
  }

  function setupTabs() {
    var header = data.header || {};
    if (!header.has_multi_turn) {
      var turnBtn = document.querySelector('.tab-btn[data-tab="turn"]');
      var turnPanel = el("panel-turn");
      if (turnBtn) turnBtn.style.display = "none";
      if (turnPanel) turnPanel.style.display = "none";
    }
    document.querySelectorAll(".tab-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        activateTab(btn.getAttribute("data-tab"));
      });
    });
    activateTab("summary");
  }

  function setupScaleToggle() {
    document.querySelectorAll("[data-scale]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        scaleMode = btn.getAttribute("data-scale");
        document.querySelectorAll("[data-scale]").forEach(function (b) {
          b.classList.toggle("active", b.getAttribute("data-scale") === scaleMode);
        });
        renderCharts();
      });
    });
  }

  function renderHeader() {
    var h = data.header || {};
    setText("meta-model", h.model || "N/A");
    setText("meta-target", h.target || "N/A");
    setText("meta-profile", h.profile || "N/A");
    setText("meta-time", h.timestamp || "N/A");
    setText("meta-version", h.guidellm_version || "N/A");
    setupKpiRunSelect();
    applyKpisFromSelectedRun();
  }

  function peakIndex() {
    var idx = data.header && data.header.peak_index;
    return idx == null ? 0 : idx;
  }

  function selectedRunIndex() {
    if (kpiRunIndex == null) return peakIndex();
    return kpiRunIndex;
  }

  function selectedRun() {
    var rows = runs();
    return rows[selectedRunIndex()] || rows[0] || {};
  }

  function applyKpisFromSelectedRun() {
    var run = selectedRun();
    setText("kpi-rps", fmt(run.request_rate));
    setText("kpi-out-tps", fmt(run.output_tps));
    setText("kpi-total-tps", fmt(run.total_tps));
    setText("kpi-lat-p95", fmt(run.request_latency_p95_ms));
    setText("kpi-lat-p99", fmt(run.request_latency_p99_ms));
    setText("kpi-ttft-p95", fmt(run.ttft_p95_ms));
    setText("kpi-ttft-p99", fmt(run.ttft_p99_ms));
    setText("kpi-itl-p95", fmt(run.itl_p95_ms));
    setText("kpi-itl-p99", fmt(run.itl_p99_ms));
    setText("kpi-error", pct(run.error_rate));
  }

  function setupKpiRunSelect() {
    var picker = el("kpi-run-picker");
    var select = el("kpi-run-select");
    if (!picker || !select) return;
    var rows = runs();
    var multi = isMultiRun();
    picker.hidden = !multi;
    if (!multi) {
      kpiRunIndex = 0;
      return;
    }
    if (kpiRunIndex == null) kpiRunIndex = peakIndex();
    clear(select);
    rows.forEach(function (run, idx) {
      var opt = document.createElement("option");
      opt.value = String(idx);
      var label = runCategoryLabel(run);
      if (idx === peakIndex()) label += " — peak tok/s";
      opt.textContent = label;
      if (idx === kpiRunIndex) opt.selected = true;
      select.appendChild(opt);
    });
    select.onchange = function () {
      kpiRunIndex = Number(select.value) || 0;
      applyKpisFromSelectedRun();
      renderSelectedRunCharts();
    };
  }

  function niceNum(range, round) {
    var exp = Math.floor(Math.log10(range || 1));
    var frac = range / Math.pow(10, exp);
    var nice;
    if (round) {
      if (frac < 1.5) nice = 1;
      else if (frac < 3) nice = 2;
      else if (frac < 7) nice = 5;
      else nice = 10;
    } else {
      if (frac <= 1) nice = 1;
      else if (frac <= 2) nice = 2;
      else if (frac <= 5) nice = 5;
      else nice = 10;
    }
    return nice * Math.pow(10, exp);
  }

  function scale(value, min, max, a, b, log) {
    if (log) {
      var minL = Math.log10(Math.max(min, 1e-9));
      // Must match axis tick math: use the passed max, not min*10.
      // Forcing a full decade here shrunk bars while ticks still used xExt.max.
      var maxL = Math.log10(Math.max(max, 1e-8));
      if (maxL <= minL) maxL = minL + 1;
      var vL = Math.log10(Math.max(value, 1e-9));
      return a + ((vL - minL) / (maxL - minL || 1)) * (b - a);
    }
    return a + ((value - min) / (max - min || 1)) * (b - a);
  }

  function extent(values, log) {
    var nums = values.filter(function (v) {
      return v != null && !Number.isNaN(Number(v)) && (!log || Number(v) > 0);
    }).map(Number);
    if (!nums.length) return { min: 0, max: 1 };
    var min = Math.min.apply(null, nums);
    var max = Math.max.apply(null, nums);
    if (min === max) {
      if (log) return { min: min / 2 || 0.1, max: max * 2 || 1 };
      return { min: min * 0.9, max: max * 1.1 || 1 };
    }
    return { min: min, max: max };
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function svgEl(name, attrs) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.keys(attrs || {}).forEach(function (key) {
      node.setAttribute(key, attrs[key]);
    });
    return node;
  }

  // Lightweight hover tooltips — no chart library, one shared DOM node.
  var chartTipEl = null;
  function ensureChartTip() {
    if (chartTipEl) return chartTipEl;
    chartTipEl = document.createElement("div");
    chartTipEl.className = "chart-tip";
    chartTipEl.hidden = true;
    document.body.appendChild(chartTipEl);
    return chartTipEl;
  }

  function escapeTip(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function tipMarkup(title, lines) {
    var html = "<div class='chart-tip-title'>" + escapeTip(title) + "</div>";
    (lines || []).forEach(function (line) {
      html += "<div>" + escapeTip(line) + "</div>";
    });
    return html;
  }

  function moveChartTip(evt) {
    var tip = ensureChartTip();
    var pad = 12;
    var x = evt.clientX + pad;
    var y = evt.clientY + pad;
    tip.hidden = false;
    var rect = tip.getBoundingClientRect();
    if (x + rect.width > window.innerWidth - 8) x = evt.clientX - rect.width - pad;
    if (y + rect.height > window.innerHeight - 8) y = evt.clientY - rect.height - pad;
    tip.style.left = Math.max(8, x) + "px";
    tip.style.top = Math.max(8, y) + "px";
  }

  function bindChartTip(node, title, lines) {
    if (!node) return;
    node.style.cursor = "pointer";
    var html = tipMarkup(title, lines);
    node.addEventListener("pointerenter", function (evt) {
      var tip = ensureChartTip();
      tip.innerHTML = html;
      tip.hidden = false;
      moveChartTip(evt);
    });
    node.addEventListener("pointermove", moveChartTip);
    node.addEventListener("pointerleave", function () {
      if (chartTipEl) chartTipEl.hidden = true;
    });
  }

  function drawAxes(svg, plot, xExt, yExt, xLabel, yLabel, logX, logY, skipXTicks) {
    svg.appendChild(
      svgEl("rect", {
        x: plot.left,
        y: plot.top,
        width: plot.width,
        height: plot.height,
        fill: "#fff",
        stroke: COLORS.line,
      })
    );
    for (var i = 0; i <= 4; i++) {
      var yRatio = i / 4;
      var y = plot.top + plot.height - yRatio * plot.height;
      svg.appendChild(
        svgEl("line", {
          x1: plot.left,
          y1: y,
          x2: plot.left + plot.width,
          y2: y,
          stroke: COLORS.line,
          "stroke-dasharray": "3 4",
        })
      );
      var yVal = logY
        ? Math.pow(10, Math.log10(Math.max(yExt.min, 1e-9)) + yRatio * (Math.log10(Math.max(yExt.max, 1e-8)) - Math.log10(Math.max(yExt.min, 1e-9))))
        : yExt.min + yRatio * (yExt.max - yExt.min);
      var label = svgEl("text", {
        x: plot.left - 8,
        y: y + 4,
        "text-anchor": "end",
        fill: COLORS.muted,
        "font-size": "11",
      });
      label.textContent = fmt(yVal);
      svg.appendChild(label);
    }
    if (!skipXTicks) {
      for (var j = 0; j <= 4; j++) {
        var xRatio = j / 4;
        var x = plot.left + xRatio * plot.width;
        var xVal = logX
          ? Math.pow(10, Math.log10(Math.max(xExt.min, 1e-9)) + xRatio * (Math.log10(Math.max(xExt.max, 1e-8)) - Math.log10(Math.max(xExt.min, 1e-9))))
          : xExt.min + xRatio * (xExt.max - xExt.min);
        var xlab = svgEl("text", {
          x: x,
          y: plot.top + plot.height + 18,
          "text-anchor": "middle",
          fill: COLORS.muted,
          "font-size": "11",
        });
        xlab.textContent = fmt(xVal);
        svg.appendChild(xlab);
      }
    }
    if (xLabel) {
      var xl = svgEl("text", {
        x: plot.left + plot.width / 2,
        y: plot.top + plot.height + (skipXTicks ? 40 : 36),
        "text-anchor": "middle",
        fill: COLORS.ink,
        "font-size": "12",
      });
      xl.textContent = xLabel;
      svg.appendChild(xl);
    }
    if (yLabel) {
      var yl = svgEl("text", {
        x: 16,
        y: plot.top + plot.height / 2,
        "text-anchor": "middle",
        fill: COLORS.ink,
        "font-size": "12",
        transform: "rotate(-90 16 " + (plot.top + plot.height / 2) + ")",
      });
      yl.textContent = yLabel;
      svg.appendChild(yl);
    }
  }

  function linePath(points) {
    if (!points.length) return "";
    return points
      .map(function (p, idx) {
        return (idx ? "L" : "M") + p[0] + " " + p[1];
      })
      .join(" ");
  }

  // Distinct markers so multi-series latency charts stay readable without color.
  function drawMarker(svg, shape, x, y, color, size) {
    var s = size == null ? 4 : size;
    var node;
    if (shape === "square") {
      node = svgEl("rect", {
        x: x - s,
        y: y - s,
        width: s * 2,
        height: s * 2,
        fill: color,
        stroke: "#fff",
        "stroke-width": "1",
      });
    } else if (shape === "diamond") {
      node = svgEl("polygon", {
        points:
          x +
          "," +
          (y - s - 0.5) +
          " " +
          (x + s + 0.5) +
          "," +
          y +
          " " +
          x +
          "," +
          (y + s + 0.5) +
          " " +
          (x - s - 0.5) +
          "," +
          y,
        fill: color,
        stroke: "#fff",
        "stroke-width": "1",
      });
    } else if (shape === "triangle") {
      node = svgEl("polygon", {
        points:
          x +
          "," +
          (y - s - 0.5) +
          " " +
          (x + s + 0.5) +
          "," +
          (y + s) +
          " " +
          (x - s - 0.5) +
          "," +
          (y + s),
        fill: color,
        stroke: "#fff",
        "stroke-width": "1",
      });
    } else if (shape === "triangle-down") {
      node = svgEl("polygon", {
        points:
          (x - s - 0.5) +
          "," +
          (y - s) +
          " " +
          (x + s + 0.5) +
          "," +
          (y - s) +
          " " +
          x +
          "," +
          (y + s + 0.5),
        fill: color,
        stroke: "#fff",
        "stroke-width": "1",
      });
    } else if (shape === "cross") {
      node = svgEl("path", {
        d:
          "M" +
          (x - s) +
          " " +
          y +
          "H" +
          (x + s) +
          "M" +
          x +
          " " +
          (y - s) +
          "V" +
          (y + s),
        fill: "none",
        stroke: color,
        "stroke-width": "2.2",
        "stroke-linecap": "round",
      });
    } else if (shape === "x") {
      node = svgEl("path", {
        d:
          "M" +
          (x - s) +
          " " +
          (y - s) +
          "L" +
          (x + s) +
          " " +
          (y + s) +
          "M" +
          (x + s) +
          " " +
          (y - s) +
          "L" +
          (x - s) +
          " " +
          (y + s),
        fill: "none",
        stroke: color,
        "stroke-width": "2.2",
        "stroke-linecap": "round",
      });
    } else {
      // Default: circle
      node = svgEl("circle", {
        cx: x,
        cy: y,
        r: s,
        fill: color,
        stroke: "#fff",
        "stroke-width": "1",
      });
    }
    svg.appendChild(node);
    return node;
  }

  function drawLineChart(containerId, series, options) {
    var host = el(containerId);
    if (!host) return;
    clear(host);
    var width = host.clientWidth || 480;
    var height = host.clientHeight || 280;
    var plot = { left: 52, top: 16, width: width - 72, height: height - 64 };
    var log = scaleMode === "log";
    var logX = !!(options && options.logX && log);
    var logY = !!(options && options.logY && log);
    var xs = [];
    var ys = [];
    series.forEach(function (s) {
      s.points.forEach(function (p) {
        if (p[0] != null) xs.push(p[0]);
        if (p[1] != null) ys.push(p[1]);
      });
    });
    var xExt = extent(xs, logX);
    var yExt = extent(ys, logY);
    var svg = svgEl("svg", {
      viewBox: "0 0 " + width + " " + height,
      role: "img",
    });
    drawAxes(
      svg,
      plot,
      xExt,
      yExt,
      options && options.xLabel,
      options && options.yLabel,
      logX,
      logY
    );
    series.forEach(function (s) {
      var rawPts = s.points.filter(function (p) {
        return p[0] != null && p[1] != null && (!logX || p[0] > 0) && (!logY || p[1] > 0);
      });
      var pts = rawPts.map(function (p) {
        return [
          scale(p[0], xExt.min, xExt.max, plot.left, plot.left + plot.width, logX),
          scale(p[1], yExt.min, yExt.max, plot.top + plot.height, plot.top, logY),
        ];
      });
      if (!pts.length) return;
      svg.appendChild(
        svgEl("path", {
          d: linePath(pts),
          fill: "none",
          stroke: s.color,
          "stroke-width": s.width || 2.5,
          "stroke-dasharray": s.dash || null,
        })
      );
      pts.forEach(function (p, pi) {
        var marker = drawMarker(svg, s.shape || "circle", p[0], p[1], s.color, 4);
        // Larger invisible hit target for easier hover.
        var hit = svgEl("circle", {
          cx: p[0],
          cy: p[1],
          r: 10,
          fill: "transparent",
        });
        svg.appendChild(hit);
        var xLabel = (options && options.xLabel) || "X";
        var yLabel = (options && options.yLabel) || "Y";
        bindChartTip(hit, s.label || "Series", [
          xLabel + ": " + fmt(rawPts[pi][0]),
          yLabel + ": " + fmt(rawPts[pi][1]),
        ]);
        if (marker) bindChartTip(marker, s.label || "Series", [
          xLabel + ": " + fmt(rawPts[pi][0]),
          yLabel + ": " + fmt(rawPts[pi][1]),
        ]);
      });
    });
    var legendY = 14;
    var legendGap = Math.max(78, Math.min(110, plot.width / Math.max(series.length, 1)));
    series.forEach(function (s, idx) {
      var x = plot.left + idx * legendGap;
      svg.appendChild(
        svgEl("line", {
          x1: x,
          y1: legendY - 3,
          x2: x + 14,
          y2: legendY - 3,
          stroke: s.color,
          "stroke-width": "2.5",
          "stroke-dasharray": s.dash || null,
        })
      );
      drawMarker(svg, s.shape || "circle", x + 7, legendY - 3, s.color, 3.5);
      var t = svgEl("text", {
        x: x + 18,
        y: legendY,
        fill: COLORS.muted,
        "font-size": "11",
      });
      t.textContent = s.label;
      svg.appendChild(t);
    });
    host.appendChild(svg);
  }

  function drawStackedBars(containerId, categories, stacks, options) {
    var host = el(containerId);
    if (!host) return;
    clear(host);
    var width = host.clientWidth || 480;
    var height = host.clientHeight || 280;
    var plot = { left: 52, top: 28, width: width - 72, height: height - 78 };
    var totals = categories.map(function (_, i) {
      return stacks.reduce(function (sum, s) {
        return sum + (Number(s.values[i]) || 0);
      }, 0);
    });
    var yExt = extent(totals.concat([0]), false);
    yExt.min = 0;
    var svg = svgEl("svg", { viewBox: "0 0 " + width + " " + height });
    drawAxes(
      svg,
      plot,
      { min: 0, max: 1 },
      yExt,
      options && options.xLabel,
      options && options.yLabel,
      false,
      false,
      true
    );
    var slot = plot.width / Math.max(categories.length, 1);
    var barW = Math.max(18, Math.min(64, slot * 0.45));
    categories.forEach(function (label, i) {
      var x = plot.left + i * slot + (slot - barW) / 2;
      var yBase = plot.top + plot.height;
      stacks.forEach(function (stack) {
        var val = Number(stack.values[i]) || 0;
        var h = ((val - yExt.min) / (yExt.max - yExt.min || 1)) * plot.height;
        yBase -= h;
        var bar = svgEl("rect", {
          x: x,
          y: yBase,
          width: barW,
          height: Math.max(h, 0),
          fill: stack.color,
        });
        svg.appendChild(bar);
        bindChartTip(bar, label, [(stack.label || "Series") + ": " + fmt(val)]);
      });
      var t = svgEl("text", {
        x: x + barW / 2,
        y: plot.top + plot.height + 18,
        "text-anchor": "middle",
        fill: COLORS.muted,
        "font-size": "11",
      });
      t.textContent = label;
      svg.appendChild(t);
    });
    // Legend for stack colors (only when labels are provided)
    var labeled = stacks.filter(function (stack) {
      return !!stack.label;
    });
    if (labeled.length) {
      var legendY = 14;
      labeled.forEach(function (stack, idx) {
        var lx = plot.left + idx * 100;
        svg.appendChild(
          svgEl("rect", {
            x: lx,
            y: legendY - 8,
            width: 10,
            height: 10,
            fill: stack.color,
            rx: 2,
          })
        );
        var lt = svgEl("text", {
          x: lx + 14,
          y: legendY,
          fill: COLORS.muted,
          "font-size": "11",
        });
        lt.textContent = stack.label;
        svg.appendChild(lt);
      });
    }
    host.appendChild(svg);
  }

  function drawGroupedBars(containerId, categories, groups, options) {
    var host = el(containerId);
    if (!host) return;
    clear(host);
    var width = host.clientWidth || 480;
    var height = host.clientHeight || 280;
    var plot = { left: 52, top: 28, width: width - 72, height: height - 78 };
    // Bars always baseline at 0. True log cannot include 0, so the log toggle
    // only applies to line charts — otherwise bar length stops encoding magnitude.
    var logY = false;
    var all = [];
    groups.forEach(function (g) {
      g.values.forEach(function (v) {
        if (v != null) all.push(v);
      });
    });
    var yExt = extent(all, false);
    yExt.min = Math.min(0, yExt.min);
    var svg = svgEl("svg", { viewBox: "0 0 " + width + " " + height });
    drawAxes(
      svg,
      plot,
      { min: 0, max: 1 },
      yExt,
      options && options.xLabel,
      options && options.yLabel,
      false,
      logY,
      true
    );

    var nGroups = Math.max(groups.length, 1);
    var nCats = Math.max(categories.length, 1);
    var slot = plot.width / nCats;
    var innerGap = 6;
    var sidePad = Math.max(10, slot * 0.18);
    var usable = Math.max(24, slot - sidePad * 2);
    var barW = Math.max(
      10,
      Math.min(36, (usable - innerGap * (nGroups - 1)) / nGroups)
    );
    var clusterW = nGroups * barW + (nGroups - 1) * innerGap;

    categories.forEach(function (label, i) {
      var clusterLeft = plot.left + i * slot + (slot - clusterW) / 2;
      groups.forEach(function (g, gi) {
        var val = g.values[i];
        if (val == null || (logY && val <= 0)) return;
        var x = clusterLeft + gi * (barW + innerGap);
        var y = scale(val, yExt.min, yExt.max, plot.top + plot.height, plot.top, logY);
        var bar = svgEl("rect", {
          x: x,
          y: y,
          width: barW,
          height: Math.max(0, plot.top + plot.height - y),
          fill: g.color,
          rx: 2,
        });
        svg.appendChild(bar);
        bindChartTip(bar, label, [(g.label || "Value") + ": " + fmt(val)]);
      });
      var t = svgEl("text", {
        x: plot.left + i * slot + slot / 2,
        y: plot.top + plot.height + 18,
        "text-anchor": "middle",
        fill: COLORS.ink,
        "font-size": "12",
        "font-weight": "600",
      });
      t.textContent = label;
      svg.appendChild(t);
    });

    // Color legend for P95 / P99 (or other group labels)
    var legendY = 14;
    groups.forEach(function (g, idx) {
      var lx = plot.left + idx * 70;
      svg.appendChild(
        svgEl("rect", {
          x: lx,
          y: legendY - 8,
          width: 10,
          height: 10,
          fill: g.color,
          rx: 2,
        })
      );
      var lt = svgEl("text", {
        x: lx + 14,
        y: legendY,
        fill: COLORS.muted,
        "font-size": "11",
      });
      lt.textContent = g.label || ("Series " + (idx + 1));
      svg.appendChild(lt);
    });
    host.appendChild(svg);
  }

  function longestLabelWidth(labels, charPx) {
    var maxLen = 0;
    labels.forEach(function (label) {
      maxLen = Math.max(maxLen, String(label || "").length);
    });
    return Math.max(36, Math.min(88, maxLen * (charPx || 6.5) + 10));
  }

  function drawHBarLegend(svg, plot, items) {
    var legendY = 14;
    items.forEach(function (item, idx) {
      var lx = plot.left + idx * 88;
      svg.appendChild(
        svgEl("rect", {
          x: lx,
          y: legendY - 8,
          width: 10,
          height: 10,
          fill: item.color,
          rx: 2,
        })
      );
      var lt = svgEl("text", {
        x: lx + 14,
        y: legendY,
        fill: COLORS.muted,
        "font-size": "11",
      });
      lt.textContent = item.label || ("Series " + (idx + 1));
      svg.appendChild(lt);
    });
  }

  function drawHorizontalGroupedBars(containerId, categories, groups, options) {
    var host = el(containerId);
    if (!host) return;
    clear(host);
    var nCats = Math.max(categories.length, 1);
    var nGroups = Math.max(groups.length, 1);
    var rowH = Math.max(26, 10 + nGroups * 12);
    var width = host.clientWidth || 480;
    var height = Math.max(160, 36 + nCats * rowH + 48);
    host.style.height = height + "px";
    var labelW = longestLabelWidth(categories);
    var plot = {
      left: labelW,
      top: 28,
      width: Math.max(120, width - labelW - 28),
      height: height - 64,
    };
    // Bars always baseline at 0 (see drawGroupedBars). Log toggle is for lines.
    var logX = false;
    var all = [];
    groups.forEach(function (g) {
      g.values.forEach(function (v) {
        if (v != null) all.push(v);
      });
    });
    var xExt = extent(all, false);
    xExt.min = Math.min(0, xExt.min);
    var svg = svgEl("svg", {
      viewBox: "0 0 " + width + " " + height,
      width: String(width),
      height: String(height),
    });

    // Plot frame + vertical value grid
    svg.appendChild(
      svgEl("rect", {
        x: plot.left,
        y: plot.top,
        width: plot.width,
        height: plot.height,
        fill: "#fff",
        stroke: COLORS.line,
      })
    );
    for (var i = 0; i <= 4; i++) {
      var xRatio = i / 4;
      var x = plot.left + xRatio * plot.width;
      var xVal = logX
        ? Math.pow(
            10,
            Math.log10(Math.max(xExt.min, 1e-9)) +
              xRatio *
                (Math.log10(Math.max(xExt.max, 1e-8)) -
                  Math.log10(Math.max(xExt.min, 1e-9)))
          )
        : xExt.min + xRatio * (xExt.max - xExt.min);
      svg.appendChild(
        svgEl("line", {
          x1: x,
          y1: plot.top,
          x2: x,
          y2: plot.top + plot.height,
          stroke: COLORS.line,
          "stroke-dasharray": "3 4",
        })
      );
      var xt = svgEl("text", {
        x: x,
        y: plot.top + plot.height + 16,
        "text-anchor": "middle",
        fill: COLORS.muted,
        "font-size": "11",
      });
      xt.textContent = fmt(xVal);
      svg.appendChild(xt);
    }
    if (options && options.xLabel) {
      var xl = svgEl("text", {
        x: plot.left + plot.width / 2,
        y: height - 8,
        "text-anchor": "middle",
        fill: COLORS.ink,
        "font-size": "12",
      });
      xl.textContent = options.xLabel;
      svg.appendChild(xl);
    }

    var slot = plot.height / nCats;
    var innerGap = 3;
    var barH = Math.max(
      6,
      Math.min(14, (slot * 0.7 - innerGap * (nGroups - 1)) / nGroups)
    );
    var clusterH = nGroups * barH + (nGroups - 1) * innerGap;

    categories.forEach(function (label, ci) {
      var midY = plot.top + ci * slot + slot / 2;
      var clusterTop = midY - clusterH / 2;
      var yl = svgEl("text", {
        x: plot.left - 8,
        y: midY + 4,
        "text-anchor": "end",
        fill: COLORS.muted,
        "font-size": "11",
        "font-weight": "400",
      });
      yl.textContent = label;
      svg.appendChild(yl);

      groups.forEach(function (g, gi) {
        var val = g.values[ci];
        if (val == null || (logX && val <= 0)) return;
        var y = clusterTop + gi * (barH + innerGap);
        var x1 = plot.left;
        var x2 = scale(val, xExt.min, xExt.max, plot.left, plot.left + plot.width, logX);
        var bar = svgEl("rect", {
          x: x1,
          y: y,
          width: Math.max(0, x2 - x1),
          height: barH,
          fill: g.color,
          rx: 2,
        });
        svg.appendChild(bar);
        bindChartTip(bar, label, [(g.label || "Value") + ": " + fmt(val)]);
      });
    });

    drawHBarLegend(svg, plot, groups);
    host.appendChild(svg);
  }

  function drawHorizontalStackedBars(containerId, categories, stacks, options) {
    var host = el(containerId);
    if (!host) return;
    clear(host);
    var nCats = Math.max(categories.length, 1);
    var rowH = 28;
    var width = host.clientWidth || 480;
    var height = Math.max(160, 36 + nCats * rowH + 48);
    host.style.height = height + "px";
    var labelW = longestLabelWidth(categories);
    var plot = {
      left: labelW,
      top: 28,
      width: Math.max(120, width - labelW - 28),
      height: height - 64,
    };
    var totals = categories.map(function (_, i) {
      return stacks.reduce(function (sum, s) {
        return sum + (Number(s.values[i]) || 0);
      }, 0);
    });
    var xExt = extent(totals.concat([0]), false);
    xExt.min = 0;
    var svg = svgEl("svg", {
      viewBox: "0 0 " + width + " " + height,
      width: String(width),
      height: String(height),
    });

    svg.appendChild(
      svgEl("rect", {
        x: plot.left,
        y: plot.top,
        width: plot.width,
        height: plot.height,
        fill: "#fff",
        stroke: COLORS.line,
      })
    );
    for (var i = 0; i <= 4; i++) {
      var xRatio = i / 4;
      var x = plot.left + xRatio * plot.width;
      var xVal = xExt.min + xRatio * (xExt.max - xExt.min);
      svg.appendChild(
        svgEl("line", {
          x1: x,
          y1: plot.top,
          x2: x,
          y2: plot.top + plot.height,
          stroke: COLORS.line,
          "stroke-dasharray": "3 4",
        })
      );
      var xt = svgEl("text", {
        x: x,
        y: plot.top + plot.height + 16,
        "text-anchor": "middle",
        fill: COLORS.muted,
        "font-size": "11",
      });
      xt.textContent = fmt(xVal);
      svg.appendChild(xt);
    }
    if (options && options.xLabel) {
      var xl = svgEl("text", {
        x: plot.left + plot.width / 2,
        y: height - 8,
        "text-anchor": "middle",
        fill: COLORS.ink,
        "font-size": "12",
      });
      xl.textContent = options.xLabel;
      svg.appendChild(xl);
    }

    var slot = plot.height / nCats;
    var barH = Math.max(10, Math.min(18, slot * 0.55));
    categories.forEach(function (label, ci) {
      var midY = plot.top + ci * slot + slot / 2;
      var y = midY - barH / 2;
      var yl = svgEl("text", {
        x: plot.left - 8,
        y: midY + 4,
        "text-anchor": "end",
        fill: COLORS.muted,
        "font-size": "11",
        "font-weight": "400",
      });
      yl.textContent = label;
      svg.appendChild(yl);

      var xCursor = plot.left;
      var rowTotal = totals[ci] || 0;
      stacks.forEach(function (stack) {
        var val = Number(stack.values[ci]) || 0;
        var w = ((val - xExt.min) / (xExt.max - xExt.min || 1)) * plot.width;
        var bar = svgEl("rect", {
          x: xCursor,
          y: y,
          width: Math.max(w, 0),
          height: barH,
          fill: stack.color,
        });
        svg.appendChild(bar);
        var pctShare = rowTotal ? (val / rowTotal) * 100 : 0;
        bindChartTip(bar, label, [
          (stack.label || "Series") + ": " + fmt(val),
          pctShare.toFixed(1) + "% of row",
        ]);
        xCursor += w;
      });
    });

    var labeled = stacks.filter(function (stack) {
      return !!stack.label;
    });
    if (labeled.length) drawHBarLegend(svg, plot, labeled);
    host.appendChild(svg);
  }

  function drawPieChart(containerId, slices, options) {
    var host = el(containerId);
    if (!host) return;
    clear(host);
    host.classList.add("chart-pie");
    var width = host.clientWidth || 480;
    var height = 160;
    host.style.height = height + "px";
    var svg = svgEl("svg", {
      viewBox: "0 0 " + width + " " + height,
      width: String(width),
      height: String(height),
      preserveAspectRatio: "xMinYMid meet",
    });
    var total = slices.reduce(function (sum, s) {
      return sum + Math.max(0, Number(s.value) || 0);
    }, 0);
    if (total <= 0) {
      var empty = svgEl("text", {
        x: width / 2,
        y: height / 2,
        "text-anchor": "middle",
        fill: COLORS.muted,
        "font-size": "13",
      });
      empty.textContent = (options && options.emptyLabel) || "No requests";
      svg.appendChild(empty);
      host.appendChild(svg);
      return;
    }

    var radius = 52;
    var cx = 28 + radius;
    var cy = height / 2;
    var angle = -Math.PI / 2;
    var innerR = Math.round(radius * 0.58);

    function polar(a, r) {
      return [cx + Math.cos(a) * r, cy + Math.sin(a) * r];
    }

    // Full annulus (even-odd): outer CW circle + inner CCW circle.
    function fullDonutPath(outer, inner) {
      return (
        "M " +
        (cx - outer) +
        " " +
        cy +
        " a " +
        outer +
        " " +
        outer +
        " 0 1 0 " +
        outer * 2 +
        " 0 a " +
        outer +
        " " +
        outer +
        " 0 1 0 -" +
        outer * 2 +
        " 0" +
        " M " +
        (cx - inner) +
        " " +
        cy +
        " a " +
        inner +
        " " +
        inner +
        " 0 1 1 " +
        inner * 2 +
        " 0 a " +
        inner +
        " " +
        inner +
        " 0 1 1 -" +
        inner * 2 +
        " 0"
      );
    }

    var nonzero = [];
    slices.forEach(function (s) {
      if (Math.max(0, Number(s.value) || 0) > 0) nonzero.push(s);
    });
    var isFull = nonzero.length === 1;

    if (isFull) {
      var full = nonzero[0];
      var fullVal = Math.max(0, Number(full.value) || 0);
      var ring = svgEl("path", {
        d: fullDonutPath(radius, innerR),
        fill: full.color,
        "fill-rule": "evenodd",
        stroke: "#fff",
        "stroke-width": "1.5",
      });
      svg.appendChild(ring);
      bindChartTip(ring, full.label || "Slice", [
        "Count: " + fmt(fullVal),
        "100%",
      ]);
      var pctLabel = svgEl("text", {
        x: String(cx),
        y: String(cy + 5),
        "text-anchor": "middle",
        fill: COLORS.ink,
        "font-size": "15",
        "font-weight": "700",
      });
      pctLabel.textContent = "100%";
      svg.appendChild(pctLabel);
    } else {
      slices.forEach(function (slice) {
        var value = Math.max(0, Number(slice.value) || 0);
        if (value <= 0) return;
        var sweep = (value / total) * Math.PI * 2;
        var start = angle;
        var end = angle + sweep;
        angle = end;
        var p0 = polar(start, radius);
        var p1 = polar(end, radius);
        var large = sweep > Math.PI ? 1 : 0;
        var d =
          "M " +
          cx +
          " " +
          cy +
          " L " +
          p0[0] +
          " " +
          p0[1] +
          " A " +
          radius +
          " " +
          radius +
          " 0 " +
          large +
          " 1 " +
          p1[0] +
          " " +
          p1[1] +
          " Z";
        svg.appendChild(
          svgEl("path", {
            d: d,
            fill: slice.color,
            stroke: "#fff",
            "stroke-width": "1.5",
          })
        );
        var sliceNode = svg.lastChild;
        bindChartTip(sliceNode, slice.label || "Slice", [
          "Count: " + fmt(value),
          ((value / total) * 100).toFixed(1) + "%",
        ]);
      });
    }

    var legendX = cx + radius + 24;
    var legendY = Math.max(20, cy - ((slices.length - 1) * 22) / 2);
    slices.forEach(function (slice, idx) {
      var value = Math.max(0, Number(slice.value) || 0);
      var pctVal = total ? (value / total) * 100 : 0;
      var y = legendY + idx * 28;
      svg.appendChild(
        svgEl("rect", {
          x: legendX,
          y: y - 8,
          width: 10,
          height: 10,
          fill: slice.color,
          rx: 2,
        })
      );
      var label = svgEl("text", {
        x: legendX + 16,
        y: y,
        fill: COLORS.ink,
        "font-size": "12",
        "font-weight": "600",
      });
      label.textContent = slice.label;
      svg.appendChild(label);
      var detail = svgEl("text", {
        x: legendX + 16,
        y: y + 14,
        fill: COLORS.muted,
        "font-size": "11",
      });
      detail.textContent = fmt(value) + " (" + pctVal.toFixed(1) + "%)";
      svg.appendChild(detail);
    });

    host.appendChild(svg);
  }

  function runs() {
    return data.runs || [];
  }

  function isMultiRun() {
    return !!(data.header && data.header.multi_run) || runs().length > 1;
  }

  function setCardVisible(id, visible) {
    var node = el(id);
    if (node) node.style.display = visible ? "" : "none";
  }

  function setCopy(id, text) {
    var node = el(id);
    if (node) node.textContent = text;
  }

  function runCategoryLabel(r) {
    if (r.label) return r.label;
    if (r.concurrency != null) {
      var c = Number(r.concurrency);
      if (!Number.isNaN(c) && Math.abs(c - Math.round(c)) < 1e-6) {
        return "concurrent@" + Math.round(c);
      }
      return "concurrent@" + fmt(r.concurrency, 1);
    }
    return r.strategy || "run";
  }

  // Compact axis labels for charts — full concurrent@N stays in tables/dropdown.
  function runChartLabel(r) {
    if (r.concurrency != null) {
      var c = Number(r.concurrency);
      if (!Number.isNaN(c) && Math.abs(c - Math.round(c)) < 1e-6) {
        return "@" + Math.round(c);
      }
      return "@" + fmt(r.concurrency, 1);
    }
    var full = runCategoryLabel(r);
    return full.length > 10 ? full.slice(0, 9) + "…" : full;
  }

  function renderSingleRunLatency(run, details) {
    // E2E is shown in KPIs / comparison table — keep this chart on first-token scale.
    var firstLabels = ["TTFT"];
    var firstP95 = [run.ttft_p95_ms];
    var firstP99 = [run.ttft_p99_ms];
    if (details.show_ttfot) {
      firstLabels.push("TTFOT");
      firstP95.push(run.ttfot_p95_ms);
      firstP99.push(run.ttfot_p99_ms);
    }
    drawHorizontalGroupedBars(
      "chart-lat-e2e",
      firstLabels,
      [
        { label: "P95", color: COLORS.p95, values: firstP95 },
        { label: "P99", color: COLORS.p99, values: firstP99 },
      ],
      { xLabel: "Latency (ms)", logX: true }
    );

    var genLabels = ["ITL", "TPOT"];
    var genP95 = [run.itl_p95_ms, run.tpot_p95_ms];
    var genP99 = [run.itl_p99_ms, run.tpot_p99_ms];
    drawHorizontalGroupedBars(
      "chart-lat-gen",
      genLabels,
      [
        { label: "P95", color: COLORS.p95, values: genP95 },
        { label: "P99", color: COLORS.p99, values: genP99 },
      ],
      { xLabel: "ms / token", logX: true }
    );
  }

  function renderMultiRunLatency(rows, details) {
    var cats = rows.map(runChartLabel);
    var firstGroups = [
      {
        label: "TTFT P95",
        color: COLORS.ttft,
        values: rows.map(function (r) {
          return r.ttft_p95_ms;
        }),
      },
      {
        label: "TTFT P99",
        color: COLORS.ttftAlt,
        values: rows.map(function (r) {
          return r.ttft_p99_ms;
        }),
      },
    ];
    if (details.show_ttfot) {
      firstGroups.push(
        {
          label: "TTFOT P95",
          color: COLORS.ttfot,
          values: rows.map(function (r) {
            return r.ttfot_p95_ms;
          }),
        },
        {
          label: "TTFOT P99",
          color: COLORS.ttfotAlt,
          values: rows.map(function (r) {
            return r.ttfot_p99_ms;
          }),
        }
      );
    }
    drawHorizontalGroupedBars("chart-lat-e2e", cats, firstGroups, {
      xLabel: "Latency (ms)",
      logX: true,
    });

    drawHorizontalGroupedBars(
      "chart-lat-gen",
      cats,
      [
        {
          label: "ITL P95",
          color: COLORS.p95,
          values: rows.map(function (r) {
            return r.itl_p95_ms;
          }),
        },
        {
          label: "ITL P99",
          color: COLORS.p99,
          values: rows.map(function (r) {
            return r.itl_p99_ms;
          }),
        },
        {
          label: "TPOT P95",
          color: COLORS.tpot,
          values: rows.map(function (r) {
            return r.tpot_p95_ms;
          }),
        },
        {
          label: "TPOT P99",
          color: COLORS.tpotAlt,
          values: rows.map(function (r) {
            return r.tpot_p99_ms;
          }),
        },
      ],
      { xLabel: "ms / token", logX: true }
    );
  }

  function renderEfficiencyStats(run) {
    var host = el("efficiency-stats");
    var chart = el("chart-efficiency");
    if (!host) return;
    var conc = run.concurrency;
    var totalEff = conc && conc > 0 ? (run.total_tps || 0) / conc : null;
    var outEff = conc && conc > 0 ? (run.output_tps || 0) / conc : null;
    var inEff = conc && conc > 0 ? (run.input_tps || 0) / conc : null;
    if (chart) {
      clear(chart);
      chart.style.display = "none";
    }
    host.hidden = false;
    clear(host);

    function addStat(labelText, helpKey, valueText, subText) {
      var stat = document.createElement("div");
      stat.className = "efficiency-stat";
      var label = document.createElement("div");
      label.className = "eff-label";
      label.appendChild(labelWithHelp(labelText, helpKey));
      var value = document.createElement("div");
      value.className = "eff-value";
      value.textContent = valueText;
      var sub = document.createElement("div");
      sub.className = "eff-sub";
      sub.textContent = subText;
      stat.appendChild(label);
      stat.appendChild(value);
      stat.appendChild(sub);
      host.appendChild(stat);
    }

    addStat("Total tok/s per concurrent", "efficiency", fmt(totalEff), "Overall efficiency");
    addStat("Output tok/s per concurrent", "out_tps", fmt(outEff), "Generated tokens");
    addStat("Input tok/s per concurrent", "in_tps", fmt(inEff), "Prompt tokens");
    addStat("Concurrency", "concurrency", fmt(conc), "Observed parallel requests");
    addStat("ITL P95", "itl", fmt(run.itl_p95_ms) + " ms", "Inter-token latency");
    addStat("TPOT P95", "tpot", fmt(run.tpot_p95_ms) + " ms", "Time per output token");
  }

  function renderSingleRunThroughput(run) {
    var chartEff = el("chart-efficiency");
    if (chartEff) chartEff.style.display = "";

    drawHorizontalGroupedBars(
      "chart-tps-conc",
      ["Req/s", "Input tok/s", "Output tok/s", "Total tok/s"],
      [
        {
          label: "Mean",
          color: COLORS.p99,
          values: [
            run.request_rate,
            run.input_tps,
            run.output_tps,
            run.total_tps,
          ],
        },
      ],
      { xLabel: "Rate", logX: true }
    );

    renderEfficiencyStats(run);
  }

  function renderMultiRunThroughput(rows) {
    var host = el("efficiency-stats");
    if (host) {
      host.hidden = true;
      host.innerHTML = "";
    }
    var chartEff = el("chart-efficiency");
    if (chartEff) chartEff.style.display = "";

    var cats = rows.map(runChartLabel);
    drawHorizontalStackedBars(
      "chart-tps-conc",
      cats,
      [
        {
          label: "Input",
          color: COLORS.input,
          values: rows.map(function (r) {
            return r.input_tps || 0;
          }),
        },
        {
          label: "Output",
          color: COLORS.output,
          values: rows.map(function (r) {
            return r.output_tps || 0;
          }),
        },
      ],
      { xLabel: "Tokens/sec" }
    );

    drawHorizontalGroupedBars(
      "chart-efficiency",
      cats,
      [
        {
          label: "Tok/s / concurrent",
          color: COLORS.p95,
          values: rows.map(function (r) {
            return r.concurrency && r.concurrency > 0
              ? (r.total_tps || 0) / r.concurrency
              : 0;
          }),
        },
      ],
      { xLabel: "Tokens/sec per concurrent request" }
    );
  }


  function renderSelectedRunCharts() {
    var details = data.details || {};
    var multi = isMultiRun();
    var run = selectedRun();
    var label = runCategoryLabel(run);
    setTitleWithHelp(
      "title-breakdown",
      multi ? "Latency components — " + label : "Latency components",
      helpLatComponents(!!details.show_ttfot)
    );
    setCopy(
      "hint-breakdown",
      details.show_ttfot
        ? multi
          ? "TTFT, TTFOT, ITL, and TPOT (P95/P99) for the selected run. Request latency is in the KPIs."
          : "TTFT, TTFOT, ITL, and TPOT (P95/P99) for this run. Request latency is in the KPIs."
        : multi
          ? "TTFT, ITL, and TPOT (P95/P99) for the selected run. Request latency is in the KPIs."
          : "TTFT, ITL, and TPOT (P95/P99) for this run. Request latency is in the KPIs."
    );
    var breakdownLabels = ["TTFT", "ITL", "TPOT"];
    var p95vals = [run.ttft_p95_ms, run.itl_p95_ms, run.tpot_p95_ms];
    var p99vals = [run.ttft_p99_ms, run.itl_p99_ms, run.tpot_p99_ms];
    if (details.show_ttfot) {
      breakdownLabels.splice(1, 0, "TTFOT");
      p95vals.splice(1, 0, run.ttfot_p95_ms);
      p99vals.splice(1, 0, run.ttfot_p99_ms);
    }
    drawHorizontalGroupedBars(
      "chart-breakdown",
      breakdownLabels,
      [
        { label: "P95", color: COLORS.p95, values: p95vals },
        { label: "P99", color: COLORS.p99, values: p99vals },
      ],
      { xLabel: "Latency (ms)", logX: true }
    );
  }

  function renderCharts() {
    var rows = runs();
    var details = data.details || {};
    var multi = isMultiRun();
    var selected = selectedRun();

    setCardVisible("toolbar-performance", multi);
    setCardVisible("card-load-latency", multi);
    setCopy("title-compare", multi ? "Run comparison" : "Run summary");
    setCopy(
      "hint-status",
      multi
        ? "Successful, incomplete, and errored requests for each run."
        : "Share of successful, incomplete, and errored requests for this run."
    );

    if (multi) {
      setTitleWithHelp(
        "title-lat-e2e",
        details.show_ttfot ? "TTFT & TTFOT by concurrency" : "TTFT by concurrency",
        helpFirstToken(!!details.show_ttfot)
      );
      setCopy("hint-lat-e2e", "P95 and P99 — lower is better. Request latency (E2E) is in the KPIs and comparison table.");
      setTitleWithHelp(
        "title-lat-gen",
        "Generation latency by concurrency",
        helpGenLatency()
      );
      setCopy("hint-lat-gen", "ITL and TPOT — lower is better.");
      setTitleWithHelp(
        "title-tps",
        "Token throughput by concurrency",
        helpTokenThroughput(true)
      );
      setCopy(
        "hint-tps",
        "Stacked input and output tokens per second at each load point."
      );
      setTitleWithHelp("title-efficiency", "Throughput efficiency", "efficiency");
      setCopy("hint-efficiency", "Total tokens/sec per concurrent request.");
    } else {
      setTitleWithHelp(
        "title-lat-e2e",
        "First-token latency",
        helpFirstToken(!!details.show_ttfot)
      );
      setCopy(
        "hint-lat-e2e",
        details.show_ttfot
          ? "TTFT and TTFOT P95 vs P99 for this run. Request latency (E2E) is in the KPIs above."
          : "TTFT P95 vs P99 for this run. Request latency (E2E) is in the KPIs above."
      );
      setTitleWithHelp("title-lat-gen", "Generation latency", helpGenLatency());
      setCopy("hint-lat-gen", "ITL and TPOT P95 vs P99 for this run.");
      setTitleWithHelp(
        "title-tps",
        "Throughput rates",
        helpTokenThroughput(false)
      );
      setCopy("hint-tps", "Mean request and token rates for this run.");
      setTitleWithHelp("title-efficiency", "Per-request efficiency", "efficiency");
      setCopy(
        "hint-efficiency",
        "Efficiency and generation cost relative to concurrency for this run."
      );
    }

    if (multi) {
      // Clearer replacement for the old saturation-knee scatter: E2E latency vs load.
      drawHorizontalGroupedBars(
        "chart-load-latency",
        rows.map(runChartLabel),
        [
          {
            label: "E2E P95",
            color: COLORS.p95,
            values: rows.map(function (r) {
              return r.request_latency_p95_ms;
            }),
          },
          {
            label: "E2E P99",
            color: COLORS.p99,
            values: rows.map(function (r) {
              return r.request_latency_p99_ms;
            }),
          },
        ],
        { xLabel: "Request latency (ms)", logX: true }
      );

      // Per-run stacked bars make success/error mix comparable across load points.
      drawHorizontalStackedBars(
        "chart-status",
        rows.map(runChartLabel),
        [
          {
            label: "Successful",
            color: COLORS.success,
            values: rows.map(function (r) {
              return r.successful;
            }),
          },
          {
            label: "Incomplete",
            color: COLORS.incomplete,
            values: rows.map(function (r) {
              return r.incomplete;
            }),
          },
          {
            label: "Errored",
            color: COLORS.errored,
            values: rows.map(function (r) {
              return r.errored;
            }),
          },
        ],
        { xLabel: "Requests" }
      );
    } else {
      drawPieChart(
        "chart-status",
        [
          {
            label: "Successful",
            color: COLORS.success,
            value: Number(selected.successful) || 0,
          },
          {
            label: "Incomplete",
            color: COLORS.incomplete,
            value: Number(selected.incomplete) || 0,
          },
          {
            label: "Errored",
            color: COLORS.errored,
            value: Number(selected.errored) || 0,
          },
        ],
        { emptyLabel: "No requests" }
      );
    }

    if (multi) {
      renderMultiRunLatency(rows, details);
      renderMultiRunThroughput(rows);
    } else {
      renderSingleRunLatency(selected, details);
      renderSingleRunThroughput(selected);
    }

    renderSelectedRunCharts();

    // By turn
    var turns = data.by_turn || [];
    if (turns.length) {
      drawLineChart(
        "chart-turn-lat",
        [
          {
            label: "E2E P95",
            color: COLORS.p95,
            shape: "circle",
            points: turns.map(function (t) {
              return [t.turn_index, t.request_latency_p95_ms];
            }),
          },
          {
            label: "E2E P99",
            color: COLORS.p99,
            shape: "square",
            points: turns.map(function (t) {
              return [t.turn_index, t.request_latency_p99_ms];
            }),
          },
          {
            label: "TTFT P95",
            color: COLORS.ttft,
            shape: "triangle",
            points: turns.map(function (t) {
              return [t.turn_index, t.ttft_p95_ms];
            }),
          },
          {
            label: "ITL P95",
            color: COLORS.tpot,
            shape: "diamond",
            points: turns.map(function (t) {
              return [t.turn_index, t.itl_p95_ms];
            }),
          },
        ],
        { xLabel: "Turn index", yLabel: "Latency (ms)", logY: true }
      );
      drawLineChart(
        "chart-turn-tokens",
        [
          {
            label: "Prompt median",
            color: COLORS.input,
            shape: "circle",
            points: turns.map(function (t) {
              return [t.turn_index, t.prompt_tokens_median];
            }),
          },
          {
            label: "Prompt P95",
            color: COLORS.p99,
            shape: "square",
            points: turns.map(function (t) {
              return [t.turn_index, t.prompt_tokens_p95];
            }),
          },
        ],
        { xLabel: "Turn index", yLabel: "Prompt tokens" }
      );
    }
  }

  function renderTableTo(host, headers, rows) {
    if (!host) return;
    clear(host);
    if (!rows.length) {
      host.innerHTML = '<div class="empty">No data for this section.</div>';
      return;
    }
    var table = document.createElement("table");
    var thead = document.createElement("thead");
    var hr = document.createElement("tr");
    headers.forEach(function (h) {
      var th = document.createElement("th");
      if (h && typeof h === "object") {
        th.appendChild(labelWithHelp(h.text, h.help));
      } else {
        th.textContent = h;
      }
      hr.appendChild(th);
    });
    thead.appendChild(hr);
    table.appendChild(thead);
    var tbody = document.createElement("tbody");
    rows.forEach(function (row) {
      var tr = document.createElement("tr");
      row.forEach(function (cell) {
        var td = document.createElement("td");
        td.textContent = cell;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    var wrap = document.createElement("div");
    wrap.className = "table-wrap";
    wrap.appendChild(table);
    host.appendChild(wrap);
  }

  function renderTable(containerId, headers, rows) {
    renderTableTo(el(containerId), headers, rows);
  }

  function renderTables() {
    var rows = runs();
    renderTable(
      "table-compare",
      [
        "Run",
        headerCell("RPS", "rps"),
        headerCell("Concurrency", "concurrency"),
        headerCell("Out tok/s", "out_tps"),
        headerCell("Total tok/s", "total_tps"),
        headerCell("E2E P95", "e2e"),
        headerCell("E2E P99", "e2e"),
        headerCell("TTFT P95", "ttft"),
        headerCell("TTFT P99", "ttft"),
        headerCell("ITL P95", "itl"),
        headerCell("ITL P99", "itl"),
        headerCell("Error %", "error_rate"),
      ],
      rows.map(function (r) {
        return [
          runCategoryLabel(r),
          fmt(r.request_rate),
          fmt(r.concurrency),
          fmt(r.output_tps),
          fmt(r.total_tps),
          fmt(r.request_latency_p95_ms),
          fmt(r.request_latency_p99_ms),
          fmt(r.ttft_p95_ms),
          fmt(r.ttft_p99_ms),
          fmt(r.itl_p95_ms),
          fmt(r.itl_p99_ms),
          pct(r.error_rate),
        ];
      })
    );

    var details = data.details || {};
    setTitleWithHelp(
      "title-extra-latency",
      "Extra latency",
      helpExtraLatency(!!details.show_ttfot)
    );
    var extraHeaders = [
      "Run",
      headerCell("TPOT P95", "tpot"),
      headerCell("TPOT P99", "tpot"),
    ];
    if (details.show_ttfot) {
      extraHeaders.push(headerCell("TTFOT P95", "ttfot"), headerCell("TTFOT P99", "ttfot"));
    }
    renderTable(
      "table-extra-latency",
      extraHeaders,
      (details.extra_latency_rows || []).map(function (r) {
        var row = [r.strategy, fmt(r.tpot_p95_ms), fmt(r.tpot_p99_ms)];
        if (details.show_ttfot) row.push(fmt(r.ttfot_p95_ms), fmt(r.ttfot_p99_ms));
        return row;
      })
    );

    renderTable(
      "table-request-size",
      [
        "Run",
        headerCell("Prompt median", "prompt_tokens"),
        headerCell("Prompt P95", "prompt_tokens"),
        headerCell("Output median", "output_tokens"),
        headerCell("Output P95", "output_tokens"),
      ],
      (details.request_size_rows || []).map(function (r) {
        return [
          r.strategy,
          fmt(r.prompt_tokens_median),
          fmt(r.prompt_tokens_p95),
          fmt(r.output_tokens_median),
          fmt(r.output_tokens_p95),
        ];
      })
    );

    var modalityHost = el("modality-sections");
    if (modalityHost) {
      clear(modalityHost);
      var sections = details.modality_sections || {};
      Object.keys(sections).forEach(function (modality) {
        var section = sections[modality];
        var card = document.createElement("div");
        card.className = "card";
        var title = document.createElement("h3");
        var heading = document.createElement("span");
        heading.className = "card-heading";
        heading.textContent = (section.label || modality) + " metrics";
        title.appendChild(heading);
        var modalityHelp = helpIcon("modality");
        if (modalityHelp) title.appendChild(modalityHelp);
        card.appendChild(title);

        (section.metrics || []).forEach(function (metric, metricIdx) {
          var block = document.createElement("div");
          block.className = "modality-metric";
          var heading = document.createElement("h4");
          heading.textContent = metric.label || metric.key;
          block.appendChild(heading);

          var metricRows = metric.rows || [];
          if (metricRows.length > 1 || isMultiRun()) {
            // Multi-run: compact 5-column table per metric group.
            // Build into the host node directly — it may not be in the document yet.
            var tableHost = document.createElement("div");
            block.appendChild(tableHost);
            card.appendChild(block);
            renderTableTo(
              tableHost,
              [
                "Run",
                headerCell("In mean", "in_out"),
                headerCell("In P95", "p95"),
                headerCell("Out mean", "in_out"),
                headerCell("Out P95", "p95"),
              ],
              metricRows.map(function (row) {
                return [
                  row.strategy,
                  fmt(row.input_mean),
                  fmt(row.input_p95),
                  fmt(row.output_mean),
                  fmt(row.output_p95),
                ];
              })
            );
            return;
          }

          // Single-run: compact input/output rows (avoid full-width sparse cards).
          metricRows.forEach(function (row) {
            var compact = document.createElement("div");
            compact.className = "modality-compact";
            ["input", "output"].forEach(function (side) {
              var mean = row[side + "_mean"];
              var p95 = row[side + "_p95"];
              if (mean == null && p95 == null) return;
              var sideRow = document.createElement("div");
              sideRow.className = "modality-compact-row";
              var sideLabel = document.createElement("div");
              sideLabel.className = "side-label";
              sideLabel.appendChild(labelWithHelp(side, "in_out"));
              var sideVals = document.createElement("div");
              sideVals.className = "side-vals";
              var meanSpan = document.createElement("span");
              meanSpan.innerHTML = "<strong>" + fmt(mean) + "</strong>";
              meanSpan.appendChild(labelWithHelp("mean", "mean"));
              var p95Span = document.createElement("span");
              p95Span.innerHTML = "<strong>" + fmt(p95) + "</strong>";
              p95Span.appendChild(labelWithHelp("p95", "p95"));
              sideVals.appendChild(meanSpan);
              sideVals.appendChild(p95Span);
              sideRow.appendChild(sideLabel);
              sideRow.appendChild(sideVals);
              compact.appendChild(sideRow);
            });
            block.appendChild(compact);
          });
          card.appendChild(block);
        });
        modalityHost.appendChild(card);
      });
    }

    var rttCard = el("rtt-card");
    if (rttCard) {
      rttCard.style.display = details.show_rtt ? "block" : "none";
      if (details.show_rtt) {
        renderTable(
          "table-rtt",
          [
            "Run",
            headerCell("Avg RTT P95", "avg_rtt"),
            headerCell("Avg RTT P99", "avg_rtt"),
            headerCell("Last RTT P95", "last_rtt"),
            headerCell("Last RTT P99", "last_rtt"),
          ],
          (details.rtt_rows || []).map(function (r) {
            return [
              r.strategy,
              fmt(r.avg_p95_ms),
              fmt(r.avg_p99_ms),
              fmt(r.last_p95_ms),
              fmt(r.last_p99_ms),
            ];
          })
        );
      }
    }

    var turns = data.by_turn || [];
    if (data.turn_note) setText("turn-note", data.turn_note);
    renderTable(
      "table-turn",
      [
        headerCell("Turn", "turn"),
        "Count",
        headerCell("History median", "history_median"),
        headerCell("Prompt median", "prompt_tokens"),
        headerCell("Prompt P95", "prompt_tokens"),
        headerCell("E2E P95", "e2e"),
        headerCell("E2E P99", "e2e"),
        headerCell("TTFT P95", "ttft"),
        headerCell("ITL P95", "itl"),
      ],
      turns.map(function (t) {
        return [
          t.turn_index,
          t.count,
          fmt(t.history_len_median),
          fmt(t.prompt_tokens_median),
          fmt(t.prompt_tokens_p95),
          fmt(t.request_latency_p95_ms),
          fmt(t.request_latency_p99_ms),
          fmt(t.ttft_p95_ms),
          fmt(t.itl_p95_ms),
        ];
      })
    );
  }

  function init() {
    hydrateHelp(document);
    renderHeader();
    setupTabs();
    setupScaleToggle();
    renderTables();
    renderCharts();
    window.addEventListener("resize", function () {
      renderCharts();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
