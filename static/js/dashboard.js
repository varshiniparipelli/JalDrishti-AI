/**
 * JalDrishti-AI — Dashboard JavaScript
 * Fetches real MIS 78th Round API endpoints and renders Plotly charts.
 */

/* ── Fetch helper ──────────────────────────────────────────────────── */
const api = async (path, opts = {}) => {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`API ${path} → ${r.status}`);
  return r.json();
};

/* ── Load everything ──────────────────────────────────────────────── */
async function loadAll() {
  const tasks = [
    loadKPIs(),
    loadStateChart(),
    loadSectorChart(),
    loadSourceChart(),
    loadGapChart(),
    loadBottomChart(),
    loadTopChart(),
    loadSDG(),
    loadAgeChart(),
    loadGenderChart(),
  ];
  // Settle all — never let one failure block the others
  const results = await Promise.allSettled(tasks);
  results.forEach((r, i) => {
    if (r.status === "rejected") {
      console.warn(`Dashboard task ${i} failed:`, r.reason?.message || r.reason);
    }
  });
}

/* ── KPIs ─────────────────────────────────────────────────────────── */
async function loadKPIs() {
  const { data } = await api("/api/kpis");
  const set = (id, val) => {
    const el = document.querySelector(`#${id} .kpi-value`);
    if (el) el.textContent = val;
    document.getElementById(id)?.classList.remove("skeleton");
  };
  set("kpi-national",  data.national_improved_pct + "%");
  set("kpi-rural",     data.rural_improved_pct    + "%");
  set("kpi-urban",     data.urban_improved_pct    + "%");
  set("kpi-piped",     data.piped_water_pct       + "%");
  set("kpi-gap",       data.rural_urban_gap       + " pp");
  set("kpi-above90",   data.states_above_90pct + " / " + data.states_covered);
}

/* ── State Chart (horizontal bar — all 37 states) ────────────────── */
async function loadStateChart() {
  const { data } = await api("/api/state-analysis");
  const states = data.map(d => d.state);
  const all    = data.map(d => d.improved_all_pct);
  const rural  = data.map(d => d.improved_rural_pct);
  const urban  = data.map(d => d.improved_urban_pct);

  const colorAll = all.map(v =>
    v === null ? "#ccc" : v >= 90 ? "#24a148" : v >= 70 ? "#0062fe" : "#da1e28"
  );

  Plotly.newPlot("chart-state", [
    {
      name: "All Sectors", type: "bar", orientation: "h",
      x: all, y: states,
      marker: { color: colorAll },
      text: all.map(v => v !== null ? v + "%" : "N/A"),
      textposition: "outside",
      hovertemplate: "<b>%{y}</b><br>All: %{x}%<extra></extra>",
    },
  ], {
    margin: { t: 10, l: 200, r: 80, b: 40 },
    xaxis: { title: "% with Improved Drinking Water Source", range: [0, 115] },
    yaxis: { autorange: "reversed" },
    plot_bgcolor: "transparent", paper_bgcolor: "transparent",
    font: { family: "Segoe UI, sans-serif", size: 11 },
    shapes: [{
      type: "line", x0: 90, x1: 90, y0: -0.5, y1: states.length - 0.5,
      line: { color: "#f1c21b", width: 2, dash: "dash" },
    }],
    annotations: [{
      x: 90, y: states.length - 1,
      text: "90% target", showarrow: false,
      font: { color: "#f1c21b", size: 11 }, xanchor: "left",
    }],
  }, { responsive: true, displayModeBar: false });
}

/* ── Sector Chart ────────────────────────────────────────────────── */
async function loadSectorChart() {
  const { data } = await api("/api/sector-analysis");
  const sectors = data.map(d => d.sector);
  const imp   = data.map(d => d.improved_pct);
  const piped = data.map(d => d.piped_pct);

  Plotly.newPlot("chart-sector", [
    {
      name: "Improved Source", type: "bar",
      x: sectors, y: imp,
      marker: { color: ["#0062fe","#00b4d8","#6929c4"] },
      text: imp.map(v => v + "%"), textposition: "outside",
    },
    {
      name: "Piped Water", type: "bar",
      x: sectors, y: piped,
      marker: { color: ["#d4e0ff","#c0f0fa","#e0d4ff"] },
      text: piped.map(v => v + "%"), textposition: "outside",
    },
  ], {
    barmode: "group",
    margin: { t: 20, l: 50, r: 20, b: 50 },
    yaxis: { title: "% Population", range: [0, 120] },
    legend: { orientation: "h", y: -0.25 },
    plot_bgcolor: "transparent", paper_bgcolor: "transparent",
    font: { family: "Segoe UI, sans-serif", size: 12 },
  }, { responsive: true, displayModeBar: false });
}

/* ── Source Distribution (Grouped bar by sector) ─────────────────── */
async function loadSourceChart() {
  const { data } = await api("/api/water-sources");

  const improved = data.filter(d => d.source === "Improved Source of Drinking Water");
  const piped    = data.filter(d => d.source === "Piped Water into Dwelling or Yard/plot");

  Plotly.newPlot("chart-source", [
    {
      name: "Improved Source",
      type: "bar",
      x: improved.map(d => d.sector),
      y: improved.map(d => d.value_pct),
      marker: { color: "#0062fe" },
      text: improved.map(d => d.value_pct + "%"), textposition: "outside",
    },
    {
      name: "Piped Water",
      type: "bar",
      x: piped.map(d => d.sector),
      y: piped.map(d => d.value_pct),
      marker: { color: "#00b4d8" },
      text: piped.map(d => d.value_pct + "%"), textposition: "outside",
    },
  ], {
    barmode: "group",
    margin: { t: 10, l: 50, r: 20, b: 50 },
    yaxis: { title: "% Population", range: [0, 120] },
    xaxis: { title: "Sector" },
    legend: { orientation: "h", y: -0.25 },
    plot_bgcolor: "transparent", paper_bgcolor: "transparent",
    font: { family: "Segoe UI, sans-serif", size: 12 },
  }, { responsive: true, displayModeBar: false });
}

/* ── Gap Chart (horizontal bar) ──────────────────────────────────── */
async function loadGapChart() {
  const { data } = await api("/api/state-gap");

  const states = data.map(d => d.state);
  const gaps   = data.map(d => d.gap);
  const colors = gaps.map(g =>
    g >= 20 ? "#da1e28" : g >= 10 ? "#f1c21b" : "#24a148"
  );

  Plotly.newPlot("chart-gap", [{
    type: "bar", orientation: "h",
    x: gaps, y: states,
    marker: { color: colors },
    text: gaps.map(g => (g >= 0 ? "+" : "") + g + " pp"),
    textposition: "outside",
    hovertemplate: "<b>%{y}</b><br>Gap: %{x} pp<extra></extra>",
    customdata: data.map(d => [d.rural_pct, d.urban_pct]),
    hovertemplate: "<b>%{y}</b><br>Rural: %{customdata[0]}%<br>Urban: %{customdata[1]}%<br>Gap: %{x} pp<extra></extra>",
  }], {
    margin: { t: 10, l: 200, r: 80, b: 50 },
    xaxis: { title: "Urban − Rural Gap (percentage points)", range: [-5, Math.max(...gaps) + 12] },
    yaxis: { autorange: "reversed" },
    shapes: [{ type: "line", x0: 0, x1: 0, y0: -0.5, y1: states.length - 0.5,
               line: { color: "#5a6a7e", width: 1 } }],
    plot_bgcolor: "transparent", paper_bgcolor: "transparent",
    font: { family: "Segoe UI, sans-serif", size: 11 },
  }, { responsive: true, displayModeBar: false });
}

/* ── Bottom 10 ───────────────────────────────────────────────────── */
async function loadBottomChart() {
  const { data } = await api("/api/bottom-states?n=10");

  Plotly.newPlot("chart-bottom", [{
    type: "bar",
    x: data.map(d => d.improved_pct),
    y: data.map(d => d.state),
    orientation: "h",
    marker: { color: data.map(d => d.improved_pct < 70 ? "#da1e28" : "#f1c21b") },
    text: data.map(d => d.improved_pct + "%"), textposition: "outside",
    hovertemplate: "<b>%{y}</b><br>%{x}%<extra></extra>",
  }], {
    margin: { t: 10, l: 200, r: 70, b: 40 },
    xaxis: { title: "Improved Access %", range: [0, 108] },
    yaxis: { autorange: "reversed" },
    plot_bgcolor: "transparent", paper_bgcolor: "transparent",
    font: { family: "Segoe UI, sans-serif", size: 11 },
  }, { responsive: true, displayModeBar: false });
}

/* ── Top 10 ──────────────────────────────────────────────────────── */
async function loadTopChart() {
  const { data } = await api("/api/top-states?n=10");

  Plotly.newPlot("chart-top", [{
    type: "bar",
    x: data.map(d => d.improved_pct),
    y: data.map(d => d.state),
    orientation: "h",
    marker: { color: "#24a148" },
    text: data.map(d => d.improved_pct + "%"), textposition: "outside",
    hovertemplate: "<b>%{y}</b><br>%{x}%<extra></extra>",
  }], {
    margin: { t: 10, l: 200, r: 70, b: 40 },
    xaxis: { title: "Improved Access %", range: [0, 108] },
    yaxis: { autorange: "reversed" },
    plot_bgcolor: "transparent", paper_bgcolor: "transparent",
    font: { family: "Segoe UI, sans-serif", size: 11 },
  }, { responsive: true, displayModeBar: false });
}

/* ── SDG Progress ────────────────────────────────────────────────── */
async function loadSDG() {
  const { data } = await api("/api/sdg-progress");
  document.getElementById("sdg-bar").style.width = data.current_coverage_pct + "%";
  document.getElementById("sdg-pct").textContent  = data.current_coverage_pct + "%";

  const statsEl = document.getElementById("sdg-stats");
  const items = [
    { val: data.baseline_2015_pct + "%",           lbl: "2015 Baseline (est.)" },
    { val: data.current_coverage_pct + "%",        lbl: "Current (MIS 78th Rd)" },
    { val: data.sdg_target_pct + "%",              lbl: "SDG 6.1 Target" },
    { val: data.gap_pct + " pp",                   lbl: "Remaining Gap" },
    { val: "+" + data.annual_improvement_rate + "%/yr", lbl: "Annual Rate" },
    { val: data.projected_2030_pct + "%",          lbl: "Projected 2030" },
    { val: data.on_track ? "Yes" : "At Risk",      lbl: "On Track?" },
  ];
  statsEl.innerHTML = items.map(i =>
    `<div class="sdg-stat-item">
       <div class="stat-val">${i.val}</div>
       <div class="stat-lbl">${i.lbl}</div>
     </div>`
  ).join("");

  Plotly.newPlot("chart-sdg", [{
    type: "bar",
    x: ["2015 Baseline", "Current (2023)", "Projected (2030)", "SDG Target (2030)"],
    y: [data.baseline_2015_pct, data.current_coverage_pct, data.projected_2030_pct, 100],
    marker: { color: ["#5a6a7e", "#0062fe", "#00b4d8", "#24a148"] },
    text: [data.baseline_2015_pct + "%", data.current_coverage_pct + "%",
           data.projected_2030_pct + "%", "100%"],
    textposition: "outside",
  }], {
    margin: { t: 10, l: 50, r: 20, b: 60 },
    yaxis: { range: [0, 115], title: "% Population with Improved Water Access" },
    plot_bgcolor: "transparent", paper_bgcolor: "transparent",
    font: { family: "Segoe UI, sans-serif", size: 12 },
  }, { responsive: true, displayModeBar: false });
}

/* ── Age Group Chart ─────────────────────────────────────────────── */
async function loadAgeChart() {
  const { data } = await api("/api/age-group-analysis");
  if (!data || data.length === 0) {
    document.getElementById("chart-age").innerHTML =
      '<p style="text-align:center;color:#5a6a7e;padding:2rem;">Age group breakdown not available in this dataset.</p>';
    return;
  }
  Plotly.newPlot("chart-age", [{
    type: "bar",
    x: data.map(d => d.age_group),
    y: data.map(d => d.improved_pct),
    marker: { color: data.map(d => d.improved_pct >= 90 ? "#24a148" : d.improved_pct >= 70 ? "#0062fe" : "#da1e28") },
    text: data.map(d => d.improved_pct + "%"),
    textposition: "outside",
    hovertemplate: "<b>%{x}</b><br>%{y}%<extra></extra>",
  }], {
    margin: { t: 10, l: 50, r: 20, b: 80 },
    yaxis: { title: "% with Improved Access", range: [0, 112] },
    xaxis: { title: "Age Group", tickangle: -30 },
    plot_bgcolor: "transparent", paper_bgcolor: "transparent",
    font: { family: "Segoe UI, sans-serif", size: 11 },
  }, { responsive: true, displayModeBar: false });
}

/* ── Gender Chart ────────────────────────────────────────────────── */
async function loadGenderChart() {
  const { data } = await api("/api/gender-analysis");
  if (!data || data.length === 0) {
    document.getElementById("chart-gender").innerHTML =
      '<p style="text-align:center;color:#5a6a7e;padding:2rem;">Gender breakdown not available in this dataset.</p>';
    return;
  }
  const colors = ["#0062fe", "#00b4d8", "#6929c4", "#24a148"];
  Plotly.newPlot("chart-gender", [{
    type: "bar",
    x: data.map(d => d.gender),
    y: data.map(d => d.improved_pct),
    marker: { color: data.map((_, i) => colors[i % colors.length]) },
    text: data.map(d => d.improved_pct + "%"),
    textposition: "outside",
    hovertemplate: "<b>%{x}</b><br>%{y}%<extra></extra>",
  }], {
    margin: { t: 10, l: 50, r: 20, b: 60 },
    yaxis: { title: "% with Improved Access", range: [0, 112] },
    xaxis: { title: "Gender" },
    plot_bgcolor: "transparent", paper_bgcolor: "transparent",
    font: { family: "Segoe UI, sans-serif", size: 12 },
  }, { responsive: true, displayModeBar: false });
}

/* ── National Insight (IBM Granite) ─────────────────────────────── */
async function loadNationalInsight() {
  const body   = document.getElementById("insight-body");
  const btn    = document.querySelector("#insight-card .btn-primary");
  body.innerHTML = '<span style="opacity:.6;font-style:italic;">Requesting insight from IBM Granite via watsonx.ai…</span>';
  if (btn) btn.disabled = true;

  try {
    const { insight } = await api("/api/watsonx/national-insight");
    body.innerHTML = `<p style="line-height:1.75;">${insight.replace(/\n/g, "<br>")}</p>
      <div style="font-size:.75rem;color:#5a6a7e;margin-top:.75rem;">
        IBM watsonx.ai &middot; granite-4-h-small &middot; Grounded in MIS 78th Round data
      </div>`;
  } catch (err) {
    body.innerHTML = `<span style="color:#da1e28;">Could not load insight: ${err.message}. Ensure WATSONX_API_KEY is configured.</span>`;
  } finally {
    if (btn) btn.disabled = false;
  }
}

/* ── Predictor ───────────────────────────────────────────────────── */
function getFormData() {
  const f = document.getElementById("predict-form");
  return {
    state:  f.state.value,
    sector: f.sector.value,
  };
}

async function trainAndPredict() {
  const resultEl = document.getElementById("predict-result");
  resultEl.textContent = "Training GradientBoosting model on MIS 78th Round data…";
  resultEl.className   = "predict-result";
  resultEl.classList.remove("hidden");

  try {
    const trainRes = await api("/api/train", { method: "POST" });
    const tr = trainRes.data;
    await runPredict(`Model trained — 5-Fold CV MAE: ${tr.loo_mae_pp} pp | Train MAE: ${tr.train_mae_pp} pp on ${tr.n_samples} samples`);
  } catch (err) {
    resultEl.textContent = "Training error: " + err.message;
  }
}

async function runPredict(trainNote = null) {
  const resultEl = document.getElementById("predict-result");
  if (!trainNote) {
    resultEl.textContent = "Predicting…";
    resultEl.className   = "predict-result";
    resultEl.classList.remove("hidden");
  }

  try {
    const { data } = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(getFormData()),
    });

    const pct = data.predicted_improved_pct;
    const isHigh = pct >= 90;
    const isMid  = pct >= 70;
    resultEl.className = "predict-result " + (isHigh ? "positive" : isMid ? "" : "negative");
    resultEl.innerHTML = `
      <div class="result-title">
        ${isHigh ? "✅" : isMid ? "ℹ️" : "⚠️"}&nbsp;
        <strong>${data.state} — ${data.sector}</strong>
      </div>
      <div style="font-size:2rem;font-weight:800;color:${isHigh?"#24a148":isMid?"#0062fe":"#da1e28"};margin:.4rem 0;">
        ${pct}%
      </div>
      <div class="result-prob">${data.interpretation}</div>
      ${trainNote ? `<div class="result-prob" style="margin-top:.5rem;font-style:italic;font-size:.82rem;">${trainNote}</div>` : ""}
      <div class="result-prob" style="margin-top:.3rem;font-size:.8rem;opacity:.7;">
        IBM Watson ML · GradientBoostingRegressor · MIS 78th Round training data
      </div>`;
  } catch (err) {
    const el = document.getElementById("predict-result");
    el.className = "predict-result negative";
    el.innerHTML = `<div class="result-title">Error</div><div>${err.message}</div>`;
  }
}

/* ── Watson Assistant Chat ───────────────────────────────────────── */
let watsonSessionId = null;

const localAnswers = [
  [/improved.*(source|water)/i,  "Improved drinking water sources include piped water into dwelling or yard/plot, protected dug wells, boreholes, protected springs, and rainwater collection — as defined by WHO/JMP standards."],
  [/sdg.*(6|target|goal)/i,      "SDG 6.1 targets universal and equitable access to safe and affordable drinking water for all by 2030. India's current MIS 78th Round coverage shows a gap from 100%, requiring accelerated intervention."],
  [/rural/i,                     "According to MIS 78th Round, rural India has lower improved water access than urban areas — with a significant rural-urban gap persisting across most states."],
  [/urban/i,                     "Urban households generally have higher improved water access. MIS 78th Round shows urban access is substantially higher than rural across all states."],
  [/piped/i,                     "Piped water into dwelling or yard/plot is the highest-quality improved source. All-India piped water coverage is significantly lower than overall improved water access, especially in rural areas."],
  [/bihar/i,                     "Bihar shows very high improved water access (99%+) in MIS 78th Round — but piped water coverage remains low (~12% rural), suggesting reliance on tube wells and boreholes."],
  [/lakshadweep/i,               "Lakshadweep shows 100% urban improved water access in MIS 78th Round, reflecting concentrated island infrastructure investment."],
  [/gap|disparity|inequal/i,     "The Rural–Urban gap in improved water access varies widely by state. Some states have gaps of 20+ percentage points, requiring targeted rural infrastructure investment."],
  [/assam/i,                     "Assam has a notable rural-urban disparity — 90.8% improved access overall (Rural), but only 3.3% have piped water, indicating heavy reliance on boreholes and tube wells."],
  [/bottom|worst|low/i,          "States with the lowest improved water access in MIS 78th Round are identified in the Bottom 10 chart — these are priority regions for SDG 6.1 intervention."],
  [/top|best|high/i,             "Top-performing states have improved water access above 98-100%, showing that universal coverage is achievable with sustained investment."],
  [/predict|model|ml|ai/i,       "The AI Predictor uses a GradientBoosting regressor trained on the real MIS 78th Round data. Select a state and sector to predict its improved water access percentage."],
];

function localChatResponse(text) {
  for (const [pattern, answer] of localAnswers) {
    if (pattern.test(text)) return answer;
  }
  return "I can answer questions about MIS 78th Round data, SDG 6.1 progress, state-level access, rural–urban disparities, piped water coverage, or the AI predictor. Try asking about a specific state, rural access, or the SDG gap!";
}

const WATSONX_PROJECT_ID = "d35a6f68-c129-4634-b823-ff9583f7fde6";

async function sendChat() {
  const input = document.getElementById("chat-input");
  const text  = input.value.trim();
  if (!text) return;
  input.value = "";

  appendMessage("user", text);
  const typingId = "typing-" + Date.now();
  appendMessage("bot", `<span id="${typingId}" style="opacity:.5;font-style:italic;">Thinking...</span>`);

  const removeTyping = () =>
    document.getElementById(typingId)?.closest(".chat-msg")?.remove();

  try {
    // 1st: IBM watsonx.ai Granite
    const wx = await api("/api/watsonx/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    removeTyping();
    appendMessage("bot", wx.response +
      `<div style="font-size:.72rem;opacity:.45;margin-top:.3rem;">
        IBM watsonx.ai &middot; granite-4-h-small &middot; Project ${WATSONX_PROJECT_ID.slice(0,8)}&hellip;
       </div>`);
  } catch {
    try {
      // 2nd: Watson Assistant
      if (!watsonSessionId) {
        const { data } = await api("/api/chat/session", { method: "POST" });
        watsonSessionId = data.session_id;
      }
      const { data } = await api("/api/chat/message", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: watsonSessionId, text }),
      });
      removeTyping();
      appendMessage("bot", data.response_text);
    } catch {
      // Local keyword fallback
      removeTyping();
      appendMessage("bot", localChatResponse(text));
    }
  }
}

function appendMessage(role, text) {
  const msgs = document.getElementById("chat-messages");
  const div  = document.createElement("div");
  div.className = "chat-msg " + role;
  div.innerHTML = `
    <span class="chat-avatar">${role === "bot" ? "💧" : "👤"}</span>
    <div class="chat-bubble">${text}</div>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

/* ── Auto-load on DOMContentLoaded ───────────────────────────────── */
document.addEventListener("DOMContentLoaded", loadAll);
