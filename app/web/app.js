/* Pi ANC Demo 仪表盘前端逻辑 */
"use strict";

// ---------- 常量与 DOM ----------
const $ = (id) => document.getElementById(id);

const LIVE_MS = 3000;
const GRID_POLL_MS = 1500;

const mapCanvas = $("map");
const mapCtx = mapCanvas.getContext("2d");
const specCanvas = $("spectrum");
const specCtx = specCanvas.getContext("2d");

// 地图状态
const mapState = {
  surface: null,      // { x:[], y:[], z:[[...]] }
  points: [],         // [{x,y,spl_db,source_hits}]
  source: null,       // 噪声源 (x, y)
  sourceManual: false,
  selected: null,     // 点选位置
  quietZone: null,    // 建议静音区
  geometry: null,     // 像素映射参数
  zoom: null,
};

// ---------- 工具 ----------
function fmt(v, unit = "") {
  if (v === null || v === undefined) return "—";
  return `${v}${unit}`;
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function heatColor(t) {
  // t ∈ [0,1]：蓝 → 青 → 黄
  const r = Math.round(30 + 210 * Math.max(0, t - 0.5) * 2);
  const g = Math.round(30 + 160 * t);
  const b = Math.round(120 + 120 * (1 - t));
  return `rgb(${r},${g},${b})`;
}

// ---------- 实时监控轮询 ----------
async function pollLive() {
  try {
    const res = await fetch("/api/live");
    const s = await res.json();
    updateLive(s);
  } catch {
    setPiStatus("offline", "离线");
  }
}

function updateLive(s) {
  // Pi 状态
  if (s.mic_ok === false) {
    setPiStatus("paused", "⚠ 无麦克风: " + (s.error || "未检测到输入设备"));
  } else if (s.error) {
    setPiStatus("paused", "异常: " + s.error);
  } else if (!s.running) {
    setPiStatus("off", "未运行");
  } else if (s.paused) {
    setPiStatus("paused", "测量中");
  } else {
    setPiStatus("on", "工作中");
  }
  $("cpu-temp").textContent = fmt(s.cpu_temp_c, "°C");
  $("uptime").textContent = fmt(s.uptime_s === null ? null : fmtUptime(s.uptime_s));
  $("last-sample").textContent = fmt(s.last_update_age_s, "s前");

  // 噪声
  const spl = s.mic_ok === false ? null : (s.spl_db ?? s.rms_db);
  $("spl-num").textContent = spl === null ? "--" : spl.toFixed(1);
  $("spl-num").classList.toggle("dim", spl === null);
  $("dominant").textContent = fmt(s.mic_ok === false ? null : s.dominant_freq, " Hz");
  $("source-guess").textContent = s.mic_ok === false
    ? "无输入信号"
    : (s.source_guess
      ? `${s.source_guess} (${Math.round((s.source_confidence || 0) * 100)}%)`
      : "未识别");

  const bands = s.band_spl_db || {};
  $("bands").textContent = s.mic_ok === false
    ? "无有效输入信号"
    : Object.entries(bands).map(([k, v]) => `${k}: ${v}dB`).join(" · ");

  if (s.spectrum_freqs && s.spectrum_db && s.mic_ok !== false) {
    drawSpectrum(s.spectrum_freqs, s.spectrum_db);
  }
}

// ---------- 麦克风标定 ----------
async function pollCalib() {
  try {
    const res = await fetch("/api/calibration");
    const r = await res.json();
    if (r.configured) {
      $("calib-offset").textContent = `${r.offset_db} dB`;
      $("spl-unit").textContent = "dB SPL";
      const info = r.info || {};
      $("calib-note").textContent = `已标定：${info.calibrated_at || ""}，参考 ${info.known_spl_db} dB，
      测得 ${info.measured_rms_db} dBFS → 偏移 ${r.offset_db} dB。当前读数为绝对声压级。`;
    } else {
      $("calib-offset").textContent = "未标定（dBFS 相对值）";
      $("spl-unit").textContent = "dB";
      $("calib-note").textContent = "用手机声级计 APP 在麦克风处读出真实 SPL，填入后标定，仪表盘与 ANC 报告即显示绝对 dB SPL（当前未标定时显示 dBFS 相对值，数值偏小属正常）。";
    }
  } catch { /* 标定服务不可用时忽略 */ }
}

$("calib-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  setButton($("calib-start"), true, "标定中…（请保持环境噪声稳定）");
  const body = {
    known_spl: parseFloat($("calib-known-spl").value) || 75,
    duration_s: parseFloat($("calib-duration").value) || 8,
  };
  try {
    const res = await fetch("/api/calibration", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const r = await res.json();
    if (r.ok) {
      $("calib-offset").textContent = `${r.offset_db} dB`;
      $("spl-unit").textContent = "dB SPL";
      $("calib-note").textContent = `标定成功：参考 ${r.known_spl_db} dB，测得 ${r.measured_rms_db} dBFS，
      偏移 ${r.offset_db} dB。现在显示的是绝对 dB SPL。`;
    } else {
      $("calib-note").textContent = "标定失败: " + (r.error || "未知错误");
    }
  } catch {
    $("calib-note").textContent = "标定失败: 无法连接服务";
  }
  setButton($("calib-start"), false, "开始标定");
});

function setPiStatus(cls, text) {
  $("pi-dot").className = "dot " + cls;
  $("pi-status").textContent = text;
}

function fmtUptime(s) {
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h${Math.floor((s % 3600) / 60)}m`;
}

// ---------- 频谱绘制 ----------
function drawSpectrum(freqs, db) {
  const w = specCanvas.width, h = specCanvas.height;
  specCtx.clearRect(0, 0, w, h);
  if (!freqs.length) return;

  const dmin = Math.min(...db) - 3;
  const dmax = Math.max(...db) + 3;
  const n = freqs.length;

  // 网格
  specCtx.strokeStyle = "rgba(124,136,173,0.25)";
  specCtx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = Math.round(h - (i / 4) * (h - 20) - 10);
    specCtx.beginPath();
    specCtx.moveTo(40, y);
    specCtx.lineTo(w, y);
    specCtx.stroke();
  }

  // 波形
  specCtx.strokeStyle = "#22d3ee";
  specCtx.lineWidth = 1.5;
  specCtx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = 40 + (i / (n - 1)) * (w - 50);
    const y = h - 10 - ((db[i] - dmin) / (dmax - dmin)) * (h - 20);
    i === 0 ? specCtx.moveTo(x, y) : specCtx.lineTo(x, y);
  }
  specCtx.stroke();

  // X 轴标签（对数刻度感）
  specCtx.fillStyle = "rgba(124,136,173,0.8)";
  specCtx.font = "10px sans-serif";
  const maxF = freqs[n - 1];
  for (const f of [50, 100, 250, 500, 1000, 2000, 4000, 8000]) {
    if (f > maxF) continue;
    const x = 40 + (Math.log(f) - Math.log(50)) / (Math.log(maxF) - Math.log(50)) * (w - 50);
    specCtx.fillText(f >= 1000 ? `${f / 1000}k` : `${f}`, x, h - 2);
  }
}

// ---------- 网格测量 ----------
async function startGrid(ev) {
  ev.preventDefault();
  const payload = {
    origin_x: parseFloat($("origin-x").value),
    origin_y: parseFloat($("origin-y").value),
    size_x: parseFloat($("size-x").value),
    size_y: parseFloat($("size-y").value),
    step: parseFloat($("step").value),
    per_point_s: parseFloat($("per-point-s").value),
    height_m: parseFloat($("height-m").value) || 0.5,
    synthetic: $("synthetic").checked,
  };
  setButton($("grid-start"), true, "测量中…");
  try {
    const res = await fetch("/api/grid/measure", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const r = await res.json();
    if (!r.started) {
      $("progress-label").textContent = r.message || "启动失败";
    }
    pollGrid();
  } catch {
    $("progress-label").textContent = "启动失败";
    setButton($("grid-start"), false, "开始测量");
  }
}

function setButton(btn, disabled, text) {
  btn.disabled = disabled;
  btn.textContent = text;
}

async function pollGrid() {
  try {
    const res = await fetch("/api/grid/status");
    const s = await res.json();
    renderGridStatus(s);
    if (s.state === "running") {
      setTimeout(pollGrid, GRID_POLL_MS);
    } else {
      setButton($("grid-start"), false, "开始测量");
    }
  } catch {
    setButton($("grid-start"), false, "开始测量");
  }
}

function renderGridStatus(s) {
  const fill = $("progress-fill");
  const label = $("progress-label");

  if (s.state === "idle") {
    fill.style.width = "0%";
    label.textContent = "空闲";
    return;
  }
  if (s.state === "running") {
    fill.style.width = `${Math.round((s.progress || 0) * 100)}%`;
    label.textContent = `测量中 ${s.current_point || ""}`;
    return;
  }
  if (s.state === "error") {
    fill.style.width = "100%";
    label.textContent = `失败: ${s.error || ""}`;
    return;
  }
  if (s.state === "done") {
    fill.style.width = "100%";
    label.textContent = "完成";
    mapState.surface = s.surface;
    mapState.points = s.points || [];
    if (!mapState.sourceManual) {
      // 自动源 = 最高 SPL 点（带 z 高度）
      const loud = [...mapState.points].sort((a, b) => b.spl_db - a.spl_db)[0];
      const z = parseFloat($("height-m").value) || 0.5;
      mapState.source = loud ? { x: loud.x, y: loud.y, z } : null;
    }
    mapState.quietZone = s.quiet_zone || null;
    renderRecommend(s);
    drawMap();
  }
}

function renderRecommend(s) {
  const panel = $("recommend-panel");
  const body = $("recommend-body");
  if (!s.quiet_zone) {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");
  const qz = s.quiet_zone;
  const src = qz.source_pos_m, pos = qz.quiet_pos_m;
  // 3D 打印机位置
  $("printer-x").textContent = fmt(src[0].toFixed(2), " m");
  $("printer-y").textContent = fmt(src[1].toFixed(2), " m");
  $("printer-z").textContent = fmt(src[2].toFixed(2), " m");
  // 建议静音区
  let html = `<div class="row"><span>建议静音区</span><strong>(${pos[0].toFixed(2)}, ${pos[1].toFixed(2)}, ${pos[2].toFixed(2)})</strong></div>`;
  html += `<div class="row"><span>距噪声源</span><strong>${qz.distance_m}m</strong></div>`;
  html += `<div class="row"><span>安静区直径</span><strong>${qz.zone_of_quiet_diameter_m}m</strong></div>`;
  html += `<div class="row"><span>主频</span><strong>${qz.dominant_freq_hz}Hz</strong></div>`;
  if (s.recommendation) {
    const rec = s.recommendation;
    html += `<div class="row"><span>建议降噪</span><strong>${rec.anc_worthwhile ? "需要" : "暂不需要"}</strong></div>`;
  }
  body.innerHTML = html;
}

// ---------- 地图绘制 ----------
function computeGeometry() {
  const pad = 40;
  const w = mapCanvas.width, h = mapCanvas.height;
  let minX = 0, maxX = 1, minY = 0, maxY = 1;

  if (mapState.points.length) {
    const xs = mapState.points.map((p) => p.x);
    const ys = mapState.points.map((p) => p.y);
    minX = Math.min(...xs); maxX = Math.max(...xs);
    minY = Math.min(...ys); maxY = Math.max(...ys);
  }
  if (mapState.surface && mapState.surface.x.length) {
    minX = Math.min(minX, mapState.surface.x[0]);
    maxX = Math.max(maxX, mapState.surface.x[mapState.surface.x.length - 1]);
    minY = Math.min(minY, mapState.surface.y[0]);
    maxY = Math.max(maxY, mapState.surface.y[mapState.surface.y.length - 1]);
  }
  const range = Math.max(maxX - minX, maxY - minY, 0.5);
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const scale = (Math.min(w, h) - pad * 2) / range;
  return {
    minX: cx - range / 2, maxX: cx + range / 2,
    minY: cy - range / 2, maxY: cy + range / 2,
    scale, pad, w, h,
  };
}

function toPx(geo, x, y) {
  const px = geo.pad + (x - geo.minX) / (geo.maxX - geo.minX) * (geo.w - geo.pad * 2);
  const py = geo.h - geo.pad - (y - geo.minY) / (geo.maxY - geo.minY) * (geo.h - geo.pad * 2);
  return [px, py];
}

function toWorld(geo, px, py) {
  const x = geo.minX + (px - geo.pad) / (geo.w - geo.pad * 2) * (geo.maxX - geo.minX);
  const y = geo.minY + (geo.h - geo.pad - py) / (geo.h - geo.pad * 2) * (geo.maxY - geo.minY);
  return [x, y];
}

function drawMap() {
  const geo = computeGeometry();
  mapState.geometry = geo;
  mapCtx.clearRect(0, 0, geo.w, geo.h);
  mapCtx.fillStyle = "#0d1424";
  mapCtx.fillRect(0, 0, geo.w, geo.h);

  // 热力面
  if (mapState.surface && mapState.surface.z.length) {
    const { x, y, z } = mapState.surface;
    let zmin = Infinity, zmax = -Infinity;
    for (const row of z) for (const v of row) {
      if (v !== null) { zmin = Math.min(zmin, v); zmax = Math.max(zmax, v); }
    }
    if (zmin === Infinity) { zmin = -60; zmax = -30; }
    const span = Math.max(zmax - zmin, 1e-6);

    for (let j = 0; j < y.length; j++) {
      for (let i = 0; i < x.length; i++) {
        const v = z[j][i];
        if (v === null) continue;
        const [px1] = toPx(geo, x[i], y[j]);
        const [px2] = toPx(geo, x[Math.min(i + 1, x.length - 1)], y[Math.min(j + 1, y.length - 1)]);
        const size = Math.max(2, Math.abs(px2 - px1) + 1);
        mapCtx.fillStyle = heatColor((v - zmin) / span);
        mapCtx.fillRect(px1 - size / 2, px2 - size / 2, size, size);
      }
    }
  }

  // 测点
  for (const p of mapState.points) {
    const [px, py] = toPx(geo, p.x, p.y);
    mapCtx.beginPath();
    mapCtx.arc(px, py, 3, 0, Math.PI * 2);
    mapCtx.fillStyle = "#dbe4ff";
    mapCtx.fill();
  }

  // 建议静音区（圆圈，直径 = 安静区直径）
  if (mapState.quietZone) {
    const qz = mapState.quietZone;
    const [cx0, cy0] = toPx(geo, qz.quiet_pos_m[0], qz.quiet_pos_m[1]);
    const d = qz.zone_of_quiet_diameter_m;
    const [dx] = toPx(geo, qz.quiet_pos_m[0] + d, qz.quiet_pos_m[1]);
    const radiusPx = Math.abs(dx - cx0);
    mapCtx.beginPath();
    mapCtx.arc(cx0, cy0, radiusPx, 0, Math.PI * 2);
    mapCtx.strokeStyle = "#34d399";
    mapCtx.lineWidth = 2;
    mapCtx.stroke();
    mapCtx.fillStyle = "rgba(52,211,153,0.12)";
    mapCtx.fill();
    mapCtx.fillStyle = "#34d399";
    mapCtx.font = "12px sans-serif";
    mapCtx.fillText("建议静音区", cx0 + 6, cy0 - 6);
  }

  // 噪声源
  if (mapState.source) {
    const [sx, sy] = toPx(geo, mapState.source.x, mapState.source.y);
    mapCtx.beginPath();
    mapCtx.arc(sx, sy, 8, 0, Math.PI * 2);
    mapCtx.strokeStyle = "#f87171";
    mapCtx.lineWidth = 2;
    mapCtx.stroke();
    mapCtx.fillStyle = "rgba(248,113,113,0.3)";
    mapCtx.fill();
    mapCtx.fillStyle = "#f87171";
    mapCtx.font = "12px sans-serif";
    const zTxt = mapState.source.z !== undefined ? `z=${mapState.source.z.toFixed(2)}` : "";
    mapCtx.fillText(`噪声源 ${zTxt}`, sx + 10, sy + 4);
  }

  // 选中点
  if (mapState.selected) {
    const [sx, sy] = toPx(geo, mapState.selected.x, mapState.selected.y);
    mapCtx.beginPath();
    mapCtx.arc(sx, sy, 5, 0, Math.PI * 2);
    mapCtx.strokeStyle = "#22d3ee";
    mapCtx.lineWidth = 2;
    mapCtx.stroke();
    mapCtx.fillStyle = "rgba(34,211,238,0.25)";
    mapCtx.fill();
    mapCtx.fillStyle = "#22d3ee";
    mapCtx.font = "12px sans-serif";
    mapCtx.fillText(`(${mapState.selected.x.toFixed(2)}, ${mapState.selected.y.toFixed(2)})`, sx + 8, sy - 8);
  }

  // 坐标轴
  mapCtx.strokeStyle = "rgba(124,136,173,0.5)";
  mapCtx.lineWidth = 1;
  mapCtx.beginPath();
  const [ox0, oy0] = toPx(geo, geo.minX, 0);
  const [ox1, oy1] = toPx(geo, geo.maxX, 0);
  const [oyA] = toPx(geo, 0, geo.minY);
  const [oyB] = toPx(geo, 0, geo.maxY);
  mapCtx.moveTo(ox0, oy0); mapCtx.lineTo(ox1, oy0);
  mapCtx.moveTo(oyA, oy0); mapCtx.lineTo(oyB, oy0);
  mapCtx.stroke();
  mapCtx.fillStyle = "rgba(124,136,173,0.9)";
  mapCtx.font = "11px sans-serif";
  mapCtx.fillText("x (m)", ox1 - 30, oy0 - 6);
  mapCtx.fillText("y (m)", oyB + 8, oy0 + 12);
}

// ---------- 地图交互 ----------
mapCanvas.addEventListener("click", async (ev) => {
  if (!mapState.geometry) return;
  const rect = mapCanvas.getBoundingClientRect();
  const px = (ev.clientX - rect.left) * (mapCanvas.width / rect.width);
  const py = (ev.clientY - rect.top) * (mapCanvas.height / rect.height);
  const [x, y] = toWorld(mapState.geometry, px, py);
  mapState.selected = { x, y };
  drawMap();
  if (!mapState.source) {
    $("qz-panel").classList.remove("hidden");
    $("qz-verdict").textContent = "请先标记噪声源";
    $("qz-verdict").className = "";
    return;
  }
  await queryQuietZone(x, y);
});

async function queryQuietZone(x, y) {
  try {
    const z = parseFloat($("height-m").value) || 0.5;
    const res = await fetch("/api/quiet-zone", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        x, y, z,
        source_x: mapState.source.x,
        source_y: mapState.source.y,
        source_z: mapState.source.z || z,
      }),
    });
    const r = await res.json();
    if (r.error) {
      $("qz-panel").classList.remove("hidden");
      $("qz-verdict").textContent = r.error;
      return;
    }
    const f = r.feasibility;
    const sel = r.selected_m, src = r.source_m;
    $("qz-panel").classList.remove("hidden");
    $("qz-pos").textContent = `(${sel[0].toFixed(2)}, ${sel[1].toFixed(2)}, ${sel[2].toFixed(2)})`;
    $("qz-source-pos").textContent = `(${src[0].toFixed(2)}, ${src[1].toFixed(2)}, ${src[2].toFixed(2)})`;
    $("qz-dist").textContent = `${f.distance_m}m`;
    $("qz-delay").textContent = `${f.propagation_delay_us}µs`;
    $("qz-diameter").textContent = `${f.zone_of_quiet_diameter_m}m`;
    const verdictEl = $("qz-verdict");
    verdictEl.textContent = {
      good: "可行（安静区足够大）",
      marginal: "勉强（距离/尺寸临界）",
      poor: "不可行（太远/安静区太小）",
    }[f.verdict] || f.verdict;
    verdictEl.className = f.verdict;
  } catch {
    $("qz-panel").classList.remove("hidden");
    $("qz-verdict").textContent = "查询失败";
  }
}

$("mark-source").addEventListener("click", () => {
  if (!mapState.selected) {
    $("qz-panel").classList.remove("hidden");
    $("qz-verdict").textContent = "请先在地图上点选一个点";
    $("qz-verdict").className = "";
    return;
  }
  mapState.source = {
    ...mapState.selected,
    z: parseFloat($("height-m").value) || 0.5,
  };
  mapState.sourceManual = true;
  drawMap();
});

$("reset-map").addEventListener("click", () => {
  mapState.surface = null;
  mapState.points = [];
  mapState.selected = null;
  mapState.quietZone = null;
  mapState.sourceManual = false;
  $("recommend-panel").classList.add("hidden");
  $("qz-panel").classList.add("hidden");
  drawMap();
});

// ---------- ANC 实时降噪 ----------
const ANC_POLL_MS = 1000;
const ancTrend = $("anc-trend");
const ancTrendCtx = ancTrend.getContext("2d");
let ancHistory = [];       // 实时 SPL 曲线
let ancReportShown = false;
let ancSourceInjected = false;

function setAncButton(disabled, text) {
  $("anc-start").disabled = disabled;
  $("anc-start").textContent = text;
}

function ancPhaseText(p) {
  return { idle: "空闲", baseline: "采集中（ANC off）", cancelling: "降噪中（ANC on）", done: "完成", error: "错误" }[p] || p;
}

function renderAncReport(r) {
  const body = $("anc-report-body");
  if (!r.found) {
    body.innerHTML = "<p>暂无完整报告。</p>";
    return;
  }
  let peaks = "";
  if (r.peak_reductions && r.peak_reductions.length) {
    peaks = r.peak_reductions.map(p =>
      `<div class="row"><span>${p.freq.toFixed(0)} Hz</span><strong>${p.reduction_db.toFixed(1)} dB 降低</strong></div>`
    ).join("");
  }
  body.innerHTML = `
    <div class="row"><span>基频 f0</span><strong>${r.f0_hz ? r.f0_hz.toFixed(1) + " Hz" : "—"}</strong></div>
    <div class="row"><span>基线 SPL</span><strong>${r.baseline_spl_db ?? "—"} dB</strong></div>
    <div class="row"><span>降噪后 SPL</span><strong>${r.cancelling_spl_db ?? "—"} dB</strong></div>
    <div class="row"><span>音调降噪（中位）</span><strong class="good">${r.tone_reduction_db ?? "—"} dB</strong></div>
    <div class="row"><span>宽带降噪</span><strong class="good">${r.broadband_reduction_db.toFixed(1)} dB</strong></div>
    <div class="row"><span>A 加权降噪</span><strong class="good">${r.a_weighted_reduction_db.toFixed(1)} dB</strong></div>
    ${peaks ? `<div class="row"><span>音调峰值</span></div>${peaks}` : ""}
  `;
  $("anc-report").classList.remove("hidden");
  ancReportShown = true;
}

async function pollAnc() {
  try {
    const res = await fetch("/api/anc/live/status");
    const st = await res.json();
    $("anc-state").textContent = st.state === "idle" ? "空闲" : st.state;
    $("anc-phase").textContent = ancPhaseText(st.phase);
    $("anc-mic-delay-now").textContent = st.mic_delay_ms != null ? st.mic_delay_ms.toFixed(1) + " ms" : "—";
    const scanned = st.delay_scan_ms;
    if (scanned != null) {
      $("anc-mic-delay-now").textContent += "（自动校准 " + scanned.toFixed(1) + " ms）";
    }
    $("anc-f0-now").textContent = st.f0 ? st.f0.toFixed(1) + " Hz" : "—";
    $("anc-base-db").textContent = st.baseline_spl_db ?? "—";
    $("anc-now-db").textContent = st.spl_now_db ?? "—";
    const red = st.reduction_db;
    $("anc-reduction").textContent = red != null ? red.toFixed(1) + " dB" : "—";
    if (st.error) $("anc-state").textContent = "错误";

    // watchdog 状态与日志
    const wd = st.watchdog || {};
    $("anc-wd-stat").textContent = wd.enabled
      ? `开启（降 ${wd.reduce_count ?? 0} 次 / 升 ${wd.increase_count ?? 0} 次）`
      : "关闭";
    renderWdLog(wd.log || []);

    // 人声门控状态：说话声出现时静音反相输出（避免"吃掉"人声）
    const vg = st.voice_gate || {};
    if (vg.mute) {
      $("anc-voice-gate").textContent = "静音（检测到人声）";
      $("anc-voice-gate").className = "warn";
    } else {
      $("anc-voice-gate").textContent = "正常";
      $("anc-voice-gate").className = "good";
    }

    // 风噪门控状态：低频能量占比超阈值时静音反相输出
    const wg = st.wind_gate || {};
    if (wg.enabled) {
      const ratioStr = wg.diff_ratio != null ? `差分 ${wg.diff_ratio.toFixed(2)}`
        : wg.lf_ratio != null ? `LF ${wg.lf_ratio.toFixed(2)}` : "—";
      if (wg.mute) {
        $("anc-wind-gate-now").textContent = `静音（${ratioStr}）`;
        $("anc-wind-gate-now").className = "warn";
      } else {
        $("anc-wind-gate-now").textContent = `正常（${ratioStr}）`;
        $("anc-wind-gate-now").className = "good";
      }
    } else {
      $("anc-wind-gate-now").textContent = "关闭";
      $("anc-wind-gate-now").className = "";
    }

    if (st.phase === "cancelling") {
      setAncButton(true, "降噪中…");
      if (st.spl_now_db != null) {
        ancHistory.push({ t: st.elapsed_s, v: st.spl_now_db });
        if (ancHistory.length > 180) ancHistory = ancHistory.slice(-180);
        drawAncTrend();
      }
      $("anc-report").classList.add("hidden");
      ancReportShown = false;
    } else if (st.phase === "done" && !ancReportShown) {
      setAncButton(false, "开始 ANC");
      const rep = await (await fetch("/api/anc/live/report")).json();
      renderAncReport(rep);
    } else if (st.state === "idle" || st.state === "error") {
      setAncButton(false, "开始 ANC");
    }
  } catch { /* 轮询失败忽略 */ }
}

async function pollAncSource() {
  try {
    const res = await fetch("/api/anc/source");
    const r = await res.json();
    if (!r.found) {
      $("anc-source-name").textContent = "—";
      $("anc-source-conf").textContent = "—";
      $("anc-source-f0").textContent = "—";
      $("anc-source-feas").textContent = "—";
      $("anc-source-notes").textContent = r.error || "暂无分析数据";
      return;
    }
    $("anc-source-name").textContent = r.source_name || r.source_id || "未识别";
    $("anc-source-conf").textContent = r.confidence
      ? Math.round(r.confidence * 100) + "%" : "—";
    $("anc-source-f0").textContent = r.recommended_f0
      ? r.recommended_f0.toFixed(1) + " Hz" : "—";
    const feas = r.feasibility || {};
    if (r.is_voice) {
      $("anc-source-feas").textContent = "不适用（检测到人声）";
      $("anc-source-notes").textContent =
        "检测到人声/语音：ANC 只对稳态周期噪声（电机/风扇/打印机）有效，"
        + "对说话声会反向消除、像\"吃掉\"人声。请等人声停止，或只对设备噪声启动 ANC。";
      return; // 人声不作为 ANC 目标：不预置 f0，不提示启动
    }
    $("anc-source-feas").textContent = r.anc_worthwhile
      ? "值得（音调成分可消除）" : "收益有限";
    let notes = r.notes || "";
    if (r.anc_worthwhile && r.recommended_f0) {
      notes = `建议以 ${r.recommended_f0.toFixed(1)} Hz 为基频启动 ANC。` + notes;
    }
    $("anc-source-notes").textContent = notes || "—";

    // 识别结果预置 f0：仅在用户未手动输入时自动填入
    if (!ancSourceInjected && r.recommended_f0 && !$("anc-f0").value) {
      $("anc-f0").value = r.recommended_f0.toFixed(1);
      ancSourceInjected = true;
    }
  } catch { /* 轮询失败忽略 */ }
}

function drawAncTrend() {
  const ctx = ancTrendCtx, c = ancTrend;
  ctx.clearRect(0, 0, c.width, c.height);
  if (ancHistory.length < 2) return;
  const pad = 8, w = c.width - 2 * pad, h = c.height - 2 * pad;
  const vs = ancHistory.map(p => p.v);
  const lo = Math.floor(Math.min(...vs) / 5) * 5;
  const hi = Math.ceil(Math.max(...vs) / 5) * 5;
  const span = Math.max(hi - lo, 5);
  ctx.strokeStyle = "#7dd3fc";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ancHistory.forEach((p, i) => {
    const x = pad + (i / (ancHistory.length - 1)) * w;
    const y = pad + (1 - (p.v - lo) / span) * h;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "rgba(148,163,184,.7)";
  ctx.font = "11px sans-serif";
  ctx.fillText(`${lo} dB`, 4, h + pad);
  ctx.fillText(`${hi} dB`, 4, 12);
}

let wdLogShown = null;
function renderWdLog(log) {
  const box = $("anc-wd-log");
  const list = $("anc-wd-log-list");
  const key = JSON.stringify(log);
  if (key === wdLogShown) return;
  wdLogShown = key;
  if (!log.length) {
    box.classList.add("hidden");
    list.innerHTML = "";
    return;
  }
  box.classList.remove("hidden");
  list.innerHTML = log.slice(-10).reverse().map((e) => {
    const cls = e.action === "reduce_gain" ? "wd-reduce"
      : e.action === "increase_gain" ? "wd-increase" : "";
    return `<li class="${cls}">${esc(e.msg)}<span class="wd-t">${e.t}s</span></li>`;
  }).join("");
}

$("anc-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  setAncButton(true, "启动中…");
  ancHistory = [];
  wdLogShown = null;
  drawAncTrend();
  const body = {
    synthetic: $("anc-synthetic").checked,
    watchdog_enabled: $("anc-watchdog").checked,
    wind_gate_enabled: $("anc-wind-gate").checked,
    dual_mic: $("anc-dual-mic").checked,
    f0: parseFloat($("anc-f0").value) || null,
    gain: parseFloat($("anc-gain").value) || 0.5,
    mic_delay_ms: parseFloat($("anc-mic-delay").value) ?? 5.0,
    input_highpass_hz: parseFloat($("anc-highpass").value) || 0,
    baseline_s: parseFloat($("anc-baseline").value) || 5,
    duration_s: parseFloat($("anc-duration").value) || 60,
  };
  const res = await fetch("/api/anc/live/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const r = await res.json();
  if (!r.started) {
    setAncButton(false, "开始 ANC");
    $("anc-state").textContent = r.message || "启动失败";
  }
});

$("anc-stop").addEventListener("click", async () => {
  await fetch("/api/anc/live/stop", { method: "POST" });
  setAncButton(false, "开始 ANC");
});

// ---------- Kimi 噪声检测 Agent ----------
const AGENT_POLL_MS = 1500;
let agentLastState = null;

function setAgentButton(disabled, text) {
  $("agent-start").disabled = disabled;
  $("agent-start").textContent = text;
}

function agentStateText(st) {
  if (st.state === "running") return "检测中…";
  if (st.state === "done") return "已完成";
  if (st.state === "error") return "失败";
  return st.configured ? "空闲" : "未配置";
}

function renderAgentResult(r) {
  const box = $("agent-result");
  if (!r) {
    box.classList.add("hidden");
    return;
  }
  box.classList.remove("hidden");
  const signalMap = {
    environment_noise: "环境噪声",
    acoustic_feedback: "声反馈（啸叫）",
    suspected_feedback: "疑似啸叫",
    uncertain: "不确定",
  };
  $("agent-signal").textContent = signalMap[r.signal_class] || "—";
  $("agent-signal").className = (r.signal_class === "acoustic_feedback" || r.signal_class === "suspected_feedback")
    ? "bad" : "";
  $("agent-howling").textContent = r.is_howling ? "是" : "否";
  $("agent-howling").className = r.is_howling ? "bad" : "good";
  $("agent-source").textContent = r.source_name || r.source_id || "未识别";
  $("agent-conf").textContent = r.confidence != null
    ? `${Math.round(r.confidence * 100)}%` : "—";
  const freq = r.dominant_freq_hz != null ? `${r.dominant_freq_hz.toFixed(1)} Hz` : "—";
  const f0 = r.recommended_f0_hz != null ? `${r.recommended_f0_hz.toFixed(1)} Hz` : "—";
  $("agent-freq").textContent = `${freq} / ${f0}`;
  $("agent-anc").textContent = r.anc_worthwhile ? "值得（可做 ANC）" : "收益有限";
  $("agent-anc").className = r.anc_worthwhile ? "good" : "";
  $("agent-summary").textContent = r.summary || "";

  let html = "";
  if (r.howling_freq_hz != null) {
    html += `<p class="hint">啸叫/声反馈候选频率：${esc(r.howling_freq_hz.toFixed(1))} Hz</p>`;
  }
  if (r.anc_adjustments && r.anc_adjustments.length) {
    html += `<h3>ANC 调节（已执行）</h3><ul class="agent-list">` +
      r.anc_adjustments.map((a) => `<li>${esc(a.action || "")}${a.value != null ? ` = ${esc(String(a.value))}` : ""} — ${esc(a.reason || "")}</li>`).join("") + `</ul>`;
  }
  if (r.reasons && r.reasons.length) {
    html += `<h3>判据</h3><ul class="agent-list">` +
      r.reasons.map((s) => `<li>${esc(s)}</li>`).join("") + `</ul>`;
  }
  if (r.actions && r.actions.length) {
    html += `<h3>建议动作</h3><ul class="agent-list">` +
      r.actions.map((s) => `<li>${esc(s)}</li>`).join("") + `</ul>`;
  }
  if (r.reference_mic) {
    html += `<p class="hint">参考麦克风：${esc(r.reference_mic)}</p>`;
  }
  if (r.quiet_zone_advice) {
    html += `<p class="hint">静音区：${esc(r.quiet_zone_advice)}</p>`;
  }
  $("agent-detail").innerHTML = html;
}

async function pollAgent() {
  try {
    const res = await fetch("/api/agent/status");
    const st = await res.json();
    $("agent-state").textContent = agentStateText(st);
    $("agent-state").className = st.state === "error" ? "bad" : "";
    $("agent-config").textContent = st.configured
      ? "可用" : (st.configured_reason || "未配置");
    $("agent-model").textContent = st.configured ? (st.model || "kimi-k3") : "—";
    $("agent-elapsed").textContent = st.elapsed_s != null ? st.elapsed_s + " s" : "—";
    $("agent-tools").textContent = (st.tools_used || []).length
      ? st.tools_used.join(", ") : "—";

    if (st.state === "running") {
      setAgentButton(true, "检测中…");
    } else {
      setAgentButton(false, "开始 AI 检测");
      // 任务结束且结论变化时刷新结果展示
      const changed = JSON.stringify(st.result) !== JSON.stringify(agentLastState);
      if (st.state === "done" && changed) {
        agentLastState = st.result;
        renderAgentResult(st.result);
      }
      if (st.state === "error") {
        $("agent-result").classList.remove("hidden");
        $("agent-summary").textContent = "检测失败: " + (st.error || "");
      }
    }
  } catch { /* 轮询失败忽略 */ }
}

$("agent-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  setAgentButton(true, "启动中…");
  agentLastState = null;
  $("agent-result").classList.add("hidden");
  const body = {
    fresh_sample: $("agent-fresh").checked,
    duration_s: parseFloat($("agent-duration").value) || 3,
    synthetic: $("agent-synthetic").checked,
  };
  const res = await fetch("/api/agent/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const r = await res.json();
  if (!r.started) {
    setAgentButton(false, "开始 AI 检测");
    $("agent-state").textContent = r.message || "启动失败";
  }
  pollAgent();
});

// ---------- 启动 ----------
$("grid-form").addEventListener("submit", startGrid);
pollLive();
setInterval(pollLive, LIVE_MS);
pollCalib();  // 麦克风标定状态
setInterval(pollCalib, 15000);
pollGrid();  // 恢复已完成的网格测量结果（刷新页面后）
pollAncSource();
setInterval(pollAncSource, 3000);
pollAnc();
setInterval(pollAnc, ANC_POLL_MS);
pollAgent();
setInterval(pollAgent, AGENT_POLL_MS);
drawMap();
