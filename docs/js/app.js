import { Scene, Detector, processVideo, RED, GREEN } from "./engine.js";

const COLORS = {
  jaywalking: "#f97316", failure_to_yield: "#ef4444", red_light: "#dc2626", stop_line: "#f59e0b",
  stopped_vehicle: "#d946ef", congestion: "#0ea5e9", wrong_way: "#a855f7", near_miss: "#eab308",
  accident: "#b91c1c", illegal_u_turn: "#84cc16",
};
const CLS_COLORS = { person: "#38bdf8", car: "#22c55e", bus: "#f59e0b", truck: "#fb923c", motorcycle: "#e879f9", bicycle: "#a78bfa" };
const $ = s => document.querySelector(s);
const el = (tag, attrs = {}, html = "") => { const e = document.createElement(tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); e.innerHTML = html; return e; };
const fmt = s => { s = Math.max(0, s); const m = Math.floor(s / 60); return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`; };

$("#menu").onclick = () => $("#nav").classList.toggle("open");
document.querySelectorAll("#nav a").forEach(a => a.onclick = () => $("#nav").classList.remove("open"));

const site = await (await fetch("data/site.json", { cache: "no-cache" })).json();
const videos = {};
for (const v of site.videos) videos[v.name] = await (await fetch(`data/${v.stem}.json`, { cache: "no-cache" })).json();

// ------------------------------------------------------------------ hero
$("#hero-stats").innerHTML = site.stats.map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");

// -------------------------------------------------------------- pipeline
$("#pipeline").innerHTML = pipelineSVG();
function pipelineSVG() {
  const W = 132, H = 52;
  const boxes = [
    ["Video (.mp4)", "any resolution", 8, 49],
    ["Registration", "SIFT + RANSAC", 160, 49],
    ["YOLO11s + ByteTrack", "1280 px · 7.5 fps", 312, 14],
    ["Signal lamp", "red / green pixels", 312, 86],
    ["Tracks", "canonical ground pts", 464, 49],
    ["Rules × 7", "zones · crossings · flow", 616, 49],
    ["Events", "[start, end, class]", 768, 49],
  ];
  let s = `<svg viewBox="0 0 910 168" xmlns="http://www.w3.org/2000/svg" font-family="Inter,sans-serif">
  <defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="#93a1b0"/></marker></defs>`;
  boxes.forEach(([t, sub, x, y], i) => {
    const col = i === 5 ? "#2dd4bf" : (i === 6 ? "#f59e0b" : "#93a1b0");
    s += `<rect x="${x}" y="${y}" width="${W}" height="${H}" rx="9" fill="none" stroke="${col}" stroke-width="1.5"/>
      <text x="${x + W / 2}" y="${y + 22}" text-anchor="middle" font-size="13" font-weight="600" fill="currentColor">${t}</text>
      <text x="${x + W / 2}" y="${y + 39}" text-anchor="middle" font-size="10.5" fill="#93a1b0">${sub}</text>`;
  });
  const arr = (x1, y1, x2, y2) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#93a1b0" stroke-width="1.4" marker-end="url(#ar)"/>`;
  s += arr(140, 75, 158, 75) + arr(292, 70, 310, 42) + arr(292, 80, 310, 110)
     + arr(444, 40, 462, 66) + arr(444, 112, 614, 92) + arr(596, 75, 614, 75) + arr(748, 75, 766, 75);
  s += `<text x="8" y="160" font-size="11" fill="#93a1b0">Part B (causal): YOLO11n + ByteTrack on every ~6th frame → pairwise time-to-collision → smoothed risk ∈ [0, 1]</text>`;
  return s + "</svg>";
}

// ------------------------------------------------------------------ rules
$("#rules").innerHTML = site.rules.map(r => `<div class="rule" style="--c:${COLORS[r.id] || "#2dd4bf"}">
  <h4>${r.id}</h4><p class="def">${r.definition}</p><p>${r.rule}</p></div>`).join("");

// ---------------------------------------------------------------- EDA table
const vt = $("#video-table");
vt.innerHTML = `<tr><th>clip</th><th class="num">duration</th><th class="num">source</th><th class="num">fps</th><th>lighting</th>
  <th class="num">view shift</th><th class="num">tracks (people / vehicles)</th><th class="num">signal cycles</th><th class="num">events</th></tr>` +
  site.videos.map(v => `<tr><td>${v.name}</td><td class="num">${fmt(v.duration)}</td><td class="num">${v.source}</td><td class="num">${v.fps.toFixed(2)}</td>
    <td>${v.lighting}</td><td class="num">${(v.shift * 100).toFixed(1)} %</td><td class="num">${v.people} / ${v.vehicles}</td>
    <td class="num">${v.cycles}</td><td class="num">${videos[v.name].events.length}</td></tr>`).join("");

// ---------------------------------------------------- counts over time chart
const countState = { video: site.videos[0].name, classes: new Set(["person", "car", "bus", "truck"]) };
function renderCountControls() {
  const c = $("#count-controls"); c.innerHTML = "";
  for (const v of site.videos) {
    const b = el("button", { class: "chip" + (countState.video === v.name ? " on" : "") }, v.name);
    b.onclick = () => { countState.video = v.name; renderCountControls(); renderCounts(); }; c.append(b);
  }
  c.append(el("span", { style: "width:14px" }));
  for (const k of Object.keys(CLS_COLORS)) {
    const b = el("button", { class: "chip" + (countState.classes.has(k) ? " on" : ""), style: `color:${countState.classes.has(k) ? CLS_COLORS[k] : ""}` }, k);
    b.onclick = () => { countState.classes.has(k) ? countState.classes.delete(k) : countState.classes.add(k); renderCountControls(); renderCounts(); };
    c.append(b);
  }
}
function lineChart(series, { w = 900, h = 220, xmax, ylabel = "", phase = null, yfix = null } = {}) {
  const pad = { l: 40, r: 10, t: 10, b: 24 };
  let ymax = 1;
  for (const s of series) for (const y of s.y) ymax = Math.max(ymax, y);
  ymax = yfix ?? Math.ceil(ymax * 1.1);
  const X = x => pad.l + (x / xmax) * (w - pad.l - pad.r), Y = y => h - pad.b - (y / ymax) * (h - pad.t - pad.b);
  let s = `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg" font-family="JetBrains Mono,monospace" font-size="10">`;
  if (phase) for (const [a, b, p] of phase) s += `<rect x="${X(a)}" y="${pad.t}" width="${Math.max(0, X(b) - X(a))}" height="${h - pad.t - pad.b}" fill="${p === RED ? "#ef4444" : "#22c55e"}" opacity=".08"/>`;
  for (let i = 0; i <= 4; i++) { const yv = ymax * i / 4; s += `<line x1="${pad.l}" x2="${w - pad.r}" y1="${Y(yv)}" y2="${Y(yv)}" stroke="#93a1b0" stroke-opacity=".15"/><text x="${pad.l - 6}" y="${Y(yv) + 3}" text-anchor="end" fill="#93a1b0">${yv.toFixed(yv < 10 ? 1 : 0)}</text>`; }
  const step = xmax > 200 ? 60 : 20;
  for (let x = 0; x <= xmax; x += step) s += `<text x="${X(x)}" y="${h - 8}" text-anchor="middle" fill="#93a1b0">${fmt(x).replace(".0", "")}</text>`;
  for (const se of series) {
    const pts = se.y.map((y, i) => `${X(se.x ? se.x[i] : i).toFixed(1)},${Y(y).toFixed(1)}`).join(" ");
    s += `<polyline points="${pts}" fill="none" stroke="${se.color}" stroke-width="1.6"/>`;
  }
  if (ylabel) s += `<text x="${pad.l + 4}" y="${pad.t + 10}" fill="#93a1b0">${ylabel}</text>`;
  return s + "</svg>";
}
function phaseRuns(ph) {
  const runs = []; let cur = null;
  for (const [t, p] of ph) {
    if (!cur || cur[2] !== p) { if (cur) { cur[1] = t; runs.push(cur); } cur = [t, t, p]; }
  }
  if (cur) { cur[1] = ph[ph.length - 1][0]; runs.push(cur); }
  return runs.filter(r => r[2] !== 0);
}
function renderCounts() {
  const d = videos[countState.video];
  const smooth = (a, k = 5) => a.map((_, i) => { const s = a.slice(Math.max(0, i - k), i + k + 1); return s.reduce((x, y) => x + y, 0) / s.length; });
  const series = [...countState.classes].filter(k => d.counts_per_sec[k]).map(k => ({ y: smooth(d.counts_per_sec[k]), color: CLS_COLORS[k] }));
  $("#count-chart").innerHTML = `<div class="legend">${[...countState.classes].map(k => `<span><i style="background:${CLS_COLORS[k]}"></i>${k}</span>`).join("")}<span><i style="background:#ef4444;opacity:.4"></i>signal red</span></div>` +
    lineChart(series, { xmax: d.duration, ylabel: "tracked objects in view (11 s moving average)", phase: phaseRuns(d.phase) });
}
renderCountControls(); renderCounts();

// signal phase chart for all clips
{
  const w = 900, rowH = 26, pad = 90, xmax = Math.max(...site.videos.map(v => v.duration));
  let s = `<svg viewBox="0 0 ${w} ${site.videos.length * rowH + 24}" xmlns="http://www.w3.org/2000/svg" font-family="JetBrains Mono,monospace" font-size="11">`;
  site.videos.forEach((v, i) => {
    const d = videos[v.name], y = i * rowH + 4;
    s += `<text x="${pad - 8}" y="${y + 14}" text-anchor="end" fill="#93a1b0">${v.stem}</text>`;
    for (const [a, b, p] of phaseRuns(d.phase)) {
      const X = x => pad + x / xmax * (w - pad - 10);
      s += `<rect x="${X(a)}" y="${y}" width="${Math.max(1, X(b) - X(a))}" height="18" rx="3" fill="${p === RED ? "#ef4444" : "#22c55e"}" opacity=".85"/>`;
    }
  });
  for (let x = 0; x <= xmax; x += 60) s += `<text x="${90 + x / xmax * (w - 100)}" y="${site.videos.length * rowH + 18}" text-anchor="middle" fill="#93a1b0">${x / 60}:00</text>`;
  $("#phase-chart").innerHTML = s + "</svg>" + `<p class="note">Boulevard phase read from the signal lamp: ${site.signal_note}</p>`;
}
$("#findings").innerHTML = site.findings.map(f => `<li>${f}</li>`).join("");

// ---------------------------------------------------------------- results
let current = site.videos[0].name;
const player = $("#player");
function renderTabs() {
  const t = $("#video-tabs"); t.innerHTML = "";
  for (const v of site.videos) {
    const b = el("button", { class: "chip" + (v.name === current ? " on" : "") }, `${v.name} · ${videos[v.name].events.length} events`);
    b.onclick = () => { current = v.name; renderTabs(); loadVideo(); }; t.append(b);
  }
}
function timeline(container, events, duration, onSeek, phase = null) {
  container.innerHTML = "";
  const classes = [...new Set(events.map(e => e[2]))].sort();
  if (!classes.length) { container.innerHTML = `<p class="muted" style="margin:6px 0 0 -100px">No events detected.</p>`; return () => {}; }
  const lanes = el("div");
  for (const c of classes) {
    const lane = el("div", { class: "lane" });
    lane.append(el("div", { class: "lane-label" }, c));
    if (phase) for (const [a, b, p] of phaseRuns(phase)) if (p === RED) lane.append(el("div", { class: "phase", style: `left:${a / duration * 100}%;width:${(b - a) / duration * 100}%;background:#ef4444` }));
    for (const [s, e] of events.filter(x => x[2] === c)) {
      const seg = el("div", { class: "seg", style: `left:${s / duration * 100}%;width:${Math.max(0.3, (e - s) / duration * 100)}%;--c:${COLORS[c] || "#2dd4bf"}`, title: `${c} ${fmt(s)}–${fmt(e)}` });
      seg.onclick = () => onSeek(s);
      lane.append(seg);
    }
    lanes.append(lane);
  }
  const cur = el("div", { class: "cursor" });
  lanes.style.position = "relative";
  lanes.append(cur);
  container.append(lanes);
  const axis = el("div", { class: "axis" });
  const step = duration > 200 ? 60 : (duration > 60 ? 20 : 5);
  for (let x = 0; x <= duration; x += step) axis.append(el("span", { style: `left:${x / duration * 100}%` }, fmt(x).replace(".0", "")));
  container.append(axis);
  return t => { cur.style.left = `${t / duration * 100}%`; };
}
let moveCursor = () => {};
function loadVideo() {
  const d = videos[current], meta = site.videos.find(v => v.name === current);
  player.src = meta.annotated;
  moveCursor = timeline($("#timeline"), d.events, d.duration, s => { player.currentTime = Math.max(0, s - 1); player.play(); }, d.phase);
  const risk = d.risk || [];
  $("#risk-chart").innerHTML = risk.length ? lineChart([{ x: risk.map(r => r[0]), y: risk.map(r => r[1]), color: "#f59e0b" }], { h: 150, xmax: d.duration, ylabel: "Part B risk: P(accident within 5 s)", yfix: 1 })
    : `<p class="muted">Risk curve not available.</p>`;
  const tb = $("#events-table");
  tb.innerHTML = `<tr><th>#</th><th>class</th><th class="num">start</th><th class="num">end</th><th class="num">length</th></tr>` +
    d.events.map((e, i) => `<tr class="clickable" data-t="${e[0]}"><td>${i + 1}</td><td><span style="color:${COLORS[e[2]]}">■</span> ${e[2]}</td><td class="num">${fmt(e[0])}</td><td class="num">${fmt(e[1])}</td><td class="num">${(e[1] - e[0]).toFixed(1)} s</td></tr>`).join("");
  tb.querySelectorAll("tr.clickable").forEach(r => r.onclick = () => { player.currentTime = Math.max(0, +r.dataset.t - 1); player.play(); player.scrollIntoView({ behavior: "smooth", block: "center" }); });
}
player.addEventListener("timeupdate", () => moveCursor(player.currentTime));
renderTabs(); loadVideo();

function exampleCards(container, list) {
  container.innerHTML = list.map(x => `<div class="example" data-v="${x.video}" data-t="${x.t}"><img src="${x.img}" loading="lazy" alt="${x.cls}">
    <div><b style="color:${COLORS[x.cls] || "#f59e0b"}">${x.cls}</b> · ${x.video} @ ${fmt(x.t)}<br><span class="muted">${x.note || ""}</span></div></div>`).join("");
  container.querySelectorAll(".example").forEach(c => c.onclick = () => {
    current = c.dataset.v; renderTabs(); loadVideo();
    player.addEventListener("loadedmetadata", () => { player.currentTime = Math.max(0, +c.dataset.t - 1); }, { once: true });
    player.scrollIntoView({ behavior: "smooth", block: "center" });
  });
}
exampleCards($("#examples"), site.examples);
exampleCards($("#failures"), site.failures);
$("#dev-intro").innerHTML = site.dev.intro;
$("#dev-table").innerHTML = `<tr>${site.dev.header.map((h, i) => `<th class="${i ? "num" : ""}">${h}</th>`).join("")}</tr>` +
  site.dev.rows.map(r => `<tr>${r.map((c, i) => `<td class="${i ? "num" : ""}">${c}</td>`).join("")}</tr>`).join("");

// ------------------------------------------------------------- operator view
{
  const cls = [...new Set(Object.values(videos).flatMap(v => v.events.map(e => e[2])))].sort();
  $("#ops-table").innerHTML = `<tr><th>clip</th>${cls.map(c => `<th class="num">${c}</th>`).join("")}<th class="num">peds crossing A against signal</th><th class="num">vehicles / min over stop line (green)</th></tr>` +
    site.videos.map(v => { const d = videos[v.name], o = d.ops || {};
      return `<tr><td>${v.name}</td>${cls.map(c => `<td class="num">${d.events.filter(e => e[2] === c).length}</td>`).join("")}
      <td class="num">${o.ped_against_pct != null ? o.ped_against_pct.toFixed(0) + " %" : "–"}</td><td class="num">${o.veh_per_min_green != null ? o.veh_per_min_green.toFixed(1) : "–"}</td></tr>`; }).join("");
}
$("#abl-intro").textContent = site.ablations.intro;
$("#abl-table").innerHTML = `<tr>${site.ablations.header.map(h => `<th>${h}</th>`).join("")}</tr>` + site.ablations.rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("");

// ---------------------------------------------------------------- report
$("#report-body").innerHTML = site.report;
$("#team-cards").innerHTML = site.team.map(m => `<div class="member"><h3>${m.name}</h3><div class="role">${m.role}</div>
  <p>${m.did}</p>${m.projects ? `<p class="muted">Previous work: ${m.projects}</p>` : ""}
  <p>${(m.links || []).map(([t, u]) => u ? `<a href="${u}" target="_blank" rel="noopener">${t}</a>` : `<span class="todo">${t}</span>`).join(" · ")}</p></div>`).join("");
$("#link-list").innerHTML = site.links.map(([t, u, n]) => `<li><a href="${u}" target="_blank" rel="noopener">${t}</a>${n ? ` <span class="muted">— ${n}</span>` : ""}</li>`).join("");

// ------------------------------------------------------------------ demo
const sceneCfg = await (await fetch("data/scene.json", { cache: "no-cache" })).json();
const flow = await (await fetch("data/flow.json", { cache: "no-cache" })).json();
const scene = new Scene(sceneCfg, flow);
let detector = null, demoResult = null;
const status = t => { $("#demo-status").textContent = t; };
async function ensureModel() {
  if (detector) return;
  status("Loading YOLO11n (10 MB)…");
  ort.env.wasm.wasmPaths = new URL("vendor/ort/", location.href).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
  detector = new Detector("models/yolo11n.onnx", 640);
  await detector.init();
  status(`Model ready (${detector.backend}).`);
}
async function runDemo(url, name) {
  try {
    await ensureModel();
    const v = $("#demo-video");
    $("#demo-view").classList.remove("hidden");
    v.src = url;
    await new Promise((res, rej) => { v.onloadedmetadata = res; v.onerror = () => rej(new Error("This browser cannot decode the file. Try an H.264 .mp4.")); });
    if (v.duration > 150) status(`Video is ${fmt(v.duration)} long; analysing the first 2:30.`);
    $("#demo-download").disabled = true;
    const res = await processVideo(v, detector, scene, {
      fps: +$("#demo-fps").value || 5, maxSec: 150,
      onProgress: (p, sec, n) => { $("#demo-bar").style.width = `${(p * 100).toFixed(1)}%`; status(`Analysing ${name}: ${(p * 100).toFixed(0)} % · ${sec.toFixed(0)} s · ${n} objects in frame`); },
    });
    demoResult = res;
    status(`Done in ${res.seconds.toFixed(0)} s on ${res.backend}: ${res.events.length} events. Press play to see the tracks.`);
    showDemo(res);
  } catch (e) {
    console.error(e); status("Error: " + e.message);
  }
}
function showDemo(res) {
  $("#demo-results").classList.remove("hidden");
  const v = $("#demo-video"), cv = $("#demo-canvas");
  v.controls = true; v.currentTime = 0;
  const hot = new Map();
  for (const [c, a, b, tids] of res.evidence) for (const t of tids) { if (!hot.has(t)) hot.set(t, []); hot.get(t).push([a, b, c]); }
  const draw = () => {
    cv.width = v.clientWidth; cv.height = v.clientHeight;
    const g = cv.getContext("2d"); g.clearRect(0, 0, cv.width, cv.height);
    const k = Math.min(res.frames.length - 1, Math.round(v.currentTime * res.fps));
    const f = res.frames[k]; if (!f) return;
    for (const d of f.dets) {
      const hits = (hot.get(d.tid) || []).filter(([a, b]) => v.currentTime >= a - 0.3 && v.currentTime <= b + 0.3);
      g.strokeStyle = hits.length ? (COLORS[hits[0][2]] || "#ef4444") : (d.cls === 0 ? "#38bdf8" : "#22c55e");
      g.lineWidth = hits.length ? 3 : 1;
      g.strokeRect(d.x1 * cv.width, d.y1 * cv.height, (d.x2 - d.x1) * cv.width, (d.y2 - d.y1) * cv.height);
      if (hits.length) { g.fillStyle = g.strokeStyle; g.font = "12px Inter"; g.fillText(hits[0][2], d.x1 * cv.width, d.y1 * cv.height - 3); }
    }
    moveDemo(v.currentTime);
  };
  v.ontimeupdate = draw; v.onseeked = draw; window.onresize = draw;
  const phaseArr = res.sigT.map((t, i) => [t, res.phase[i]]);
  const moveDemo = timeline($("#demo-timeline"), res.events, res.duration, s => { v.currentTime = Math.max(0, s - 1); v.play(); }, phaseArr);
  $("#demo-risk").innerHTML = lineChart([{ x: res.risk.map(r => r[0]), y: res.risk.map(r => r[1]), color: "#f59e0b" }], { h: 150, xmax: res.duration, ylabel: "risk: P(accident within 5 s)", yfix: 1 });
  $("#demo-events").innerHTML = `<tr><th>class</th><th class="num">start</th><th class="num">end</th></tr>` +
    (res.events.length ? res.events.map(e => `<tr class="clickable" data-t="${e[0]}"><td><span style="color:${COLORS[e[2]]}">■</span> ${e[2]}</td><td class="num">${fmt(e[0])}</td><td class="num">${fmt(e[1])}</td></tr>`).join("") : `<tr><td colspan="3" class="muted">No events in this clip.</td></tr>`);
  $("#demo-events").querySelectorAll("tr.clickable").forEach(r => r.onclick = () => { v.currentTime = Math.max(0, +r.dataset.t - 1); v.play(); });
  $("#demo-download").disabled = false;
  draw();
}
$("#file").onchange = e => {
  const f = e.target.files[0]; if (!f) return;
  if (f.size > 500e6) { status("File is larger than 500 MB."); return; }
  runDemo(URL.createObjectURL(f), f.name);
};
$("#sample-btn").onclick = () => runDemo(site.demo_sample, "sample clip");
$("#demo-download").onclick = () => {
  if (!demoResult) return;
  const blob = new Blob([JSON.stringify({ events: demoResult.events, risk: demoResult.risk }, null, 1)], { type: "application/json" });
  const a = el("a", { href: URL.createObjectURL(blob), download: "events.json" }); a.click();
};
