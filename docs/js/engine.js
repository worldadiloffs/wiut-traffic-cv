// In-browser version of the pipeline (live demo).
// Same scene layout and rules as src/traffic/*.py, lighter detector
// (YOLO11n, 640 px, ONNX Runtime Web) and a greedy IoU tracker instead of
// ByteTrack. Everything runs on the visitor's machine; nothing is uploaded.

const KEEP = { 0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck" };
const VEHICLES = new Set([2, 3, 5, 7]);
const CAR_LIKE = new Set([2, 5, 7]);
const ANY_VEHICLE = new Set([1, 2, 3, 5, 7]);
const TWO_WHEEL = new Set([1, 3]);
export const RED = 1, GREEN = -1, UNKNOWN = 0;

export const P = {
  still: 0.12, moving: 0.35,
  jay_min_dur: 1.2, jay_cross_buffer: 0.04, jay_min_height: 0.045, jay_kerb_margin: 0.008, jay_min_path: 0.03, jay_min_net: 0.02,
  fty_ped_dist: 0.16, fty_min_samples: 2, fty_u_margin: 0.12, fty_max_cross: 6.0, fty_ped_moving: 0.25,
  red_grace: 0.8, red_tail: 3.0, red_platoon: 3,
  stop_min_still: 2.0, stop_margin: 0.012,
  stopped_min: 10.0, queue_gap: 0.035, junction_wait: 45.0,
  cong_min_veh: 6, cong_speed: 0.10, cong_min_dur: 12.0,
  ww_coh: 0.85, ww_cnt: 25, ww_cos: -0.6, ww_min_dur: 1.5, ww_min_dist: 0.05,
};
const MERGE_GAP = { jaywalking: 2.0, failure_to_yield: 0.5, congestion: 5.0 };

// ------------------------------------------------------------ geometry ----
function inPoly(x, y, poly) {
  let c = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi) c = !c;
  }
  return c;
}
function edgeDist(x, y, poly) {
  let best = Infinity;
  for (let i = 0; i < poly.length; i++) {
    const [ax, ay] = poly[i], [bx, by] = poly[(i + 1) % poly.length];
    const abx = bx - ax, aby = by - ay;
    let t = ((x - ax) * abx + (y - ay) * aby) / Math.max(abx * abx + aby * aby, 1e-12);
    t = Math.max(0, Math.min(1, t));
    const d = Math.hypot(x - (ax + t * abx), y - (ay + t * aby));
    if (d < best) best = d;
  }
  return best;
}
function distToPoly(x, y, poly) { return inPoly(x, y, poly) ? 0 : edgeDist(x, y, poly); }

export class Scene {
  constructor(cfg, flow) {
    const [w, h] = cfg.reference_size;
    const n = pts => pts.map(([x, y]) => [x / w, y / h]);
    const obj = o => Object.fromEntries(Object.entries(o).filter(([k]) => !k.startsWith("_")).map(([k, v]) => [k, n(v)]));
    this.crosswalks = obj(cfg.crosswalks);
    this.stopLine = n(cfg.stop_line_A);
    this.zones = obj(cfg.zones);
    this.road = Object.values(obj(cfg.road || {}));
    this.islands = Object.values(obj(cfg.islands || {}));
    this.redLamp = n([cfg.signal.red_lamp])[0];
    this.greenLamp = n([cfg.signal.green_lamp])[0];
    this.flow = flow;   // {w, h, vec:[[x,y]...], coh:[], cnt:[]}
  }
  inZone(x, y, name) { return inPoly(x, y, this.zones[name]); }
  onRoad(x, y, margin = 0) {
    let on = false;
    for (const poly of this.road) if (inPoly(x, y, poly) && (margin <= 0 || edgeDist(x, y, poly) > margin)) { on = true; break; }
    if (!on) return false;
    for (const poly of this.islands) if (inPoly(x, y, poly)) return false;
    return true;
  }
  crosswalkU(x, y, name) {
    const p = this.crosswalks[name];
    const a = [(p[0][0] + p[1][0]) / 2, (p[0][1] + p[1][1]) / 2], b = [(p[2][0] + p[3][0]) / 2, (p[2][1] + p[3][1]) / 2];
    const d = [b[0] - a[0], b[1] - a[1]];
    return ((x - a[0]) * d[0] + (y - a[1]) * d[1]) / (d[0] * d[0] + d[1] * d[1]);
  }
  sideOfStopLine(x, y) {
    const [[x1, y1], [x2, y2]] = this.stopLine;
    return -((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1));
  }
  flowAt(x, y) {
    const f = this.flow;
    if (!f) return null;
    const xi = Math.min(f.w - 1, Math.max(0, Math.floor(x * f.w)));
    const yi = Math.min(f.h - 1, Math.max(0, Math.floor(y * f.h)));
    const k = yi * f.w + xi;
    return { vec: f.vec[k], coh: f.coh[k], cnt: f.cnt[k] };
  }
}

// ------------------------------------------------------------ detector ----
export class Detector {
  constructor(url, size = 640) { this.url = url; this.size = size; }
  async init() {
    const opts = { executionProviders: [], graphOptimizationLevel: "all" };
    let gpuOk = false;
    try { gpuOk = !!(navigator.gpu && await navigator.gpu.requestAdapter()); } catch (e) { gpuOk = false; }
    if (gpuOk) opts.executionProviders.push("webgpu");
    opts.executionProviders.push("wasm");
    try {
      this.session = await ort.InferenceSession.create(this.url, opts);
      this.backend = opts.executionProviders[0];
    } catch (e) {
      this.session = await ort.InferenceSession.create(this.url, { executionProviders: ["wasm"] });
      this.backend = "wasm";
    }
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.canvas.height = this.size;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
  }
  async detect(source, sw, sh, conf = 0.25) {
    const S = this.size, r = Math.min(S / sw, S / sh);
    const nw = Math.round(sw * r), nh = Math.round(sh * r), px = (S - nw) / 2, py = (S - nh) / 2;
    this.ctx.fillStyle = "rgb(114,114,114)";
    this.ctx.fillRect(0, 0, S, S);
    this.ctx.drawImage(source, 0, 0, sw, sh, px, py, nw, nh);
    const data = this.ctx.getImageData(0, 0, S, S).data;
    const area = S * S, input = new Float32Array(3 * area);
    for (let i = 0; i < area; i++) {
      input[i] = data[4 * i] / 255; input[i + area] = data[4 * i + 1] / 255; input[i + 2 * area] = data[4 * i + 2] / 255;
    }
    const feeds = {}; feeds[this.session.inputNames[0]] = new ort.Tensor("float32", input, [1, 3, S, S]);
    const out = (await this.session.run(feeds))[this.session.outputNames[0]];
    const [, C, N] = out.dims, o = out.data;  // (1, 84, 8400)
    const boxes = [];
    for (let i = 0; i < N; i++) {
      let best = -1, bs = conf;
      for (const c of [0, 1, 2, 3, 5, 7]) { const s = o[(4 + c) * N + i]; if (s > bs) { bs = s; best = c; } }
      if (best < 0) continue;
      const cx = o[i], cy = o[N + i], w = o[2 * N + i], h = o[3 * N + i];
      boxes.push({ cls: best, conf: bs,
        x1: (cx - w / 2 - px) / r / sw, y1: (cy - h / 2 - py) / r / sh,
        x2: (cx + w / 2 - px) / r / sw, y2: (cy + h / 2 - py) / r / sh });
    }
    return nms(boxes, 0.5);
  }
}
function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy, u = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return u > 0 ? inter / u : 0;
}
function nms(boxes, thr) {
  boxes.sort((a, b) => b.conf - a.conf);
  const keep = [];
  for (const b of boxes) {
    if (keep.every(k => (k.cls !== b.cls && !(VEHICLES.has(k.cls) && VEHICLES.has(b.cls))) || iou(k, b) < thr)) keep.push(b);
  }
  return keep;
}

// ------------------------------------------------------------- tracker ----
export class IoUTracker {
  constructor(maxAge = 8) { this.maxAge = maxAge; this.tracks = []; this.next = 1; }
  update(dets) {
    const pairs = [];
    this.tracks.forEach((t, ti) => dets.forEach((d, di) => {
      const sameKind = (t.cls === 0) === (d.cls === 0);
      if (!sameKind) return;
      // predict with last velocity
      const pb = { x1: t.box.x1 + t.v[0], y1: t.box.y1 + t.v[1], x2: t.box.x2 + t.v[0], y2: t.box.y2 + t.v[1] };
      const s = iou(pb, d);
      if (s > 0.2) pairs.push([s, ti, di]);
    }));
    pairs.sort((a, b) => b[0] - a[0]);
    const usedT = new Set(), usedD = new Set();
    for (const [, ti, di] of pairs) {
      if (usedT.has(ti) || usedD.has(di)) continue;
      usedT.add(ti); usedD.add(di);
      const t = this.tracks[ti], d = dets[di];
      t.v = [((d.x1 + d.x2) - (t.box.x1 + t.box.x2)) / 2, (d.y2 - t.box.y2)];
      t.box = d; t.age = 0; t.hits++; d.tid = t.id;
      if (t.cls !== d.cls && t.hits < 3) t.cls = d.cls;
    }
    dets.forEach((d, di) => {
      if (usedD.has(di) || d.conf < 0.35) return;
      const t = { id: this.next++, box: d, v: [0, 0], age: 0, hits: 1, cls: d.cls };
      d.tid = t.id; this.tracks.push(t);
    });
    this.tracks.forEach((t, ti) => { if (!usedT.has(ti)) t.age++; });
    this.tracks = this.tracks.filter(t => t.age <= this.maxAge);
    return dets.filter(d => d.tid);
  }
}

// -------------------------------------------------------------- signal ----
export function lampScores(ctx, W, H, scene) {
  const r = Math.max(2, Math.round(6 * W / 1280));
  const read = (c, f) => {
    const x = Math.round(c[0] * W) - r, y = Math.round(c[1] * H) - r;
    const d = ctx.getImageData(Math.max(0, x), Math.max(0, y), 2 * r + 1, 2 * r + 1).data;
    let best = -999;
    for (let i = 0; i < d.length; i += 4) best = Math.max(best, f(d[i], d[i + 1], d[i + 2]));
    return best;
  };
  return [read(scene.redLamp, (R, G, B) => R - Math.max(G, B)), read(scene.greenLamp, (R, G, B) => G - R)];
}
function percentile(arr, p) {
  const a = [...arr].sort((x, y) => x - y);
  if (!a.length) return 0;
  const k = (a.length - 1) * p / 100, lo = Math.floor(k), hi = Math.ceil(k);
  return a[lo] + (a[hi] - a[lo]) * (k - lo);
}
export function phaseSeries(scores, fpsSample) {
  const hr = percentile(scores.map(s => s[0]), 95), hg = percentile(scores.map(s => s[1]), 95);
  if (hr < 15 || hg < 15) return scores.map(() => UNKNOWN);
  const raw = scores.map(([a, b]) => {
    const r = a / hr, g = b / hg;
    if (r > 0.45 && r > g + 0.2) return RED;
    if (g > 0.45 && g > r + 0.2) return GREEN;
    return UNKNOWN;
  });
  const hold = Math.max(2, Math.round(0.5 * fpsSample));
  let state = UNKNOWN, cand = UNKNOWN, count = 0;
  const ph = raw.map(r => {
    if (r === UNKNOWN) return state;
    if (r === state) { cand = r; count = 0; return state; }
    if (r === cand) count++; else { cand = r; count = 1; }
    if (count >= hold) { state = r; count = 0; }
    return state;
  });
  return ph.slice(hold).concat(new Array(Math.min(hold, ph.length)).fill(ph[ph.length - 1]));
}

// -------------------------------------------------------------- tracks ----
function smooth(t, xs, win) {
  const out = new Array(xs.length);
  let lo = 0, hi = 0, sum = 0;
  for (let i = 0; i < t.length; i++) {
    while (hi < t.length && t[hi] <= t[i] + win / 2) sum += xs[hi++];
    while (t[lo] < t[i] - win / 2) sum -= xs[lo++];
    out[i] = sum / (hi - lo);
  }
  return out;
}
export function buildTracks(frames) {
  const map = new Map();
  for (const f of frames) for (const d of f.dets) {
    if (!map.has(d.tid)) map.set(d.tid, { tid: d.tid, t: [], box: [], clsW: {} });
    const tr = map.get(d.tid);
    tr.t.push(f.t); tr.box.push(d); tr.clsW[d.cls] = (tr.clsW[d.cls] || 0) + d.conf;
  }
  const tracks = [];
  for (const tr of map.values()) {
    if (tr.t.length < 3) continue;
    tr.cls = +Object.entries(tr.clsW).sort((a, b) => b[1] - a[1])[0][0];
    const gx = tr.box.map(b => (b.x1 + b.x2) / 2), gy = tr.box.map(b => b.y2);
    const size = tr.box.map(b => Math.max(b.y2 - b.y1, 1e-3));
    tr.gx = smooth(tr.t, gx, 0.6); tr.gy = smooth(tr.t, gy, 0.6);
    const ss = smooth(tr.t, size, 1.0);
    tr.vx = []; tr.vy = []; tr.speed = [];
    for (let i = 0; i < tr.t.length; i++) {
      let lo = i, hi = i;
      while (lo > 0 && tr.t[lo - 1] >= tr.t[i] - 0.5) lo--;
      while (hi < tr.t.length - 1 && tr.t[hi + 1] <= tr.t[i] + 0.5) hi++;
      const dt = Math.max(tr.t[hi] - tr.t[lo], 1e-3);
      const vx = hi > lo ? (tr.gx[hi] - tr.gx[lo]) / dt : 0, vy = hi > lo ? (tr.gy[hi] - tr.gy[lo]) / dt : 0;
      tr.vx.push(vx); tr.vy.push(vy); tr.speed.push(Math.hypot(vx, vy) / ss[i]);
    }
    tr.at = t => { let k = tr.t.findIndex(x => x >= t); return k < 0 ? tr.t.length - 1 : k; };
    tracks.push(tr);
  }
  return tracks;
}
function riderIds(tracks, kinds = TWO_WHEEL) {
  const wheels = new Map();
  for (const tr of tracks) if (kinds.has(tr.cls)) tr.t.forEach((t, i) => {
    const k = t.toFixed(3); if (!wheels.has(k)) wheels.set(k, []); wheels.get(k).push(tr.box[i]);
  });
  const riders = new Set();
  for (const p of tracks) {
    if (p.cls !== 0) continue;
    let hits = 0;
    p.t.forEach((t, i) => {
      const w = wheels.get(t.toFixed(3)); if (!w) return;
      const pb = p.box[i], pa = (pb.x2 - pb.x1) * (pb.y2 - pb.y1);
      if (w.some(b => Math.max(0, Math.min(pb.x2, b.x2) - Math.max(pb.x1, b.x1)) * Math.max(0, Math.min(pb.y2, b.y2) - Math.max(pb.y1, b.y1)) > 0.35 * pa)) hits++;
    });
    if (hits > 0.4 * p.t.length) riders.add(p.tid);
  }
  return riders;
}

// --------------------------------------------------------------- rules ----
function runsOf(mask) {
  const out = []; let s = -1;
  mask.forEach((m, i) => { if (m && s < 0) s = i; else if (!m && s >= 0) { out.push([s, i - 1]); s = -1; } });
  if (s >= 0) out.push([s, mask.length - 1]);
  return out;
}
function timeRuns(t, mask, maxGap, minLen) {
  const merged = [];
  for (const [i, j] of runsOf(mask)) {
    const s = t[i], e = t[j];
    if (merged.length && s - merged[merged.length - 1][1] <= maxGap) merged[merged.length - 1][1] = e;
    else merged.push([s, e]);
  }
  return merged.filter(([s, e]) => e - s >= minLen);
}
function union(iv, gap) {
  iv = iv.filter(([s, e]) => e > s).sort((a, b) => a[0] - b[0]);
  const out = [];
  for (const [s, e] of iv) {
    if (out.length && s <= out[out.length - 1][1] + gap) out[out.length - 1][1] = Math.max(out[out.length - 1][1], e);
    else out.push([s, e]);
  }
  return out;
}
function phaseAt(sigT, phase, t) {
  let k = -1;
  for (let i = 0; i < sigT.length && sigT[i] <= t; i++) k = i;
  return k < 0 ? (phase[0] ?? 0) : phase[k];
}
function phaseIntervals(sigT, phase, v) {
  const out = []; let s = null;
  sigT.forEach((t, i) => { if (phase[i] === v && s === null) s = t; else if (phase[i] !== v && s !== null) { out.push([s, t]); s = null; } });
  if (s !== null) out.push([s, sigT[sigT.length - 1]]);
  return out;
}

export function detectEvents(tracks, scene, sigT, phase, duration) {
  const riders = riderIds(tracks), motoRiders = riderIds(tracks, new Set([3]));
  const peds = tracks.filter(t => t.cls === 0 && !riders.has(t.tid) && t.t.length >= 4);
  const crossers = tracks.filter(t => t.cls === 0 && !motoRiders.has(t.tid) && t.t.length >= 4);
  const veh = tracks.filter(t => ANY_VEHICLE.has(t.cls) && t.t.length >= 4);
  const hasSignal = phase.some(p => p !== 0);
  const evidence = [];
  const ev = (name, s, e, tids) => { evidence.push([name, s, e, tids]); return [s, e]; };
  const pedAt = new Map();
  for (const p of crossers) p.t.forEach((t, i) => { const k = t.toFixed(3); if (!pedAt.has(k)) pedAt.set(k, []); pedAt.get(k).push([p.gx[i], p.gy[i], p.speed[i]]); });
  const vehAt = new Map();
  for (const v of veh) if (CAR_LIKE.has(v.cls)) v.t.forEach((t, i) => { const k = t.toFixed(3); if (!vehAt.has(k)) vehAt.set(k, []); vehAt.get(k).push([v.tid, v.gx[i], v.gy[i], v.speed[i], v.box[i]]); });
  const out = {};

  // jaywalking
  out.jaywalking = [];
  for (const p of peds) {
    const flag = p.t.map((_, i) => {
      const x = p.gx[i], y = p.gy[i];
      if (!scene.onRoad(x, y, P.jay_kerb_margin) || scene.inZone(x, y, "bus_stop") || scene.inZone(x, y, "far_corner")) return false;
      if (p.box[i].y2 - p.box[i].y1 < P.jay_min_height) return false;
      return Object.values(scene.crosswalks).every(poly => distToPoly(x, y, poly) >= P.jay_cross_buffer);
    });
    for (const [a, b] of timeRuns(p.t, flag, 0.8, P.jay_min_dur)) {
      const i = p.at(a), j = p.at(b);
      let path = 0; for (let k = i + 1; k <= j; k++) path += Math.hypot(p.gx[k] - p.gx[k - 1], p.gy[k] - p.gy[k - 1]);
      if (path >= P.jay_min_path && Math.hypot(p.gx[j] - p.gx[i], p.gy[j] - p.gy[i]) >= P.jay_min_net) out.jaywalking.push(ev("jaywalking", a, b, [p.tid]));
    }
  }
  // failure to yield
  out.failure_to_yield = [];
  for (const v of veh) {
    if (!VEHICLES.has(v.cls)) continue;
    for (const [name, poly] of Object.entries(scene.crosswalks)) {
      const inside = v.t.map((_, i) => inPoly(v.gx[i], v.gy[i], poly) && v.box[i].y2 < 0.96);
      for (const [i, j] of runsOf(inside)) {
        const sp = v.speed.slice(i, j + 1).sort((a, b) => a - b);
        if (v.t[j] - v.t[i] > P.fty_max_cross || sp[Math.floor(sp.length / 2)] < P.moving) continue;
        let hits = 0;
        for (let k = i; k <= j; k++) {
          const ps = pedAt.get(v.t[k].toFixed(3)) || [];
          if (ps.some(([x, y, s]) => {
            if (!inPoly(x, y, poly)) return false;
            const u = scene.crosswalkU(x, y, name);
            if (u <= P.fty_u_margin || u >= 1 - P.fty_u_margin) return false;
            if (!(s > P.fty_ped_moving || (u > 0.2 && u < 0.8))) return false;   // walking, or standing mid-crossing
            return Math.hypot(x - v.gx[k], (y - v.gy[k]) * 9 / 16) < P.fty_ped_dist;
          })) hits++;
        }
        if (hits >= P.fty_min_samples) out.failure_to_yield.push(ev("failure_to_yield", v.t[i], v.t[j] + 0.15, [v.tid]));
      }
    }
  }
  // red light + stop line
  out.red_light = []; out.stop_line = [];
  if (hasSignal) {
    const reds = phaseIntervals(sigT, phase, RED), greens = phaseIntervals(sigT, phase, GREEN).map(g => g[0]);
    const cands = [], crosses = [];
    for (const v of veh) {
      if (!VEHICLES.has(v.cls)) continue;
      const side = v.t.map((_, i) => scene.sideOfStopLine(v.gx[i], v.gy[i]));
      let k = -1;
      for (let i = 0; i < side.length - 1; i++) {
        const app = scene.inZone(v.gx[i], v.gy[i], "near_approach") || scene.inZone(v.gx[i], v.gy[i], "past_stop_line");
        if (side[i] > 0 && side[i + 1] <= 0 && app) { k = i + 1; break; }
      }
      if (k >= 0) {
        const tc = v.t[k]; crosses.push(tc);
        if (reds.some(([s, e]) => s + P.red_grace <= tc && tc <= e - P.red_tail)) {
          let jIn = -1;
          for (let i = k; i < v.t.length; i++) if (scene.inZone(v.gx[i], v.gy[i], "intersection")) { jIn = i; break; }
          if (jIn >= 0) {
            const minSp = Math.min(...v.speed.slice(k, jIn + 1));
            if (!(minSp < P.still && v.t[jIn] - tc > 3)) {
              let jOut = v.t.length - 1;
              for (let i = jIn; i < v.t.length; i++) if (!scene.inZone(v.gx[i], v.gy[i], "intersection")) { jOut = i; break; }
              cands.push([tc, Math.min(v.t[jOut], tc + 12), v.tid]);
            }
          }
        }
      }
      // stop line
      const flag = v.t.map((t, i) => scene.inZone(v.gx[i], v.gy[i], "past_stop_line") && side[i] < -P.stop_margin && v.speed[i] < P.still && phaseAt(sigT, phase, t) === RED);
      for (const [s] of timeRuns(v.t, flag, 1.0, P.stop_min_still)) {
        const nxt = greens.find(g => g > s);
        out.stop_line.push(ev("stop_line", s, nxt ?? duration, [v.tid]));
      }
    }
    for (const [tc, te, tid] of cands) if (crosses.filter(c => Math.abs(c - tc) <= 2).length < P.red_platoon) out.red_light.push(ev("red_light", tc, te, [tid]));
  }
  // stopped vehicle
  out.stopped_vehicle = [];
  for (const v of veh) {
    if (!CAR_LIKE.has(v.cls) || v.t[v.t.length - 1] - v.t[0] < P.stopped_min) continue;
    const still = v.t.map((_, i) => v.speed[i] < P.still && scene.onRoad(v.gx[i], v.gy[i]));
    for (const [s, e] of timeRuns(v.t, still, 1.5, P.stopped_min)) {
      const i = v.at(s), j = v.at(e), m = Math.floor((i + j) / 2), x = v.gx[m], y = v.gy[m], b = v.box[m];
      if (v.cls === 5 && scene.inZone(x, y, "bus_stop")) continue;
      if (scene.inZone(x, y, "near_approach") || scene.inZone(x, y, "past_stop_line")) continue;
      if (scene.inZone(x, y, "intersection") && e - s < P.junction_wait) continue;
      if (Object.values(scene.crosswalks).some(poly => distToPoly(x, y, poly) < 0.03)) continue;
      if (b.x1 < 0.005 || b.x2 > 0.995 || b.y2 > 0.995) continue;
      let q = 0, n = 0;
      for (let k = i; k <= j; k += 3) {
        n++;
        const others = vehAt.get(v.t[k].toFixed(3)) || [];
        if (others.some(([tid, , , sp, ob]) => tid !== v.tid && sp < P.moving && boxGap(ob, v.box[k]) < P.queue_gap)) q++;
      }
      if (q > 0.5 * n) continue;
      out.stopped_vehicle.push(ev("stopped_vehicle", s, e, [v.tid]));
    }
  }
  // congestion
  out.congestion = [];
  const times = [...vehAt.keys()].map(Number).sort((a, b) => a - b);
  for (const zone of ["near_approach", "far_carriageway"]) {
    const flag = times.map(t => {
      const items = vehAt.get(t.toFixed(3)).filter(it => scene.inZone(it[1], it[2], zone) && !scene.inZone(it[1], it[2], "bus_stop"));
      if (items.length < P.cong_min_veh) return false;
      const sp = items.map(it => it[3]).sort((a, b) => a - b);
      const ok = sp[Math.floor(sp.length / 2)] < P.cong_speed && sp.filter(s => s < P.moving).length / sp.length > 0.8;
      return ok && (!hasSignal || phaseAt(sigT, phase, t) === GREEN);
    });
    out.congestion.push(...timeRuns(times, flag, 3.0, P.cong_min_dur));
  }
  // wrong way
  out.wrong_way = [];
  if (scene.flow) for (const v of veh) {
    if (!VEHICLES.has(v.cls)) continue;
    const against = v.t.map((_, i) => {
      if (v.speed[i] <= P.moving) return false;
      const f = scene.flowAt(v.gx[i], v.gy[i]);
      if (!f || f.coh <= P.ww_coh || f.cnt <= P.ww_cnt || scene.inZone(v.gx[i], v.gy[i], "intersection")) return false;
      if (Object.values(scene.crosswalks).some(poly => distToPoly(v.gx[i], v.gy[i], poly) < 0.02)) return false;
      const n = Math.hypot(v.vx[i], v.vy[i]) || 1e-9;
      return (v.vx[i] * f.vec[0] + v.vy[i] * f.vec[1]) / n < P.ww_cos;
    });
    for (const [s, e] of timeRuns(v.t, against, 1.0, P.ww_min_dur)) {
      const i = v.at(s), j = v.at(e);
      if (Math.hypot(v.gx[j] - v.gx[i], v.gy[j] - v.gy[i]) >= P.ww_min_dist) out.wrong_way.push(ev("wrong_way", s, e, [v.tid]));
    }
  }
  const events = [];
  for (const [name, iv] of Object.entries(out)) for (const [s, e] of union(iv, MERGE_GAP[name] ?? 0.3)) {
    const a = Math.max(0, s), b = Math.min(duration, e);
    if (b - a >= 0.2) events.push([+a.toFixed(2), +b.toFixed(2), name]);
  }
  events.sort((a, b) => a[0] - b[0]);
  return { events, evidence };
}
function boxGap(a, b) {
  const dx = Math.max(0, Math.max(a.x1, b.x1) - Math.min(a.x2, b.x2));
  const dy = Math.max(0, Math.max(a.y1, b.y1) - Math.min(a.y2, b.y2));
  return Math.hypot(dx, dy);
}

// ---------------------------------------------------------------- risk ----
// Causal time-to-collision score, same formula as src/traffic/risk.py.
export class Risk {
  constructor() { this.hist = new Map(); this.pairs = new Map(); this.ema = 0; }
  step(t, dets) {
    const items = [];
    for (const d of dets) {
      if (!(VEHICLES.has(d.cls) || d.cls === 0)) continue;
      if (!this.hist.has(d.tid)) this.hist.set(d.tid, []);
      const h = this.hist.get(d.tid);
      h.push([t, (d.x1 + d.x2) / 2, d.y2, d.x2 - d.x1, d.y2 - d.y1]);
      if (h.length > 8) h.shift();
      if (h.length >= 3) items.push([d.tid, d.cls, h]);
    }
    let raw = 0;
    const nextPairs = new Map();
    for (let a = 0; a < items.length; a++) for (let b = a + 1; b < items.length; b++) {
      const [ta, ca, ha] = items[a], [tb, cb, hb] = items[b];
      if (!VEHICLES.has(ca) && !VEHICLES.has(cb)) continue;
      const va = vel(ha), vb = vel(hb);
      const ra = 0.5 * Math.max(ha[ha.length - 1][3], 0.3 * ha[ha.length - 1][4]);
      const rb = 0.5 * Math.max(hb[hb.length - 1][3], 0.3 * hb[hb.length - 1][4]);
      const pa = ha[ha.length - 1], pb = hb[hb.length - 1];
      const dp = [pa[1] - pb[1], pa[2] - pb[2]], dv = [va[0] - vb[0], va[1] - vb[1]];
      const dv2 = dv[0] ** 2 + dv[1] ** 2; if (dv2 < 1e-9) continue;
      const ts = -(dp[0] * dv[0] + dp[1] * dv[1]) / dv2;
      const dm = Math.hypot(dp[0] + dv[0] * ts, dp[1] + dv[1] * ts), rc = ra + rb;
      const spa = Math.hypot(...va) / (2 * ra), spb = Math.hypot(...vb) / (2 * rb);
      if (!(ts > 0 && ts < 3.5 && dm < 0.5 * rc && (spa > 0.3 || spb > 0.3))) continue;
      if (Math.hypot(...dp) <= 0.6 * rc) continue;
      const key = ta + ":" + tb, c = (this.pairs.get(key) || 0) + 1; nextPairs.set(key, c);
      const closeness = 1 - dm / (0.5 * rc), speedF = Math.min(1, Math.sqrt(dv2) / (rc * 3));
      raw = Math.max(raw, closeness ** 2 * Math.exp(-ts / 1.5) * speedF * Math.min(1, (c - 1) / 2));
    }
    this.pairs = nextPairs;
    const al = raw > this.ema ? 0.6 : 0.15;
    this.ema = (1 - al) * this.ema + al * raw;
    return Math.min(1, Math.max(0, this.ema));
  }
}
function vel(h) {
  const a = h[0], b = h[h.length - 1], dt = Math.max(b[0] - a[0], 1e-3);
  return [(b[1] - a[1]) / dt, (b[2] - a[2]) / dt];
}

// --------------------------------------------------------------- driver ----
export async function processVideo(video, detector, scene, { fps = 5, maxSec = 150, onProgress } = {}) {
  const W = video.videoWidth, H = video.videoHeight;
  const duration = Math.min(video.duration, maxSec);
  const full = document.createElement("canvas");
  const fw = 1280, fh = Math.round(1280 * H / W);
  full.width = fw; full.height = fh;
  const fctx = full.getContext("2d", { willReadFrequently: true });
  const tracker = new IoUTracker(8), risk = new Risk();
  const frames = [], lamp = [], riskCurve = [];
  const t0 = performance.now();
  // Seek and wait until the new frame is actually presented (Chrome may still
  // hold the old frame when "seeked" fires); fall back to a short timeout.
  const seek = t => new Promise(res => {
    let finished = false;
    const finish = () => { if (!finished) { finished = true; res(); } };
    video.addEventListener("seeked", () => {
      if (video.requestVideoFrameCallback) { video.requestVideoFrameCallback(() => finish()); setTimeout(finish, 250); }
      else setTimeout(finish, 30);
    }, { once: true });
    video.currentTime = t;
  });
  const n = Math.floor(duration * fps);
  for (let i = 0; i < n; i++) {
    const t = i / fps;
    await seek(t);
    fctx.drawImage(video, 0, 0, fw, fh);
    const dets = tracker.update(await detector.detect(full, fw, fh));
    frames.push({ t, dets });
    lamp.push(lampScores(fctx, fw, fh, scene));
    riskCurve.push([+t.toFixed(2), +risk.step(t, dets).toFixed(3)]);
    if (onProgress) onProgress((i + 1) / n, (performance.now() - t0) / 1000, dets.length);
  }
  const sigT = frames.map(f => f.t);
  const phase = phaseSeries(lamp, fps);
  console.log("lamp", JSON.stringify(lamp.slice(0, 60).map(l => l.map(Math.round))));
  const tracks = buildTracks(frames);
  const { events, evidence } = detectEvents(tracks, scene, sigT, phase, duration);
  return { duration, fps, frames, events, evidence, risk: riskCurve, phase, sigT, width: W, height: H,
           seconds: (performance.now() - t0) / 1000, backend: detector.backend };
}
