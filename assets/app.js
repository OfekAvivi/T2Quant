/* T²Quant project page: replay, charts, in-browser allocator, results table. No dependencies (KaTeX optional). */
(function () {
  "use strict";
  const SITE = window.T2Q_SITE || {};
  const P = window.T2Q_PROFILES;
  const NS = "http://www.w3.org/2000/svg";
  const K = 16;
  const reduceMotion = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

  const MODEL_NAME = { llada8b_instruct: "LLaDA-8B-Instruct", llada15: "LLaDA-1.5-8B", dream: "Dream-7B-Instruct" };
  const TASK_NAME = { gsm8k: "GSM8K", humaneval: "HumanEval", arcgen: "ARC-Challenge", mmlupro: "MMLU-Pro", drop: "DROP", bbh: "BBH" };

  function svgEl(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function newSvg(box) {
    box.innerHTML = "";
    const w = Math.max(240, box.clientWidth), h = box.clientHeight || 260;
    const s = svgEl("svg", { width: w, height: h, viewBox: `0 0 ${w} ${h}` }, box);
    return { s, w, h };
  }
  const lin = (d0, d1, r0, r1) => v => r0 + (v - d0) / (d1 - d0) * (r1 - r0);
  const fmt = (v, d = 2) => Number(v).toFixed(d);
  const fmtG = v => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
  function onResize(box, draw) {
    let w = box.clientWidth;
    if ("ResizeObserver" in window) new ResizeObserver(() => { if (box.clientWidth !== w) { w = box.clientWidth; draw(); } }).observe(box);
  }

  /* ---------- tooltip ---------- */
  const tip = $("#tip");
  function showTip(html, x, y) {
    tip.innerHTML = html; tip.hidden = false;
    const r = tip.getBoundingClientRect();
    let left = x + 14, top = y + 14;
    if (left + r.width > innerWidth - 8) left = x - r.width - 14;
    if (top + r.height > innerHeight - 8) top = y - r.height - 14;
    tip.style.left = Math.max(8, left) + "px"; tip.style.top = Math.max(8, top) + "px";
  }
  const hideTip = () => { tip.hidden = true; };
  const row = (label, value, color) =>
    `<div class="r"><span>${color ? `<span class="sw" style="background:${color}"></span>` : ""}${label}</span><b>${value}</b></div>`;
  function bindTip(node, html) {
    node.addEventListener("pointermove", e => showTip(typeof html === "function" ? html() : html, e.clientX, e.clientY));
    node.addEventListener("pointerleave", hideTip);
  }
  addEventListener("scroll", hideTip, { passive: true });

  /* ---------- schedule strips ---------- */
  function renderStrip(el, bits, now = -1) {
    el.innerHTML = "";
    bits.forEach((b, k) => {
      const d = document.createElement("div");
      d.dataset.b = b; d.textContent = b;
      if (k === now) d.className = "now";
      d.title = `window ${k + 1}: ${b}-bit activations`;
      el.appendChild(d);
    });
  }

  /* ---------- exact allocator (same recurrence as the paper's Eq. 7; port of the research repo's solver) ----------
     state = running balance sum(b_k - base); keeps the best objective per balance; the earliest-found schedule wins ties. */
  function allocate(gain, levels, total, base = 4) {
    let S = new Map([[0, { o: 0, b: [] }]]);
    for (let k = 0; k < K; k++) {
      const N = new Map();
      for (const [bal, st] of S) {
        for (const L of levels) {
          const nb = bal + (L - base), no = st.o + gain[L][k], cur = N.get(nb);
          if (!cur || no > cur.o + 1e-12) N.set(nb, { o: no, b: st.b.concat(L) });
        }
      }
      S = N;
    }
    const target = total - K * base, r = S.get(target);
    if (!r) return null;
    if (target === 0 && r.o <= 1e-12) return { bits: Array(K).fill(base), o: 0 };
    return { bits: r.b, o: r.o };
  }

  /* ---------- 1. replay ---------- */
  (function replay() {
    const grid = $("#replay-grid"), strip = $("#replay-strip"), meta = $("#replay-meta");
    const N = 512, T = 128, PER = 4;
    const cells = [];
    for (let i = 0; i < N; i++) { const c = document.createElement("div"); c.className = "cell"; grid.appendChild(c); cells.push(c); }
    // Schematic unmasking order: roughly left to right with noise, like low-confidence remasking on a single block.
    let s = 20261020;
    const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
    const order = Array.from({ length: N }, (_, i) => ({ i, v: i / N + (rnd() - 0.5) * 0.5 })).sort((a, b) => a.v - b.v).map(o => o.i);
    const sched = {
      gsm8k: P.models.llada8b_instruct.gsm8k.schedule,
      humaneval: P.models.llada8b_instruct.humaneval.schedule,
      mmlupro: P.models.llada8b_instruct.mmlupro.schedule,
      uniform: Array(K).fill(4)
    };
    let task = "gsm8k", raf = 0;
    function setMeta(t) {
      const w = Math.min(K, Math.floor(Math.max(0, t - 1) / 8) + 1);
      meta.innerHTML = `<span>step <b>${t}</b> / ${T}</span><span>window <b>${t ? w : 0}</b> / ${K}${t ? ` · <b>${sched[task][w - 1]}</b>-bit` : ""}</span><span>average <b>${(sched[task].reduce((a, b) => a + b, 0) / K).toFixed(2)}</b> bits</span>`;
    }
    function paintUpTo(t) {
      for (let step = 0; step < T; step++) {
        const b = step < t ? sched[task][Math.floor(step / 8)] : null;
        for (let j = 0; j < PER; j++) {
          const c = cells[order[step * PER + j]];
          c.classList.remove("fresh");
          if (b == null) c.removeAttribute("data-b"); else c.dataset.b = b;
        }
      }
    }
    function final() { cancelAnimationFrame(raf); paintUpTo(T); renderStrip(strip, sched[task]); setMeta(T); }
    function play() {
      cancelAnimationFrame(raf);
      if (reduceMotion) return final();
      paintUpTo(0); setMeta(0); renderStrip(strip, sched[task], 0);
      const msPerStep = 26; let t0 = null, done = 0;
      const tick = now => {
        if (t0 == null) t0 = now;
        const target = Math.min(T, Math.floor((now - t0) / msPerStep));
        while (done < target) {
          const b = sched[task][Math.floor(done / 8)];
          for (let j = 0; j < PER; j++) {
            const c = cells[order[done * PER + j]];
            c.dataset.b = b; c.classList.add("fresh");
            setTimeout(() => c.classList.remove("fresh"), 220);
          }
          done++;
          if (done % 8 === 1 || done === T) renderStrip(strip, sched[task], done === T ? -1 : Math.floor((done - 1) / 8));
        }
        setMeta(done);
        if (done < T) raf = requestAnimationFrame(tick);
      };
      raf = requestAnimationFrame(tick);
    }
    $$("#replay .seg button").forEach(b => b.addEventListener("click", () => {
      $$("#replay .seg button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
      task = b.dataset.task; play();
    }));
    $("#replay-play").addEventListener("click", play);
    final(); // complete at rest; animation runs on interaction
  })();

  /* ---------- 2. sensitivity profile ---------- */
  (function profile() {
    const box = $("#chart-profile");
    const tasks = ["gsm8k", "humaneval", "mmlupro"];
    let model = "llada8b_instruct";
    function draw() {
      const { s, w, h } = newSvg(box);
      const m = { l: 44, r: 82, t: 12, b: 34 };
      const x = lin(0, 128, m.l, w - m.r), y = lin(0, 1, h - m.b, m.t);
      const ax = svgEl("g", { class: "ax" }, s);
      [0, .25, .5, .75, 1].forEach(v => {
        svgEl("line", { class: "gridline", x1: m.l, x2: w - m.r, y1: y(v), y2: y(v) }, s);
        svgEl("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end" }, ax).textContent = v === 0 ? "0" : v.toFixed(2).replace(/0$/, "");
      });
      [0, 32, 64, 96, 128].forEach(v => svgEl("text", { x: x(v), y: h - m.b + 18, "text-anchor": "middle" }, ax).textContent = v);
      svgEl("text", { x: (m.l + w - m.r) / 2, y: h - 2, "text-anchor": "middle", class: "ax" }, s).textContent = "denoising step";
      const series = tasks.map(t => {
        const v = P.models[model][t].kappa4, mx = Math.max(...v);
        return { t, v: v.map(z => z / mx), raw: v };
      });
      series.forEach(se => {
        const d = se.v.map((v, k) => `${k ? "L" : "M"}${x(k * 8 + 4).toFixed(1)},${y(v).toFixed(1)}`).join("");
        svgEl("path", { d, fill: "none", stroke: `var(--s-${se.t})`, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, s);
        se.v.forEach((v, k) => svgEl("circle", { cx: x(k * 8 + 4), cy: y(v), r: 2.6, fill: `var(--s-${se.t})`, stroke: "var(--surface)", "stroke-width": 1.5 }, s));
      });
      // direct labels at line ends, nudged apart
      const ends = series.map(se => ({ t: se.t, y: y(se.v[K - 1]) })).sort((a, b) => a.y - b.y);
      for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 15) ends[i].y = ends[i - 1].y + 15;
      ends.forEach(e => svgEl("text", { x: w - m.r + 10, y: e.y + 4, class: "lbl-ink2" }, s).textContent = TASK_NAME[e.t]);
      // crosshair
      const cross = svgEl("line", { class: "cross", y1: m.t, y2: h - m.b, x1: -10, x2: -10 }, s);
      const hit = svgEl("rect", { class: "hit", x: m.l, y: m.t, width: w - m.l - m.r, height: h - m.t - m.b }, s);
      hit.addEventListener("pointermove", e => {
        const r = s.getBoundingClientRect();
        const step = (e.clientX - r.left - m.l) / (w - m.l - m.r) * 128;
        const k = Math.max(0, Math.min(K - 1, Math.floor(step / 8)));
        cross.setAttribute("x1", x(k * 8 + 4)); cross.setAttribute("x2", x(k * 8 + 4));
        showTip(`<div class="t">Steps ${k * 8}–${k * 8 + 7} · window ${k + 1}</div>` +
          series.map(se => row(TASK_NAME[se.t], fmt(se.v[k]), `var(--s-${se.t})`)).join("") +
          `<div class="r" style="margin-top:.2rem"><span>fraction of each task's peak</span></div>`, e.clientX, e.clientY);
      });
      hit.addEventListener("pointerleave", () => { cross.setAttribute("x1", -10); cross.setAttribute("x2", -10); hideTip(); });
    }
    $$("#profile-model button").forEach(b => b.addEventListener("click", () => {
      $$("#profile-model button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
      model = b.dataset.model; draw();
    }));
    draw(); onResize(box, draw);
  })();

  /* ---------- 3. allocator ---------- */
  (function allocator() {
    const box = $("#chart-gain"), selM = $("#ex-model"), selT = $("#ex-task"), budget = $("#ex-budget");
    let res = null;
    function data() { return P.models[selM.value][selT.value]; }
    function solve() {
      const d = data(), total = Number(budget.value);
      res = allocate(d.gain, d.levels, total);
      $("#ex-budget-v").textContent = (total / K).toFixed(3).replace(/0$/, "");
      renderStrip($("#ex-strip"), res.bits);
      renderStrip($("#ex-uniform"), Array(K).fill(4));
      const up = res.bits.filter(b => b > 4).length, down = res.bits.filter(b => b < 4).length;
      const match = total === 64 ? (res.bits.join() === d.schedule.join() ? " · matches the stored schedule" : " · differs from the stored schedule") : "";
      $("#ex-summary").textContent = up + down === 0 ? "no profitable trade: uniform" : `${up} up, ${down} down${match}`;
      draw();
    }
    function draw() {
      const d = data(), g8 = d.gain[8], g3 = d.gain[3];
      const { s, w, h } = newSvg(box);
      const m = { l: 48, r: 8, t: 10, b: 24 };
      const hi = Math.max(...g8, 1e-9), lo = Math.min(...g3, -1e-9);
      const y = lin(lo, hi, h - m.b, m.t), bw = (w - m.l - m.r) / K, x = k => m.l + k * bw;
      const ax = svgEl("g", { class: "ax" }, s);
      [lo, lo / 2, 0, hi / 2, hi].forEach(v => {
        svgEl("line", { class: "gridline", x1: m.l, x2: w - m.r, y1: y(v), y2: y(v) }, s);
        svgEl("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end" }, ax).textContent = fmtG(v);
      });
      for (let k = 0; k < K; k++) if (k % (w < 520 ? 3 : 1) === 0 || k === K - 1)
        svgEl("text", { x: x(k) + bw / 2, y: h - 6, "text-anchor": "middle" }, ax).textContent = k + 1;
      const barW = Math.max(3, Math.min(22, bw * 0.56));
      for (let k = 0; k < K; k++) {
        const cx = x(k) + bw / 2;
        if (g8[k] > 0) svgEl("rect", { x: cx - barW / 2, y: y(g8[k]), width: barW, height: Math.max(0.5, y(0) - y(g8[k]) - 1), rx: 3, fill: "var(--s-up)" }, s);
        if (g3[k] < 0) svgEl("rect", { x: cx - barW / 2, y: y(0) + 1, width: barW, height: Math.max(0.5, y(g3[k]) - y(0) - 1), rx: 3, fill: "var(--s-down)" }, s);
        const hit = svgEl("rect", { class: "hit", x: x(k), y: m.t, width: bw, height: h - m.t - m.b }, s);
        const steps = d.T / K;
        bindTip(hit, () => `<div class="t">Window ${k + 1} · steps ${k * steps}–${(k + 1) * steps - 1} of ${d.T}</div>` +
          row("assigned", `${res.bits[k]} bits`) +
          row("g(8)", fmtG(d.gain[8][k]), "var(--s-up)") + row("g(6)", fmtG(d.gain[6][k])) +
          row("g(3)", fmtG(d.gain[3][k]), "var(--s-down)") +
          row("κ at 4 bits", d.kappa4[k].toExponential(2)) + row("influence ῑ", fmtG(d.influence[k])));
      }
      svgEl("line", { class: "zero", x1: m.l, x2: w - m.r, y1: y(0), y2: y(0) }, s);
    }
    [selM, selT, budget].forEach(e => e.addEventListener("input", solve));
    solve(); onResize(box, draw);
  })();

  /* ---------- 4. results ---------- */
  const COLS = ["GSM8K", "HumanEval", "ARC-C", "MMLU-Pro", "DROP", "BBH"];
  const R = { // Table 1 of the submitted paper: [avg, GSM8K, HumanEval, ARC-C, MMLU-Pro, DROP, BBH]
    dream: [
      ["FP16", 44.05, 62.50, 19.00, 87.30, 29.69, 37.89, 27.93],
      ["RTN", 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
      ["DLLMQuant", 33.99, 35.55, 9.00, 79.30, 24.80, 38.09, 17.19],
      ["FAIR-Calib", 34.81, 44.92, 4.00, 81.64, 20.70, 41.60, 16.02],
      ["STaR-Quant", 38.55, 46.09, 11.00, 83.01, 26.17, 46.68, 18.36],
      ["Random schedule", 19.61, 15.82, 4.00, 54.30, 13.67, 20.31, 9.57],
      ["DLLMQuant + T²Quant", 40.69, 53.52, 17.00, 84.18, 29.69, 40.43, 19.34],
      ["FAIR-Calib + T²Quant", 40.61, 57.03, 12.00, 82.81, 26.95, 39.45, 25.39],
      ["STaR-Quant + T²Quant", 42.10, 58.79, 20.00, 84.77, 24.02, 42.77, 22.27]],
    llada15: [
      ["FP16", 47.73, 57.03, 26.00, 82.62, 37.30, 40.43, 42.97],
      ["RTN", 0.13, 0.00, 0.00, 0.59, 0.00, 0.20, 0.00],
      ["DLLMQuant", 39.09, 35.94, 15.00, 78.52, 32.23, 34.96, 37.89],
      ["FAIR-Calib", 42.88, 47.07, 18.00, 80.86, 35.16, 36.33, 39.84],
      ["STaR-Quant", 44.09, 48.63, 20.00, 79.49, 34.96, 39.26, 42.19],
      ["Random schedule", 39.11, 31.64, 20.00, 76.56, 30.86, 35.55, 40.04],
      ["DLLMQuant + T²Quant", 42.47, 45.51, 15.00, 83.79, 33.59, 37.11, 39.84],
      ["FAIR-Calib + T²Quant", 44.82, 52.15, 16.00, 82.81, 34.77, 41.80, 41.41],
      ["STaR-Quant + T²Quant", 45.22, 55.08, 17.00, 82.23, 34.57, 42.97, 39.45]],
    llada8b_instruct: [
      ["FP16", 46.32, 52.93, 25.00, 81.84, 36.52, 39.26, 42.38],
      ["RTN", 0.26, 0.39, 0.00, 0.98, 0.00, 0.20, 0.00],
      ["DLLMQuant", 37.35, 31.84, 12.00, 77.73, 30.66, 34.96, 36.91],
      ["FAIR-Calib", 42.27, 43.75, 22.00, 78.12, 33.79, 34.96, 41.02],
      ["STaR-Quant", 43.36, 45.70, 25.00, 78.52, 32.23, 38.09, 40.62],
      ["Random schedule", 37.72, 30.08, 23.00, 75.39, 31.25, 27.73, 38.87],
      ["DLLMQuant + T²Quant", 40.88, 42.19, 15.00, 81.84, 29.88, 38.09, 38.28],
      ["FAIR-Calib + T²Quant", 44.76, 51.95, 24.00, 80.08, 33.79, 36.52, 42.19],
      ["STaR-Quant + T²Quant", 43.70, 48.83, 21.00, 80.47, 32.62, 38.28, 41.02]]
  };
  const BASES = ["DLLMQuant", "FAIR-Calib", "STaR-Quant"];
  const find = (model, name) => R[model].find(r => r[0] === name);

  (function table() {
    const tbl = $("#results-table");
    function draw(model) {
      const rows = R[model], quant = rows.filter(r => r[0] !== "FP16");
      const marks = [];
      for (let c = 1; c <= 7; c++) {
        const vals = [...new Set(quant.map(r => r[c]))].sort((a, b) => b - a);
        marks[c] = { best: vals[0], second: vals[1] };
      }
      let html = `<thead><tr><th>Method</th><th class="avg">Avg.</th>${COLS.map(c => `<th>${c}</th>`).join("")}</tr></thead><tbody>`;
      rows.forEach((r, i) => {
        const ours = r[0].includes("T²Quant");
        const cls = [ours ? "ours" : "", i === 0 || i === 5 ? "sep" : ""].join(" ").trim();
        html += `<tr${cls ? ` class="${cls}"` : ""}><td>${ours ? r[0].replace("T²Quant", "T²Quant (ours)") : r[0]}</td>`;
        for (let c = 1; c <= 7; c++) {
          const v = r[c];
          const mk = r[0] === "FP16" ? "" : v === marks[c].best ? "best" : v === marks[c].second ? "second" : "";
          let extra = "";
          if (ours && c === 1) { const base = find(model, r[0].split(" + ")[0]); const dlt = v - base[1]; extra = `<span class="delta">${dlt >= 0 ? "+" : "−"}${Math.abs(dlt).toFixed(2)}</span>`; }
          html += `<td class="${[c === 1 ? "avg" : "", r[0] === "FP16" ? "muted" : ""].join(" ").trim()}"><span class="${mk}">${v.toFixed(2)}</span>${extra}</td>`;
        }
        html += "</tr>";
      });
      tbl.innerHTML = html + "</tbody>";
    }
    $$("#table-model button").forEach(b => b.addEventListener("click", () => {
      $$("#table-model button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
      draw(b.dataset.model);
    }));
    draw("dream");
  })();

  (function dumbbell() {
    const box = $("#chart-dumbbell");
    const models = ["dream", "llada15", "llada8b_instruct"];
    function draw() {
      const { s, w, h } = newSvg(box);
      const narrow = w < 560;
      const m = { l: narrow ? 92 : 150, r: 56, t: 8, b: 30 };
      const x = lin(32, 50, m.l, w - m.r);
      const groupH = (h - m.t - m.b) / 3, rowH = groupH / 4.2;
      const ax = svgEl("g", { class: "ax" }, s);
      [32, 36, 40, 44, 48].forEach(v => {
        svgEl("line", { class: "gridline", x1: x(v), x2: x(v), y1: m.t, y2: h - m.b }, s);
        svgEl("text", { x: x(v), y: h - m.b + 18, "text-anchor": "middle" }, ax).textContent = v;
      });
      models.forEach((md, gi) => {
        const gy = m.t + gi * groupH;
        svgEl("text", { x: 0, y: gy + rowH * 0.62, class: "lbl-ink" }, s).textContent = MODEL_NAME[md];
        const fp = find(md, "FP16")[1];
        svgEl("line", { class: "ref dash", x1: x(fp), x2: x(fp), y1: gy + rowH * 0.95, y2: gy + rowH * 4 }, s);
        svgEl("text", { x: x(fp) + 4, y: gy + rowH * 0.95 + 4, class: "ax" }, s).textContent = `FP16 ${fp.toFixed(1)}`;
        BASES.forEach((b, bi) => {
          const yy = gy + rowH * (1.6 + bi * 0.85 + bi * 0.08);
          const v0 = find(md, b)[1], v1 = find(md, b + " + T²Quant")[1];
          svgEl("text", { x: 0, y: yy + 4, class: "lbl-ink2" }, s).textContent = narrow ? b.replace("Quant", "Q").replace("-Calib", "") : b;
          svgEl("line", { x1: x(v0), x2: x(v1), y1: yy, y2: yy, stroke: "var(--s-ours)", "stroke-width": 2, "stroke-opacity": .45 }, s);
          svgEl("circle", { cx: x(v0), cy: yy, r: 5, fill: "var(--surface)", stroke: "var(--s-base)", "stroke-width": 2 }, s);
          svgEl("circle", { cx: x(v1), cy: yy, r: 5.5, fill: "var(--s-ours)", stroke: "var(--surface)", "stroke-width": 2 }, s);
          svgEl("text", { x: x(Math.max(v0, v1)) + 10, y: yy + 4, class: "lbl-ink", style: "font-size:11.5px" }, s).textContent = `+${(v1 - v0).toFixed(2)}`;
          const hit = svgEl("rect", { class: "hit", x: m.l, y: yy - rowH * 0.42, width: w - m.l - m.r, height: rowH * 0.84 }, s);
          bindTip(hit, `<div class="t">${MODEL_NAME[md]} · ${b}</div>` + row("base", v0.toFixed(2), "var(--s-base)") +
            row("+ T²Quant", v1.toFixed(2), "var(--s-ours)") + row("change", `+${(v1 - v0).toFixed(2)}`) + row("FP16", fp.toFixed(2)));
        });
      });
      svgEl("text", { x: (m.l + w - m.r) / 2, y: h - 2, "text-anchor": "middle", class: "ax" }, s).textContent = "average accuracy over six benchmarks (%)";
    }
    draw(); onResize(box, draw);
  })();

  /* ---------- 5. ablations (LLaDA-8B-Instruct, GSM8K, n = 512) ---------- */
  const UNI = 44.1, FP = 52.9;
  function panel(box, { cats, yMin, yMax, yTicks, xLabel, refs, drawMarks }) {
    const { s, w, h } = newSvg(box);
    const m = { l: 34, r: 10, t: 10, b: 36 };
    const y = lin(yMin, yMax, h - m.b, m.t);
    const bw = (w - m.l - m.r) / cats.length, x = i => m.l + bw * (i + 0.5);
    const ax = svgEl("g", { class: "ax" }, s);
    yTicks.forEach(v => {
      svgEl("line", { class: "gridline", x1: m.l, x2: w - m.r, y1: y(v), y2: y(v) }, s);
      svgEl("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end" }, ax).textContent = v;
    });
    cats.forEach((c, i) => svgEl("text", { x: x(i), y: h - m.b + 16, "text-anchor": "middle" }, ax).textContent = c);
    svgEl("text", { x: (m.l + w - m.r) / 2, y: h - 3, "text-anchor": "middle", class: "ax" }, s).textContent = xLabel;
    (refs || []).forEach(r => svgEl("line", { class: "ref" + (r.dash ? " dash" : ""), x1: m.l, x2: w - m.r, y1: y(r.v), y2: y(r.v), stroke: r.color || "" , style: r.color ? `stroke:${r.color}` : "" }, s));
    drawMarks({ s, x, y, bw });
  }
  function dot(s, cx, cy, color, r = 5) { return svgEl("circle", { cx, cy, r, fill: color, stroke: "var(--surface)", "stroke-width": 2 }, s); }

  (function calsize() {
    const box = $("#chart-calsize");
    const D = [[16, 48.0, 47.3, 48.4], [32, 47.1, 47.1, 47.1], [64, 47.9, 47.7, 48.4], [128, 47.7, 47.1, 48.4], [256, 49.4, 49.4, 49.4]];
    const draw = () => panel(box, {
      cats: D.map(d => d[0]), yMin: 42, yMax: 54, yTicks: [42, 46, 50, 54], xLabel: "unlabeled calibration prompts",
      refs: [{ v: UNI, color: "var(--s-base)" }, { v: FP, dash: 1 }],
      drawMarks: ({ s, x, y, bw }) => D.forEach((d, i) => {
        if (d[3] > d[2]) svgEl("line", { x1: x(i), x2: x(i), y1: y(d[2]), y2: y(d[3]), stroke: "var(--s-ours)", "stroke-width": 2, "stroke-linecap": "round", "stroke-opacity": .5 }, s);
        dot(s, x(i), y(d[1]), "var(--s-ours)");
        const hit = svgEl("rect", { class: "hit", x: x(i) - bw / 2, y: 0, width: bw, height: 400 }, s);
        bindTip(hit, `<div class="t">${d[0]} prompts</div>` + row("T²Quant", d[0] === 256 ? d[1].toFixed(1) : `${d[1].toFixed(1)} (${d[2].toFixed(1)}–${d[3].toFixed(1)})`, "var(--s-ours)") +
          row("uniform W4A4", UNI.toFixed(1), "var(--s-base)") + row("FP16", FP.toFixed(1)));
      })
    });
    draw(); onResize(box, draw);
  })();

  (function tps() {
    const box = $("#chart-tps");
    const D = [[2, 55.3, 60.5, 64.1], [4, 44.1, 49.4, 52.9], [8, 22.7, 24.2, 31.6]];
    const draw = () => panel(box, {
      cats: D.map(d => d[0]), yMin: 20, yMax: 68, yTicks: [20, 36, 52, 68], xLabel: "tokens committed per step",
      drawMarks: ({ s, x, y, bw }) => D.forEach((d, i) => {
        const half = Math.min(26, bw * 0.3);
        svgEl("line", { class: "ref dash", x1: x(i) - half, x2: x(i) + half, y1: y(d[3]), y2: y(d[3]) }, s);
        svgEl("line", { x1: x(i), x2: x(i), y1: y(d[1]), y2: y(d[2]), stroke: "var(--s-ours)", "stroke-width": 2, "stroke-opacity": .45 }, s);
        dot(s, x(i), y(d[1]), "var(--s-base)");
        dot(s, x(i), y(d[2]), "var(--s-ours)");
        const hit = svgEl("rect", { class: "hit", x: x(i) - bw / 2, y: 0, width: bw, height: 400 }, s);
        bindTip(hit, `<div class="t">${d[0]} tokens per step</div>` + row("T²Quant", d[2].toFixed(1), "var(--s-ours)") +
          row("uniform W4A4", d[1].toFixed(1), "var(--s-base)") + row("FP16", d[3].toFixed(1)) + row("gain", `+${(d[2] - d[1]).toFixed(1)}`));
      })
    });
    draw(); onResize(box, draw);
  })();

  (function budgetPanel() {
    const box = $("#chart-budget");
    const bits = [3.5, 3.75, 4.0, 4.25, 4.5], ours = [46.1, 48.4, 49.4, 49.0, 49.0], rand = [29.7, 29.3, 28.5, 29.1, 42.6];
    const draw = () => panel(box, {
      cats: bits.map(b => b.toFixed(2)), yMin: 24, yMax: 56, yTicks: [24, 32, 40, 48, 56], xLabel: "average activation bits",
      refs: [{ v: UNI, color: "var(--s-base)" }, { v: FP, dash: 1 }],
      drawMarks: ({ s, x, y, bw }) => {
        [[ours, "var(--s-ours)"], [rand, "var(--s-random)"]].forEach(([v, c]) => {
          svgEl("path", { d: v.map((z, i) => `${i ? "L" : "M"}${x(i)},${y(z)}`).join(""), fill: "none", stroke: c, "stroke-width": 2, "stroke-linejoin": "round" }, s);
          v.forEach((z, i) => dot(s, x(i), y(z), c, 4.5));
        });
        bits.forEach((b, i) => {
          const hit = svgEl("rect", { class: "hit", x: x(i) - bw / 2, y: 0, width: bw, height: 400 }, s);
          bindTip(hit, `<div class="t">${b.toFixed(2)} average bits</div>` + row("T²Quant", ours[i].toFixed(1), "var(--s-ours)") +
            row("random, same bits", rand[i].toFixed(1), "var(--s-random)") + row("uniform W4A4 (4.00)", UNI.toFixed(1), "var(--s-base)") + row("FP16", FP.toFixed(1)));
        });
      }
    });
    draw(); onResize(box, draw);
  })();

  /* ---------- 6. repository schedule ---------- */
  renderStrip($("#repo-strip"), [8, 5, 5, 5, 5, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3]);

  /* ---------- 7. links, BibTeX, copy, math ---------- */
  (function links() {
    if (SITE.venue) $("#venue").textContent = SITE.venue;
    if (SITE.code) $("#btn-code").href = SITE.code;
    const id = (SITE.arxiv || "").trim();
    const bib = [
      "@misc{avivi2026t2quant,",
      "  title         = {Different Tasks, Different Timesteps: Temporal Task-Aware Quantization for Diffusion Language Models},",
      "  author        = {Avivi, Ofek and Rahimi, Eliron and Mendelson, Avi and Baskin, Chaim},",
      "  year          = {2026},"];
    if (id) {
      const a = $("#btn-arxiv");
      a.href = `https://arxiv.org/abs/${id}`; a.removeAttribute("aria-disabled");
      a.querySelector(".soon").textContent = id;
      const pdf = a.cloneNode(true);
      pdf.removeAttribute("id"); pdf.href = `https://arxiv.org/pdf/${id}`;
      pdf.querySelector("span").textContent = "PDF"; pdf.querySelector(".soon").remove();
      a.after(pdf);
      bib.push(`  eprint        = {${id}},`, "  archivePrefix = {arXiv},", `  url           = {https://arxiv.org/abs/${id}}`);
    } else {
      bib.push(`  note          = {${SITE.venue || "Preprint"}},`, `  url           = {${SITE.code || "https://github.com/OfekAvivi/T2Quant"}}`);
    }
    $("#bibtex").textContent = bib.join("\n") + "\n}";
  })();

  $$("[data-copy]").forEach(btn => btn.addEventListener("click", () => {
    const pre = document.getElementById(btn.dataset.copy), text = pre.textContent;
    const done = ok => { btn.textContent = ok ? "Copied" : "Select and copy"; setTimeout(() => (btn.textContent = "Copy"), 1600); };
    const select = () => { const r = document.createRange(); r.selectNodeContents(pre); const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r); };
    try {
      navigator.clipboard.writeText(text).then(() => done(true), () => { select(); done(false); });
    } catch (e) { select(); done(false); }
  }));

  if (window.katex) {
    $$(".tex").forEach(el => {
      try {
        katex.render(el.dataset.tex, el, { output: "mathml", displayMode: el.dataset.display === "1", throwOnError: false });
      } catch (e) { /* keep the plain-text fallback */ }
    });
  }

  // Display equations never scroll: shrink the font of any that is wider than its column.
  function fitMath() {
    $$(".eq").forEach(el => {
      el.style.fontSize = "";
      const base = parseFloat(getComputedStyle(el).fontSize);
      let fs = base;
      for (let i = 0; i < 6 && el.scrollWidth > el.clientWidth + 1; i++) {
        fs = Math.max(base * 0.5, fs * (el.clientWidth / el.scrollWidth) * 0.98);
        el.style.fontSize = fs + "px";
      }
    });
  }
  let fitQueued = false;
  addEventListener("resize", () => { if (!fitQueued) { fitQueued = true; requestAnimationFrame(() => { fitQueued = false; fitMath(); }); } });
  fitMath();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(fitMath);
})();
