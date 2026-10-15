(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const el = {
    feed: $("feed"), chips: $("chips"), composer: $("composer"), input: $("input"), send: $("sendBtn"), stop: $("stopBtn"), hint: $("hint"),
    docs: $("docs"), drop: $("drop"), file: $("fileInput"), upStatus: $("uploadStatus"), model: $("model"), topk: $("topk"), kOut: $("kOut"),
    pEmbed: $("pEmbed"), pGroq: $("pGroq"), pChunks: $("pChunks"), legend: $("legend"), selected: $("selected"), mapSub: $("mapSub"),
    clearHits: $("clearHits"), mapEmpty: $("mapEmpty"),
  };
  const PALETTE = ["#19C37D", "#8B7BFF", "#FFB020", "#FF6F9C", "#4DA3FF", "#B6D43A", "#22D3EE", "#C084FC"];
  const SAMPLE_IDEAS = [
    ["Explain", "What is an embedding?"],
    ["Rule", "Why must documents and questions use the same embedding model?"],
    ["Filter", "How does ChromaDB search only one file?"],
    ["Limits", "What happens when the Groq rate limit is reached?"],
    ["Other topic", "How long should biryani rice be soaked?"],
    ["Not in docs", "Who won the cricket world cup?"],
  ];
  const GENERIC_IDEAS = [["Overview", "What is this document about?"], ["Key points", "List the key points."], ["Process", "What steps or methods are described?"], ["Risks", "What limitations or risks are mentioned?"]];

  const S = { docs: [], chunks: 0, scope: null, messages: [], busy: false, abort: null, active: null, model: "", map: null, ready: false, groq: false };

  /* ---------------- helpers ---------------- */
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
  const shortName = (n) => n.replace(/\.(pdf|txt|md|markdown)$/i, "");
  const colorOf = (source) => { const i = S.docs.findIndex((d) => d.source === source); return PALETTE[(i < 0 ? 0 : i) % PALETTE.length]; };

  async function api(path, options) {
    let res;
    try { res = await fetch(path, options); } catch { throw new Error("Can't reach the server. Is it running?"); }
    let body = null; try { body = await res.json(); } catch { /* not json */ }
    if (!res.ok) throw new Error((body && typeof body.detail === "string" && body.detail) || "Request failed. Please try again.");
    return body;
  }
  async function streamSSE(path, payload, onEvent, signal) {
    const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal });
    if (!res.ok) { let d = null; try { d = await res.json(); } catch { /* */ } throw new Error((d && d.detail) || "Request failed. Please try again."); }
    const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = "";
    for (;;) {
      const { done, value } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true }); let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, i); buf = buf.slice(i + 2);
        if (block.startsWith("data: ")) { try { onEvent(JSON.parse(block.slice(6))); } catch { /* skip */ } }
      }
    }
  }
  const setHint = (t, warn = false) => { el.hint.textContent = t || ""; el.hint.classList.toggle("warn", warn); };

  /* ---------------- markdown ---------------- */
  function inline(s, cites, idx) {
    return s.replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*\w])\*([^*\s][^*]*?)\*(?![*\w])/g, "$1<em>$2</em>")
      .replace(/\[(\d+)\]/g, (m, n) => (cites.has(+n) ? `<button type="button" class="cite" data-cite="${n}" data-idx="${idx}" aria-label="Show source ${n}">${n}</button>` : m));
  }
  function md(text, cites, idx) {
    let src = esc(text); const blocks = [];
    const stash = (c) => { blocks.push(`<pre><code>${c}</code></pre>`); return `\u0000${blocks.length - 1}\u0000`; };
    src = src.replace(/```[\w-]*\n([\s\S]*?)```/g, (m, c) => stash(c)).replace(/```[\w-]*\n([\s\S]*)$/, (m, c) => stash(c));
    const out = []; let list = null; let para = [];
    const fp = () => { if (para.length) { out.push(`<p>${inline(para.join(" "), cites, idx)}</p>`); para = []; } };
    const fl = () => { if (list) { out.push(`</${list}>`); list = null; } };
    for (const raw of src.split(/\r?\n/)) {
      const line = raw.trimEnd(); let m;
      if (!line.trim()) { fp(); fl(); continue; }
      if (/^\u0000\d+\u0000$/.test(line.trim())) { fp(); fl(); out.push(line.trim()); continue; }
      if ((m = line.match(/^#{1,4}\s+(.*)$/))) { fp(); fl(); out.push(`<h3>${inline(m[1], cites, idx)}</h3>`); continue; }
      if ((m = line.match(/^\s*[-*•]\s+(.*)$/))) { fp(); if (list !== "ul") { fl(); out.push("<ul>"); list = "ul"; } out.push(`<li>${inline(m[1], cites, idx)}</li>`); continue; }
      if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) { fp(); if (list !== "ol") { fl(); out.push("<ol>"); list = "ol"; } out.push(`<li>${inline(m[1], cites, idx)}</li>`); continue; }
      fl(); para.push(line.trim());
    }
    fp(); fl();
    return out.join("").replace(/\u0000(\d+)\u0000/g, (m, n) => blocks[+n]);
  }

  /* ---------------- embedding map (canvas) ---------------- */
  const MapView = (() => {
    const cv = $("map"), ctx = cv.getContext("2d"), wrap = $("canvasWrap"), tip = $("tip");
    let pts = [], hits = new Map(), q = null, W = 0, H = 0, dpr = 1, hover = null, selId = null, animStart = 0, animUntil = 0, raf = 0, onPick = () => {};
    const bound = () => { let m = 1; pts.forEach((p) => { m = Math.max(m, Math.abs(p.x), Math.abs(p.y)); }); if (q) m = Math.max(m, Math.abs(q.x), Math.abs(q.y)); return m * 1.12; };
    const scr = (p) => { const s = (Math.min(W, H) / 2 - 24) / bound(); return { x: W / 2 + p.x * s, y: H / 2 - p.y * s }; };
    function resize() {
      const r = cv.getBoundingClientRect(); if (!r.width) return;
      dpr = window.devicePixelRatio || 1; W = r.width; H = r.height; cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); draw();
    }
    function kick(ms) { animStart = performance.now(); animUntil = animStart + ms; if (!raf) raf = requestAnimationFrame(loop); }
    function loop() { draw(); raf = performance.now() < animUntil ? requestAnimationFrame(loop) : 0; }
    function draw() {
      if (!W) return;
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = "rgba(147,162,198,.13)"; ctx.lineWidth = 1;
      for (const k of [0.33, 0.66, 1]) { ctx.beginPath(); ctx.arc(W / 2, H / 2, k * (Math.min(W, H) / 2 - 12), 0, Math.PI * 2); ctx.stroke(); }
      ctx.beginPath(); ctx.moveTo(10, H / 2); ctx.lineTo(W - 10, H / 2); ctx.moveTo(W / 2, 10); ctx.lineTo(W / 2, H - 10); ctx.stroke();
      const active = hits.size > 0; const now = performance.now(); const t = Math.min(1, (now - animStart) / 900); const ease = 1 - Math.pow(1 - t, 3);
      for (const p of pts) {
        if (hits.has(p.id)) continue;
        const s = scr(p); ctx.globalAlpha = active ? 0.2 : 0.92; ctx.fillStyle = colorOf(p.source);
        ctx.beginPath(); ctx.arc(s.x, s.y, 5.5, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
      if (q && active) {
        const qs = scr(q);
        for (const [id, h] of hits) {
          const p = pts.find((x) => x.id === id); if (!p) continue; const s = scr(p);
          ctx.strokeStyle = h.relevant ? "rgba(255,90,79,.8)" : "rgba(255,90,79,.3)"; ctx.lineWidth = 1 + Math.max(0, h.sim) * 3; ctx.setLineDash([5, 4]);
          ctx.beginPath(); ctx.moveTo(qs.x, qs.y); ctx.lineTo(qs.x + (s.x - qs.x) * ease, qs.y + (s.y - qs.y) * ease); ctx.stroke(); ctx.setLineDash([]);
        }
      }
      for (const [id, h] of hits) {
        const p = pts.find((x) => x.id === id); if (!p) continue; const s = scr(p); const col = colorOf(p.source);
        ctx.globalAlpha = h.relevant ? 1 : 0.55; ctx.fillStyle = col; ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(s.x, s.y, 10, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        ctx.fillStyle = "#0B1429"; ctx.font = "700 11px 'IBM Plex Mono', monospace"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(String(h.n), s.x, s.y + 0.5);
        ctx.globalAlpha = 1;
      }
      if (q) {
        const s = scr(q);
        const phase = ((now - animStart) % 1400) / 1400;
        if (now < animUntil) for (const off of [0, 0.5]) { const ph = (phase + off) % 1; ctx.strokeStyle = `rgba(255,90,79,${0.55 * (1 - ph)})`; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(s.x, s.y, 10 + ph * 46, 0, Math.PI * 2); ctx.stroke(); }
        ctx.save(); ctx.translate(s.x, s.y); ctx.rotate(Math.PI / 4); ctx.fillStyle = "#FF5A4F"; ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.fillRect(-7, -7, 14, 14); ctx.strokeRect(-7, -7, 14, 14); ctx.restore();
        ctx.fillStyle = "#FFB3AD"; ctx.font = "600 11px 'IBM Plex Mono', monospace"; ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillText("question", s.x + 13, s.y - 12);
      }
      for (const id of [hover, selId]) {
        if (!id) continue; const p = pts.find((x) => x.id === id); if (!p) continue; const s = scr(p);
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(s.x, s.y, 13, 0, Math.PI * 2); ctx.stroke();
      }
    }
    function nearest(ev) {
      const r = cv.getBoundingClientRect(); const mx = ev.clientX - r.left, my = ev.clientY - r.top; let best = null, bd = 14;
      for (const p of pts) { const s = scr(p); const d = Math.hypot(s.x - mx, s.y - my); if (d < bd) { bd = d; best = { p, s }; } }
      return best;
    }
    cv.addEventListener("mousemove", (ev) => {
      const n = nearest(ev); const id = n ? n.p.id : null;
      if (id !== hover) { hover = id; draw(); }
      if (!n) { tip.hidden = true; return; }
      tip.hidden = false; tip.innerHTML = `<b>${esc(shortName(n.p.source))}</b><small>${n.p.page ? "page " + n.p.page + " · " : ""}chunk ${n.p.chunk}</small>${esc(n.p.snippet.slice(0, 130))}…`;
      const tw = 240; tip.style.left = Math.min(Math.max(6, n.s.x + 14), W - tw - 6) + "px"; tip.style.top = Math.min(n.s.y + 14, H - 90) + "px";
    });
    cv.addEventListener("mouseleave", () => { hover = null; tip.hidden = true; draw(); });
    cv.addEventListener("click", (ev) => { const n = nearest(ev); if (n) { selId = n.p.id; onPick(n.p); draw(); } });
    new ResizeObserver(resize).observe(wrap); window.addEventListener("resize", resize);
    return {
      resize, draw, onPick: (f) => { onPick = f; },
      setPoints(p) { pts = p || []; if (selId && !pts.some((x) => x.id === selId)) selId = null; draw(); },
      setQuery(point, list) { q = point; hits = new Map((list || []).map((s) => [s.id, { n: s.n, sim: s.similarity, relevant: s.relevant }])); kick(1800); },
      clear() { q = null; hits = new Map(); draw(); },
      get hasQuery() { return !!q; },
    };
  })();

  /* ---------------- library ---------------- */
  function renderDocs() {
    const all = `<button class="doc ${S.scope ? "" : "on"}" data-scope=""><span class="dot" style="background:conic-gradient(${PALETTE.slice(0, 5).join(",")})"></span><div><b>All documents</b><small>${plural(S.docs.length, "document")} · ${plural(S.chunks, "chunk")}</small></div><span></span></button>`;
    el.docs.innerHTML = S.docs.length ? all + S.docs.map((d) => `<div class="doc ${S.scope === d.source ? "on" : ""}" data-scope="${esc(d.source)}" role="button" tabindex="0"><span class="dot" style="background:${colorOf(d.source)}"></span><div><b>${esc(d.source)}</b><small>${plural(d.chunks, "chunk")}${d.pages ? " · " + plural(d.pages, "page") : ""}</small></div><button class="x" data-del="${esc(d.source)}" aria-label="Remove ${esc(d.source)}" title="Remove">×</button></div>`).join("")
      : '<p class="none">No documents yet. Upload some, or load the sample documents.</p>';
    el.pChunks.querySelector("span").textContent = plural(S.chunks, "chunk");
    el.legend.innerHTML = S.docs.map((d) => `<button class="lg ${S.scope === d.source ? "on" : ""}" data-scope="${esc(d.source)}"><i class="dot" style="background:${colorOf(d.source)}"></i><span>${esc(shortName(d.source))}</span></button>`).join("");
    el.mapEmpty.hidden = S.chunks > 0;
  }
  async function loadDocs() {
    const d = await api("/api/documents"); S.docs = d.documents; S.chunks = d.chunks;
    if (S.scope && !S.docs.some((x) => x.source === S.scope)) S.scope = null;
    renderDocs(); await loadMap(); renderFeed(); renderChips();
  }
  async function loadMap() {
    if (!S.chunks) { S.map = null; MapView.setPoints([]); MapView.clear(); el.mapSub.textContent = "No chunks yet"; return; }
    S.map = await api("/api/map"); MapView.setPoints(S.map.points);
    const [a, b] = S.map.explained; el.mapSub.textContent = `${plural(S.map.points.length, "chunk")} · PCA ${Math.round(a * 100)}% + ${Math.round(b * 100)}%`;
    MapView.resize();
  }
  async function upload(files) {
    files = [...files]; if (!files.length) return;
    el.upStatus.innerHTML = `<div class="up"><div class="spin"></div><div><b>Embedding ${plural(files.length, "file")}…</b><small>Chunking, creating local embeddings, storing in ChromaDB. The first run also loads the model.</small></div></div>`;
    const form = new FormData(); files.forEach((f) => form.append("files", f, f.name));
    try {
      const d = await api("/api/documents", { method: "POST", body: form });
      el.upStatus.innerHTML = d.results.map((r) => r.status === "indexed" ? `<div class="up ok"><div><b>${esc(r.name)}</b><small>${plural(r.chunks, "chunk")} from ${plural(Math.max(r.pages, 1), "page")} stored</small></div></div>` : `<div class="up err"><div><b>${esc(r.name)}</b><small>${esc(r.error)}</small></div></div>`).join("");
      await loadDocs(); refreshHealth();
      if (!el.upStatus.querySelector(".err")) setTimeout(() => { el.upStatus.innerHTML = ""; }, 6000);
    } catch (e) { el.upStatus.innerHTML = `<div class="up err"><div><b>Upload failed</b><small>${esc(e.message)}</small></div></div>`; }
    el.file.value = "";
  }
  async function loadSample() {
    el.upStatus.innerHTML = '<div class="up"><div class="spin"></div><div><b>Indexing the sample documents…</b></div></div>';
    try { const r = await api("/api/ingest/folder", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reset: false }) }); el.upStatus.innerHTML = `<div class="up ok"><div><b>Sample documents ready</b><small>${plural(r.chunks_upserted, "chunk")} indexed</small></div></div>`; await loadDocs(); refreshHealth(); setTimeout(() => { el.upStatus.innerHTML = ""; }, 5000); }
    catch (e) { el.upStatus.innerHTML = `<div class="up err"><div><b>Could not load samples</b><small>${esc(e.message)}</small></div></div>`; }
  }

  /* ---------------- feed ---------------- */
  const stepHTML = (n, label, val, state) => `<span class="step ${state || ""}"><span class="n">${n}</span>${label}${val ? ` <b>${val}</b>` : ""}</span>`;
  function pipeHTML(m) {
    const st = m.stages || {}; const s3 = m.stats;
    const a = st.embed != null ? stepHTML(1, "Embed question", st.embed + " ms") : stepHTML(1, "Embed question", "", m.streaming ? "run" : "");
    const b = st.search != null ? stepHTML(2, "ChromaDB search", `${st.search} ms · ${plural(st.retrieved || 0, "chunk")}`) : stepHTML(2, "ChromaDB search", "", m.streaming && st.embed != null ? "run" : "");
    let c;
    if (s3 && s3.no_llm_call) c = stepHTML(3, "Groq answer", "skipped: nothing relevant", "skip");
    else if (s3) c = stepHTML(3, "Groq first token", s3.ttft_ms + " ms");
    else c = stepHTML(3, "Groq answer", "", m.streaming && m.sources ? "run" : "");
    return `<div class="pipe">${a}${b}${c}</div>`;
  }
  function sourcesHTML(m, idx) {
    if (!m.sources || !m.sources.length || m.streaming && !m.sources) return "";
    const chips = m.sources.map((s) => `<button type="button" class="src ${s.relevant ? "" : "low"} ${m.open === s.n ? "on" : ""}" data-src="${s.n}" data-idx="${idx}" aria-expanded="${m.open === s.n}" title="${s.relevant ? "" : "Below the similarity threshold, not sent to the model"}"><span class="n">${s.n}</span><span class="nm" style="color:inherit">${esc(shortName(s.source))}</span><span>${s.page ? "p." + s.page : "chunk " + s.chunk}</span><span class="sim">${s.similarity.toFixed(2)}</span></button>`).join("");
    const cards = m.sources.map((s) => `<div class="ev ${m.open === s.n ? "open" : ""}"><div><div class="evc"><div class="evt"><b>${esc(s.source)}</b><span>${s.page ? "page " + s.page + " · " : ""}chunk ${s.chunk}</span></div><div class="simbar"><i style="width:${Math.max(2, Math.round(Math.max(0, s.similarity) * 100))}%"></i></div><div class="evt" style="margin-bottom:6px"><span>cosine similarity ${s.similarity.toFixed(3)}</span><span>${s.relevant ? "sent to Groq" : "below threshold"}</span></div><p class="evx">${esc(s.text)}</p></div></div></div>`).join("");
    return `<div class="srcs"><span class="lab">${m.usable ? "Retrieved" : "Closest chunks"}</span>${chips}</div>${cards}`;
  }
  function statsHTML(m) {
    const s = m.stats; if (!s || s.no_llm_call) return "";
    return `<div class="stats"><span class="stat hot"><b>${s.tokens_per_s}</b> tok/s</span><span class="stat">${s.tokens_estimated ? "~" : ""}<b>${s.tokens}</b> tokens</span><span class="stat">total <b>${((s.total_ms + (s.embed_ms || 0) + (s.search_ms || 0)) / 1000).toFixed(2)} s</b></span><span class="stat">${esc(s.model)}</span></div>`;
  }
  function botHTML(m, idx) {
    let body;
    const cites = new Set((m.sources || []).map((s) => s.n));
    if (m.error) body = `<div class="err">${esc(m.error)}</div>`;
    else if (m.stats && m.stats.no_llm_call) body = `<div class="refuse"><div class="g">∅</div><div>${esc(m.content)}</div></div>`;
    else if (m.streaming && !m.content) body = `<div class="wait">${m.sources ? "Groq is reading the retrieved chunks…" : "Working…"}</div>`;
    else body = `<div class="content ${m.streaming ? "cursor" : ""}">${md(m.content || "", cites, idx)}</div>`;
    return `${pipeHTML(m)}${body}${m.streaming && !m.sources ? "" : sourcesHTML(m, idx)}${m.streaming ? "" : statsHTML(m)}`;
  }
  function introHTML() {
    if (!S.docs.length) return `<div class="wrap intro"><h1>Ask your documents. See the search.</h1><p>Upload PDFs, text or Markdown. Atlas embeds every chunk on your computer, stores it in ChromaDB, and lets Groq answer with citations while the map shows exactly which chunks were retrieved.</p><div class="cta"><button class="btn" data-pick>Upload documents</button><button class="btn alt" data-sample>Load sample documents</button></div></div>`;
    const sample = S.docs.some((d) => d.source === "rag_basics.md");
    const ideas = (sample ? SAMPLE_IDEAS : GENERIC_IDEAS).map(([tag, q], i) => `<button class="idea" data-idea="${i}"><small>${tag}</small>${esc(q)}</button>`).join("");
    return `<div class="wrap intro"><h1>What would you like to know?</h1><p>Pick a question, or type your own. Use the scope list to search a single document.</p><div class="ideas">${ideas}</div></div>`;
  }
  let raf = 0;
  function renderFeed() {
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => {
      const f = el.feed; const near = f.scrollHeight - f.scrollTop - f.clientHeight < 90;
      f.innerHTML = S.messages.length ? '<div class="wrap">' + S.messages.map((m, i) => m.role === "user" ? `<div class="row you"><div>${esc(m.content)}</div></div>` : `<div class="row">${botHTML(m, i)}</div>`).join("") + "</div>" : introHTML();
      if (near) f.scrollTo({ top: f.scrollHeight });
    });
  }
  const ideasNow = () => (S.docs.some((d) => d.source === "rag_basics.md") ? SAMPLE_IDEAS : GENERIC_IDEAS);
  function renderChips() {
    el.chips.hidden = !S.messages.length || !S.docs.length;
    el.chips.innerHTML = ideasNow().slice(0, 5).map(([, q], i) => `<button type="button" class="chip" data-idea="${i}" title="${esc(q)}">${esc(q)}</button>`).join("");
  }

  /* ---------------- asking ---------------- */
  async function ask(text) {
    text = (text || "").trim();
    if (S.busy) return;
    if (!text) { setHint("Type a question first.", true); el.input.focus(); return; }
    if (!S.docs.length) { setHint("No document uploaded yet. Upload a file or load the sample documents first.", true); return; }
    if (!S.groq) { setHint("GROQ_API_KEY is missing. Add it to your .env file and restart the server.", true); return; }
    setHint(""); el.input.value = ""; autogrow();
    const history = S.messages.filter((m) => !m.error && m.content).map((m) => ({ role: m.role, content: m.content })).slice(-6);
    S.messages.push({ role: "user", content: text });
    const m = { role: "assistant", content: "", streaming: true, stages: {}, sources: null, open: null }; S.messages.push(m);
    const idx = S.messages.length - 1; S.busy = true; S.abort = new AbortController(); controls(); renderFeed(); renderChips();
    try {
      await streamSSE("/api/rag", { question: text, history, top_k: Number(el.topk.value), source: S.scope, model: S.model }, (ev) => {
        if (ev.type === "stage") { m.stages[ev.name] = ev.ms; if (ev.retrieved != null) m.stages.retrieved = ev.retrieved; }
        else if (ev.type === "sources") { m.sources = ev.sources; m.usable = ev.usable; m.qpoint = ev.query_point; S.active = idx; MapView.setQuery(ev.query_point, ev.sources); el.clearHits.hidden = false; }
        else if (ev.type === "token") m.content += ev.text;
        else if (ev.type === "done") { m.stats = ev.stats; m.streaming = false; }
        else if (ev.type === "error") { m.error = ev.message; m.streaming = false; }
        renderFeed();
      }, S.abort.signal);
    } catch (e) { m.error = e.name === "AbortError" ? (m.content ? null : "Stopped.") : e.message; }
    m.streaming = false; S.busy = false; S.abort = null; controls(); renderFeed(); el.input.focus();
  }
  function controls() { el.send.hidden = S.busy; el.stop.hidden = !S.busy; }
  function autogrow() { el.input.style.height = "auto"; el.input.style.height = Math.min(el.input.scrollHeight, 150) + "px"; }
  function pickSource(idx, n) {
    const m = S.messages[idx]; if (!m || !m.sources) return;
    m.open = m.open === n ? null : n; S.active = idx;
    MapView.setQuery(m.qpoint, m.sources); el.clearHits.hidden = false;
    const s = m.sources.find((x) => x.n === n);
    if (s && m.open) showSelected({ source: s.source, page: s.page, chunk: s.chunk, snippet: s.text.replace(/\s+/g, " ").slice(0, 260) }, s.similarity);
    renderFeed();
  }
  function showSelected(p, sim) {
    el.selected.innerHTML = `<small>${esc(p.source)} · ${p.page ? "page " + p.page + " · " : ""}chunk ${p.chunk}${sim != null ? " · similarity " + sim.toFixed(2) : ""}</small>${esc(p.snippet)}${p.snippet.length >= 220 ? "…" : ""}`;
  }

  /* ---------------- health ---------------- */
  async function refreshHealth() {
    try {
      const h = await api("/api/health"); S.groq = h.groq_configured; S.ready = h.embedding.ready;
      const model = h.embedding.model.split("/").pop();
      el.pEmbed.className = "pill " + (h.embedding.ready ? "ok" : h.embedding.error ? "bad" : "loading");
      el.pEmbed.querySelector("span").textContent = h.embedding.ready ? `${model} ready` : h.embedding.error ? "Embedding model failed" : "Loading embedding model…";
      el.pEmbed.title = h.embedding.error || "";
      el.pGroq.className = "pill " + (h.groq_configured ? "ok" : "bad"); el.pGroq.querySelector("span").textContent = h.groq_configured ? "Groq ready" : "No Groq key";
      if (!h.embedding.ready && !h.embedding.error) setTimeout(refreshHealth, 2000);
      if (h.embedding.error) setHint(h.embedding.error, true);
    } catch { el.pEmbed.className = "pill bad"; el.pEmbed.querySelector("span").textContent = "Server offline"; }
  }

  /* ---------------- wiring ---------------- */
  el.drop.addEventListener("click", () => el.file.click());
  el.drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.file.click(); } });
  el.file.addEventListener("click", (e) => e.stopPropagation());
  el.file.addEventListener("change", () => upload(el.file.files));
  ["dragenter", "dragover"].forEach((t) => document.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((t) => document.addEventListener(t, (e) => { e.preventDefault(); if (t === "drop" || e.target === document.documentElement) el.drop.classList.remove("drag"); }));
  document.addEventListener("drop", (e) => { if (e.dataTransfer && e.dataTransfer.files.length) upload(e.dataTransfer.files); });
  $("sampleBtn").addEventListener("click", loadSample);
  $("resetBtn").addEventListener("click", async () => {
    if (!confirm("Delete every document from ChromaDB?")) return;
    try { await api("/api/documents/reset", { method: "POST" }); S.scope = null; S.messages = []; await loadDocs(); refreshHealth(); } catch (e) { setHint(e.message, true); }
  });
  el.topk.addEventListener("input", () => { el.kOut.textContent = el.topk.value; try { localStorage.setItem("atlas.k", el.topk.value); } catch { /* */ } });
  el.model.addEventListener("change", () => { S.model = el.model.value; try { localStorage.setItem("atlas.model", S.model); } catch { /* */ } });
  el.clearHits.addEventListener("click", () => { MapView.clear(); el.clearHits.hidden = true; });
  function setScope(s) { S.scope = S.scope === s || !s ? null : s; renderDocs(); }
  document.addEventListener("click", async (e) => {
    const del = e.target.closest("[data-del]");
    if (del) { e.stopPropagation(); try { await api("/api/documents/" + encodeURIComponent(del.dataset.del), { method: "DELETE" }); await loadDocs(); refreshHealth(); } catch (err) { setHint(err.message, true); } return; }
    const sc = e.target.closest("[data-scope]"); if (sc) { const v = sc.dataset.scope; S.scope = v && S.scope !== v ? v : null; renderDocs(); return; }
    if (e.target.closest("[data-pick]")) { el.file.click(); return; }
    if (e.target.closest("[data-sample]")) { loadSample(); return; }
    const idea = e.target.closest("[data-idea]"); if (idea) { ask(ideasNow()[+idea.dataset.idea][1]); return; }
    const cite = e.target.closest("[data-cite]"); if (cite) { pickSource(+cite.dataset.idx, +cite.dataset.cite); return; }
    const src = e.target.closest("[data-src]"); if (src) { pickSource(+src.dataset.idx, +src.dataset.src); return; }
    const go = e.target.closest("[data-go]");
    if (go) { document.body.dataset.view = go.dataset.go; document.querySelectorAll(".tabbar button").forEach((b) => b.classList.toggle("on", b === go)); setTimeout(() => MapView.resize(), 30); }
  });
  el.composer.addEventListener("submit", (e) => { e.preventDefault(); ask(el.input.value); });
  el.input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); ask(el.input.value); } });
  el.input.addEventListener("input", () => { autogrow(); if (el.input.value.trim()) setHint(""); });
  el.stop.addEventListener("click", () => { if (S.abort) S.abort.abort(); });
  MapView.onPick((p) => showSelected(p));

  /* ---------------- boot ---------------- */
  (async function init() {
    try { el.topk.value = localStorage.getItem("atlas.k") || 5; S.model = localStorage.getItem("atlas.model") || ""; } catch { /* */ }
    el.kOut.textContent = el.topk.value; renderFeed();
    refreshHealth();
    try {
      const m = await api("/api/models"); if (!m.models.includes(S.model)) S.model = m.default;
      el.model.innerHTML = m.models.map((x) => `<option ${x === S.model ? "selected" : ""}>${esc(x)}</option>`).join("");
    } catch { /* offline */ }
    try { await loadDocs(); } catch (e) { setHint(e.message, true); }
    MapView.resize();
  })();
})();
