/* Music Gen Studio front end: talks to webui.py, polls job progress, shows the library. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const el = {
    form: $("form"), title: $("title"), style: $("style"), lyrics: $("lyrics"), cot: $("cot"), seed: $("seed"),
    cfg: $("cfg"), abc: $("abc"), advanced: $("advanced"), generate: $("btn-generate"), formError: $("form-error"),
    composeHint: $("compose-hint"), pillModel: $("pill-model"), pillGpu: $("pill-gpu"),
    now: $("now"), nowTitle: $("now-title"), nowSub: $("now-sub"), stepper: $("stepper"), progress: $("progress"),
    bar: $("bar"), stats: $("stats"), queue: $("queue"), cancel: $("btn-cancel"),
    result: $("result"), resultTitle: $("result-title"), resultMeta: $("result-meta"), resultBadge: $("result-badge"),
    resultError: $("result-error"), player: $("player"), dlFlac: $("dl-flac"), dlWav: $("dl-wav"),
    load: $("btn-load"), editScore: $("btn-edit-score"), resultStyle: $("result-style"), scoreBlock: $("score-block"),
    sheet: $("sheet"), abcText: $("abc-text"), songs: $("songs"), libraryEmpty: $("library-empty"), libraryCount: $("library-count"),
  };
  const TOKENS_PER_SECOND = 25;
  const STAGE_STEPS = [
    [/^(Verifying|Resolving|Loading model)/, 0],
    [/^(Planning score|Using provided score)/, 1],
    [/^Generating song/, 2],
    [/^Synthesizing/, 3],
    [/^(Loading audio decoder|Decoding audio)/, 4],
  ];
  const state = { watching: new Set(), lastStep: 0, selected: null, song: null, songs: [], examples: null, timer: null, online: true };

  // ── helpers ────────────────────────────────────────────────────────────
  async function api(path, options) {
    const res = await fetch(path, options);
    if (!res.ok) {
      let message = `${res.status} ${res.statusText}`;
      try { const body = await res.json(); message = body.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : message; } catch {}
      throw new Error(message);
    }
    return res.json();
  }
  const fmtTime = (s) => { s = Math.max(0, Math.round(s || 0)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
  const fmtNum = (n) => (n == null ? "—" : Number(n).toLocaleString());
  const fmtDate = (t) => new Date(t * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
  const modeName = { full: "melody + chords", melody: "melody plan", off: "no plan" };
  const randomSeed = () => Math.floor(Math.random() * 2 ** 31);
  function setPill(pill, kind, text) { pill.className = `pill ${kind}`; pill.querySelector(".pill-text").textContent = text; }
  function showError(node, message) { node.textContent = message || ""; node.hidden = !message; }

  // ── form ───────────────────────────────────────────────────────────────
  function readForm() {
    const seed = el.seed.value.trim();
    const cfg = el.cfg.value.trim();
    return {
      title: el.title.value.trim(), style: el.style.value.trim(), lyrics: el.lyrics.value.trim(), cot: el.cot.value,
      seed: seed === "" ? null : Number(seed), cfg_scale: cfg === "" ? null : Number(cfg), abc: el.abc.value.trim() || null,
    };
  }
  function fillForm({ title, style, lyrics, cot, seed, cfg_scale, abc }) {
    if (title !== undefined) el.title.value = title || "";
    if (style !== undefined) el.style.value = style || "";
    if (lyrics !== undefined) el.lyrics.value = lyrics || "";
    if (cot !== undefined) el.cot.value = cot || "full";
    if (seed !== undefined) el.seed.value = seed == null ? "" : seed;
    if (cfg_scale !== undefined) el.cfg.value = cfg_scale == null ? "" : cfg_scale;
    if (abc !== undefined) { el.abc.value = abc || ""; if (abc) el.advanced.open = true; }
    saveDraft();
  }
  function saveDraft() { try { localStorage.setItem("yue2.draft", JSON.stringify(readForm())); } catch {} }
  function restoreDraft() {
    try { const draft = JSON.parse(localStorage.getItem("yue2.draft") || "null"); if (draft) fillForm(draft); } catch {}
  }
  el.form.addEventListener("input", saveDraft);
  el.form.addEventListener("change", saveDraft);
  $("btn-dice").addEventListener("click", () => { el.seed.value = randomSeed(); saveDraft(); });
  $("btn-clear-abc").addEventListener("click", () => { el.abc.value = ""; saveDraft(); });
  $("btn-example").addEventListener("click", async () => {
    const examples = await loadExamples();
    fillForm({ title: "City Lights", ...examples.song, abc: "", cfg_scale: null });
  });
  document.querySelectorAll("[data-score]").forEach((button) => button.addEventListener("click", async () => {
    const examples = await loadExamples();
    const name = button.dataset.score;
    fillForm({ abc: examples.scores[name] || "", cot: name === "melody" ? "melody" : "full" });
    if (!el.lyrics.value.trim()) fillForm({ title: "City Lights", style: examples.song.style, lyrics: examples.song.lyrics });
  }));
  async function loadExamples() { return state.examples || (state.examples = await api("/api/examples")); }
  el.lyrics.addEventListener("keydown", (event) => { if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) { event.preventDefault(); el.form.requestSubmit(); } });

  el.form.addEventListener("submit", async (event) => {
    event.preventDefault();
    showError(el.formError, "");
    const body = readForm();
    if (!body.style || !body.lyrics) return showError(el.formError, "Style and lyrics are both required.");
    if (body.abc && body.cot === "off") return showError(el.formError, "A supplied score needs the “Melody + chords” or “Melody only” plan mode.");
    el.generate.disabled = true;
    try {
      const job = await api("/api/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      state.watching.add(job.id);
      el.now.hidden = false;
      el.nowTitle.textContent = job.title;
      el.nowSub.textContent = "Queued";
      renderSteps(-1);
      poll();
    } catch (error) {
      showError(el.formError, error.message);
    } finally {
      el.generate.disabled = false;
    }
  });

  // ── status polling ────────────────────────────────────────────────────
  async function poll() {
    clearTimeout(state.timer);
    let status = null;
    try {
      status = await api("/api/status");
      if (!state.online) { state.online = true; loadLibrary(); }
      renderPills(status);
      renderNow(status);
      await resolveWatched(status);
    } catch {
      state.online = false;
      setPill(el.pillModel, "err", "Server offline");
    }
    const active = status && (status.current || status.queue.length || status.model.state === "loading");
    state.timer = setTimeout(poll, active ? 1000 : 4000);
  }

  function renderPills(status) {
    const m = status.model;
    if (m.state === "loading") setPill(el.pillModel, "busy", m.stage ? m.stage.label + "…" : "Loading model…");
    else if (m.state === "error") setPill(el.pillModel, "err", "Model failed to load");
    else if (status.current) setPill(el.pillModel, "busy", "Generating");
    else setPill(el.pillModel, "ok", ["Model ready", m.vram_mode === "low" ? "low-VRAM mode" : null, m.quantization === "fp8" ? "FP8" : null].filter(Boolean).join(" · "));
    if (status.gpu) setPill(el.pillGpu, "", `GPU ${status.gpu.used_gib.toFixed(1)} / ${status.gpu.total_gib.toFixed(1)} GB`);
    el.composeHint.textContent = m.state === "loading"
      ? "The model is still loading into memory; songs you submit now start as soon as it is ready."
      : m.state === "error" ? `Model failed to load: ${m.error}`
      : "About one minute of compute per minute of audio on this GPU. Songs queue one at a time.";
  }

  function stepFor(label) {
    for (const [pattern, step] of STAGE_STEPS) if (pattern.test(label)) return step;
    return state.lastStep;
  }
  function renderSteps(step) {
    el.stepper.querySelectorAll("li").forEach((li) => {
      const n = Number(li.dataset.step);
      li.classList.toggle("done", n < step);
      li.classList.toggle("active", n === step);
    });
  }
  function renderNow(status) {
    const current = status.current;
    const queue = status.queue || [];
    if (!current && !queue.length) {
      if (!state.watching.size) el.now.hidden = true;
      return;
    }
    el.now.hidden = false;
    if (current) {
      const stage = current.stage;
      const step = stage ? stepFor(stage.label) : state.lastStep;
      state.lastStep = step;
      el.nowTitle.textContent = current.title;
      el.nowSub.textContent = `Running · ${fmtTime(current.elapsed)}`;
      renderSteps(step);
      el.cancel.disabled = false;
      el.cancel.dataset.job = current.id;
      let stats = [];
      let fraction = null;
      if (stage) {
        if (stage.total) {
          fraction = stage.completed / stage.total;
          stats.push(`<b>${fmtNum(stage.completed)}</b> / ${fmtNum(stage.total)} ${stage.unit || ""}`);
          stats.push(`<b>${Math.round(fraction * 100)}%</b>`);
        } else if (stage.unit === "tokens") {
          stats.push(`<b>${fmtNum(stage.completed)}</b> tokens`);
          if (stage.rate) stats.push(`<b>${stage.rate.toFixed(1)}</b> tok/s`);
          if (step === 2) stats.push(`≈ <b>${fmtTime(stage.completed / TOKENS_PER_SECOND)}</b> of audio so far`);
        } else {
          stats.push(stage.label);
        }
        stats.push(`${fmtTime(stage.elapsed)} in this step`);
      } else {
        stats.push("Switching stages…");
      }
      el.stats.innerHTML = stats.join("<span class='sep'> · </span>");
      el.progress.classList.toggle("indeterminate", fraction == null);
      el.bar.style.width = fraction == null ? "" : `${Math.max(2, fraction * 100)}%`;
    } else {
      const next = queue[0];
      el.nowTitle.textContent = next.title;
      el.nowSub.textContent = status.model.state === "loading" ? "Queued · waiting for the model to load" : "Queued";
      renderSteps(-1);
      el.progress.classList.add("indeterminate");
      el.stats.textContent = "";
      el.cancel.disabled = false;
      el.cancel.dataset.job = next.id;
    }
    const waiting = current ? queue : queue.slice(1);
    el.queue.hidden = !waiting.length;
    el.queue.textContent = waiting.length ? `Up next: ${waiting.map((j) => j.title).join(", ")}` : "";
  }
  el.cancel.addEventListener("click", async () => {
    const id = el.cancel.dataset.job;
    if (!id) return;
    el.cancel.disabled = true;
    try { await api(`/api/jobs/${id}/cancel`, { method: "POST" }); } catch (error) { alert(error.message); }
  });

  async function resolveWatched(status) {
    const live = new Set([status.current && status.current.id, ...(status.queue || []).map((j) => j.id)].filter(Boolean));
    if (status.current) state.watching.add(status.current.id);   // page reloaded mid-generation
    for (const id of [...state.watching]) {
      if (live.has(id)) continue;
      state.watching.delete(id);
      let job = null;
      try { job = await api(`/api/jobs/${id}`); } catch { continue; }
      if (job.state === "done") {
        await loadLibrary();
        await openSong(job.id);
      } else if (job.state === "failed") {
        showResultError(job.title, job.error || "Generation failed.");
      } else if (job.state === "cancelled") {
        showResultError(job.title, "Cancelled.", "warn");
      }
    }
    if (!state.watching.size && !status.current && !(status.queue || []).length) el.now.hidden = true;
  }

  // ── result ─────────────────────────────────────────────────────────────
  function showResultError(title, message, kind = "err") {
    el.result.hidden = false;
    el.resultTitle.textContent = title;
    el.resultMeta.textContent = "";
    el.resultBadge.className = `badge ${kind}`;
    el.resultBadge.textContent = kind === "warn" ? "Cancelled" : "Failed";
    showError(el.resultError, message);
    el.player.removeAttribute("src");
    el.player.hidden = true;
    el.scoreBlock.hidden = true;
    el.resultStyle.textContent = "";
    [el.dlFlac, el.dlWav, el.load, el.editScore].forEach((node) => (node.hidden = true));
    state.song = null;
  }
  async function openSong(id, { scroll = true } = {}) {
    const song = await api(`/api/songs/${id}`);
    state.song = song;
    state.selected = id;
    el.result.hidden = false;
    showError(el.resultError, "");
    el.resultTitle.textContent = song.title;
    const truncated = song.truncated && (song.truncated.abc || song.truncated.semantic);
    el.resultBadge.className = `badge ${truncated ? "warn" : "ok"}`;
    el.resultBadge.textContent = truncated ? "Hit length limit" : "Complete";
    const meta = [fmtTime(song.seconds), modeName[song.cot] || song.cot, `seed ${song.seed}`];
    if (song.elapsed) meta.push(`generated in ${fmtTime(song.elapsed)}`);
    meta.push(fmtDate(song.created));
    el.resultMeta.textContent = meta.join(" · ");
    el.resultStyle.textContent = song.style;
    el.player.hidden = false;
    el.player.src = song.audio_url;
    el.dlFlac.href = song.audio_url; el.dlFlac.download = `${song.id}.flac`;
    el.dlWav.href = `/api/songs/${song.id}/audio.wav`; el.dlWav.download = `${song.id}.wav`;
    [el.dlFlac, el.dlWav, el.load].forEach((node) => (node.hidden = false));
    el.editScore.hidden = !song.score;
    renderScore(song.score);
    renderLibrary();
    if (scroll) el.result.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
  function renderScore(abc) {
    el.scoreBlock.hidden = !abc;
    if (!abc) return;
    el.abcText.textContent = abc;
    el.sheet.innerHTML = "";
    let rendered = false;
    if (window.ABCJS) {
      try {
        ABCJS.renderAbc(el.sheet, abc, { responsive: "resize", add_classes: true, paddingtop: 0, paddingbottom: 0 });
        rendered = el.sheet.querySelector("svg") != null;
      } catch (error) { console.warn("abcjs could not render this score:", error); }
    }
    selectTab(rendered ? "sheet" : "abc");
    document.querySelector(".tab[data-tab='sheet']").disabled = !rendered;
  }
  function selectTab(name) {
    document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
    el.sheet.hidden = name !== "sheet";
    el.abcText.hidden = name !== "abc";
  }
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => selectTab(tab.dataset.tab)));
  el.load.addEventListener("click", () => {
    const { request, title } = state.song;
    fillForm({ title, style: request.style, lyrics: request.lyrics, cot: request.cot, seed: request.seed, cfg_scale: request.cfg_scale ?? null, abc: request.abc || "" });
    el.style.scrollIntoView({ behavior: "smooth", block: "start" });
  });
  el.editScore.addEventListener("click", () => {
    const { request, title, score } = state.song;
    fillForm({ title: `${title} (edited)`, style: request.style, lyrics: request.lyrics, cot: request.cot === "off" ? "full" : request.cot, seed: request.seed, abc: score });
    el.abc.scrollIntoView({ behavior: "smooth", block: "center" });
    el.abc.focus();
  });

  // ── library ────────────────────────────────────────────────────────────
  async function loadLibrary() {
    try { state.songs = await api("/api/songs"); } catch { return; }
    renderLibrary();
  }
  function renderLibrary() {
    el.songs.innerHTML = "";
    el.libraryEmpty.hidden = state.songs.length > 0;
    el.libraryCount.textContent = state.songs.length ? `${state.songs.length} song${state.songs.length === 1 ? "" : "s"}` : "";
    for (const song of state.songs) {
      const li = document.createElement("li");
      li.className = `song${song.id === state.selected ? " active" : ""}`;
      li.innerHTML = `<div class="song-icon">♪</div>
        <div class="song-main"><div class="song-title"></div><div class="song-style"></div></div>
        <div class="song-meta"><div>${fmtTime(song.seconds)}</div><div>${new Date(song.created * 1000).toLocaleDateString()}</div></div>`;
      li.querySelector(".song-title").textContent = song.title;
      li.querySelector(".song-style").textContent = song.style;
      li.addEventListener("click", () => openSong(song.id).catch((error) => alert(error.message)));
      el.songs.appendChild(li);
    }
  }

  // ── boot ───────────────────────────────────────────────────────────────
  restoreDraft();
  loadLibrary().then(() => { if (state.songs.length && el.result.hidden) openSong(state.songs[0].id, { scroll: false }).catch(() => {}); });
  poll();
})();
