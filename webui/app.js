/* Music Gen Studio front end: talks to webui.py, polls job progress, shows the library. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const el = {
    form: $("form"), title: $("title"), style: $("style"), lyrics: $("lyrics"), cot: $("cot"), seed: $("seed"), takes: $("takes"),
    cfg: $("cfg"), ode: $("ode"), abc: $("abc"), abcInfo: $("abc-info"), advanced: $("advanced"), cover: $("cover"),
    coverFile: $("cover-file"), coverMelody: $("cover-melody"), coverHint: $("cover-hint"), transcribe: $("btn-transcribe"),
    coverLyrics: $("cover-lyrics"), coverLanguage: $("cover-language"),
    generate: $("btn-generate"), plan: $("btn-plan"), formError: $("form-error"), composeHint: $("compose-hint"),
    pillModel: $("pill-model"), pillGpu: $("pill-gpu"), pillShare: $("pill-share"), pillDoctor: $("pill-doctor"),
    pillUser: $("pill-user"), pillSettings: $("pill-settings"),
    now: $("now"), nowTitle: $("now-title"), nowSub: $("now-sub"), stepper: $("stepper"), progress: $("progress"),
    bar: $("bar"), stats: $("stats"), queue: $("queue"), cancel: $("btn-cancel"),
    planCard: $("plan"), planTitle: $("plan-title"), planMeta: $("plan-meta"), planSheet: $("plan-sheet"), planAbc: $("plan-abc"),
    renderPlan: $("btn-render-plan"), planEdit: $("btn-plan-edit"), planDiscard: $("btn-plan-discard"),
    result: $("result"), resultTitle: $("result-title"), resultMeta: $("result-meta"), resultBadge: $("result-badge"),
    resultError: $("result-error"), player: $("player"), altBlock: $("alt-block"), altPlayer: $("alt-player"),
    dlFlac: $("dl-flac"), dlWav: $("dl-wav"), newTake: $("btn-new-take"), restyle: $("btn-restyle"), redecode: $("btn-redecode"),
    load: $("btn-load"), editScore: $("btn-edit-score"), del: $("btn-delete"), resultStyle: $("result-style"), shareSong: $("btn-share-song"),
    scoreBlock: $("score-block"), sheet: $("sheet"), abcText: $("abc-text"),
    compare: $("compare"), compareGrid: $("compare-grid"), compareClose: $("btn-compare-close"), compareBtn: $("btn-compare"),
    songs: $("songs"), libraryEmpty: $("library-empty"), libraryCount: $("library-count"),
    libSearch: $("lib-search"), libFav: $("lib-fav"), libMine: $("lib-mine"),
    voiceBlock: $("voice-block"), voiceSelect: $("voice-select"), voiceSemitones: $("voice-semitones"), voiceBtn: $("btn-voice"),
    voiceHarmony: $("voice-harmony"), voiceDuet: $("voice-duet"), voiceDuetSections: $("voice-duet-sections"),
    voicesManage: $("btn-voices-manage"), voiceHint: $("voice-hint"), voiceVersions: $("voice-versions"),
    voicesOverlay: $("voices-overlay"), voicesClose: $("voices-close"), voiceName: $("voice-name"), voiceFile: $("voice-file"),
    voiceUpload: $("voice-upload"), voicesHint: $("voices-hint"), voicesList: $("voices-list"),
    describeText: $("describe-text"), describeBtn: $("btn-describe"), describeGo: $("btn-describe-go"), describeVoice: $("describe-voice"), describeHint: $("describe-hint"),
    presetSelect: $("preset-select"), presetSave: $("btn-preset-save"), presetRemove: $("btn-preset-remove"), stylesBtn: $("btn-styles"),
    stylesOverlay: $("styles-overlay"), stylesClose: $("styles-close"), stylesList: $("styles-list"), stylesGo: $("styles-go"), stylesHint: $("styles-hint"),
    metaBlock: $("meta-block"), fav: $("btn-fav"), rating: $("rating"), tags: $("tags"), notes: $("notes"), metaOwner: $("meta-owner"), family: $("family"),
    karaokeBlock: $("karaoke-block"), karaokeBtn: $("btn-karaoke"), dlLrc: $("dl-lrc"), dlSrt: $("dl-srt"), karaokeHint: $("karaoke-hint"), lyricsSync: $("lyrics-sync"),
    stemsBlock: $("stems-block"), stemsBtn: $("btn-stems"), stemsHint: $("stems-hint"), stemList: $("stem-list"),
    exportBlock: $("export-block"), exportSource: $("export-source"), exportFormat: $("export-format"), exportMaster: $("export-master"),
    exportFade: $("export-fade"), exportTrim: $("export-trim"), exportBtn: $("btn-export"), exportHint: $("export-hint"), exportList: $("export-list"),
    songShareOverlay: $("song-share-overlay"), songShareClose: $("song-share-close"), songShareDays: $("song-share-days"), songShareQr: $("song-share-qr"),
    songShareUrl: $("song-share-url"), songShareCopy: $("song-share-copy"), songShareOpen: $("song-share-open"),
    settingsOverlay: $("settings-overlay"), settingsClose: $("settings-close"), setName: $("set-name"), setDiscord: $("set-discord"),
    setTgToken: $("set-tg-token"), setTgChat: $("set-tg-chat"), setFailed: $("set-failed"), settingsSave: $("settings-save"), settingsTest: $("settings-test"), settingsHint: $("settings-hint"),
  };
  const TOKENS_PER_SECOND = 25;
  const STAGE_STEPS = [
    [/^(Verifying|Resolving|Loading model)/, 0],
    [/^(Planning score|Using provided score)/, 1],
    [/^Generating song/, 2],
    [/^Synthesizing/, 3],
    [/^(Loading audio decoder|Decoding audio)/, 4],
  ];
  const KIND_LABEL = { plan: "Planning", decode: "Decoding", transcribe: "Transcribing", voice: "Re-singing", stems: "Splitting stems", karaoke: "Syncing lyrics" };
  const KIND_SUB = { plan: "Planning the score", decode: "Re-decoding", transcribe: "Transcribing with SheetSage2", voice: "Singing it in another voice",
                     stems: "Splitting into stems with Demucs", karaoke: "Timing the lyrics with Whisper" };
  const KIND_WAIT = { transcribe: "SheetSage2 is listening…", voice: "Splitting the vocal off and re-singing it (about a minute per voice)…",
                      stems: "Demucs is separating the stems (about 10 s)…", karaoke: "Whisper is listening for the words (about 15 s)…" };
  const state = { watching: new Set(), lastStep: 0, selected: null, song: null, songs: [], examples: null, timer: null,
                  online: true, status: null, plan: null, scoreSource: null, compare: new Set(), sections: null,
                  voices: null, voiceReady: false, presets: { builtin: [], user: [] }, origin: null, karaoke: null,
                  filters: { search: "", fav: false, mine: false }, metaTimer: null };

  // ── helpers ────────────────────────────────────────────────────────────
  const userName = () => { try { return (localStorage.getItem("mgs.name") || "").trim(); } catch { return ""; } };
  async function api(path, options = {}) {
    const headers = Object.assign({}, options.headers || {});
    const name = userName();
    if (name) headers["X-User"] = encodeURIComponent(name).replace(/%20/g, " ").slice(0, 80);
    const res = await fetch(path, { ...options, headers });
    if (!res.ok) {
      let message = `${res.status} ${res.statusText}`;
      try { const body = await res.json(); message = body.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : message; } catch {}
      throw new Error(message);
    }
    return res.json();
  }
  const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const patch = (path, body) => api(path, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const put = (path, body) => api(path, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const fmtTime = (s) => { s = Math.max(0, Math.round(s || 0)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
  const fmtNum = (n) => (n == null ? "—" : Number(n).toLocaleString());
  const fmtDate = (t) => new Date(t * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
  const fmtBytes = (b) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.round(b / 1e3)} kB`);
  const modeName = { full: "melody + chords", melody: "melody plan", off: "no plan" };
  const randomSeed = () => Math.floor(Math.random() * 2 ** 31);
  function setPill(pill, kind, text) { pill.className = `pill ${kind}`; pill.querySelector(".pill-text").textContent = text; }
  function showError(node, message) { node.textContent = message || ""; node.hidden = !message; }
  function numberOrNull(input) { const v = input.value.trim(); return v === "" ? null : Number(v); }
  function watchJob(job, subtitle) {
    for (const j of job.jobs || [job]) state.watching.add(j.id);
    el.now.hidden = false; el.nowTitle.textContent = job.title; el.nowSub.textContent = subtitle || "Queued"; renderSteps(-1);
    poll();
  }

  // ── form ───────────────────────────────────────────────────────────────
  function readSampling(prefix) {
    const out = {};
    for (const key of ["temperature", "top_p", "top_k", "repetition_penalty", "max_tokens"]) {
      const v = numberOrNull($(`s-${prefix}-${key}`));
      if (v != null) out[key] = v;
    }
    return Object.keys(out).length ? out : null;
  }
  function readForm() {
    return {
      title: el.title.value.trim(), style: el.style.value.trim(), lyrics: el.lyrics.value.trim(), cot: el.cot.value,
      seed: numberOrNull(el.seed), cfg_scale: numberOrNull(el.cfg), abc: el.abc.value.trim() || null,
      takes: Number(el.takes.value), ode_steps: numberOrNull(el.ode),
      abc_sampling: readSampling("abc"), semantic_sampling: readSampling("semantic"),
      origin: state.origin,
    };
  }
  function fillForm({ title, style, lyrics, cot, seed, cfg_scale, abc, origin }) {
    if (title !== undefined) el.title.value = title || "";
    if (style !== undefined) el.style.value = style || "";
    if (lyrics !== undefined) el.lyrics.value = lyrics || "";
    if (cot !== undefined) el.cot.value = cot || "full";
    if (seed !== undefined) el.seed.value = seed == null ? "" : seed;
    if (cfg_scale !== undefined) el.cfg.value = cfg_scale == null ? "" : cfg_scale;
    if (abc !== undefined) { el.abc.value = abc || ""; if (abc) el.advanced.open = true; }
    if (origin !== undefined) state.origin = origin || null;
    saveDraft();
  }
  function saveDraft() { try { localStorage.setItem("yue2.draft", JSON.stringify({ ...readForm(), scoreSource: state.scoreSource })); } catch {} }
  function restoreDraft() {
    try {
      const draft = JSON.parse(localStorage.getItem("yue2.draft") || "null");
      if (!draft) return;
      fillForm(draft);
      if (draft.takes) el.takes.value = draft.takes;
      if (draft.ode_steps != null) el.ode.value = draft.ode_steps;
      for (const prefix of ["abc", "semantic"]) for (const [k, v] of Object.entries(draft[`${prefix}_sampling`] || {})) $(`s-${prefix}-${k}`).value = v;
      state.scoreSource = draft.scoreSource || null;
    } catch {}
  }
  el.form.addEventListener("input", saveDraft);
  el.form.addEventListener("change", saveDraft);
  $("btn-dice").addEventListener("click", () => { el.seed.value = randomSeed(); saveDraft(); });
  $("btn-clear-abc").addEventListener("click", () => { el.abc.value = ""; state.scoreSource = null; el.abcInfo.hidden = true; saveDraft(); });
  $("btn-example").addEventListener("click", async () => {
    const examples = await loadExamples();
    fillForm({ title: "City Lights", ...examples.song, abc: "", cfg_scale: null, origin: null });
  });
  document.querySelectorAll("[data-score]").forEach((button) => button.addEventListener("click", async () => {
    const examples = await loadExamples();
    const name = button.dataset.score;
    state.scoreSource = null;
    fillForm({ abc: examples.scores[name] || "", cot: name === "melody" ? "melody" : "full" });
    if (!el.lyrics.value.trim()) fillForm({ title: "City Lights", style: examples.song.style, lyrics: examples.song.lyrics });
  }));
  async function loadExamples() { return state.examples || (state.examples = await api("/api/examples")); }
  el.lyrics.addEventListener("keydown", (event) => { if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) { event.preventDefault(); el.form.requestSubmit(); } });

  // ── "just describe it" (local Ollama model) ───────────────────────────
  async function draftFromDescription() {
    const text = el.describeText.value.trim();
    if (text.length < 3) { el.describeHint.textContent = "Describe the song in a sentence or two first."; return null; }
    el.describeBtn.disabled = el.describeGo.disabled = true;
    el.describeHint.textContent = "Drafting a title, style and lyrics… (20-60 s on the local model)";
    try {
      const draft = await post("/api/describe", { text });
      fillForm({ title: draft.title, style: draft.style, lyrics: draft.lyrics, cot: "full", seed: null, abc: "", origin: null });
      state.scoreSource = null;
      el.describeHint.textContent = `Drafted by ${draft.model} in ${draft.seconds}s — read the lyrics over, then Generate (or edit anything first).`;
      return draft;
    } catch (error) { el.describeHint.textContent = error.message; return null; }
    finally { el.describeBtn.disabled = el.describeGo.disabled = false; }
  }
  el.describeBtn.addEventListener("click", draftFromDescription);
  el.describeGo.addEventListener("click", async () => {
    const draft = await draftFromDescription();
    if (!draft) return;
    const body = { ...readForm(), takes: 1 };
    if (el.describeVoice.value) body.then_voice = el.describeVoice.value;
    submitJobs("/api/generate", body, el.describeGo);
  });

  // ── style presets ─────────────────────────────────────────────────────
  async function loadPresets() {
    try { state.presets = await api("/api/presets"); } catch { return; }
    const current = el.presetSelect.value;
    el.presetSelect.innerHTML = '<option value="">presets…</option>';
    const groups = [["Mine", state.presets.user, "u"], ["Built-in", state.presets.builtin, "b"]];
    for (const [label, list, prefix] of groups) {
      if (!list.length) continue;
      const group = document.createElement("optgroup"); group.label = label;
      list.forEach((preset, i) => { const option = document.createElement("option"); option.value = `${prefix}${i}`; option.textContent = preset.name; group.appendChild(option); });
      el.presetSelect.appendChild(group);
    }
    if ([...el.presetSelect.options].some((o) => o.value === current)) el.presetSelect.value = current;
    el.presetRemove.hidden = !el.presetSelect.value.startsWith("u");
  }
  const presetOf = (value) => (value ? (value[0] === "u" ? state.presets.user : state.presets.builtin)[Number(value.slice(1))] : null);
  el.presetSelect.addEventListener("change", () => {
    const preset = presetOf(el.presetSelect.value);
    if (preset) { fillForm({ style: preset.style }); }
    el.presetRemove.hidden = !el.presetSelect.value.startsWith("u");
  });
  el.presetSave.addEventListener("click", async () => {
    const style = el.style.value.trim();
    if (!style) { alert("Write a style first."); return; }
    const name = prompt("Name for this style preset:", style.split(",").slice(0, 2).join(",").slice(0, 40));
    if (!name) return;
    const user = state.presets.user.filter((p) => p.name !== name.trim()).concat([{ name: name.trim(), style }]);
    try { await put("/api/presets", { presets: user }); await loadPresets(); el.presetSelect.value = `u${user.length - 1}`; el.presetRemove.hidden = false; }
    catch (error) { alert(error.message); }
  });
  el.presetRemove.addEventListener("click", async () => {
    const value = el.presetSelect.value;
    if (!value.startsWith("u")) return;
    const user = state.presets.user.filter((_, i) => i !== Number(value.slice(1)));
    try { await put("/api/presets", { presets: user }); await loadPresets(); el.presetSelect.value = ""; el.presetRemove.hidden = true; }
    catch (error) { alert(error.message); }
  });
  el.stylesBtn.addEventListener("click", async () => {
    await loadPresets();
    el.stylesList.innerHTML = "";
    for (const [prefix, list] of [["u", state.presets.user], ["b", state.presets.builtin]]) {
      list.forEach((preset, i) => {
        const li = document.createElement("li");
        li.innerHTML = `<label class="check"><input type="checkbox"><span class="name"></span></label><span class="grow"></span>`;
        li.querySelector("input").value = `${prefix}${i}`;
        li.querySelector(".name").textContent = preset.name;
        li.querySelector(".grow").textContent = preset.style;
        el.stylesList.appendChild(li);
      });
    }
    el.stylesHint.textContent = el.lyrics.value.trim() ? "" : "Write (or draft) lyrics first — every style uses the same title and lyrics.";
    el.stylesOverlay.hidden = false;
  });
  el.stylesClose.addEventListener("click", () => { el.stylesOverlay.hidden = true; });
  el.stylesGo.addEventListener("click", async () => {
    const picked = [...el.stylesList.querySelectorAll("input:checked")].map((box) => presetOf(box.value)).filter(Boolean);
    const base = readForm();
    if (!picked.length) { el.stylesHint.textContent = "Tick at least one style."; return; }
    if (!base.lyrics) { el.stylesHint.textContent = "Write (or draft) lyrics first."; return; }
    el.stylesGo.disabled = true;
    let first = null, count = 0;
    try {
      for (const preset of picked) {
        const title = `${base.title || "song"} (${preset.name})`;
        const job = await post("/api/generate", { ...base, title, style: preset.style, takes: 1, seed: null, abc: null, origin: state.song ? { song_id: state.song.id, kind: "styles" } : null });
        state.watching.add(job.id); first = first || job; count++;
      }
      el.stylesOverlay.hidden = true;
      if (first) watchJob(first, `Queued · ${count} style${count === 1 ? "" : "s"}`);
    } catch (error) { el.stylesHint.textContent = error.message; }
    finally { el.stylesGo.disabled = false; }
  });

  // ── lyric writing (local Ollama model) ───────────────────────────────
  const lyricButtons = [$("btn-lyrics-continue"), $("btn-lyrics-write")];
  function lyricHint(text) { $("lyrics-hint").textContent = text; }
  async function askLyrics(task) {
    const body = { task, lyrics: el.lyrics.value.trim(), style: el.style.value.trim(), title: el.title.value.trim(), section: $("lyrics-section").value || null };
    if (task === "continue" && !body.lyrics) return lyricHint("Write a first section (or use Write from title) before continuing.");
    if (task === "write" && !body.title && !body.style) return lyricHint("Give a title or a style first.");
    if (task === "write" && body.lyrics && !confirm("Replace the current lyrics with newly written ones? (undo is available)")) return;
    lyricButtons.forEach((b) => (b.disabled = true));
    lyricHint(task === "continue" ? "Writing the next section…" : "Writing lyrics…");
    try {
      const reply = await post("/api/lyrics", body);
      state.lyricsUndo = el.lyrics.value;
      el.lyrics.value = task === "continue" ? `${el.lyrics.value.trimEnd()}\n\n${reply.text}` : reply.text;
      saveDraft();
      $("btn-lyrics-undo").hidden = false;
      lyricHint(`Written by ${reply.model} on the ${reply.device} in ${reply.seconds}s — read it over, the model can mishear the rhythm.`);
      el.lyrics.scrollTop = el.lyrics.scrollHeight;
    } catch (error) { lyricHint(error.message); }
    finally { lyricButtons.forEach((b) => (b.disabled = false)); }
  }
  $("btn-lyrics-continue").addEventListener("click", () => askLyrics("continue"));
  $("btn-lyrics-write").addEventListener("click", () => askLyrics("write"));
  $("btn-lyrics-undo").addEventListener("click", () => {
    if (state.lyricsUndo == null) return;
    el.lyrics.value = state.lyricsUndo; state.lyricsUndo = null; $("btn-lyrics-undo").hidden = true; saveDraft(); lyricHint("Restored.");
  });

  function validate(body) {
    if (!body.style || !body.lyrics) return "Style and lyrics are both required.";
    if (body.abc && body.cot === "off") return "A supplied score needs the “Melody + chords” or “Melody only” plan mode.";
    return null;
  }
  async function submitJobs(path, body, button) {
    showError(el.formError, "");
    const problem = validate(body);
    if (problem) return showError(el.formError, problem);
    button.disabled = true;
    try {
      const reply = await post(path, body);
      watchJob(reply);
    } catch (error) {
      showError(el.formError, error.message);
    } finally {
      button.disabled = false;
    }
  }
  el.form.addEventListener("submit", (event) => { event.preventDefault(); submitJobs("/api/generate", readForm(), el.generate); });
  el.plan.addEventListener("click", () => {
    const body = { ...readForm(), takes: 1 };
    if (body.cot === "off") return showError(el.formError, "Planning a score needs the “Melody + chords” or “Melody only” mode.");
    if (body.abc) return showError(el.formError, "The composer already holds a score — clear it to plan a new one, or just Generate to render it.");
    submitJobs("/api/plan", body, el.plan);
  });

  // ── score tools ─────────────────────────────────────────────────────────
  function showInfo(text) { el.abcInfo.textContent = text; el.abcInfo.hidden = !text; }
  document.querySelectorAll("#abc-tools [data-tool]").forEach((button) => button.addEventListener("click", async () => {
    const abc = el.abc.value.trim();
    if (!abc) return showInfo("The score box is empty.");
    const tool = button.dataset.tool;
    const body = { action: tool, abc };
    if (tool === "keep-voice") body.voice = button.dataset.voice;
    if (tool === "transpose") body.semitones = button.dataset.semitones === "+" ? Number($("semitones").value || 0) : Number(button.dataset.semitones);
    if (tool === "tempo") { body.bpm = numberOrNull($("bpm")); if (!body.bpm) return showInfo("Enter a BPM first."); }
    if (tool === "compare") { if (!state.scoreSource) return showInfo("Nothing to compare with: load a score with “Edit score” first."); body.action = "compare"; body.abc = state.scoreSource; body.abc2 = abc; }
    button.disabled = true;
    try {
      const reply = await post("/api/abc/tool", body);
      if (reply.abc !== undefined) { el.abc.value = reply.abc; saveDraft(); showInfo(`${button.textContent.trim()}: done.`); }
      else if (tool === "inspect") {
        const v = reply.voices || {};
        showInfo([`tempo ${reply.bpm} BPM · ${reply.nominal_duration_seconds ? fmtTime(reply.nominal_duration_seconds) : "?"} nominal`,
                  ...Object.entries(v).map(([name, d]) => `${name}: ${d.sounding_notes} notes over ${d.measures} bars${d.keys ? " · key " + d.keys.map((k) => k[1]).join(", ") : ""}`)].join("\n"));
      } else if (tool === "compare") showInfo((reply.ok ? "Invariants hold.\n" : "Differences found.\n") + reply.report);
    } catch (error) { showInfo(error.message); }
    finally { button.disabled = false; }
  }));

  // sections editor (client-side: sections are '% name' blocks of Vocal/Ins line pairs)
  function parseSections(abc) {
    const lines = abc.replace(/\r/g, "").split("\n");
    const first = lines.findIndex((l) => l.startsWith("%"));
    if (first < 0) return null;
    const header = lines.slice(0, first);
    const sections = [];
    let current = null;
    for (const line of lines.slice(first)) {
      if (line.startsWith("%")) { current = { name: line.replace(/^%\s*/, ""), pairs: [] }; sections.push(current); }
      else if (current && line.trim()) {
        if (line.trim() === "V: Vocal") current.pairs.push([line]);
        else if (current.pairs.length) current.pairs[current.pairs.length - 1].push(line);
      }
    }
    return { header, sections };
  }
  const barsOf = (section) => section.pairs.reduce((n, pair) => n + (pair[3] || "").split("|").length - 1, 0);
  function secondsPerBar(header) {
    const q = header.find((l) => l.startsWith("Q:")), m = header.find((l) => l.startsWith("M:"));
    const bpm = q ? Number((q.match(/=(\d+)/) || [])[1]) : 120;
    const [num, den] = m ? m.slice(2).split("/").map(Number) : [4, 4];
    return num * (4 / den) * 60 / bpm;
  }
  function renderSections() {
    const { header, sections } = state.sections;
    const spb = secondsPerBar(header);
    const list = $("sections-list"); list.innerHTML = "";
    sections.forEach((section, i) => {
      const li = document.createElement("li");
      li.innerHTML = `<span class="sec-name"></span><span class="sec-bars"></span>
        <span class="sec-actions"><button type="button" data-act="up" title="Move up">↑</button><button type="button" data-act="down" title="Move down">↓</button>
        <button type="button" data-act="dup" title="Repeat this section">⧉</button><button type="button" data-act="half" title="Keep only the second half">½</button><button type="button" data-act="del" title="Delete">✕</button></span>`;
      li.querySelector(".sec-name").textContent = section.name;
      li.querySelector(".sec-bars").textContent = `${barsOf(section)} bars · ${fmtTime(barsOf(section) * spb)}`;
      li.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
        const act = b.dataset.act;
        if (act === "up" && i > 0) [sections[i - 1], sections[i]] = [sections[i], sections[i - 1]];
        if (act === "down" && i < sections.length - 1) [sections[i + 1], sections[i]] = [sections[i], sections[i + 1]];
        if (act === "dup") sections.splice(i + 1, 0, { name: section.name, pairs: section.pairs.map((p) => [...p]) });
        if (act === "half" && section.pairs.length > 1) section.pairs = section.pairs.slice(Math.floor(section.pairs.length / 2));
        if (act === "del") sections.splice(i, 1);
        renderSections();
      }));
      list.appendChild(li);
    });
    const total = sections.reduce((n, s) => n + barsOf(s), 0);
    $("sections-total").textContent = `${total} bars · about ${fmtTime(total * spb)}`;
  }
  function tagsFromScore(abc) {
    const parsed = parseSections(abc);
    if (!parsed || !parsed.sections.length) return null;
    return parsed.sections.map((s) => `[${s.name.charAt(0).toUpperCase() + s.name.slice(1)} - instrumental, no vocals]`).join("\n\n");
  }
  $("btn-tags").addEventListener("click", () => {
    const tags = tagsFromScore(el.abc.value);
    if (!tags) return showInfo("No '% section' markers found in this score.");
    fillForm({ lyrics: tags });
    showInfo("Lyrics box filled with instrumental section tags — set a style and Generate for an instrumental version of this score.");
  });
  $("btn-sections").addEventListener("click", () => {
    const parsed = parseSections(el.abc.value);
    if (!parsed || !parsed.sections.length) return showInfo("No '% section' markers found in this score.");
    state.sections = parsed;
    renderSections();
    $("sections-overlay").hidden = false;
  });
  $("sections-close").addEventListener("click", () => { $("sections-overlay").hidden = true; });
  $("sections-apply").addEventListener("click", () => {
    const { header, sections } = state.sections;
    const body = sections.flatMap((s) => [`% ${s.name}`, ...s.pairs.flat()]);
    el.abc.value = [...header, ...body].join("\n") + "\n";
    saveDraft();
    $("sections-overlay").hidden = true;
    showInfo("Sections applied — press Check to validate.");
  });

  // cover: upload a recording for transcription
  el.transcribe.addEventListener("click", async () => {
    const file = el.coverFile.files[0];
    if (!file) { el.coverHint.textContent = "Choose an audio file first."; return; }
    const data = new FormData();
    data.append("file", file); data.append("melody_only", el.coverMelody.checked ? "true" : "false"); data.append("title", file.name.replace(/\.[^.]+$/, ""));
    data.append("lyrics", el.coverLyrics.checked ? "true" : "false"); data.append("language", el.coverLanguage.value.trim());
    el.transcribe.disabled = true; el.coverHint.textContent = `Uploading ${file.name}…`;
    try {
      const job = await api("/api/transcribe", { method: "POST", body: data });
      el.coverHint.textContent = "Transcribing with SheetSage2 — about a minute for a 3-minute song" + (el.coverLyrics.checked ? ", then Whisper writes down the words" : "") + ". The score appears in the box above when done.";
      watchJob(job);
    } catch (error) { el.coverHint.textContent = error.message; }
    finally { el.transcribe.disabled = false; }
  });

  // ── status polling ────────────────────────────────────────────────────
  async function poll() {
    clearTimeout(state.timer);
    let status = null;
    try {
      status = await api("/api/status");
      state.status = status;
      if (!state.online) { state.online = true; loadLibrary(); }
      renderPills(status);
      renderNow(status);
      await resolveWatched(status);
    } catch (error) {
      state.online = false;
      const locked = /^401\b/.test(error.message);
      setPill(el.pillModel, "err", locked ? "Signed out" : "Server offline");
      if (locked) { location.href = "/login"; return; }
    }
    const active = status && (status.current || status.queue.length || status.model.state === "loading");
    state.timer = setTimeout(poll, active ? 1000 : 4000);
  }

  function renderPills(status) {
    const m = status.model;
    if (m.state === "loading") setPill(el.pillModel, "busy", m.stage ? m.stage.label + "…" : "Loading model…");
    else if (m.state === "error") setPill(el.pillModel, "err", "Model failed to load");
    else if (status.current) setPill(el.pillModel, "busy", KIND_LABEL[status.current.kind] || "Generating");
    else setPill(el.pillModel, "ok", ["Model ready", m.vram_mode === "low" ? "low-VRAM mode" : null, m.quantization === "fp8" ? "FP8" : null].filter(Boolean).join(" · "));
    if (status.gpu) setPill(el.pillGpu, "", `GPU ${status.gpu.used_gib.toFixed(1)} / ${status.gpu.total_gib.toFixed(1)} GB`);
    const share = status.share_urls || [];
    el.pillShare.hidden = !share.length;
    if (share.length) { el.pillShare.querySelector(".pill-text").textContent = "share: " + share[0].replace(/^https?:[/][/]/, ""); el.pillShare.dataset.url = share[0]; }
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
  const byLine = (job) => job.owner ? `${job.title} (${job.owner})` : job.title;
  function renderNow(status) {
    const current = status.current;
    const queue = status.queue || [];
    if (!current && !queue.length) { if (!state.watching.size) el.now.hidden = true; return; }
    el.now.hidden = false;
    if (current) {
      const kind = current.kind || "song";
      const stage = current.stage;
      el.stepper.hidden = kind !== "song";
      el.nowTitle.textContent = current.title;
      el.nowSub.textContent = `${KIND_SUB[kind] || "Running"}${current.owner ? ` · queued by ${current.owner}` : ""} · ${fmtTime(current.elapsed)}`;
      el.cancel.disabled = false;
      el.cancel.dataset.job = current.id;
      let stats = [], fraction = null;
      if (kind === "song") {
        const step = stage ? stepFor(stage.label) : state.lastStep;
        state.lastStep = step;
        renderSteps(step);
      }
      if (stage) {
        if (stage.total) {
          fraction = stage.completed / stage.total;
          stats.push(`<b>${fmtNum(stage.completed)}</b> / ${fmtNum(stage.total)} ${stage.unit || ""}`, `<b>${Math.round(fraction * 100)}%</b>`);
        } else if (stage.unit === "tokens") {
          stats.push(`<b>${fmtNum(stage.completed)}</b> tokens`);
          if (stage.rate) stats.push(`<b>${stage.rate.toFixed(1)}</b> tok/s`);
          if (/^Generating song/.test(stage.label)) stats.push(`≈ <b>${fmtTime(stage.completed / TOKENS_PER_SECOND)}</b> of audio so far`);
        } else stats.push(stage.label);
        stats.push(`${fmtTime(stage.elapsed)} in this step`);
      } else stats.push(KIND_WAIT[kind] || "Switching stages…");
      el.stats.innerHTML = stats.join("<span class='sep'> · </span>");
      el.progress.classList.toggle("indeterminate", fraction == null);
      el.bar.style.width = fraction == null ? "" : `${Math.max(2, fraction * 100)}%`;
    } else {
      const next = queue[0];
      el.stepper.hidden = false;
      el.nowTitle.textContent = next.title;
      el.nowSub.textContent = (status.model.state === "loading" ? "Queued · waiting for the model to load" : "Queued") + (next.owner ? ` · by ${next.owner}` : "");
      renderSteps(-1);
      el.progress.classList.add("indeterminate");
      el.stats.textContent = "";
      el.cancel.disabled = false;
      el.cancel.dataset.job = next.id;
    }
    const waiting = current ? queue : queue.slice(1);
    el.queue.hidden = !waiting.length;
    el.queue.textContent = waiting.length ? `Up next: ${waiting.map(byLine).join(", ")}` : "";
  }
  el.cancel.addEventListener("click", async () => {
    const id = el.cancel.dataset.job;
    if (!id) return;
    el.cancel.disabled = true;
    try { await api(`/api/jobs/${id}/cancel`, { method: "POST" }); } catch (error) { alert(error.message); }
  });

  async function resolveWatched(status) {
    const live = new Set([status.current && status.current.id, ...(status.queue || []).map((j) => j.id)].filter(Boolean));
    if (status.current) state.watching.add(status.current.id);
    for (const id of [...state.watching]) {
      if (live.has(id)) continue;
      state.watching.delete(id);
      let job = null;
      try { job = await api(`/api/jobs/${id}`); } catch { continue; }
      if (job.state === "done") {
        if (job.kind === "song") { await loadLibrary(); await openSong(job.id); }
        else if (job.kind === "plan") showPlan(job);
        else if (job.kind === "decode") { await openSong(job.result.song_id); }
        else if (job.kind === "transcribe") applyTranscription(job);
        else if (["voice", "stems", "karaoke"].includes(job.kind)) {
          if (state.song && state.song.id !== job.result.song_id) continue;   // finished for a song that is no longer open
          await openSong(job.result.song_id, { scroll: false });
          if (job.kind === "voice") el.voiceHint.hidden = true;
        }
      } else if (job.state === "failed") {
        if (job.kind === "transcribe") el.coverHint.textContent = job.error || "Transcription failed.";
        else if (job.kind === "voice" && state.song) { el.voiceHint.hidden = false; el.voiceHint.textContent = job.error; }
        else if (job.kind === "stems" && state.song) el.stemsHint.textContent = job.error;
        else if (job.kind === "karaoke" && state.song) el.karaokeHint.textContent = job.error;
        else showResultError(job.title, job.error || `${job.kind} failed.`);
      } else if (job.state === "cancelled") {
        if (job.kind === "transcribe") el.coverHint.textContent = "Cancelled.";
        else if (["voice", "stems", "karaoke"].includes(job.kind)) { if (state.song) openSong(state.song.id, { scroll: false }).catch(() => {}); }
        else showResultError(job.title, "Cancelled.", "warn");
      }
    }
    if (!state.watching.size && !status.current && !(status.queue || []).length) el.now.hidden = true;
  }

  // ── plan card ──────────────────────────────────────────────────────────
  function showPlan(job) {
    state.plan = { job, abc: job.result.abc, request: job.request };
    el.planCard.hidden = false;
    el.planTitle.textContent = job.title;
    el.planMeta.textContent = `${job.result.tokens} score tokens · ${modeName[job.request.cot] || job.request.cot} · seed ${job.request.seed}` + (job.result.truncated ? " · hit the score length limit" : "");
    renderSheet(el.planSheet, el.planAbc, job.result.abc, "plan");
    el.planCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
  el.renderPlan.addEventListener("click", () => {
    const { request, abc } = state.plan;
    const body = { ...readForm(), title: state.plan.job.title, style: request.style, lyrics: request.lyrics, cot: request.cot, seed: request.seed, cfg_scale: request.cfg_scale ?? null, abc };
    submitJobs("/api/generate", body, el.renderPlan);
  });
  el.planEdit.addEventListener("click", () => {
    const { request, abc } = state.plan;
    state.scoreSource = abc;
    fillForm({ title: state.plan.job.title, style: request.style, lyrics: request.lyrics, cot: request.cot, seed: request.seed, abc });
    el.abc.scrollIntoView({ behavior: "smooth", block: "center" });
  });
  el.planDiscard.addEventListener("click", () => { el.planCard.hidden = true; state.plan = null; });

  function applyTranscription(job) {
    state.scoreSource = null;
    fillForm({ abc: job.result.abc, cot: job.result.melody_only ? "melody" : "full", origin: null });
    const heard = (job.result.lyrics_text || "").trim();
    if ($("cover-instrumental").checked) {
      const tags = tagsFromScore(job.result.abc);
      if (tags) fillForm({ lyrics: tags, title: el.title.value.trim() || `${(job.request.filename || "cover").replace(/\.[^.]+$/, "")} (cover)` });
    } else if (heard) {
      fillForm({ lyrics: heard, title: el.title.value.trim() || `${(job.request.filename || "cover").replace(/\.[^.]+$/, "")} (cover)` });
    }
    const warnings = job.result.warnings || [];
    el.coverHint.textContent = `Transcribed ${job.request.filename || "the recording"}${job.result.seconds ? ` (${fmtTime(job.result.seconds)})` : ""} in ${fmtTime(job.result.total_seconds)}. ` +
      (warnings.length ? `Warnings: ${warnings.join("; ")}. ` : "") +
      ($("cover-instrumental").checked ? "Lyrics filled with instrumental tags — set a style and Generate." :
       heard ? "Whisper's words are in the lyrics box — fix what it misheard, set a style, and Generate." : "Now add lyrics and a style, then Generate.");
    el.cover.open = true; el.advanced.open = true;
    el.abc.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  // ── result ─────────────────────────────────────────────────────────────
  function showResultError(title, message, kind = "err") {
    el.result.hidden = false;
    el.resultTitle.textContent = title;
    el.resultMeta.textContent = "";
    el.resultBadge.className = `badge ${kind}`;
    el.resultBadge.textContent = kind === "warn" ? "Cancelled" : "Failed";
    showError(el.resultError, message);
    el.player.removeAttribute("src"); el.player.hidden = true; el.altBlock.hidden = true;
    el.scoreBlock.hidden = true; el.resultStyle.textContent = "";
    [el.dlFlac, el.dlWav, el.load, el.editScore, el.del, el.newTake, el.restyle, el.redecode, el.shareSong,
     el.metaBlock, el.karaokeBlock, el.voiceBlock, el.stemsBlock, el.exportBlock].forEach((node) => (node.hidden = true));
    state.song = null;
  }
  async function openSong(id, { scroll = true } = {}) {
    const song = await api(`/api/songs/${id}`);
    state.song = song; state.selected = id;
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
    if (el.player.dataset.song !== song.id) { el.player.src = song.audio_url; el.player.dataset.song = song.id; }
    el.altBlock.hidden = !song.alt_audio;
    if (song.alt_audio) el.altPlayer.src = song.alt_audio + "?t=" + Date.now(); else el.altPlayer.removeAttribute("src");
    el.dlFlac.href = song.audio_url; el.dlFlac.download = `${song.id}.flac`;
    el.dlWav.href = `/api/songs/${song.id}/audio.wav`; el.dlWav.download = `${song.id}.wav`;
    [el.dlFlac, el.dlWav, el.load, el.del, el.newTake, el.shareSong].forEach((node) => (node.hidden = false));
    el.editScore.hidden = el.restyle.hidden = !song.score;
    el.redecode.hidden = !(song.has_latent && song.legacy_vae_available);
    el.redecode.textContent = song.alt_audio ? "Re-decode again (legacy)" : "Re-decode (legacy)";
    renderSheet(el.sheet, el.abcText, song.score, "result");
    el.scoreBlock.hidden = !song.score;
    renderMeta(song);
    renderKaraoke(song);
    renderVoiceBlock(song);
    renderStems(song);
    renderExport(song);
    setMediaSession(song);
    try { history.replaceState(null, "", `?song=${encodeURIComponent(song.id)}`); } catch {}
    renderLibrary();
    if (scroll) el.result.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
  function renderSheet(sheetNode, abcNode, abc, target) {
    sheetNode.innerHTML = ""; abcNode.textContent = abc || "";
    if (!abc) return;
    let rendered = false;
    if (window.ABCJS) {
      try { ABCJS.renderAbc(sheetNode, abc, { responsive: "resize", add_classes: true, paddingtop: 0, paddingbottom: 0 }); rendered = sheetNode.querySelector("svg") != null; }
      catch (error) { console.warn("abcjs could not render this score:", error); }
    }
    selectTab(target, rendered ? "sheet" : "abc");
    document.querySelector(`.tab[data-target='${target}'][data-tab='sheet']`).disabled = !rendered;
  }
  function selectTab(target, name) {
    document.querySelectorAll(`.tab[data-target='${target}']`).forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
    const [sheet, abc] = target === "plan" ? [el.planSheet, el.planAbc] : [el.sheet, el.abcText];
    sheet.hidden = name !== "sheet"; abc.hidden = name !== "abc";
  }
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => selectTab(tab.dataset.target, tab.dataset.tab)));
  el.load.addEventListener("click", () => {
    const { request, title } = state.song;
    state.scoreSource = request.abc || null;
    fillForm({ title, style: request.style, lyrics: request.lyrics, cot: request.cot, seed: request.seed, cfg_scale: request.cfg_scale ?? null, abc: request.abc || "", origin: { song_id: state.song.id, kind: "load" } });
    el.style.scrollIntoView({ behavior: "smooth", block: "start" });
  });
  el.editScore.addEventListener("click", () => {
    const { request, title, score } = state.song;
    state.scoreSource = score;
    fillForm({ title: `${title} (edited)`, style: request.style, lyrics: request.lyrics, cot: request.cot === "off" ? "full" : request.cot, seed: request.seed, abc: score, origin: { song_id: state.song.id, kind: "edit" } });
    el.abc.scrollIntoView({ behavior: "smooth", block: "center" });
    el.abc.focus();
  });
  el.restyle.addEventListener("click", () => {
    const { request, title, score } = state.song;
    state.scoreSource = score;
    fillForm({ title: `${title} (new style)`, style: request.style, lyrics: request.lyrics, cot: request.cot === "off" ? "full" : request.cot, seed: null, abc: score, origin: { song_id: state.song.id, kind: "restyle" } });
    el.style.scrollIntoView({ behavior: "smooth", block: "start" });
    el.style.focus(); el.style.select();
  });
  el.newTake.addEventListener("click", () => {
    const { request, title } = state.song;
    submitJobs("/api/generate", { title: `${title} (new take)`, style: request.style, lyrics: request.lyrics, cot: request.cot, seed: null,
                                  cfg_scale: request.cfg_scale ?? null, abc: request.abc || null, takes: 1, origin: { song_id: state.song.id, kind: "take" } }, el.newTake);
  });
  el.redecode.addEventListener("click", async () => {
    el.redecode.disabled = true;
    try { watchJob(await api(`/api/songs/${state.song.id}/redecode`, { method: "POST" })); }
    catch (error) { alert(error.message); }
    finally { el.redecode.disabled = false; }
  });
  el.del.addEventListener("click", async () => {
    const song = state.song;
    if (!song) return;
    if (!confirm(`Delete "${song.title}"?\n\nIt moves to the trash folder on the PC (trash\\${song.id}), so it can be restored by hand.`)) return;
    // release every player that may hold a suspended download of this song's files
    for (const audio of [el.player, el.altPlayer, ...el.voiceVersions.querySelectorAll("audio"), ...el.stemList.querySelectorAll("audio"), ...el.compareGrid.querySelectorAll("audio")]) { audio.pause(); audio.removeAttribute("src"); audio.load(); }
    delete el.player.dataset.song;
    el.del.disabled = true;
    try {
      const reply = await api(`/api/songs/${song.id}`, { method: "DELETE" });
      state.song = null; state.selected = null; state.compare.delete(song.id);
      el.result.hidden = true;
      await loadLibrary();
      if (reply.deferred) el.libraryCount.textContent = "deleted — the folder moves to trash once every player lets go of it";
      if (state.songs.length) openSong(state.songs[0].id, { scroll: false }).catch(() => {});
    } catch (error) {
      alert(error.message);
      openSong(song.id, { scroll: false }).catch(() => {});
    } finally { el.del.disabled = false; }
  });

  // ── favourites, rating, tags, notes, version family ───────────────────
  function renderMeta(song) {
    const meta = song.meta || {};
    el.metaBlock.hidden = false;
    el.fav.textContent = meta.favorite ? "★" : "☆";
    el.fav.classList.toggle("on", !!meta.favorite);
    el.rating.innerHTML = "";
    for (let n = 1; n <= 5; n++) {
      const star = document.createElement("button");
      star.type = "button"; star.className = `rate${n <= (meta.rating || 0) ? " on" : ""}`; star.textContent = "★"; star.title = `${n} / 5`;
      star.addEventListener("click", () => saveMeta({ rating: meta.rating === n ? 0 : n }));
      el.rating.appendChild(star);
    }
    el.tags.value = (meta.tags || []).join(", ");
    el.notes.value = meta.notes || "";
    el.metaOwner.textContent = meta.owner ? `by ${meta.owner}` : "";
    const family = song.family || {};
    const parts = [];
    if (family.parent) parts.push(`<span>from <a href="#" data-song="${family.parent.id}"></a></span>`);
    if ((family.children || []).length) parts.push(`<span>versions: ${family.children.map((c) => `<a href="#" data-song="${c.id}"></a>`).join(", ")}</span>`);
    el.family.hidden = !parts.length;
    el.family.innerHTML = parts.join(" · ");
    const titles = [family.parent, ...(family.children || [])].filter(Boolean);
    el.family.querySelectorAll("a").forEach((a) => {
      const item = titles.find((t) => t.id === a.dataset.song);
      a.textContent = item ? item.title : a.dataset.song;
      a.addEventListener("click", (e) => { e.preventDefault(); openSong(a.dataset.song).catch((error) => alert(error.message)); });
    });
  }
  async function saveMeta(changes) {
    if (!state.song) return;
    try {
      const meta = await patch(`/api/songs/${state.song.id}/meta`, changes);
      state.song.meta = meta;
      const item = state.songs.find((s) => s.id === state.song.id);
      if (item) Object.assign(item, { favorite: meta.favorite, rating: meta.rating, tags: meta.tags });
      renderMeta(state.song); renderLibrary();
    } catch (error) { alert(error.message); }
  }
  el.fav.addEventListener("click", () => saveMeta({ favorite: !(state.song && state.song.meta && state.song.meta.favorite) }));
  el.tags.addEventListener("change", () => saveMeta({ tags: el.tags.value.split(",").map((t) => t.trim()).filter(Boolean) }));
  el.notes.addEventListener("input", () => { clearTimeout(state.metaTimer); state.metaTimer = setTimeout(() => saveMeta({ notes: el.notes.value }), 800); });
  el.notes.addEventListener("blur", () => { clearTimeout(state.metaTimer); if (state.song && el.notes.value !== (state.song.meta.notes || "")) saveMeta({ notes: el.notes.value }); });

  // ── karaoke (synced lyrics) ───────────────────────────────────────────
  const hasWords = (song) => /[^\s\[\]]/.test((song.request.lyrics || "").replace(/\[[^\]]*\]/g, ""));
  async function renderKaraoke(song) {
    state.karaoke = null;
    el.lyricsSync.hidden = true; el.lyricsSync.innerHTML = "";
    el.karaokeHint.textContent = "";
    el.karaokeBlock.hidden = !(song.voice_ready && hasWords(song));
    el.dlLrc.hidden = el.dlSrt.hidden = !song.karaoke;
    if (!song.karaoke) { el.karaokeBtn.textContent = "Karaoke: sync lyrics"; return; }
    el.karaokeBtn.textContent = "Re-sync lyrics";
    el.dlLrc.href = song.karaoke.lrc; el.dlLrc.download = `${song.id}.lrc`;
    el.dlSrt.href = song.karaoke.srt; el.dlSrt.download = `${song.id}.srt`;
    if (song.karaoke.matched_ratio != null && song.karaoke.matched_ratio < 0.6) el.karaokeHint.textContent = `Only ${Math.round(song.karaoke.matched_ratio * 100)}% of the words were recognised — timings may drift.`;
    try {
      const data = await fetch(`${song.karaoke.json}?t=${Date.now()}`, { headers: userName() ? { "X-User": userName() } : {} }).then((r) => r.json());
      if (!state.song || state.song.id !== song.id) return;
      state.karaoke = data.lines || [];
      el.lyricsSync.hidden = false;
      let section = null;
      state.karaoke.forEach((line, i) => {
        if (line.section !== section) { section = line.section; const h = document.createElement("div"); h.className = "sync-section"; h.textContent = section || ""; el.lyricsSync.appendChild(h); }
        const node = document.createElement("div");
        node.className = "sync-line"; node.dataset.index = i; node.textContent = line.text; node.title = `${fmtTime(line.start)}${line.estimated ? " (estimated)" : ""}`;
        node.addEventListener("click", () => { el.player.currentTime = line.start; el.player.play().catch(() => {}); });
        el.lyricsSync.appendChild(node);
      });
      highlightLine();
    } catch (error) { el.karaokeHint.textContent = `Could not load the sync: ${error.message}`; }
  }
  function highlightLine() {
    if (!state.karaoke || el.lyricsSync.hidden) return;
    const t = el.player.currentTime;
    let active = -1;
    state.karaoke.forEach((line, i) => { if (t >= line.start - 0.15) active = i; });
    el.lyricsSync.querySelectorAll(".sync-line").forEach((node) => {
      const i = Number(node.dataset.index);
      const on = i === active && t <= state.karaoke[i].end + 0.6;
      node.classList.toggle("active", on);
      node.classList.toggle("past", i < active);
      if (on && !el.player.paused) node.scrollIntoView({ block: "nearest", behavior: "smooth" });
    });
  }
  el.player.addEventListener("timeupdate", highlightLine);
  el.karaokeBtn.addEventListener("click", async () => {
    if (!state.song) return;
    el.karaokeBtn.disabled = true; el.karaokeHint.textContent = "Queued…";
    try { watchJob(await api(`/api/songs/${state.song.id}/karaoke`, { method: "POST" })); el.karaokeHint.textContent = "Timing the lyrics — the synced view appears here when done (about 15 s)."; }
    catch (error) { el.karaokeHint.textContent = error.message; }
    finally { el.karaokeBtn.disabled = false; }
  });

  // ── stems ─────────────────────────────────────────────────────────────
  const STEM_LABELS = { accompaniment: "Instrumental (no vocals)", vocals: "Acapella (vocals only)", drums: "Drums", bass: "Bass", other: "Other (keys, guitars, synths)" };
  function renderStems(song) {
    el.stemsBlock.hidden = !(song.voice_ready || song.stems);
    el.stemsHint.textContent = "";
    el.stemList.innerHTML = "";
    el.stemsBtn.hidden = !!song.stems;
    if (!song.stems) return;
    for (const [name, label] of Object.entries(STEM_LABELS)) {
      if (!song.stems[name]) continue;
      const block = document.createElement("div");
      block.className = "alt stem";
      block.innerHTML = `<div class="sub"><span></span><a class="btn-mini" download>Download</a></div><audio controls preload="none"></audio>`;
      block.querySelector("span").textContent = label;
      const link = block.querySelector("a"); link.href = song.stems[name]; link.download = `${song.id}-${name}.flac`;
      block.querySelector("audio").src = song.stems[name];
      el.stemList.appendChild(block);
    }
  }
  el.stemsBtn.addEventListener("click", async () => {
    if (!state.song) return;
    el.stemsBtn.disabled = true; el.stemsHint.textContent = "Queued…";
    try { watchJob(await api(`/api/songs/${state.song.id}/stems`, { method: "POST" })); el.stemsHint.textContent = "Separating — the stems appear here when done (about 10 s)."; }
    catch (error) { el.stemsHint.textContent = error.message; }
    finally { el.stemsBtn.disabled = false; }
  });

  // ── export (mastered MP3 / WAV / FLAC) ────────────────────────────────
  function renderExport(song) {
    el.exportBlock.hidden = !song.voice_ready;
    el.exportHint.textContent = "";
    el.exportSource.innerHTML = "";
    const sources = [["audio.flac", "original"], ...(song.voice_versions || []).map((v) => [v.file, `in ${v.name}'s voice`])];
    for (const [file, label] of sources) { const option = document.createElement("option"); option.value = file; option.textContent = label; el.exportSource.appendChild(option); }
    el.exportList.innerHTML = "";
    for (const item of song.exports || []) {
      const row = document.createElement("div");
      row.className = "export-row";
      row.innerHTML = `<a download></a><span class="sub"></span>`;
      const link = row.querySelector("a"); link.href = item.url; link.download = item.name; link.textContent = item.name;
      row.querySelector("span").textContent = fmtBytes(item.bytes);
      el.exportList.appendChild(row);
    }
  }
  el.exportBtn.addEventListener("click", async () => {
    if (!state.song) return;
    el.exportBtn.disabled = true; el.exportHint.textContent = "Exporting… (a few seconds)";
    try {
      const reply = await post(`/api/songs/${state.song.id}/export`, { format: el.exportFormat.value, source: el.exportSource.value, normalize: el.exportMaster.checked,
                                                                    fade_out: Number(el.exportFade.value) || 0, trim: el.exportTrim.checked });
      el.exportHint.textContent = `Done in ${reply.total_seconds}s` + (reply.lufs_after != null ? ` · ${reply.lufs_before} → ${reply.lufs_after} LUFS, peak ${reply.peak_dbfs} dBFS` : "") + ".";
      await openSong(state.song.id, { scroll: false });
      const link = [...el.exportList.querySelectorAll("a")].find((a) => a.download === reply.download);
      if (link) link.classList.add("fresh");
    } catch (error) { el.exportHint.textContent = error.message; }
    finally { el.exportBtn.disabled = false; }
  });

  // ── share a song (listen-only link) ───────────────────────────────────
  async function refreshShareLink() {
    if (!state.song) return;
    try {
      const reply = await api(`/api/songs/${state.song.id}/share?days=${el.songShareDays.value}`);
      el.songShareUrl.textContent = reply.url;
      el.songShareOpen.href = reply.url;
      el.songShareQr.src = `/api/qr.svg?text=${encodeURIComponent(reply.url)}`;
    } catch (error) { el.songShareUrl.textContent = error.message; }
  }
  el.shareSong.addEventListener("click", () => { el.songShareOverlay.hidden = false; refreshShareLink(); });
  el.songShareDays.addEventListener("change", refreshShareLink);
  el.songShareClose.addEventListener("click", () => { el.songShareOverlay.hidden = true; });
  el.songShareCopy.addEventListener("click", async () => {
    const url = el.songShareUrl.textContent;
    try { await navigator.clipboard.writeText(url); el.songShareCopy.textContent = "Copied"; setTimeout(() => (el.songShareCopy.textContent = "Copy link"), 1500); }
    catch { prompt("Link:", url); }
  });

  // ── media session (lock-screen controls on phones) ────────────────────
  function setMediaSession(song) {
    if (!("mediaSession" in navigator)) return;
    try {
      navigator.mediaSession.metadata = new MediaMetadata({ title: song.title, artist: "Music Gen Studio", album: song.style.slice(0, 80),
        artwork: [{ src: "/static/icons/icon-512.png", sizes: "512x512", type: "image/png" }, { src: "/static/icons/icon-192.png", sizes: "192x192", type: "image/png" }] });
      const step = (delta) => {
        const ids = visibleSongs().map((s) => s.id);
        const next = ids[(ids.indexOf(state.selected) + delta + ids.length) % ids.length];
        if (next) openSong(next, { scroll: false }).then(() => el.player.play().catch(() => {})).catch(() => {});
      };
      navigator.mediaSession.setActionHandler("play", () => el.player.play());
      navigator.mediaSession.setActionHandler("pause", () => el.player.pause());
      navigator.mediaSession.setActionHandler("previoustrack", () => step(-1));
      navigator.mediaSession.setActionHandler("nexttrack", () => step(1));
      navigator.mediaSession.setActionHandler("seekto", (d) => { if (d.seekTime != null) el.player.currentTime = d.seekTime; });
    } catch {}
  }

  // ── sing it in my voice ───────────────────────────────────────────────
  const autoOption = document.createElement("option");
  autoOption.value = "auto"; autoOption.textContent = "auto (fit my range)";
  el.voiceSemitones.appendChild(autoOption);
  for (let s = 12; s >= -12; s--) {
    const option = document.createElement("option");
    option.value = s; option.textContent = s === 0 ? "same pitch" : (s > 0 ? `+${s}` : `${s}`) + (Math.abs(s) === 12 ? " (octave)" : "");
    el.voiceSemitones.appendChild(option);
  }
  el.voiceSemitones.value = "0";
  function fillVoiceSelect(select, keep, empty) {
    const current = select.value;
    select.innerHTML = "";
    if (empty) { const option = document.createElement("option"); option.value = ""; option.textContent = empty; select.appendChild(option); }
    for (const voice of state.voices || []) {
      const option = document.createElement("option");
      option.value = voice.name; option.textContent = voice.seconds ? `${voice.name} (${fmtTime(voice.seconds)})` : voice.name;
      select.appendChild(option);
    }
    if (keep && [...select.options].some((o) => o.value === current)) select.value = current;
  }
  async function loadVoices() {
    try {
      const reply = await api("/api/voices");
      state.voices = reply.voices; state.voiceReady = reply.ready;
    } catch { state.voices = state.voices || []; }
    fillVoiceSelect(el.voiceSelect, true, state.voices.length ? null : "no voices yet");
    fillVoiceSelect(el.voiceDuet, true, "none");
    fillVoiceSelect(el.describeVoice, true, "the original vocal");
    el.voiceSelect.disabled = el.voiceBtn.disabled = !state.voices.length || !state.voiceReady;
    renderVoicesList();
    return state.voices;
  }
  function renderVoiceBlock(song) {
    el.voiceHint.hidden = true;
    el.voiceBlock.hidden = !(song.voice_ready || (song.voice_versions || []).length);
    el.voiceVersions.innerHTML = "";
    for (const version of song.voice_versions || []) {
      const block = document.createElement("div");
      block.className = "alt";
      block.innerHTML = `<div class="sub"><span>In <b></b>'s voice</span><a class="btn-mini" download>Download</a></div><audio controls preload="metadata"></audio>`;
      block.querySelector("b").textContent = version.name.replace(/-harmony/, " + harmonies").replace(/-duet-/, ", duet with ");
      const link = block.querySelector("a"); link.href = version.url; link.download = `${song.id}-voice-${version.name}.flac`;
      block.querySelector("audio").src = `${version.url}?t=${Math.round(version.modified || 0)}`;
      el.voiceVersions.appendChild(block);
    }
    if (song.voice_ready && state.voices === null) loadVoices().catch(() => {});
  }
  el.voiceBtn.addEventListener("click", async () => {
    const song = state.song, voice = el.voiceSelect.value;
    if (!song || !voice) return;
    const pitch = el.voiceSemitones.value;
    const body = { voice, semitones: pitch === "auto" ? 0 : Number(pitch) || 0, auto_f0: pitch === "auto", steps: 30,
                   harmonies: el.voiceHarmony.value ? el.voiceHarmony.value.split(",").map(Number) : [],
                   duet_voice: el.voiceDuet.value || null, duet_sections: [el.voiceDuetSections.value] };
    if (body.duet_voice === voice) { el.voiceHint.hidden = false; el.voiceHint.textContent = "Pick a different voice for the duet."; return; }
    if (body.duet_voice && !hasWords(song)) { el.voiceHint.hidden = false; el.voiceHint.textContent = "A duet needs sung lyrics (the sections come from the lyric timings)."; return; }
    el.voiceBtn.disabled = true;
    el.voiceHint.hidden = false; el.voiceHint.textContent = "Queued…";
    try {
      const job = await post(`/api/songs/${song.id}/voice`, body);
      const passes = 1 + body.harmonies.length + (body.duet_voice ? 1 : 0);
      el.voiceHint.textContent = `Working — "${job.title}" appears below this player when done (${passes} conversion pass${passes === 1 ? "" : "es"}, about a minute each).`;
      watchJob(job);
    } catch (error) { el.voiceHint.textContent = error.message; }
    finally { el.voiceBtn.disabled = !state.voices || !state.voices.length; }
  });
  function renderVoicesList() {
    el.voicesList.innerHTML = "";
    if (!state.voiceReady) el.voicesHint.textContent = "Voice conversion is not set up on this PC — see README: Sing it in your voice.";
    else el.voicesHint.textContent = state.voices.length ? "" : "No voices yet. Add a clip to get started.";
    for (const voice of state.voices || []) {
      const li = document.createElement("li");
      li.innerHTML = `<span class="name"></span><span class="grow"></span><button class="btn-mini" type="button">Remove</button>`;
      li.querySelector(".name").textContent = voice.name;
      li.querySelector(".grow").textContent = `${voice.file}${voice.seconds ? ` · ${fmtTime(voice.seconds)}` : ""}`;
      li.querySelector("button").addEventListener("click", async () => {
        if (!confirm(`Remove the voice "${voice.name}"? Songs already sung in it keep their audio.`)) return;
        try { await api(`/api/voices/${encodeURIComponent(voice.name)}`, { method: "DELETE" }); await loadVoices(); }
        catch (error) { el.voicesHint.textContent = error.message; }
      });
      el.voicesList.appendChild(li);
    }
  }
  el.voicesManage.addEventListener("click", async () => { el.voicesOverlay.hidden = false; await loadVoices(); });
  el.voicesClose.addEventListener("click", () => { el.voicesOverlay.hidden = true; });
  el.voiceUpload.addEventListener("click", async () => {
    const file = el.voiceFile.files[0];
    if (!file) { el.voicesHint.textContent = "Choose a recording first (wav, flac, mp3 or ogg)."; return; }
    const data = new FormData();
    data.append("file", file); data.append("name", el.voiceName.value.trim());
    el.voiceUpload.disabled = true; el.voicesHint.textContent = "Uploading…";
    try {
      const reply = await api("/api/voices", { method: "POST", body: data });
      el.voiceFile.value = ""; el.voiceName.value = "";
      await loadVoices();
      el.voiceSelect.value = reply.voice.name;
      el.voicesHint.textContent = `Added "${reply.voice.name}" (${fmtTime(reply.voice.seconds)}). Close this and press "Sing it in this voice" on any song.`;
    } catch (error) { el.voicesHint.textContent = error.message; }
    finally { el.voiceUpload.disabled = false; }
  });

  // ── library & compare ─────────────────────────────────────────────────
  async function loadLibrary() {
    try { state.songs = await api("/api/songs"); } catch { return; }
    renderLibrary();
  }
  function visibleSongs() {
    const q = state.filters.search.trim().toLowerCase();
    const me = userName().toLowerCase();
    return state.songs.filter((song) => {
      if (state.filters.fav && !song.favorite) return false;
      if (state.filters.mine && (song.owner || "").toLowerCase() !== me) return false;
      if (!q) return true;
      return [song.title, song.style, (song.tags || []).join(" "), song.owner || ""].join(" ").toLowerCase().includes(q);
    });
  }
  function renderLibrary() {
    el.songs.innerHTML = "";
    const songs = visibleSongs();
    el.libraryEmpty.hidden = state.songs.length > 0;
    el.libraryCount.textContent = state.songs.length ? (songs.length === state.songs.length ? `${state.songs.length} song${state.songs.length === 1 ? "" : "s"}` : `${songs.length} of ${state.songs.length}`) : "";
    for (const song of songs) {
      const li = document.createElement("li");
      li.className = `song${song.id === state.selected ? " active" : ""}`;
      li.innerHTML = `<label class="song-pick" title="Select for comparison"><input type="checkbox"></label><div class="song-icon">♪</div>
        <div class="song-main"><div class="song-title"></div><div class="song-style"></div></div>
        <div class="song-meta"><div class="song-badges"></div><div>${fmtTime(song.seconds)}</div><div>${new Date(song.created * 1000).toLocaleDateString()}</div></div>`;
      li.querySelector(".song-title").textContent = song.title;
      li.querySelector(".song-style").textContent = [(song.tags || []).map((t) => `#${t}`).join(" "), song.style].filter(Boolean).join(" · ");
      const badges = [];
      if (song.favorite) badges.push(["★", "favourite"]);
      if (song.rating) badges.push([`${song.rating}/5`, "rating"]);
      if (song.has_voice) badges.push(["🎤", "has a voice version"]);
      if (song.has_karaoke) badges.push(["🎶", "lyrics synced"]);
      if (song.owner) badges.push([song.owner, "queued by"]);
      li.querySelector(".song-badges").innerHTML = badges.map(() => `<span class="song-badge"></span>`).join("");
      li.querySelectorAll(".song-badge").forEach((node, i) => { node.textContent = badges[i][0]; node.title = badges[i][1]; });
      const box = li.querySelector("input");
      box.checked = state.compare.has(song.id);
      box.addEventListener("click", (e) => { e.stopPropagation(); box.checked ? state.compare.add(song.id) : state.compare.delete(song.id); updateCompareButton(); });
      li.querySelector(".song-pick").addEventListener("click", (e) => e.stopPropagation());
      li.addEventListener("click", () => openSong(song.id).catch((error) => alert(error.message)));
      el.songs.appendChild(li);
    }
    if (state.songs.length && !songs.length) { const p = document.createElement("li"); p.className = "empty"; p.textContent = "No songs match this filter."; el.songs.appendChild(p); }
    updateCompareButton();
  }
  el.libSearch.addEventListener("input", () => { state.filters.search = el.libSearch.value; renderLibrary(); });
  el.libFav.addEventListener("click", () => { state.filters.fav = !state.filters.fav; el.libFav.classList.toggle("on", state.filters.fav); renderLibrary(); });
  el.libMine.addEventListener("click", () => {
    if (!userName()) { openSettings(); return; }
    state.filters.mine = !state.filters.mine; el.libMine.classList.toggle("on", state.filters.mine); renderLibrary();
  });
  function updateCompareButton() {
    const n = state.compare.size;
    el.compareBtn.hidden = n < 2;
    el.compareBtn.textContent = `Compare selected (${n})`;
  }
  el.compareBtn.addEventListener("click", async () => {
    el.compareGrid.innerHTML = "";
    for (const id of state.compare) {
      let song;
      try { song = await api(`/api/songs/${id}`); } catch { continue; }
      const card = document.createElement("div");
      card.className = "compare-card";
      card.innerHTML = `<div class="compare-title"></div><div class="sub"></div><audio controls preload="metadata"></audio><div class="compare-style"></div>`;
      card.querySelector(".compare-title").textContent = song.title;
      card.querySelector(".sub").textContent = `${fmtTime(song.seconds)} · ${modeName[song.cot] || song.cot} · seed ${song.seed}`;
      card.querySelector("audio").src = song.audio_url;
      card.querySelector(".compare-style").textContent = song.style;
      el.compareGrid.appendChild(card);
    }
    el.compare.hidden = false;
    el.compare.scrollIntoView({ behavior: "smooth", block: "nearest" });
  });
  el.compareClose.addEventListener("click", () => { el.compare.hidden = true; el.compareGrid.innerHTML = ""; });

  // ── settings (name + notifications) ──────────────────────────────────
  function renderUserPill() { const name = userName(); el.pillUser.querySelector(".pill-text").textContent = name ? `You: ${name}` : "Who's this?"; }
  async function openSettings() {
    el.setName.value = userName();
    el.settingsHint.textContent = "";
    try {
      const s = await api("/api/settings");
      el.setDiscord.value = s.discord_webhook || ""; el.setTgToken.value = s.telegram_bot_token || ""; el.setTgChat.value = s.telegram_chat_id || "";
      el.setFailed.checked = s.notify_failed !== false;
    } catch (error) { el.settingsHint.textContent = error.message; }
    el.settingsOverlay.hidden = false;
    el.setName.focus();
  }
  el.pillUser.addEventListener("click", openSettings);
  el.pillSettings.addEventListener("click", openSettings);
  el.settingsClose.addEventListener("click", () => { el.settingsOverlay.hidden = true; });
  el.settingsSave.addEventListener("click", async () => {
    try { localStorage.setItem("mgs.name", el.setName.value.trim().slice(0, 40)); } catch {}
    renderUserPill();
    el.settingsSave.disabled = true;
    try {
      await put("/api/settings", { discord_webhook: el.setDiscord.value.trim(), telegram_bot_token: el.setTgToken.value.trim(), telegram_chat_id: el.setTgChat.value.trim(), notify_failed: el.setFailed.checked });
      el.settingsHint.textContent = "Saved.";
      renderLibrary();
      setTimeout(() => { el.settingsOverlay.hidden = true; }, 500);
    } catch (error) { el.settingsHint.textContent = error.message; }
    finally { el.settingsSave.disabled = false; }
  });
  el.settingsTest.addEventListener("click", async () => {
    el.settingsHint.textContent = "Sending…";
    try { await put("/api/settings", { discord_webhook: el.setDiscord.value.trim(), telegram_bot_token: el.setTgToken.value.trim(), telegram_chat_id: el.setTgChat.value.trim() }); await post("/api/settings/test", {}); el.settingsHint.textContent = "Test message sent — check Discord / Telegram."; }
    catch (error) { el.settingsHint.textContent = error.message; }
  });

  // ── share & system overlays ────────────────────────────────────────────
  el.pillShare.addEventListener("click", () => {
    const urls = state.status ? (state.status.share_urls || []) : [];
    if (!urls.length) return;
    const main = urls[0];
    $("share-qr").src = `/api/qr.svg?text=${encodeURIComponent(main)}`;
    $("share-url").textContent = main;
    $("share-list").innerHTML = "";
    for (const url of urls) {
      const li = document.createElement("li");
      li.innerHTML = `<span>${/trycloudflare|ts\.net/.test(url) ? "internet" : "this network"}</span><code></code>`;
      li.querySelector("code").textContent = url;
      $("share-list").appendChild(li);
    }
    $("share-overlay").hidden = false;
  });
  $("share-close").addEventListener("click", () => { $("share-overlay").hidden = true; });
  $("share-copy").addEventListener("click", async () => {
    const url = $("share-url").textContent;
    try { await navigator.clipboard.writeText(url); $("share-copy").textContent = "Copied"; setTimeout(() => ($("share-copy").textContent = "Copy address"), 1500); }
    catch { prompt("Address:", url); }
  });
  el.pillDoctor.addEventListener("click", async () => {
    let d;
    try { d = await api("/api/doctor"); } catch (error) { alert(error.message); return; }
    const rows = [
      ["GPU", d.gpu ? `${d.gpu.name} · ${d.gpu.total_gib} GiB (free ${d.gpu.free_gib}) · cc ${d.gpu.capability}` : "none"],
      ["Mode", `${d.vram_mode} VRAM · quantization ${d.quantization} · graph attention ${d.graph_attention} · model ${d.model_state}${d.model_error ? " (" + d.model_error + ")" : ""}`],
      ["Software", `Python ${d.python} · torch ${d.versions.torch} · CUDA ${d.cuda} · cuDNN ${d.cudnn} · transformers ${d.versions.transformers} · yue2-infer ${d.versions["yue2-infer"]}`],
      ...Object.entries(d.models).map(([name, m]) => [name, m.present ? `${m.size_gb} GB · ${m.path}` : `not downloaded (${m.path})`]),
      ["Cover feature", d.sheetsage2_env && d.models.SheetSage2.present ? "SheetSage2 ready" + (d.ffmpeg ? " · ffmpeg found" : " · no ffmpeg (wav/flac/mp3/ogg still work)") : "SheetSage2 not set up"],
      ["Lyric model", !d.lyrics_model.running ? "Ollama not running" : d.lyrics_model.downloaded ? `${d.lyrics_model.name} ready (Ollama)` : `Ollama running, ${d.lyrics_model.name} not downloaded`],
      ["Voice tools", d.voice_conversion.env && d.voice_conversion.seed_vc ? `Seed-VC + Demucs + Whisper (${d.whisper_model}) ready · ${d.voice_conversion.voices} reference voice${d.voice_conversion.voices === 1 ? "" : "s"}` : "not set up (see README: Sing it in your voice)"],
      ["Notifications", [d.notifications.discord ? "Discord" : null, d.notifications.telegram ? "Telegram" : null].filter(Boolean).join(" + ") || "none configured (Settings)"],
      ["Sharing", (d.cloudflared ? `cloudflared: ${d.cloudflared}` : "cloudflared not found") + (d.share_urls.length ? ` · ${d.share_urls.join(", ")}` : "")],
      ["Storage", `${d.disk_free_gb} GB free · ${d.songs} songs · ${d.plans} plans · ${d.trash} in trash`],
      ["Weights", d.weights ? Object.entries(d.weights).map(([k, v]) => `${k}: ${(v.files && Object.values(v.files)[0] && Object.values(v.files)[0].sha256 || "").slice(0, 12)}…`).join(" · ") : "—"],
    ];
    $("doctor-list").innerHTML = rows.map(() => `<dt></dt><dd></dd>`).join("");
    const dts = $("doctor-list").querySelectorAll("dt"), dds = $("doctor-list").querySelectorAll("dd");
    rows.forEach(([k, v], i) => { dts[i].textContent = k; dds[i].textContent = v; });
    $("doctor-overlay").hidden = false;
  });
  $("doctor-close").addEventListener("click", () => { $("doctor-overlay").hidden = true; });
  document.querySelectorAll(".overlay").forEach((o) => o.addEventListener("click", (e) => { if (e.target === o) o.hidden = true; }));

  // ── theme ──────────────────────────────────────────────────────────────
  // Light is the default; html[data-theme="dark"] switches to dark. With nothing saved the attribute stays off so the
  // stylesheet may follow prefers-color-scheme. Inside the hub the bridge owns data-theme, so the saved value is left alone.
  const THEME_KEY = "mgs_theme";
  const root = document.documentElement;
  function applyTheme(theme) {
    root.dataset.theme = theme;
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", theme === "dark" ? "#141416" : "#FFFFFF");
  }
  function currentTheme() {
    if (root.dataset.theme) return root.dataset.theme;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  if (!root.dataset.hubTheme) {
    let saved = null;
    try { saved = localStorage.getItem(THEME_KEY); } catch {}
    if (saved === "dark" || saved === "light") applyTheme(saved);
  }
  $("btn-theme").addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    applyTheme(next);
    try { localStorage.setItem(THEME_KEY, next); } catch {}
  });

  // ── boot ───────────────────────────────────────────────────────────────
  restoreDraft();
  renderUserPill();
  loadPresets();
  loadVoices().catch(() => {});
  const wanted = new URLSearchParams(location.search).get("song");
  loadLibrary().then(() => {
    const first = wanted && state.songs.some((s) => s.id === wanted) ? wanted : (state.songs[0] || {}).id;
    if (first && el.result.hidden) openSong(first, { scroll: !!wanted }).catch(() => {});
  });
  poll();
  if ("serviceWorker" in navigator && (location.protocol === "https:" || location.hostname === "localhost" || location.hostname === "127.0.0.1")) {
    navigator.serviceWorker.register("/sw.js").catch(() => {});
  }
})();
