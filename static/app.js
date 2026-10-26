(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const screens = ["screen-capture", "screen-loading", "screen-result"];

  const ICONS = {
    wet: '<path d="M5 19c0-8 6-14 14-14 0 8-6 14-14 14z"/><path d="M5 19l8-8"/>',
    dry: '<path d="M3 7l9-4 9 4-9 4z"/><path d="M3 7v10l9 4 9-4V7"/><path d="M12 11v10"/>',
    hazardous: '<path d="M12 3l10 18H2z"/><path d="M12 10v4.5"/><path d="M12 17.5h.01"/>',
    ewaste: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
    sanitary: '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/>',
    reuse: '<path d="M20 11a8 8 0 0 0-14.9-3M4 13a8 8 0 0 0 14.9 3"/><path d="M5 3v5h5M19 21v-5h-5"/>',
    unknown: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6V14"/><path d="M12 17h.01"/>',
    alert: '<path d="M12 3l10 18H2z"/><path d="M12 10v4.5"/><path d="M12 17.5h.01"/>',
    check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    list: '<path d="M9 6h11M9 12h11M9 18h11"/><path d="M4 6h.01M4 12h.01M4 18h.01"/>',
    pin: '<path d="M12 21s-7-6.2-7-11.5a7 7 0 0 1 14 0C19 14.8 12 21 12 21z"/><circle cx="12" cy="9.5" r="2.5"/>',
    share: '<path d="M12 3v12"/><path d="M7 8l5-5 5 5"/><path d="M5 13v6a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-6"/>',
    chevron: '<path d="M9 6l6 6-6 6"/>',
    camera: '<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
  };

  const EXAMPLES = [
    { label: "Swollen phone battery", tone: "red" },
    { label: "Greasy pizza box", tone: "green" },
    { label: "Expired tablets", tone: "red" },
    { label: "Old phone charger", tone: "purple" },
    { label: "Broken CFL bulb", tone: "red" },
    { label: "Plastic water bottle", tone: "blue" },
  ];

  const BINS = [
    { stream: "wet", label: "Wet waste", tone: "green", text: "Food scraps, peels, flowers, garden waste" },
    { stream: "dry", label: "Dry waste", tone: "blue", text: "Clean plastic, paper, glass, metal, cloth" },
    { stream: "hazardous", label: "Hazardous", tone: "red", text: "Batteries, medicines, CFLs, needles, paint" },
    { stream: "ewaste", label: "E-waste", tone: "purple", text: "Phones, chargers, cables, LED bulbs" },
  ];

  const TIPS = [
    "Segregating waste into wet, dry and domestic hazardous at home is required under India's Solid Waste Management Rules, 2016.",
    "Rinsing containers keeps a whole batch of dry waste clean enough to recycle.",
    "Damaged lithium batteries can catch fire inside garbage trucks. Always keep them out of bins.",
    "Burning waste in the open is prohibited. It releases toxic smoke into the air you breathe.",
    "Composting food waste at home returns nutrients to the soil and keeps it out of landfills.",
    "Old phones contain metals like copper and gold that authorised recyclers can recover.",
    "Greasy paper can't be recycled, but it can be composted with wet waste.",
  ];

  const LOADING_STEPS_MS = [0, 1600, 3600];
  const HISTORY_KEY = "wg-history";
  const STATS_KEY = "wg-stats";
  const HISTORY_LIMIT = 8;

  const state = {
    config: null,
    image: null, previewUrl: null, thumb: null,
    result: null, current: null, resultThumb: null,
  };

  // ---------- helpers ----------
  function h(tag, props, ...children) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "style") el.style.cssText = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) {
      if (c == null || c === false) continue;
      el.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return el;
  }

  function icon(name) {
    const span = document.createElement("span");
    span.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[name] || ICONS.unknown}</svg>`;
    return span.firstChild;
  }

  const store = {
    get(key, fallback) {
      try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
    },
    set(key, value) {
      try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode or full */ }
    },
  };

  let toastTimer;
  function toast(message) {
    const el = $("toast");
    el.textContent = message;
    el.hidden = false;
    el.style.animation = "none";
    void el.offsetWidth;
    el.style.animation = "";
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, 2600);
  }

  function haptic(pattern) {
    try { navigator.vibrate?.(pattern); } catch { /* unsupported */ }
  }

  function show(id) {
    for (const s of screens) $(s).hidden = s !== id;
    window.scrollTo({ top: 0 });
  }

  function itemLabel(key) {
    return state.config?.item_types.find((t) => t.key === key)?.label || key;
  }

  function conditionLabel(key) {
    return state.config?.conditions.find((c) => c.key === key)?.label || key;
  }

  function timeAgo(ts) {
    const s = Math.round((Date.now() - ts) / 1000);
    if (s < 60) return "just now";
    if (s < 3600) return `${Math.floor(s / 60)} min ago`;
    if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
    return `${Math.floor(s / 86400)} d ago`;
  }

  async function api(path, options = {}, timeoutMs = 70000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(path, { ...options, signal: controller.signal });
      const data = await res.json().catch(() => ({}));
      if (res.status === 413) throw new Error("That photo is too large. Please try a smaller one.");
      if (!res.ok) throw new Error(data.detail || `Something went wrong (${res.status}). Please try again.`);
      return data;
    } catch (err) {
      if (err.name === "AbortError") throw new Error("This is taking too long. Check your connection and try again.");
      if (err instanceof TypeError) throw new Error("Can't reach the server. Check your connection and try again.");
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }

  function loadImage(blob) {
    const url = URL.createObjectURL(blob);
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = url;
    }).finally(() => setTimeout(() => URL.revokeObjectURL(url), 0));
  }

  function drawScaled(img, maxSide) {
    const scale = Math.min(1, maxSide / Math.max(img.naturalWidth, img.naturalHeight));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(img.naturalWidth * scale);
    canvas.height = Math.round(img.naturalHeight * scale);
    canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
    return canvas;
  }

  // Shrink phone photos (often 4–12 MB) to ~1280 px before upload; also make a tiny thumbnail for history.
  async function prepareImage(file) {
    try {
      const img = await loadImage(file);
      const upload = await new Promise((resolve) => drawScaled(img, 1280).toBlob(resolve, "image/jpeg", 0.85));
      const thumbCanvas = drawScaled(img, 160);
      return { blob: upload || file, thumb: thumbCanvas.toDataURL("image/jpeg", 0.7) };
    } catch {
      return { blob: file, thumb: null };
    }
  }

  // ---------- history & impact ----------
  const scans = {
    all: () => store.get(HISTORY_KEY, []),
    save: (items) => store.set(HISTORY_KEY, items.slice(0, HISTORY_LIMIT)),
    stats: () => store.get(STATS_KEY, { sorted: 0, hazardous: 0 }),
  };

  function recordNewResult() {
    const { result, current } = state;
    const stats = scans.stats();
    stats.sorted += 1;
    if (result.guidance.hazard) stats.hazardous += 1;
    store.set(STATS_KEY, stats);
    scans.save([
      { id: result.request_id, ts: Date.now(), thumb: state.resultThumb, result, current, hazardCounted: result.guidance.hazard },
      ...scans.all().filter((e) => e.id !== result.request_id),
    ]);
  }

  function updateHistoryEntry() {
    const items = scans.all();
    const entry = items.find((e) => e.id === state.result.request_id);
    if (!entry) return;
    const hazard = state.result.guidance.hazard;
    if (entry.hazardCounted !== hazard) {
      const stats = scans.stats();
      stats.hazardous = Math.max(0, stats.hazardous + (hazard ? 1 : -1));
      store.set(STATS_KEY, stats);
      entry.hazardCounted = hazard;
    }
    entry.result = state.result;
    entry.current = state.current;
    scans.save(items);
  }

  function entryTitle(result, current) {
    return current.item_type !== result.identification.item_type ? itemLabel(current.item_type) : result.identification.item_name;
  }

  function chevron() {
    const svg = icon("chevron");
    svg.classList.add("chev");
    return svg;
  }

  function renderImpact() {
    const items = scans.all();
    const stats = scans.stats();
    $("impact-block").hidden = stats.sorted === 0;
    $("stat-sorted").textContent = stats.sorted;
    $("stat-hazard").textContent = stats.hazardous;
    $("recent").replaceChildren(...items.slice(0, 4).map((e) => {
      const g = e.result.guidance;
      return h("li", null,
        h("button", { type: "button", class: `tone-${g.tone}`, onclick: () => openFromHistory(e) },
          e.thumb ? h("img", { src: e.thumb, alt: "" }) : h("span", { class: "ph" }, icon(g.stream)),
          h("span", { class: "r-main" },
            h("span", { class: "r-title" }, entryTitle(e.result, e.current)),
            h("span", { class: "r-sub" }, h("span", { class: "dot" }), `${g.stream_label} · ${timeAgo(e.ts)}`)),
          chevron()));
    }));
  }

  function openFromHistory(entry) {
    state.result = entry.result;
    state.current = entry.current;
    state.resultThumb = entry.thumb;
    renderResult();
    show("screen-result");
    window.history.pushState({ screen: "result" }, "");
  }

  // ---------- capture screen ----------
  function updateSubmit() {
    $("submit").disabled = !state.image && !$("note").value.trim();
  }

  function showCaptureError(message) {
    const el = $("capture-error");
    el.textContent = message;
    el.hidden = !message;
  }

  async function handleFile(file) {
    if (!file) return;
    if (file.type && !file.type.startsWith("image/")) {
      showCaptureError("That file isn't an image. Please choose a photo.");
      return;
    }
    showCaptureError("");
    const { blob, thumb } = await prepareImage(file);
    state.image = blob;
    state.thumb = thumb;
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = URL.createObjectURL(blob);
    $("preview").src = state.previewUrl;
    $("vf-empty").hidden = true;
    $("vf-photo").hidden = false;
    $("viewfinder").classList.add("has-photo");
    $("gallery-btn").hidden = true;
    haptic(10);
    updateSubmit();
  }

  function clearPhoto() {
    state.image = null;
    state.thumb = null;
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = null;
    $("preview").removeAttribute("src");
    $("vf-empty").hidden = false;
    $("vf-photo").hidden = true;
    $("viewfinder").classList.remove("has-photo");
    $("gallery-btn").hidden = false;
    for (const id of ["camera-input", "retake-input", "gallery-input"]) $(id).value = "";
    updateSubmit();
  }

  function startLoading() {
    const img = $("loading-img");
    img.hidden = !state.previewUrl;
    $("loading-placeholder").hidden = !!state.previewUrl;
    if (state.previewUrl) img.src = state.previewUrl;

    const steps = [...$("progress").children];
    const setStep = (n) => steps.forEach((li, i) => {
      li.classList.toggle("done", i < n);
      li.classList.toggle("active", i === n);
    });
    setStep(0);
    const timers = LOADING_STEPS_MS.slice(1).map((ms, i) => setTimeout(() => setStep(i + 1), ms));

    let tip = Math.floor(Math.random() * TIPS.length);
    const tipEl = $("tip-text");
    tipEl.textContent = TIPS[tip];
    const tipTimer = setInterval(() => {
      tipEl.style.opacity = 0;
      setTimeout(() => { tip = (tip + 1) % TIPS.length; tipEl.textContent = TIPS[tip]; tipEl.style.opacity = 1; }, 300);
    }, 5000);

    show("screen-loading");
    return {
      finish: async () => {
        timers.forEach(clearTimeout);
        clearInterval(tipTimer);
        setStep(steps.length);
        await new Promise((r) => setTimeout(r, 280));
      },
      cancel: () => { timers.forEach(clearTimeout); clearInterval(tipTimer); },
    };
  }

  async function submit() {
    showCaptureError("");
    const form = new FormData();
    if (state.image) form.append("image", state.image, "photo.jpg");
    form.append("note", $("note").value.trim());
    form.append("region", $("region").value);

    const loading = startLoading();
    try {
      const data = await api("/api/identify", { method: "POST", body: form });
      await loading.finish();
      state.result = data;
      state.current = {
        item_type: data.identification.item_type,
        material: data.identification.material,
        condition: data.identification.condition,
        confirmed: false,
      };
      state.resultThumb = state.thumb;
      recordNewResult();
      renderResult({ fresh: true });
      show("screen-result");
      window.history.pushState({ screen: "result" }, "");
      haptic(data.guidance.hazard ? [40, 70, 40] : 15);
    } catch (err) {
      loading.cancel();
      show("screen-capture");
      showCaptureError(err.message);
    }
  }

  function runExample(text) {
    clearPhoto();
    $("note").value = text;
    updateSubmit();
    submit();
  }

  function reset() {
    clearPhoto();
    $("note").value = "";
    state.result = null;
    state.current = null;
    showCaptureError("");
    updateSubmit();
    renderImpact();
    show("screen-capture");
  }

  // ---------- result screen ----------
  async function refreshGuidance(changes, markConfirmed) {
    const root = $("screen-result");
    root.classList.add("updating");
    try {
      const next = { ...state.current, ...changes };
      const data = await api("/api/guidance", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          item_type: next.item_type, material: next.material, condition: next.condition,
          region: $("region").value, request_id: state.result.request_id,
        }),
      }, 20000);
      const changedStream = data.guidance.stream !== state.result.guidance.stream;
      state.current = { ...next, item_type: data.item_type, condition: data.condition, confirmed: state.current.confirmed || markConfirmed };
      state.result.guidance = data.guidance;
      updateHistoryEntry();
      renderResult({ fresh: changedStream });
      if (changedStream && data.guidance.hazard) haptic([40, 70, 40]);
    } catch (err) {
      toast(err.message);
    } finally {
      root.classList.remove("updating");
    }
  }

  function renderHero(ident, cur, g) {
    const changed = cur.item_type !== ident.item_type;
    const meta = [itemLabel(cur.item_type)];
    if (!changed && ident.material !== "unknown") meta.push(ident.material);
    if (cur.condition !== "unknown") meta.push(conditionLabel(cur.condition).toLowerCase());

    const pct = Math.round(ident.confidence * 100);
    const level = pct >= 75 ? ["lvl-high", "High confidence"] : pct >= 50 ? ["lvl-mid", "Medium confidence"] : ["lvl-low", "Low confidence"];
    const bar = h("span");
    requestAnimationFrame(() => requestAnimationFrame(() => { bar.style.width = `${Math.max(pct, 4)}%`; }));

    const thumb = state.resultThumb
      ? h("img", { class: "thumb", src: state.resultThumb, alt: "Your photo" })
      : h("div", { class: `thumb tone-${g.tone}`, "aria-hidden": "true" }, icon(g.stream));

    return h("div", { class: "result-hero" },
      thumb,
      h("div", { class: "info" },
        h("p", { class: "eyebrow" }, cur.confirmed ? "You confirmed" : "We think this is"),
        h("h2", null, changed ? itemLabel(cur.item_type) : ident.item_name),
        h("p", { class: "meta" }, meta.join(" · ")),
        cur.confirmed
          ? h("span", { class: "pill-confirmed" }, icon("check"), "Confirmed by you")
          : h("div", { class: `confidence ${level[0]}` },
              h("div", { class: "confidence-bar", role: "img", "aria-label": `${pct}% confident` }, bar),
              h("span", { class: "confidence-label" }, `${level[1]} · ${pct}%`))));
  }

  function renderConfirm(confirm) {
    const chips = confirm.candidates.map((c) =>
      h("button", {
        class: `chip${c.hazardous ? " hazard" : ""}`, type: "button",
        onclick: () => refreshGuidance({ item_type: c.item_type }, true),
      }, c.hazardous ? icon("alert") : null, c.label));
    chips.push(h("button", {
      class: "chip link", type: "button",
      onclick: () => { const d = $("correct"); d.open = true; d.scrollIntoView({ behavior: "smooth", block: "center" }); },
    }, "Something else"));
    return h("div", { class: "card confirm", role: "group", "aria-label": "Confirm the item" },
      h("p", null, confirm.reason),
      h("div", { class: "chips" }, chips));
  }

  function renderSteps(steps) {
    const count = h("span", { class: "count" }, `0/${steps.length}`);
    const list = h("ol", { class: "steps" });
    let done = 0;
    steps.forEach((s, i) => {
      const btn = h("button", { class: "step", type: "button", "aria-pressed": "false" },
        h("span", { class: "num" }, h("b", null, i + 1), icon("check")),
        h("span", { class: "txt" }, s));
      btn.addEventListener("click", () => {
        const on = btn.classList.toggle("done");
        btn.setAttribute("aria-pressed", String(on));
        done += on ? 1 : -1;
        count.textContent = `${done}/${steps.length}`;
        haptic(8);
        if (on && done === steps.length) toast("All prepared. It's ready to go!");
      });
      list.append(h("li", null, btn));
    });
    return h("section", { class: "card" },
      h("h3", { class: "card-title" }, icon("list"), "How to prepare it", count),
      list,
      h("p", { class: "steps-hint" }, "Tap each step as you do it."));
  }

  function renderCorrection(cur) {
    const groups = {};
    for (const t of state.config.item_types) (groups[t.group] ||= []).push(t);
    const typeSelect = h("select", { id: "fix-type" },
      Object.entries(groups).map(([group, items]) =>
        h("optgroup", { label: group },
          items.map((t) => h("option", { value: t.key, selected: t.key === cur.item_type }, t.label)))));
    const condSelect = h("select", { id: "fix-condition" },
      state.config.conditions.map((c) => h("option", { value: c.key, selected: c.key === cur.condition }, c.label)));

    return h("details", { class: "card", id: "correct" },
      h("summary", null, "Not right? Correct it"),
      h("div", { class: "form-grid" },
        h("label", null, "What is it?", typeSelect),
        h("label", null, "What condition is it in?", condSelect),
        h("button", {
          class: "btn btn-primary", type: "button",
          onclick: () => refreshGuidance({ item_type: typeSelect.value, condition: condSelect.value }, true),
        }, "Update guidance")));
  }

  function renderFeedback() {
    const box = h("div", { class: "card" });
    const send = async (helpful, comment = "") => {
      box.replaceChildren(h("p", { class: "thanks" }, icon("check"), "Thanks! Your feedback helps improve the rules."));
      try {
        await api("/api/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            request_id: state.result.request_id, helpful, comment,
            item_type: state.current.item_type, condition: state.current.condition,
            rule_id: state.result.guidance.rule_id,
          }),
        }, 15000);
      } catch { /* feedback is best-effort */ }
    };
    const askMore = () => {
      const text = h("textarea", { rows: "2", maxlength: "500", placeholder: "What was wrong? (optional)" });
      box.replaceChildren(
        h("p", { class: "feedback-title" }, "Sorry about that. What should it have said?"),
        h("div", { class: "feedback-more" }, text,
          h("button", { class: "btn btn-primary btn-small", type: "button", onclick: () => send(false, text.value.trim()) }, "Send")));
      text.focus();
    };
    box.append(h("div", { class: "feedback" },
      h("p", null, "Was this helpful?"),
      h("div", { class: "actions" },
        h("button", { class: "btn btn-ghost btn-small", type: "button", onclick: () => send(true) }, "Yes"),
        h("button", { class: "btn btn-ghost btn-small", type: "button", onclick: askMore }, "No"))));
    return box;
  }

  async function shareResult() {
    const g = state.result.guidance;
    const title = entryTitle(state.result, state.current);
    const text = [
      `${title}: ${g.stream_label}`,
      "",
      "How to prepare it:",
      ...g.prep_steps.map((s, i) => `${i + 1}. ${s}`),
      "",
      `Where: ${g.where_to_take}`,
    ].join("\n");
    const url = location.origin;
    try {
      if (navigator.share) {
        await navigator.share({ title: `How to dispose of: ${title}`, text, url });
      } else {
        await navigator.clipboard.writeText(`${text}\n\n${url}`);
        toast("Copied to clipboard");
      }
    } catch (err) {
      if (err?.name !== "AbortError") toast("Couldn't share right now");
    }
  }

  function renderResult({ fresh = false } = {}) {
    const { identification: ident, confirm, guidance: g } = state.result;
    const cur = state.current;
    const root = $("screen-result");

    const children = [
      renderHero(ident, cur, g),
      ident.reason && !cur.confirmed ? h("p", { class: "reason" }, `"${ident.reason}"`) : null,
      confirm.needed && !cur.confirmed ? renderConfirm(confirm) : null,

      h("div", { class: `stream tone-${g.tone}${g.hazard ? " is-hazard" : ""}` },
        h("div", { class: "badge" }, icon(g.stream)),
        h("div", { style: "position:relative;z-index:1" },
          h("p", { class: "eyebrow" }, "Goes to"),
          h("p", { class: "stream-name" }, g.stream_label))),

      g.hazard
        ? h("div", { class: "alert alert-danger", role: "alert" }, icon("alert"),
            h("div", null, h("strong", null, "Keep this out of regular bins"), g.hazard_note))
        : g.hazard_note
          ? h("div", { class: "alert alert-warn" }, icon("alert"),
              h("div", null, h("strong", null, "Handle with care"), g.hazard_note))
          : null,

      renderSteps(g.prep_steps),

      h("section", { class: "card" },
        h("h3", { class: "card-title" }, icon("pin"), "Where to take it"),
        h("p", { class: "where" }, g.where_to_take)),

      h("div", { class: "btn-row" },
        h("button", { class: "btn btn-secondary", type: "button", onclick: shareResult, "aria-label": "Share this guidance" }, icon("share"), "Share"),
        h("button", { class: "btn btn-primary", type: "button", onclick: () => window.history.back() }, icon("camera"), "Scan another")),

      h("div", { style: "height:12px" }),
      renderCorrection(cur),
      renderFeedback(),
      h("p", { class: "source" }, `Rule ${g.rule_id} · ${g.source}`),
    ].filter(Boolean);

    children.forEach((c, i) => c.style.setProperty("--i", i));
    root.classList.toggle("reveal", fresh);
    root.replaceChildren(...children);
  }

  // ---------- init ----------
  function renderStaticBlocks() {
    $("examples").replaceChildren(...EXAMPLES.map((ex) =>
      h("button", { class: `chip tone-${ex.tone}`, type: "button", onclick: () => runExample(ex.label) },
        h("span", { class: "dot" }), ex.label)));
    $("bins").replaceChildren(...BINS.map((b) =>
      h("div", { class: `bin tone-${b.tone}` },
        h("span", { class: "bin-icon" }, icon(b.stream)),
        h("strong", null, b.label),
        h("span", null, b.text))));
  }

  async function init() {
    for (const id of ["camera-input", "retake-input", "gallery-input"]) {
      $(id).addEventListener("change", (e) => handleFile(e.target.files[0]));
    }
    $("clear-photo").addEventListener("click", clearPhoto);
    $("note").addEventListener("input", updateSubmit);
    $("submit").addEventListener("click", submit);
    window.addEventListener("popstate", () => { if (!$("screen-result").hidden) reset(); });

    renderStaticBlocks();

    try {
      state.config = await api("/api/config", {}, 15000);
    } catch (err) {
      showCaptureError(err.message);
      return;
    }
    renderImpact();

    const region = $("region");
    const saved = store.get("region", null);
    for (const r of state.config.regions) region.append(h("option", { value: r.key }, r.label));
    region.value = state.config.regions.some((r) => r.key === saved) ? saved : state.config.default_region;
    region.addEventListener("change", () => {
      store.set("region", region.value);
      if (state.result && !$("screen-result").hidden) refreshGuidance({}, false);
    });
  }

  init();
})();
