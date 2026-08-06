"use strict";

/* ==========================================================================
   Sprite Editor — SPA vanilla contra la API REST del pipeline.
   El aid se parsea de la ruta /editor/{aid}. Sin dependencias externas.
   ========================================================================== */

/* --------------------------------------------------------------- contexto */

const AID = (() => {
  const parts = window.location.pathname.split("/").filter(Boolean);
  const i = parts.indexOf("editor");
  if (i >= 0 && parts[i + 1]) return decodeURIComponent(parts[i + 1]);
  return parts.length ? decodeURIComponent(parts[parts.length - 1]) : "";
})();

const API = `/api/animations/${encodeURIComponent(AID)}`;

const state = {
  frameset: null,     // manifiesto actual (version=working)
  images: [],         // Image precargada por cuadro
  selected: -1,       // índice de cuadro seleccionado (-1 = ninguno)
  current: 0,         // cuadro visible en el preview
  playing: true,
  speed: 1,
  cacheBust: Date.now(),
  busy: false,        // hay una edición en vuelo
  dragFrom: null,     // índice origen del drag & drop
  lastTick: 0,
  acc: 0,             // ms acumulados dentro del cuadro actual
};

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ toast */

let toastTimer = null;

function toast(msg) {
  const el = $("toast");
  el.textContent = String(msg);
  el.classList.remove("hidden");
  // reinicia la animación de entrada
  el.classList.remove("show");
  void el.offsetWidth;
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.classList.remove("show");
    el.classList.add("hidden");
  }, 4000);
}

/* -------------------------------------------------------------------- API */

async function apiFetch(path, opts = {}, { silent = false } = {}) {
  let res;
  try {
    res = await fetch(path, opts);
  } catch (err) {
    if (!silent) toast(`Error de red: ${err.message}`);
    throw err;
  }
  if (!res.ok) {
    let detail = `Error ${res.status}`;
    try {
      const data = await res.json();
      if (data && data.detail !== undefined) {
        detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
      }
    } catch (_) {
      /* cuerpo no JSON: se queda el código de estado */
    }
    if (!silent) toast(detail);
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  const ctype = res.headers.get("content-type") || "";
  return ctype.includes("application/json") ? res.json() : res;
}

function frameURL(file) {
  return `${API}/frames/${encodeURIComponent(file)}?version=working&v=${state.cacheBust}`;
}

/* --------------------------------------------------------- carga de datos */

function loadImage(url) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(img); // un cuadro roto no bloquea el editor
    img.src = url;
  });
}

async function preloadImages() {
  const fs = state.frameset;
  state.images = fs ? await Promise.all(fs.frames.map((f) => loadImage(frameURL(f.file)))) : [];
}

async function refreshFrameset({ silent = false } = {}) {
  state.cacheBust = Date.now(); // cache-busting tras cada operación
  const fs = await apiFetch(`${API}/frameset?version=working`, {}, { silent });
  state.frameset = fs;
  await preloadImages();
  const n = fs.frames.length;
  if (state.selected >= n) state.selected = n - 1;
  if (state.current >= n) state.current = Math.max(0, n - 1);
  state.acc = 0;
  renderTimeline();
  updateFrameActions();
  fitPreviewCanvas();
  drawPreview();
}

async function loadAnimationInfo({ silent = false } = {}) {
  const anim = await apiFetch(API, {}, { silent });
  $("anim-name").textContent = anim.name || AID;
  const actionEl = $("anim-action");
  actionEl.textContent = anim.action || "";
  actionEl.hidden = !anim.action;
  const statusEl = $("anim-status");
  statusEl.textContent = anim.status || "?";
  statusEl.dataset.status = anim.status || "";
  return anim;
}

/* Mientras la animación se procesa, sondea hasta que esté lista. */
async function pollUntilReady() {
  let anim = null;
  try {
    anim = await loadAnimationInfo({ silent: true });
  } catch (_) {
    /* reintenta más abajo */
  }
  const status = anim ? anim.status : "";
  if (status === "ready" || status === "failed") {
    if (status === "failed" && anim && anim.error) toast(anim.error);
    try {
      await refreshFrameset({ silent: true });
    } catch (_) {
      /* puede no haber frameset aún (fallo temprano) */
    }
    return;
  }
  try {
    await refreshFrameset({ silent: true });
  } catch (_) {
    /* aún sin cuadros: normal durante la generación */
  }
  setTimeout(pollUntilReady, 2500);
}

/* --------------------------------------------------------------- ediciones */

async function sendEdit(op) {
  if (state.busy) return;
  state.busy = true;
  document.body.classList.add("busy");
  try {
    await apiFetch(`${API}/edits`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(op),
    });
    await refreshFrameset(); // recarga el frameset working tras CADA operación
  } catch (_) {
    /* el toast ya se mostró en apiFetch */
  } finally {
    state.busy = false;
    document.body.classList.remove("busy");
  }
}

async function undoEdit() {
  if (state.busy) return;
  state.busy = true;
  document.body.classList.add("busy");
  try {
    await apiFetch(`${API}/undo`, { method: "POST" });
    await refreshFrameset();
    toast("Edición deshecha");
  } catch (_) {
    /* toast ya mostrado */
  } finally {
    state.busy = false;
    document.body.classList.remove("busy");
  }
}

/* ---------------------------------------------------------------- preview */

function fitPreviewCanvas() {
  const canvas = $("preview-canvas");
  const img = state.images.find((im) => im && im.naturalWidth);
  if (img) {
    if (canvas.width !== img.naturalWidth) canvas.width = img.naturalWidth;
    if (canvas.height !== img.naturalHeight) canvas.height = img.naturalHeight;
  }
}

function effectiveDuration(i) {
  const fs = state.frameset;
  const f = fs && fs.frames[i];
  const base = f && f.duration_ms > 0 ? f.duration_ms : 100;
  return base / state.speed; // duration_ms escalado por la velocidad
}

function drawPreview() {
  const canvas = $("preview-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const fs = state.frameset;
  if (!fs || !fs.frames.length) {
    $("frame-counter").textContent = "0 / 0";
    return;
  }
  const img = state.images[state.current];
  if (img && img.naturalWidth) {
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  }
  $("frame-counter").textContent = `${state.current + 1} / ${fs.frames.length}`;
  markPlayingThumb();
}

function markPlayingThumb() {
  const cells = $("timeline").children;
  for (let i = 0; i < cells.length; i++) {
    cells[i].classList.toggle("playing", i === state.current);
  }
}

function tick(ts) {
  requestAnimationFrame(tick);
  const fs = state.frameset;
  if (!fs || !fs.frames.length) return;
  if (!state.playing) {
    state.lastTick = ts;
    return;
  }
  if (!state.lastTick) state.lastTick = ts;
  const dt = Math.min(ts - state.lastTick, 500);
  state.lastTick = ts;
  state.acc += dt;
  let dur = effectiveDuration(state.current);
  let guard = 0;
  while (state.acc >= dur && guard++ < fs.frames.length * 4) {
    state.acc -= dur;
    state.current = (state.current + 1) % fs.frames.length;
    dur = effectiveDuration(state.current);
  }
  drawPreview();
}

function setPlaying(playing) {
  state.playing = playing;
  $("btn-play").innerHTML = playing ? "&#10074;&#10074;" : "&#9654;";
  $("btn-play").title = playing ? "Pausa" : "Play";
}

/* --------------------------------------------------------------- timeline */

function frameIsFlagged(f) {
  return (f.flags || []).some((fl) => fl === "anomaly" || fl === "review");
}

function renderTimeline() {
  const tl = $("timeline");
  tl.innerHTML = "";
  const fs = state.frameset;
  if (!fs || !fs.frames.length) {
    const empty = document.createElement("div");
    empty.className = "timeline-empty";
    empty.textContent = "Sin cuadros todavía";
    tl.appendChild(empty);
    return;
  }
  fs.frames.forEach((f, i) => {
    const cell = document.createElement("div");
    cell.className = "thumb";
    cell.dataset.index = String(i);
    cell.draggable = true;
    if (i === state.selected) cell.classList.add("selected");
    const flagged = frameIsFlagged(f);
    if (flagged) cell.classList.add("flagged");

    const img = document.createElement("img");
    img.src = frameURL(f.file);
    img.alt = `Cuadro ${i}`;
    img.draggable = false;
    cell.appendChild(img);

    if (flagged) {
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = "!";
      badge.title = (f.flags || []).join(", ");
      cell.appendChild(badge);
    }

    const num = document.createElement("span");
    num.className = "thumb-num";
    num.textContent = String(i);
    cell.appendChild(num);

    if (i === state.selected) {
      const input = document.createElement("input");
      input.type = "number";
      input.className = "dur-input";
      input.min = "10";
      input.step = "10";
      input.value = String(f.duration_ms);
      input.title = "Duración del cuadro (ms)";
      input.addEventListener("click", (e) => e.stopPropagation());
      input.addEventListener("pointerdown", (e) => e.stopPropagation());
      input.addEventListener("change", () => {
        const v = parseInt(input.value, 10);
        if (Number.isFinite(v) && v > 0) {
          sendEdit({ op: "set_duration", index: i, duration_ms: v });
        } else {
          input.value = String(f.duration_ms);
        }
      });
      cell.appendChild(input);
    } else {
      const dur = document.createElement("span");
      dur.className = "thumb-dur";
      dur.textContent = `${f.duration_ms} ms`;
      cell.appendChild(dur);
    }

    cell.addEventListener("click", () => selectFrame(i));

    /* ---- drag & drop nativo para reordenar ---- */
    cell.addEventListener("dragstart", (e) => {
      state.dragFrom = i;
      cell.classList.add("dragging");
      e.dataTransfer.effectAllowed = "move";
      e.dataTransfer.setData("text/plain", String(i));
    });
    cell.addEventListener("dragend", () => {
      state.dragFrom = null;
      cell.classList.remove("dragging");
      clearDropMarks();
    });
    cell.addEventListener("dragover", (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
      cell.classList.add("drop-target");
    });
    cell.addEventListener("dragleave", () => cell.classList.remove("drop-target"));
    cell.addEventListener("drop", (e) => {
      e.preventDefault();
      clearDropMarks();
      let from = state.dragFrom;
      if (from === null || from === undefined) {
        from = parseInt(e.dataTransfer.getData("text/plain"), 10);
      }
      state.dragFrom = null;
      if (!Number.isFinite(from) || from === i) return;
      // Permutación completa: mover `from` a la posición `i`.
      const order = fs.frames.map((_, k) => k);
      order.splice(from, 1);
      order.splice(i, 0, from);
      state.selected = i;
      sendEdit({ op: "reorder", order });
    });

    tl.appendChild(cell);
  });
  markPlayingThumb();
}

function clearDropMarks() {
  document.querySelectorAll(".thumb.drop-target").forEach((el) => el.classList.remove("drop-target"));
}

function selectFrame(i) {
  state.selected = i;
  renderTimeline();
  updateFrameActions();
  const cell = $("timeline").querySelector(`.thumb[data-index="${i}"]`);
  if (cell) cell.scrollIntoView({ block: "nearest", inline: "nearest" });
}

function updateFrameActions() {
  const fs = state.frameset;
  const has = fs && state.selected >= 0 && state.selected < fs.frames.length;
  $("selected-label").textContent = has ? String(state.selected) : "—";
  $("btn-delete").disabled = !has || fs.frames.length <= 1;
  $("btn-duplicate").disabled = !has;
  $("btn-interpolate").disabled = !has;
  $("btn-retouch").disabled = !has;
}

/* --------------------------------------------------------- modal de retoque */

const retouch = {
  index: -1,
  img: null,
  scale: 1,      // display px por px natural
  mask: null,    // canvas offscreen a tamaño natural, trazos blancos
  painting: false,
  lastPt: null,
  hoverPt: null,
};

function brushRadius() {
  return parseInt($("brush-radius").value, 10) || 18;
}

function openRetouch(i) {
  const img = state.images[i];
  if (!img || !img.naturalWidth) {
    toast("El cuadro aún no está cargado");
    return;
  }
  retouch.index = i;
  retouch.img = img;
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  retouch.scale = 512 / Math.max(w, h); // ampliado, lado mayor = 512 px
  const canvas = $("retouch-canvas");
  canvas.width = Math.max(1, Math.round(w * retouch.scale));
  canvas.height = Math.max(1, Math.round(h * retouch.scale));
  retouch.mask = document.createElement("canvas");
  retouch.mask.width = w;
  retouch.mask.height = h;
  retouch.painting = false;
  retouch.lastPt = null;
  retouch.hoverPt = null;
  $("retouch-index").textContent = String(i);
  $("retouch-modal").classList.remove("hidden");
  redrawRetouch();
}

function closeRetouch() {
  $("retouch-modal").classList.add("hidden");
  retouch.img = null;
  retouch.mask = null;
}

function redrawRetouch() {
  const canvas = $("retouch-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!retouch.img) return;
  ctx.imageSmoothingEnabled = retouch.scale < 1;
  ctx.drawImage(retouch.img, 0, 0, canvas.width, canvas.height);
  // superposición roja semitransparente donde hay máscara
  const tint = document.createElement("canvas");
  tint.width = retouch.mask.width;
  tint.height = retouch.mask.height;
  const tctx = tint.getContext("2d");
  tctx.drawImage(retouch.mask, 0, 0);
  tctx.globalCompositeOperation = "source-in";
  tctx.fillStyle = "#ff2d2d";
  tctx.fillRect(0, 0, tint.width, tint.height);
  ctx.globalAlpha = 0.55;
  ctx.drawImage(tint, 0, 0, canvas.width, canvas.height);
  ctx.globalAlpha = 1;
  // contorno del pincel bajo el cursor
  if (retouch.hoverPt) {
    ctx.beginPath();
    ctx.arc(retouch.hoverPt.x, retouch.hoverPt.y, brushRadius(), 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

function canvasPoint(e, canvas) {
  const r = canvas.getBoundingClientRect();
  return {
    x: (e.clientX - r.left) * (canvas.width / r.width),
    y: (e.clientY - r.top) * (canvas.height / r.height),
  };
}

function paintStroke(from, to) {
  const ctx = retouch.mask.getContext("2d");
  const s = retouch.scale;
  const r = Math.max(brushRadius() / s, 0.5); // radio en px naturales
  ctx.strokeStyle = "#ffffff";
  ctx.fillStyle = "#ffffff";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = r * 2;
  ctx.beginPath();
  ctx.moveTo(from.x / s, from.y / s);
  ctx.lineTo(to.x / s, to.y / s);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(to.x / s, to.y / s, r, 0, Math.PI * 2);
  ctx.fill();
}

function bindRetouchCanvas() {
  const canvas = $("retouch-canvas");
  canvas.addEventListener("pointerdown", (e) => {
    if (!retouch.img) return;
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);
    retouch.painting = true;
    const p = canvasPoint(e, canvas);
    retouch.lastPt = p;
    retouch.hoverPt = p;
    paintStroke(p, p);
    redrawRetouch();
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!retouch.img) return;
    const p = canvasPoint(e, canvas);
    retouch.hoverPt = p;
    if (retouch.painting && retouch.lastPt) {
      paintStroke(retouch.lastPt, p);
      retouch.lastPt = p;
    }
    redrawRetouch();
  });
  const stop = () => {
    retouch.painting = false;
    retouch.lastPt = null;
  };
  canvas.addEventListener("pointerup", stop);
  canvas.addEventListener("pointercancel", stop);
  canvas.addEventListener("pointerleave", () => {
    retouch.hoverPt = null;
    if (retouch.img) redrawRetouch();
  });
}

function clearMask() {
  if (!retouch.mask) return;
  retouch.mask.getContext("2d").clearRect(0, 0, retouch.mask.width, retouch.mask.height);
  redrawRetouch();
}

async function applyRetouch() {
  if (!retouch.img || !retouch.mask) return;
  const index = retouch.index;
  const w = retouch.mask.width;
  const h = retouch.mask.height;
  // PNG en escala de grises: fondo negro, pintado blanco, tamaño natural del cuadro.
  const out = document.createElement("canvas");
  out.width = w;
  out.height = h;
  const ctx = out.getContext("2d");
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(retouch.mask, 0, 0);
  const b64 = out.toDataURL("image/png").split(",")[1];
  closeRetouch();
  await sendEdit({ op: "retouch", index, mask_png_base64: b64 });
}

/* ------------------------------------------------------- modal de export */

function openExport() {
  $("export-result").hidden = true;
  $("export-result").innerHTML = "";
  $("export-modal").classList.remove("hidden");
}

function closeExport() {
  $("export-modal").classList.add("hidden");
}

async function doExport() {
  const formats = Array.from(
    document.querySelectorAll('#export-modal input[name="fmt"]:checked'),
  ).map((c) => c.value);
  if (!formats.length) {
    toast("Selecciona al menos un formato");
    return;
  }
  const btn = $("btn-do-export");
  btn.disabled = true;
  try {
    const res = await apiFetch(`${API}/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ formats }),
    });
    renderExportResult(res);
  } catch (_) {
    /* toast ya mostrado */
  } finally {
    btn.disabled = false;
  }
}

function renderExportResult(res) {
  const box = $("export-result");
  box.innerHTML = "";
  const title = document.createElement("h3");
  title.textContent = "Archivos generados";
  box.appendChild(title);
  const list = document.createElement("ul");
  (res.files || []).forEach((file) => {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = `${API}/exports/${encodeURIComponent(res.export_id)}/${file}`;
    a.textContent = file;
    a.setAttribute("download", "");
    li.appendChild(a);
    list.appendChild(li);
  });
  if (!list.children.length) {
    const li = document.createElement("li");
    li.textContent = "(vacío)";
    list.appendChild(li);
  }
  box.appendChild(list);
  box.hidden = false;
}

/* ------------------------------------------------------------------- init */

function bindUI() {
  $("btn-play").addEventListener("click", () => setPlaying(!state.playing));
  $("speed-slider").addEventListener("input", (e) => {
    state.speed = parseFloat(e.target.value) || 1;
    $("speed-value").textContent = `${state.speed}x`;
  });
  $("btn-undo").addEventListener("click", undoEdit);
  $("btn-export").addEventListener("click", openExport);

  $("btn-delete").addEventListener("click", () => {
    if (state.selected >= 0) sendEdit({ op: "delete", index: state.selected });
  });
  $("btn-duplicate").addEventListener("click", () => {
    if (state.selected >= 0) sendEdit({ op: "duplicate", index: state.selected });
  });
  $("btn-interpolate").addEventListener("click", () => {
    if (state.selected >= 0) sendEdit({ op: "interpolate", after_index: state.selected });
  });
  $("btn-retouch").addEventListener("click", () => {
    if (state.selected >= 0) openRetouch(state.selected);
  });

  bindRetouchCanvas();
  $("brush-radius").addEventListener("input", () => {
    $("brush-radius-value").textContent = `${brushRadius()} px`;
    if (retouch.img) redrawRetouch();
  });
  $("btn-mask-clear").addEventListener("click", clearMask);
  $("btn-mask-cancel").addEventListener("click", closeRetouch);
  $("btn-mask-apply").addEventListener("click", applyRetouch);

  $("btn-do-export").addEventListener("click", doExport);
  $("btn-export-close").addEventListener("click", closeExport);

  // clic en el fondo del modal = cerrar
  ["retouch-modal", "export-modal"].forEach((id) => {
    $(id).addEventListener("click", (e) => {
      if (e.target === $(id)) $(id).classList.add("hidden");
    });
  });
}

async function init() {
  bindUI();
  setPlaying(true);
  requestAnimationFrame(tick);
  await pollUntilReady();
}

document.addEventListener("DOMContentLoaded", init);
