let fpsChart, tempChart;

function makeChart(ctx, label, color) {
  return new Chart(ctx, {
    type: "line",
    data: { labels: [], datasets: [{ label, data: [], borderColor: color, fill: false }] },
    options: {
      animation: false,
      responsive: true,
      plugins: { legend: { labels: { color: "rgba(255,255,255,.75)" } } },
      scales: {
        x: { display: false },
        y: { ticks: { color: "rgba(255,255,255,.65)" }, grid: { color: "rgba(255,255,255,.08)" } },
      }
    }
  });
}

const lastTextById = new Map();
function setTextWithFlash(id, nextText) {
  const el = document.getElementById(id);
  if (!el) return;
  const prev = lastTextById.get(id);
  el.textContent = nextText;
  if (prev !== undefined && prev !== nextText) {
    el.classList.remove("flash");
    void el.offsetWidth;
    el.classList.add("flash");
  }
  lastTextById.set(id, nextText);
}

async function fetchLatest() {
  try {
    const res = await fetch("/latest");
    const json = await res.json();
    if (!json.ok) return;
    const d = json.data;

    setTextWithFlash("fps", d.fps.toFixed(1));
    setTextWithFlash("temp", d.cpu_temp_c.toFixed(1));
    setTextWithFlash("latency", d.latency_ms.toFixed(1));
    setTextWithFlash("detect", String(d.detect_count));

    const tempEl = document.getElementById("temp");
    tempEl.classList.remove("warn", "danger");
    if (d.cpu_temp_c > 80) tempEl.classList.add("danger");
    else if (d.cpu_temp_c > 75) tempEl.classList.add("warn");
  } catch (_) {}
}

async function fetchSeries() {
  try {
    const res = await fetch("/timeseries?minutes=10");
    const json = await res.json();
    if (!json.ok) return;

    const labels = json.data.map(x => new Date(x.ts).toLocaleTimeString());
    const fps = json.data.map(x => x.fps);
    const temp = json.data.map(x => x.cpu_temp_c);

    fpsChart.data.labels = labels;
    fpsChart.data.datasets[0].data = fps;
    fpsChart.update();

    tempChart.data.labels = labels;
    tempChart.data.datasets[0].data = temp;
    tempChart.update();
  } catch (_) {}
}

async function tickMetrics() {
  await fetchLatest();
  await fetchSeries();
}

// ─── Live Camera (snapshot vs Tailscale full stream) ─────────────────────────

const VideoMode = Object.freeze({ SNAPSHOT: "snapshot", FULL: "full" });

let currentMode = VideoMode.SNAPSHOT;
let snapshotTimer = null;
let cachedStreamUrl = null;

function elements() {
  return {
    img: document.getElementById("videoImg"),
    overlay: document.getElementById("videoOverlay"),
    badge: document.getElementById("videoMode"),
    btnSnap: document.getElementById("btnSnapshot"),
    btnFull: document.getElementById("btnFullStream"),
    tsUrl: document.getElementById("tailscaleUrl"),
  };
}

function setActiveButton(mode) {
  const { btnSnap, btnFull, badge } = elements();
  btnSnap.classList.toggle("active", mode === VideoMode.SNAPSHOT);
  btnFull.classList.toggle("active", mode === VideoMode.FULL);
  badge.textContent = mode === VideoMode.SNAPSHOT ? "snapshot 1fps" : "Tailscale MJPEG";
}

function startSnapshotMode() {
  const { img, overlay } = elements();
  currentMode = VideoMode.SNAPSHOT;
  setActiveButton(currentMode);
  overlay.hidden = true;
  if (snapshotTimer) clearInterval(snapshotTimer);
  const refresh = () => { img.src = `/snapshot.jpg?t=${Date.now()}`; };
  refresh();
  snapshotTimer = setInterval(refresh, 1000);
}

async function startFullStreamMode() {
  const { img, overlay, tsUrl } = elements();

  if (!cachedStreamUrl) {
    try {
      const res = await fetch("/stream-config");
      const json = await res.json();
      cachedStreamUrl = json.stream_url || null;
    } catch (_) {
      cachedStreamUrl = null;
    }
  }

  if (!cachedStreamUrl) {
    overlay.hidden = false;
    tsUrl.textContent = "(Pi chưa đăng ký URL stream)";
    return;
  }

  currentMode = VideoMode.FULL;
  setActiveButton(currentMode);
  if (snapshotTimer) { clearInterval(snapshotTimer); snapshotTimer = null; }
  overlay.hidden = true;
  tsUrl.textContent = cachedStreamUrl;

  // Probe: if browser cannot reach Tailscale IP within ~3s, fall back.
  const probeTimer = setTimeout(() => {
    if (currentMode === VideoMode.FULL && !img.complete) {
      handleFullStreamError();
    }
  }, 3500);

  img.onload = () => clearTimeout(probeTimer);
  img.onerror = () => { clearTimeout(probeTimer); handleFullStreamError(); };
  img.src = cachedStreamUrl;
}

function handleFullStreamError() {
  const { overlay, tsUrl } = elements();
  overlay.hidden = false;
  tsUrl.textContent = cachedStreamUrl || "(không có)";
  // Behind the overlay, keep snapshot fallback running so something is visible.
  startSnapshotMode();
  setActiveButton(VideoMode.SNAPSHOT);
}

async function showRegisteredStreamUrl() {
  const { tsUrl } = elements();
  if (!tsUrl) return;
  try {
    const res = await fetch("/stream-config");
    const json = await res.json();
    const u = json.stream_url || null;
    cachedStreamUrl = u;
    tsUrl.textContent = u || "--";
  } catch (_) {
    tsUrl.textContent = "--";
  }
}

window.onload = () => {
  fpsChart = makeChart(document.getElementById("fpsChart"), "FPS", "#2563eb");
  tempChart = makeChart(document.getElementById("tempChart"), "CPU Temp", "#dc2626");

  const { btnSnap, btnFull } = elements();
  btnSnap.addEventListener("click", () => startSnapshotMode());
  btnFull.addEventListener("click", () => startFullStreamMode());

  startSnapshotMode();
  showRegisteredStreamUrl();
  tickMetrics();
  setInterval(tickMetrics, 2000);

  // ─── Servo Control ────────────────────────────────────────────────────────
  const btnCCW = document.getElementById("btnServoCCW");
  const btnCW = document.getElementById("btnServoCW");

  function sendServoCmd(dir) {
    fetch("/servo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ direction: dir })
    }).catch(() => {});
  }

  const startCCW = (e) => { e.preventDefault(); btnCCW.classList.add("active"); sendServoCmd(-1); };
  const startCW  = (e) => { e.preventDefault(); btnCW.classList.add("active"); sendServoCmd(1);  };
  const stop     = (e) => { e.preventDefault(); btnCCW.classList.remove("active"); btnCW.classList.remove("active"); sendServoCmd(0);  };

  btnCCW.addEventListener("mousedown", startCCW);
  btnCCW.addEventListener("touchstart", startCCW);
  btnCW.addEventListener("mousedown", startCW);
  btnCW.addEventListener("touchstart", startCW);

  // Stop on mouse up or mouse leave
  ["mouseup", "mouseleave", "touchend", "touchcancel"].forEach(evt => {
    btnCCW.addEventListener(evt, stop);
    btnCW.addEventListener(evt, stop);
  });
};
