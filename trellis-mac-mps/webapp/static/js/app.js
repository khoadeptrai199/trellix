const $ = (id) => document.getElementById(id);

const fileInput = $("file-input");
const dropzone = $("dropzone");
const dropzoneLabel = $("dropzone-label");
const previewImg = $("preview-img");
const examplesGrid = $("examples-grid");
const generateBtn = $("generate-btn");
const progressPanel = $("progress-panel");
const progressStage = $("progress-stage");
const progressBarFill = $("progress-bar-fill");
const progressDetail = $("progress-detail");
const progressElapsed = $("progress-elapsed");
const downloadRow = $("download-row");
const dlGlb = $("dl-glb");
const dlPly = $("dl-ply");
const historyList = $("history-list");

const livePreviewPanel = $("live-preview-panel");
const liveGsItem = $("live-gs-item");
const liveGsGif = $("live-gs-gif");
const liveMeshItem = $("live-mesh-item");
const liveMeshGif = $("live-mesh-gif");
const liveRfItem = $("live-rf-item");
const liveRfGif = $("live-rf-gif");
const liveMvItem = $("live-mv-item");
const liveMvImg = $("live-mv-img");
const liveMvCounter = $("live-mv-counter");

let selectedFile = null;
let selectedExample = null;
let pollTimer = null;

// Rough overall-progress weighting per stage, purely cosmetic (the real detail line below the bar
// always shows the exact tqdm text) -- stage durations vary wildly (diffusion sampling ~1min vs
// texture-bake optimization ~5-10min), so this just keeps the bar moving roughly monotonically
// instead of claiming a precise percentage we can't actually compute cheaply.
const STAGE_WEIGHTS = [
  "Starting", "Loading model", "Model loaded",
  "Generating shape (diffusion sampling)", "Shape generation complete",
  "Rendering preview video", "Gaussian preview done",
  "Rendering mesh preview", "Mesh preview done",
  "Post-processing mesh", "Simplifying mesh (fallback decimator)",
  "Filling holes", "Hole-filling done",
  "Rendering multi-view captures", "Multi-view capture done",
  "Optimizing baked texture", "Exporting .glb", "Saving .ply",
  "Finalizing", "Done",
];

function stageProgressPct(stage) {
  const idx = STAGE_WEIGHTS.indexOf(stage);
  if (idx < 0) return 5;
  return Math.round(((idx + 1) / STAGE_WEIGHTS.length) * 100);
}

function setPreview(url) {
  previewImg.src = url;
  previewImg.hidden = false;
  dropzoneLabel.hidden = true;
}

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) {
    selectedFile = fileInput.files[0];
    selectedExample = null;
    setPreview(URL.createObjectURL(selectedFile));
    clearExampleSelection();
    generateBtn.disabled = false;
  }
});

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) {
    selectedFile = e.dataTransfer.files[0];
    selectedExample = null;
    setPreview(URL.createObjectURL(selectedFile));
    clearExampleSelection();
    generateBtn.disabled = false;
  }
});

function clearExampleSelection() {
  examplesGrid.querySelectorAll("img.selected").forEach((el) => el.classList.remove("selected"));
}

async function loadExamples() {
  const res = await fetch("/api/examples");
  const examples = await res.json();
  examplesGrid.innerHTML = "";
  for (const ex of examples) {
    const img = document.createElement("img");
    img.src = ex.url;
    img.title = ex.name;
    img.addEventListener("click", () => {
      selectedExample = ex.name;
      selectedFile = null;
      fileInput.value = "";
      setPreview(ex.url);
      clearExampleSelection();
      img.classList.add("selected");
      generateBtn.disabled = false;
    });
    examplesGrid.appendChild(img);
  }
}

function gatherOptions() {
  const opts = {
    seed: $("opt-seed").value,
    simplify: $("opt-simplify").value,
    texture_size: $("opt-texture-size").value,
    fill_holes_resolution: $("opt-fill-res").value,
    fill_holes_num_views: $("opt-fill-views").value,
    multiview_nviews: $("opt-mv-views").value,
  };
  if ($("opt-skip-previews").checked) opts.skip_previews = "1";
  if ($("opt-mesh-only").checked) opts.mesh_only = "1";
  if (!$('opt-mesh-only').checked && $("opt-radiance-field").checked) opts.include_radiance_field = "1";
  return opts;
}

function resetLivePreview() {
  livePreviewPanel.hidden = true;
  for (const item of [liveGsItem, liveMeshItem, liveRfItem, liveMvItem]) item.hidden = true;
  liveGsGif.src = "";
  liveMeshGif.src = "";
  liveRfGif.src = "";
  liveMvImg.src = "";
  liveMvCounter.textContent = "";
}

function updateLivePreview(preview) {
  if (!preview) return;
  let anyVisible = false;

  if (preview.gaussian_gif) {
    liveGsGif.src = preview.gaussian_gif;
    liveGsItem.hidden = false;
    anyVisible = true;
  }
  if (preview.mesh_gif) {
    liveMeshGif.src = preview.mesh_gif;
    liveMeshItem.hidden = false;
    anyVisible = true;
  }
  if (preview.radiance_field_gif) {
    liveRfGif.src = preview.radiance_field_gif;
    liveRfItem.hidden = false;
    anyVisible = true;
  }
  if (preview.multiview_live && preview.multiview_live.image) {
    liveMvImg.src = preview.multiview_live.image;
    liveMvCounter.textContent = `${preview.multiview_live.current} / ${preview.multiview_live.total}`;
    liveMvItem.hidden = false;
    anyVisible = true;
  }

  if (anyVisible) livePreviewPanel.hidden = false;
}

generateBtn.addEventListener("click", async () => {
  const form = new FormData();
  if (selectedFile) {
    form.append("image", selectedFile);
  } else if (selectedExample) {
    form.append("example", selectedExample);
  } else {
    return;
  }
  const opts = gatherOptions();
  for (const [k, v] of Object.entries(opts)) form.append(k, v);
  form.append("out_prefix", (selectedExample || selectedFile.name).replace(/\.[^.]+$/, "").slice(0, 40));

  generateBtn.disabled = true;
  progressPanel.hidden = false;
  progressPanel.classList.remove("error");
  downloadRow.hidden = true;
  progressStage.textContent = "Starting…";
  progressBarFill.style.width = "2%";
  progressDetail.textContent = "";
  resetLivePreview();

  const res = await fetch("/api/generate", { method: "POST", body: form });
  const data = await res.json();
  if (data.error) {
    progressStage.textContent = "Error";
    progressDetail.textContent = data.error;
    progressPanel.classList.add("error");
    generateBtn.disabled = false;
    return;
  }
  pollStatus(data.job_id);
});

function formatElapsed(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s elapsed`;
}

function pollStatus(jobId) {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    const res = await fetch(`/api/status/${jobId}`);
    const status = await res.json();
    if (status.error && !status.stage) {
      clearInterval(pollTimer);
      return;
    }
    progressStage.textContent = status.stage;
    progressDetail.textContent = status.detail;
    progressElapsed.textContent = formatElapsed(status.elapsed);
    progressBarFill.style.width = `${stageProgressPct(status.stage)}%`;
    updateLivePreview(status.preview);

    if (status.error) {
      progressPanel.classList.add("error");
      clearInterval(pollTimer);
      generateBtn.disabled = false;
      loadHistory();
      return;
    }

    if (status.status === "done") {
      progressBarFill.style.width = "100%";
      clearInterval(pollTimer);
      generateBtn.disabled = false;
      showResult(status.result);
      loadHistory();
    }
  }, 1500);
}

function showResult(result) {
  if (!result) return;
  if (result.glb) {
    window.trellisViewer.load(result.glb);
    dlGlb.href = result.glb;
    downloadRow.hidden = false;
  }
  if (result.ply) {
    dlPly.href = result.ply;
  }
}

async function loadHistory() {
  const res = await fetch("/api/jobs");
  const jobs = await res.json();
  historyList.innerHTML = "";
  for (const job of jobs) {
    const el = document.createElement("div");
    el.className = "history-item";
    el.innerHTML = `
      <div class="h-name">${job.out_prefix}</div>
      <div class="h-status status-${job.status}">${job.status} &mdash; ${job.stage}</div>
    `;
    if (job.status === "done" && job.result && job.result.glb) {
      el.addEventListener("click", () => showResult(job.result));
    }
    historyList.appendChild(el);
  }
}

loadExamples();
loadHistory();
setInterval(loadHistory, 5000);
