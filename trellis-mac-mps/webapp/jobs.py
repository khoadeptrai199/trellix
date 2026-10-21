"""
In-memory background job manager for the local TRELLIS web app.

Design: each generation job is a *subprocess* running scripts/run_real_pipeline.py, not an
in-process function call. Why: that script is the exact, already-verified pipeline path (binned
Metal rasterizer, decimate_pro fallback, etc. -- see APPLE_SILICON.md) used throughout manual
testing this session, so reusing it via subprocess means the web app can't drift from a
independently-tested code path. It also gets us progress reporting almost for free -- the script's
tqdm output already narrates every stage in detail, so a poller just tails+greps that log instead
of needing custom instrumentation plumbed through the whole pipeline.

This is deliberately single-process, in-memory, no persistence -- fine for a local, single-user
tool. Restarting the Flask process loses job history (generated files on disk are untouched).
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOB_LOG_DIR = os.path.join(REPO_ROOT, 'webapp', 'job_logs')
UPLOAD_DIR = os.path.join(REPO_ROOT, 'webapp', 'uploads')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'outputs')
PIPELINE_SCRIPT = os.path.join(REPO_ROOT, 'scripts', 'run_real_pipeline.py')

os.makedirs(JOB_LOG_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

_jobs = {}
_lock = threading.Lock()

# Substrings that show up in scripts/run_real_pipeline.py's own print()/tqdm output, in the order
# they occur -- scanning forward through the whole log and keeping the last match gives "the most
# recent stage reached" for free, no custom instrumentation needed in the pipeline script itself.
_STAGE_MARKERS = [
    ("Loading pipeline", "Loading model"),
    ("Pipeline loaded", "Model loaded"),
    ("Sampling:", "Generating shape (diffusion sampling)"),
    ("Pipeline.run() completed", "Shape generation complete"),
    ("Rendering radiance-field", "Rendering octree preview (slow, ~18s/frame)"),
    ("radiance-field video rendered", "Octree preview done"),
    ("Rendering gaussian video", "Rendering preview video"),
    ("gaussian video rendered", "Gaussian preview done"),
    ("Rendering mesh normal video", "Rendering mesh preview"),
    ("mesh video rendered", "Mesh preview done"),
    ("Baking texture", "Post-processing mesh"),
    ("decimate_pro", "Simplifying mesh (fallback decimator)"),
    ("Rasterizing:", "Filling holes"),
    ("After remove invisible faces", "Hole-filling done"),
    ("Rendering multi-view", "Rendering multi-view captures"),
    ("multi-view renders ->", "Multi-view capture done"),
    ("optimizing", "Optimizing baked texture"),
    ("glb exported", "Exporting .glb"),
    ("ply saved", "Saving .ply"),
    ("Untextured GLB saved", "Done"),
    ("TOTAL TIME", "Finalizing"),
    ("=== DONE ===", "Done"),
]


def _parse_progress(log_path):
    try:
        with open(log_path, 'r', errors='replace') as f:
            content = f.read()
    except FileNotFoundError:
        return {"stage": "Starting", "detail": "", "error": False}

    if "Traceback (most recent call last)" in content:
        lines = [l for l in re.split(r'[\r\n]+', content) if l.strip()]
        return {"stage": "Error", "detail": "\n".join(lines[-20:]), "error": True}

    stage = "Starting"
    for marker, label in _STAGE_MARKERS:
        if marker in content:
            stage = label

    lines = [l for l in re.split(r'[\r\n]+', content) if l.strip()]
    detail = lines[-1] if lines else ""
    return {"stage": stage, "detail": detail, "error": False}


def start_job(image_path, out_prefix, options=None):
    """Launches scripts/run_real_pipeline.py as a subprocess. Returns a job_id immediately;
    poll get_job_status(job_id) for progress."""
    options = options or {}
    job_id = uuid.uuid4().hex[:12]
    log_path = os.path.join(JOB_LOG_DIR, f"{job_id}.log")

    cmd = [sys.executable, PIPELINE_SCRIPT, image_path, out_prefix, '--out-dir', OUTPUT_DIR]
    for flag, key in [
        ('--seed', 'seed'),
        ('--simplify', 'simplify'),
        ('--texture-size', 'texture_size'),
        ('--fill-holes-resolution', 'fill_holes_resolution'),
        ('--fill-holes-num-views', 'fill_holes_num_views'),
        ('--multiview-resolution', 'multiview_resolution'),
        ('--multiview-nviews', 'multiview_nviews'),
    ]:
        val = options.get(key)
        if val not in (None, ''):
            cmd += [flag, str(val)]
    if options.get('skip_previews'):
        cmd.append('--skip-previews')
    if options.get('mesh_only'):
        cmd.append('--mesh-only')
    if options.get('include_radiance_field'):
        cmd.append('--include-radiance-field')

    log_f = open(log_path, 'w')
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=REPO_ROOT)

    with _lock:
        _jobs[job_id] = {
            'id': job_id,
            'image': image_path,
            'out_prefix': out_prefix,
            'log_path': log_path,
            'status': 'running',
            'started': time.time(),
            'finished': None,
            'returncode': None,
        }

    def _waiter():
        rc = proc.wait()
        log_f.close()
        with _lock:
            _jobs[job_id]['status'] = 'done' if rc == 0 else 'error'
            _jobs[job_id]['finished'] = time.time()
            _jobs[job_id]['returncode'] = rc

    threading.Thread(target=_waiter, daemon=True).start()
    return job_id


def _result_files(out_prefix):
    def rel(name):
        path = os.path.join(OUTPUT_DIR, name)
        return f"/outputs/{name}" if os.path.exists(path) else None

    return {
        'glb': rel(f"{out_prefix}.glb"),
        'ply': rel(f"{out_prefix}.ply"),
        'gaussian_video': rel(f"{out_prefix}_gs.mp4"),
        'mesh_video': rel(f"{out_prefix}_mesh.mp4"),
    }


def _bust(name):
    """/outputs/<name>?v=<mtime> -- run_real_pipeline.py overwrites the gif/live-preview files
    in place as generation progresses, so a plain URL would serve a browser-cached stale image;
    mtime changes every time the file is rewritten, which is all a cache-busting key needs to do."""
    path = os.path.join(OUTPUT_DIR, name)
    if not os.path.exists(path):
        return None
    return f"/outputs/{name}?v={int(os.path.getmtime(path) * 1000)}"


def _preview_state(out_prefix):
    """Signals available *during* a run, well before the job finishes: gif previews (written
    right after diffusion sampling, long before hole-filling/texture-bake), and the multi-view
    render's current frame + progress (written on every one of the ~100 views, see
    scripts/run_real_pipeline.py)."""
    state = {
        'gaussian_gif': _bust(f"{out_prefix}_gs.gif"),
        'mesh_gif': _bust(f"{out_prefix}_mesh.gif"),
        'radiance_field_gif': _bust(f"{out_prefix}_rf.gif"),
        'multiview_live': None,
    }
    status_path = os.path.join(OUTPUT_DIR, f"{out_prefix}_multiview_live.json")
    try:
        with open(status_path, 'r') as f:
            progress = json.load(f)
        state['multiview_live'] = {
            'image': _bust(f"{out_prefix}_multiview_live.png"),
            'current': progress.get('current'),
            'total': progress.get('total'),
        }
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return state


def get_job_status(job_id):
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        job = dict(job)  # shallow copy so we can read fields outside the lock

    progress = _parse_progress(job['log_path'])
    result = _result_files(job['out_prefix']) if job['status'] == 'done' else None

    return {
        'id': job_id,
        'status': job['status'],
        'out_prefix': job['out_prefix'],
        'image': os.path.basename(job['image']),
        'elapsed': (job['finished'] or time.time()) - job['started'],
        'stage': progress['stage'],
        'detail': progress['detail'],
        'error': progress['error'],
        'result': result,
        'preview': _preview_state(job['out_prefix']),
    }


def list_jobs():
    with _lock:
        ids = list(_jobs.keys())
    return [get_job_status(jid) for jid in reversed(ids)]
