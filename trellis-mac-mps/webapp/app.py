"""
Local web UI for the TRELLIS Apple Silicon port: upload an image (or pick an example), generate a
3D asset in the background, watch real progress, view the result in-browser (three.js), download
.glb/.ply/preview videos. See APPLE_SILICON.md for the pipeline this drives.

Run: python3 webapp/app.py  (serves http://127.0.0.1:5050)
"""
import os
import sys
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

import jobs

REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) + '/..'
REPO_ROOT = os.path.abspath(REPO_ROOT)
EXAMPLES_DIR = os.path.join(REPO_ROOT, 'assets', 'example_image')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB upload cap


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/examples')
def api_examples():
    files = sorted(f for f in os.listdir(EXAMPLES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    return jsonify([{'name': f, 'url': f'/examples/{f}'} for f in files])


@app.route('/examples/<path:filename>')
def serve_example(filename):
    return send_from_directory(EXAMPLES_DIR, filename)


@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(jobs.OUTPUT_DIR, filename)


@app.route('/APPLE_SILICON.md')
def serve_apple_silicon_doc():
    return send_from_directory(REPO_ROOT, 'APPLE_SILICON.md', mimetype='text/plain')


@app.route('/api/generate', methods=['POST'])
def api_generate():
    out_prefix = request.form.get('out_prefix', '').strip()
    if not out_prefix:
        out_prefix = f"job_{uuid.uuid4().hex[:8]}"
    out_prefix = secure_filename(out_prefix) or f"job_{uuid.uuid4().hex[:8]}"

    image_path = None
    upload = request.files.get('image')
    if upload and upload.filename:
        filename = secure_filename(upload.filename)
        image_path = os.path.join(jobs.UPLOAD_DIR, f"{out_prefix}_{filename}")
        upload.save(image_path)
    else:
        example = request.form.get('example', '')
        candidate = secure_filename(example)
        candidate_path = os.path.join(EXAMPLES_DIR, candidate)
        if candidate and os.path.exists(candidate_path):
            image_path = candidate_path

    if not image_path:
        return jsonify({'error': 'No image uploaded and no valid example selected.'}), 400

    options = {}
    for key in ('seed', 'simplify', 'texture_size', 'fill_holes_resolution',
                'fill_holes_num_views', 'multiview_resolution', 'multiview_nviews'):
        val = request.form.get(key)
        if val:
            options[key] = val
    if request.form.get('skip_previews') in ('1', 'true', 'on'):
        options['skip_previews'] = True
    if request.form.get('mesh_only') in ('1', 'true', 'on'):
        options['mesh_only'] = True
    if request.form.get('include_radiance_field') in ('1', 'true', 'on'):
        options['include_radiance_field'] = True

    job_id = jobs.start_job(image_path, out_prefix, options)
    return jsonify({'job_id': job_id})


@app.route('/api/status/<job_id>')
def api_status(job_id):
    status = jobs.get_job_status(job_id)
    if status is None:
        return jsonify({'error': 'unknown job id'}), 404
    return jsonify(status)


@app.route('/api/jobs')
def api_jobs():
    return jsonify(jobs.list_jobs())


if __name__ == '__main__':
    print(f"TRELLIS web UI -- repo root: {REPO_ROOT}")
    print("Open http://127.0.0.1:5050 in your browser.")
    app.run(host='127.0.0.1', port=5050, debug=False, threaded=True)
