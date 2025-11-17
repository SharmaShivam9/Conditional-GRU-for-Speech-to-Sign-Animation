# app.py
from flask import Flask, request, jsonify, send_file
import os
import shutil
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow cross-origin requests from your frontend

# ---------- CONFIG ----------
GIF_DATASET_DIR = r"D:\Des646\AI\web\output_gifs_dataset"   # where model writes GIFs (random names)
LATEST_GIF_PATH = os.path.join(GIF_DATASET_DIR, "latest.gif")
# If you want latest.gif elsewhere, change LATEST_GIF_PATH accordingly.

os.makedirs(GIF_DATASET_DIR, exist_ok=True)

# ---------- Helpers ----------
def list_gifs():
    """Return list of gif filenames (excluding latest.gif) sorted by mtime descending."""
    try:
        files = [f for f in os.listdir(GIF_DATASET_DIR) if f.lower().endswith(".gif")]
    except Exception as e:
        print("[ERROR] cannot list GIF directory:", e)
        return []
    files = [f for f in files if f.lower() != "latest.gif"]
    files_full = [os.path.join(GIF_DATASET_DIR, f) for f in files]
    files_full.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files_full

def copy_newest_to_latest():
    """Copy newest (by mtime) gif to latest.gif; return (True, source_path) or (False, reason)."""
    gifs = list_gifs()
    if not gifs:
        return False, "No GIF files found in dataset folder"
    newest = gifs[0]
    try:
        shutil.copyfile(newest, LATEST_GIF_PATH)
        return True, newest
    except Exception as e:
        return False, f"Copy failed: {e}"

# ---------- Endpoints ----------

@app.route("/api/refresh_gif", methods=["POST", "GET"])
def refresh_gif():
    """
    Force-copy the newest GIF into latest.gif and return the filename.
    Frontend should call this right after backend/model finishes.
    """
    ok, info = copy_newest_to_latest()
    if ok:
        print("[server] refresh_gif -> copied:", info)
        return jsonify({"ok": True, "copied_from": os.path.basename(info), "gif_url": "/api/get_gif"})
    else:
        print("[server] refresh_gif -> failed:", info)
        return jsonify({"ok": False, "error": info}), 404

@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Main endpoint: the frontend sends sentence. Server will:
      - (OPTIONAL) run your ML pipeline here (not included)
      - Wait/attempt to copy newest gif to latest.gif
    If your ML code runs externally and writes new gif into GIF_DATASET_DIR,
    this endpoint will wait up to WAIT_TIMEOUT seconds for a new file and then copy it.
    """
    data = request.get_json(silent=True) or {}
    sentence = data.get("sentence", "")[:1000]
    print("[server] /api/generate received sentence:", repr(sentence))

    # If you want server to run your model inline, place code here.
    # Example (pseudo):
    # embed = vectorize_sentence(sentence)
    # y_pred = predict_from_embedding(embed)
    # generate_stick_figure_gif_from_array(y_pred, os.path.join(GIF_DATASET_DIR, "model_out_<ts>.gif"))

    # Strategy: attempt immediate copy; if no gif exists, wait up to WAIT_TIMEOUT for a new one
    WAIT_TIMEOUT = 40  # seconds
    POLL_INTERVAL = 1.0

    start = time.time()
    # record existing newest file at start
    initial = list_gifs()
    initial_newest = initial[0] if initial else None
    print("[server] initial newest:", initial_newest)

    # If there is already a gif, copy it immediately (so UI always sees something)
    if initial_newest:
        ok, info = copy_newest_to_latest()
        if ok:
            return jsonify({"gif_url": "/api/get_gif", "copied_from": os.path.basename(info)})

    # Otherwise wait for up to WAIT_TIMEOUT for a new file to appear
    print(f"[server] waiting up to {WAIT_TIMEOUT}s for model to generate a new gif...")
    while time.time() - start < WAIT_TIMEOUT:
        time.sleep(POLL_INTERVAL)
        current = list_gifs()
        current_newest = current[0] if current else None

        if current_newest and current_newest != initial_newest:
            ok, info = copy_newest_to_latest()
            if ok:
                return jsonify({"gif_url": "/api/get_gif", "copied_from": os.path.basename(info)})
            else:
                return jsonify({"error": info}), 500

    # Timeout
    print("[server] timeout waiting for gif")
    return jsonify({"error": "No GIF produced within timeout."}), 500

@app.route("/api/get_gif")
def get_gif():
    if not os.path.exists(LATEST_GIF_PATH):
        print("[server] GET /api/get_gif -> latest.gif not found at:", LATEST_GIF_PATH)
        return jsonify({"error": "GIF not available"}), 404
    # serve file
    try:
        print("[server] GET /api/get_gif -> serving:", LATEST_GIF_PATH)
        return send_file(LATEST_GIF_PATH, mimetype="image/gif")
    except Exception as e:
        print("[server] GET /api/get_gif -> send_file failed:", e)
        return jsonify({"error": f"send_file failed: {e}"}), 500

# simple health
@app.route("/api/ping")
def ping():
    return jsonify({"ok": True})

if __name__ == "__main__":
    print("Starting server. GIF_DATASET_DIR =", GIF_DATASET_DIR)

    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
