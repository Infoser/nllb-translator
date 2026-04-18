import os
import sys
import io
import traceback
import threading
import torch
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import snapshot_download

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app      = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'chhattisgarhi-secret')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ── HuggingFace model repo (set via env var or default) ───────────────────────
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "infoser090/AksharaDhara")

# ── State shared with the loading endpoint ────────────────────────────────────
load_state = {
    "step"    : 0,       # 0-4
    "percent" : 0,       # 0-100
    "message" : "शुरू हो रहा है…",
    "ready"   : False,
    "error"   : None,
}

model     = None
tokenizer = None
DEVICE    = None
FORCED_BOS_ID = None
USE_CUDA  = False
_infer_lock = threading.Lock()

# ── Loader (runs in background thread) ───────────────────────────────────────
def load_model():
    global model, tokenizer, DEVICE, FORCED_BOS_ID, USE_CUDA

    def update(step, pct, msg):
        load_state.update(step=step, percent=pct, message=msg)
        print(f"[LOAD {pct:3d}%] {msg}")

    try:
        update(0, 5, "Device जाँच रहे हैं…")

        USE_CUDA = torch.cuda.is_available()
        if USE_CUDA:
            DEVICE = torch.device("cuda:0")
            torch.cuda.set_device(0)
            gpu_name = torch.cuda.get_device_name(0)
            update(1, 10, f"GPU मिला: {gpu_name}")
        else:
            DEVICE = torch.device("cpu")
            update(1, 10, "GPU नहीं मिला — CPU मोड में चल रहा है")

        # Download model from HuggingFace Hub (cached after first run)
        local_model_path = os.environ.get("LOCAL_MODEL_PATH", "")
        if local_model_path and os.path.isdir(local_model_path):
            model_path = local_model_path
            update(1, 20, f"लोकल मॉडल मिला: {model_path}")
        else:
            update(1, 15, f"मॉडल डाउनलोड हो रहा है: {HF_MODEL_REPO}…")
            model_path = snapshot_download(HF_MODEL_REPO)
            update(2, 25, "मॉडल डाउनलोड हो गया ✓")

        update(2, 30, "Tokenizer लोड हो रहा है…")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Language config
        SRC_LANG = "hne_Deva"
        TGT_LANG = "hin_Deva"
        tokenizer.src_lang = SRC_LANG
        FORCED_BOS_ID = tokenizer.convert_tokens_to_ids(TGT_LANG)

        dtype = torch.float16 if USE_CUDA else torch.float32
        device_label = "GPU" if USE_CUDA else "CPU"
        update(3, 55, f"मॉडल {device_label} पर लोड हो रहा है… (इसमें समय लग सकता है)")
        m = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to(DEVICE)
        m.eval()
        model = m

        if USE_CUDA:
            vram = torch.cuda.memory_allocated(0) / (1024**3)
            update(4, 90, f"वार्मअप हो रहा है… ({vram:.2f} GB VRAM)")
        else:
            update(4, 90, "वार्मअप हो रहा है…")

        # One warm-up pass so first real request is fast
        dummy = tokenizer("नमस्कार", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            model.generate(
                **dummy,
                forced_bos_token_id=FORCED_BOS_ID,
                max_length=32,
                num_beams=1,
            )

        update(4, 100, "तैयार है! ✓")
        load_state["ready"] = True
        print("\n[OK] Model fully loaded and warmed up.\n")

    except Exception as e:
        traceback.print_exc()
        load_state["error"]   = str(e)
        load_state["message"] = f"त्रुटि: {e}"
        load_state["percent"] = 0


# Start loading immediately when app starts
_loader_thread = threading.Thread(target=load_model, daemon=True)
_loader_thread.start()


# ── Translate ─────────────────────────────────────────────────────────────────
def translate(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    with _infer_lock:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=FORCED_BOS_ID,
                max_length=256,
                num_beams=4,
                early_stopping=True,
            )
        return tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True)[0]


# ── HTTP routes ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/load-status")
def load_status():
    """Polled by the UI loading bar every 500 ms."""
    return jsonify(load_state)


@app.route("/gpu-info")
def gpu_info():
    if not load_state["ready"]:
        return jsonify({"error": "Model not ready"}), 503
    info = {
        "model_device" : str(next(model.parameters()).device),
    }
    if USE_CUDA:
        alloc = torch.cuda.memory_allocated(0) / (1024**2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info.update({
            "gpu_name"            : torch.cuda.get_device_name(0),
            "vram_total_gb"       : round(total, 1),
            "memory_allocated_mb" : round(alloc, 1),
        })
    else:
        info.update({
            "gpu_name"            : "CPU (no GPU)",
            "vram_total_gb"       : 0,
            "memory_allocated_mb" : 0,
        })
    return jsonify(info)


# ── WebSocket ─────────────────────────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    emit('status', {'ready': load_state['ready'], 'percent': load_state['percent']})


@socketio.on('translate')
def on_translate(data):
    if not load_state["ready"]:
        emit('translate_error', {'error': 'मॉडल अभी लोड हो रहा है, कृपया प्रतीक्षा करें।'})
        return
    text = (data.get('text') or '').strip()
    if not text:
        emit('translate_error', {'error': 'कृपया पाठ दर्ज करें।'})
        return

    emit('translate_start', {})
    try:
        result = translate(text)
        gpu_mb = round(torch.cuda.memory_allocated(0) / (1024**2), 1) if USE_CUDA else 0
        emit('translate_result', {'translation': result, 'input': text, 'gpu_mb': gpu_mb})
    except Exception as e:
        traceback.print_exc()
        emit('translate_error', {'error': str(e)})


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[OK] Flask starting on port {port} — model loading in background thread\n")
    socketio.run(app, host="0.0.0.0", port=port, debug=False)
