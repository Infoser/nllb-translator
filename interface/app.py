import sys
import io
import traceback
import torch
from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = Flask(__name__)

# ── GPU Diagnostics ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  GPU / CUDA DIAGNOSTICS")
print("="*60)
print(f"  PyTorch version      : {torch.__version__}")
print(f"  CUDA available       : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  CUDA version         : {torch.version.cuda}")
    print(f"  GPU count            : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        vram  = props.total_memory / (1024**3)
        print(f"  GPU [{i}]              : {props.name}  ({vram:.1f} GB VRAM)")
else:
    print("  WARNING: CUDA NOT AVAILABLE")
    print("  Fix: pip install torch --index-url https://download.pytorch.org/whl/cu121")
print("="*60 + "\n")

# ── Hard-require GPU ─────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    raise RuntimeError(
        "\n\nCUDA GPU not detected. Reinstall PyTorch with GPU support:\n"
        "  pip uninstall torch torchvision torchaudio\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
        "Then restart this script."
    )

DEVICE = torch.device("cuda:0")
torch.cuda.set_device(0)
print(f"[OK] Using GPU: {torch.cuda.get_device_name(0)}\n")

# ── Load model onto GPU ──────────────────────────────────────────────────────
model_path = r"nllb_training\nllb_finetuned_final"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading model onto GPU with float16 ...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,      # half-precision: faster + less VRAM
    device_map={"": 0},             # force ALL layers to cuda:0
)
model.eval()

# Verify
param_device = str(next(model.parameters()).device)
vram_used    = torch.cuda.memory_allocated(0) / (1024**3)
print(f"[OK] Model device   : {param_device}")
print(f"[OK] VRAM used      : {vram_used:.2f} GB\n")

# ── Language codes ───────────────────────────────────────────────────────────
SRC_LANG = "hne_Deva"
TGT_LANG = "hin_Deva"
tokenizer.src_lang = SRC_LANG
FORCED_BOS_ID = tokenizer.convert_tokens_to_ids(TGT_LANG)
print(f"[OK] forced_bos_token_id for '{TGT_LANG}' = {FORCED_BOS_ID}\n")


# ── Translate helper ─────────────────────────────────────────────────────────
def translate(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(DEVICE)                    # inputs on GPU

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=FORCED_BOS_ID,
            max_length=256,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True)[0]


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/gpu-info")
def gpu_info():
    """Called by the UI on load to show GPU badge."""
    alloc = torch.cuda.memory_allocated(0) / (1024**2)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return jsonify({
        "gpu_name"            : torch.cuda.get_device_name(0),
        "cuda_available"      : True,
        "vram_total_gb"       : round(total, 1),
        "memory_allocated_mb" : round(alloc, 1),
        "model_device"        : str(next(model.parameters()).device),
    })


@app.route("/translate", methods=["POST"])
def translate_route():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        result  = translate(text)
        gpu_mem = round(torch.cuda.memory_allocated(0) / (1024**2), 1)
        print(f"[translate] in='{text[:60]}' | out='{result[:60]}' | GPU {gpu_mem} MB")
        return jsonify({"translation": result, "input": text})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/translate-batch", methods=["POST"])
def translate_batch_route():
    data  = request.get_json(force=True)
    texts = data.get("texts", [])
    if not texts or not isinstance(texts, list):
        return jsonify({"error": "Provide a 'texts' list"}), 400
    try:
        results = [{"input": t, "translation": translate(t)} for t in texts]
        return jsonify({"results": results})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # threaded=False is CRITICAL — CUDA does not like multiple threads
    # sharing the same context without explicit management
    app.run(debug=False, port=5000, threaded=False)
