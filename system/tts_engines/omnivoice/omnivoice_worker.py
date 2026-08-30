r"""
OmniVoice sidecar worker for the AllTalk "omnivoice" TTS engine.

This script is NOT run by AllTalk's Python environment. It is spawned by
system/tts_engines/omnivoice/model_engine.py using the OmniVoice venv interpreter
(configured per machine in omnivoice_local.json or via the OMNIVOICE_VENV_PYTHON
environment variable), because
the omnivoice package requires transformers>=5.x while AllTalk runs 4.x.

Protocol: JSON lines over stdin/stdout.
  Request  (one JSON object per line on stdin):
    {"id": <int>, "type": "ping"}
    {"id": <int>, "type": "load", "model_path": str, "device": "cuda:0"|"cpu", "dtype": "float16"|"float32"}
    {"id": <int>, "type": "unload"}
    {"id": <int>, "type": "generate", "text": str, "language": str|null,
     "ref_audio": str|null, "ref_text": str|null, "speed": float|null,
     "output_file": str, ...optional generate kwargs}
  Response (one JSON object per line on stdout, matching "id"):
    {"id": <int>, "type": "pong", "ok": true}
    {"id": <int>, "type": "loaded", "ok": true, "sampling_rate": 24000}
    {"id": <int>, "type": "unloaded", "ok": true}
    {"id": <int>, "type": "generated", "ok": true, "output_file": str, "sampling_rate": 24000, "duration": float}
    {"id": <int>, "type": "error", "ok": false, "detail": str}

ALL logging/output goes to stderr (redirected to omnivoice_worker.log by the
engine). stdout is reserved exclusively for the JSON protocol.
"""
import json
import logging
import sys
import time

# Make sure the protocol channel is UTF-8 and line-buffered no matter what.
try:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
log = logging.getLogger("omnivoice_worker")

model = None
loaded_model_path = None


def _emit(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _worker_load(req):
    global model, loaded_model_path
    import torch

    model_path = req["model_path"]
    device = req.get("device", "cpu")
    dtype_name = req.get("dtype", "float32")
    dtype = torch.float16 if dtype_name == "float16" else torch.float32

    from omnivoice.models.omnivoice import OmniVoice

    log.info("Loading OmniVoice model from %s (device=%s dtype=%s)", model_path, device, dtype_name)
    model = OmniVoice.from_pretrained(model_path, device_map=device, dtype=dtype)
    loaded_model_path = model_path
    log.info("OmniVoice model loaded. sampling_rate=%s", model.sampling_rate)
    return {"type": "loaded", "ok": True, "sampling_rate": int(model.sampling_rate)}


def _worker_unload(req):
    global model, loaded_model_path
    import torch

    log.info("Unloading OmniVoice model")
    model = None
    loaded_model_path = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"type": "unloaded", "ok": True}


def _worker_generate(req):
    import numpy as np
    import soundfile as sf

    if model is None:
        raise RuntimeError("No model loaded. Send a 'load' request first.")

    text = req["text"]
    language = req.get("language")
    ref_audio = req.get("ref_audio")
    ref_text = req.get("ref_text")
    speed = req.get("speed")
    output_file = req["output_file"]

    # Lazy ASR fallback: if a reference audio was given without its transcript,
    # load the (locally cached) Whisper model so _preprocess_all can transcribe.
    if ref_text is None and ref_audio is not None and getattr(model, "_asr_pipe", None) is None:
        log.info("ref_text missing; loading ASR model for auto-transcription")
        model.load_asr_model()

    gen_kwargs = {
        key: req[key]
        for key in (
            "num_step",
            "guidance_scale",
            "t_shift",
            "denoise",
            "postprocess_output",
            "layer_penalty_factor",
            "position_temperature",
            "class_temperature",
        )
        if key in req
    }

    log.info("Generating TTS (chars=%d, language=%s, ref_audio=%s, speed=%s)",
             len(text or ""), language, ref_audio, speed)
    gen_start = time.time()
    audios = model.generate(
        text=text,
        language=language,
        ref_audio=ref_audio,
        ref_text=ref_text,
        speed=speed,
        **gen_kwargs,
    )
    audio = audios[0]

    # Write 16-bit PCM WAV so the AllTalk side can stream raw int16 chunks.
    if audio.dtype != np.int16:
        if audio.dtype in (np.float32, np.float64):
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767.0).astype(np.int16)
        else:
            audio = audio.astype(np.int16)
    sf.write(output_file, audio, model.sampling_rate)
    gen_elapsed = time.time() - gen_start
    duration = float(audio.shape[0] / model.sampling_rate)
    log.info("Generated %.2fs of audio in %.2fs -> %s", duration, gen_elapsed, output_file)
    return {
        "type": "generated",
        "ok": True,
        "output_file": output_file,
        "sampling_rate": int(model.sampling_rate),
        "duration": round(duration, 3),
    }


def main():
    log.info("OmniVoice worker started (python %s)", sys.version.split()[0])
    _emit({"type": "worker_ready", "ok": True, "python": sys.version.split()[0]})
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            log.warning("Skipping non-JSON stdin line (%s): %s", e, line[:120])
            continue

        rid = req.get("id")
        rtype = req.get("type")
        log.info("Request id=%s type=%s", rid, rtype)
        try:
            if rtype == "ping":
                resp = {"type": "pong", "ok": True}
            elif rtype == "load":
                resp = _worker_load(req)
            elif rtype == "unload":
                resp = _worker_unload(req)
            elif rtype == "generate":
                resp = _worker_generate(req)
            elif rtype == "quit":
                _emit({"id": rid, "type": "quitting", "ok": True})
                _worker_unload(req)
                break
            else:
                resp = {"type": "error", "ok": False, "detail": f"Unknown request type '{rtype}'"}
            _emit({"id": rid, **resp})
        except Exception as e:
            log.exception("Request id=%s type=%s failed", rid, rtype)
            _emit({"id": rid, "type": "error", "ok": False, "detail": f"{type(e).__name__}: {e}"})
    log.info("OmniVoice worker exiting")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        log.exception("Worker crashed")
        try:
            _emit({"id": None, "type": "error", "ok": False, "detail": "Worker crashed (see log)"})
        except Exception:
            pass
        sys.exit(1)
