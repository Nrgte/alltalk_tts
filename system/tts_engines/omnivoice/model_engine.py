###############################################
# DONT CHANGE # These are base imports needed #
###############################################
import glob
import io
import os
import sys
import json
import time
import wave
import asyncio
import atexit
import ctypes
import logging
import signal
import torch
import numpy as np
from pathlib import Path
from fastapi import (HTTPException)
logging.disable(logging.WARNING)
###############################################
# DONT CHANGE # Get Pytorch & Python versions #
###############################################
pytorch_version = torch.__version__
cuda_version = torch.version.cuda
major, minor, micro = sys.version_info[:3]
python_version = f"{major}.{minor}.{micro}"
try:
    import deepspeed
    deepspeed_available = True
except ImportError:
    deepspeed_available = False
    pass

#############################################################################################################
#############################################################################################################
# CHANGE ME # Run any specifc imports, requirements or setup any global vaiables needed for this TTS Engine #
#############################################################################################################
#############################################################################################################
# The OmniVoice model cannot be imported into AllTalk's Python environment (it requires
# transformers>=5.x, AllTalk runs 4.x), so this engine drives a persistent sidecar worker
# process (omnivoice_worker.py) running under the OmniVoice venv interpreter, over a
# JSON-lines stdin/stdout protocol. The worker loads the model once and writes the
# generated 16-bit PCM WAV files, which this engine streams back as needed.
#
# Both paths below are machine specific and are NOT hardcoded here. They are
# read from the local (git-ignored) config file omnivoice_local.json (copy it
# from omnivoice_local.json.example and adjust) and can be overridden with the
# environment variables OMNIVOICE_VENV_PYTHON and OMNIVOICE_MODEL_ROOT, which
# take precedence over the file.
OMNIVOICE_LOCAL_CONFIG = Path(__file__).parent / "omnivoice_local.json"
try:
    with open(OMNIVOICE_LOCAL_CONFIG, "r", encoding="utf-8") as f:
        _omnivoice_local_cfg = json.load(f)
except (OSError, json.JSONDecodeError):
    _omnivoice_local_cfg = {}
OMNIVOICE_VENV_PYTHON = os.environ.get("OMNIVOICE_VENV_PYTHON") or _omnivoice_local_cfg.get("OMNIVOICE_VENV_PYTHON")
OMNIVOICE_MODEL_ROOT = os.environ.get("OMNIVOICE_MODEL_ROOT") or _omnivoice_local_cfg.get("OMNIVOICE_MODEL_ROOT")
OMNIVOICE_HUB_CACHE = Path(OMNIVOICE_MODEL_ROOT) / "hub" if OMNIVOICE_MODEL_ROOT else None
WORKER_SCRIPT = Path(__file__).parent / "omnivoice_worker.py"
WORKER_LOG_FILE = Path(__file__).parent / "omnivoice_worker.log"
WORKER_LOG_MAX_BYTES = 1024 * 1024
WORKER_LOG_BACKUP = WORKER_LOG_FILE.with_suffix(".log.old")
DEFAULT_HF_MODEL_ID = "k2-fsa/OmniVoice"

WORKER_PING_TIMEOUT = 30
WORKER_LOAD_TIMEOUT = 900
WORKER_UNLOAD_TIMEOUT = 120
WORKER_GENERATE_TIMEOUT = 900
STREAM_CHUNK_SECONDS = 0.25


class WorkerNotRunningError(RuntimeError):
    pass


class WorkerDeadError(RuntimeError):
    pass


def _process_alive(pid):
    """OS-level liveness check that works even after the asyncio loop is closed."""
    if sys.platform == "win32":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_RUNNING = 259
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        exit_code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return bool(ok) and exit_code.value == STILL_RUNNING
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _process_kill_hard(pid):
    """OS-level kill that works even after the asyncio loop is closed."""
    if sys.platform == "win32":
        PROCESS_TERMINATE = 0x0001
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
        if handle:
            ctypes.windll.kernel32.TerminateProcess(handle, 1)
            ctypes.windll.kernel32.CloseHandle(handle)
        return
    os.kill(pid, signal.SIGKILL)


#############################################################
# DONT CHANGE # Do not change the Class name from tts_class #
#############################################################
class tts_class:
    def __init__(self):
        ########################################################################
        # DONT CHANGE # Sets up the base variables required for any tts engine #
        ########################################################################
        self.branding = None
        self.this_dir = Path(__file__).parent.resolve()
        self.main_dir = Path(__file__).parent.parent.parent.parent.resolve()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cuda_is_available = torch.cuda.is_available()
        self.tts_generating_lock = False
        self.tts_stop_generation = False
        self.tts_narrator_generatingtts = False
        self.model = None
        self.is_tts_model_loaded = False
        self.current_model_loaded = None
        self.available_models = None
        self.setup_has_run = False
        ##############################################################################################
        # DONT CHANGE # Load in a list of the available TTS engines and the currently set TTS engine #
        ##############################################################################################
        tts_engines_file = os.path.join(self.main_dir, "system", "tts_engines", "tts_engines.json")
        with open(tts_engines_file, "r") as f:
            tts_engines_data = json.load(f)
        self.engines_available = [engine["name"] for engine in tts_engines_data["engines_available"]]
        self.engine_loaded = tts_engines_data["engine_loaded"]
        self.selected_model = tts_engines_data["selected_model"]
        ############################################################################
        # DONT CHANGE # Pull out all the settings for the currently set TTS engine #
        ############################################################################
        with open(os.path.join(self.this_dir, "model_settings.json"), "r") as f:
            tts_model_loaded = json.load(f)
        # Access the model details
        self.manufacturer_name = tts_model_loaded["model_details"]["manufacturer_name"]
        self.manufacturer_website = tts_model_loaded["model_details"]["manufacturer_website"]
        # Access the features the model is capable of:
        self.audio_format = tts_model_loaded["model_capabilties"]["audio_format"]
        self.deepspeed_capable = tts_model_loaded["model_capabilties"]["deepspeed_capable"]
        self.deepspeed_available = 'deepspeed' in globals()
        self.generationspeed_capable = tts_model_loaded["model_capabilties"]["generationspeed_capable"]
        self.languages_capable = tts_model_loaded["model_capabilties"]["languages_capable"]
        self.lowvram_capable = tts_model_loaded["model_capabilties"]["lowvram_capable"]
        self.multimodel_capable = tts_model_loaded["model_capabilties"]["multimodel_capable"]
        self.repetitionpenalty_capable = tts_model_loaded["model_capabilties"]["repetitionpenalty_capable"]
        self.streaming_capable = tts_model_loaded["model_capabilties"]["streaming_capable"]
        self.temperature_capable = tts_model_loaded["model_capabilties"]["temperature_capable"]
        self.multivoice_capable = tts_model_loaded["model_capabilties"]["multivoice_capable"]
        self.pitch_capable = tts_model_loaded["model_capabilties"]["pitch_capable"]
        # Access the current enginesettings
        self.def_character_voice = tts_model_loaded["settings"]["def_character_voice"]
        self.def_narrator_voice = tts_model_loaded["settings"]["def_narrator_voice"]
        self.deepspeed_enabled = tts_model_loaded["settings"]["deepspeed_enabled"]
        self.engine_installed = tts_model_loaded["settings"]["engine_installed"]
        self.generationspeed_set = tts_model_loaded["settings"]["generationspeed_set"]
        self.lowvram_enabled = tts_model_loaded["settings"]["lowvram_enabled"]
        # Check if someone has enabled lowvram on a system that's not CUDA enabled
        self.lowvram_enabled = False if not torch.cuda.is_available() else self.lowvram_enabled
        self.repetitionpenalty_set = tts_model_loaded["settings"]["repetitionpenalty_set"]
        self.temperature_set = tts_model_loaded["settings"]["temperature_set"]
        self.pitch_set = tts_model_loaded["settings"]["pitch_set"]
        # Gather the OpenAI API Voice Mappings
        self.openai_alloy = tts_model_loaded["openai_voices"]["alloy"]
        self.openai_echo = tts_model_loaded["openai_voices"]["echo"]
        self.openai_fable = tts_model_loaded["openai_voices"]["fable"]
        self.openai_nova = tts_model_loaded["openai_voices"]["nova"]
        self.openai_onyx = tts_model_loaded["openai_voices"]["onyx"]
        self.openai_shimmer = tts_model_loaded["openai_voices"]["shimmer"]
        ###################################################################
        # DONT CHANGE #  Load params and api_defaults from confignew.json #
        ###################################################################
        # Define the path to the confignew.json file
        configfile_path = self.main_dir / "confignew.json"
        # Load config file and get settings
        with open(configfile_path, "r") as configfile:
            configfile_data = json.load(configfile)
        self.branding = configfile_data.get("branding", "")
        self.params = configfile_data
        self.debug_tts = configfile_data.get("debugging").get("debug_tts")
        self.debug_tts_variables = configfile_data.get("debugging").get("debug_tts_variables")

        # OmniVoice sidecar worker state
        self.worker_proc = None
        self.worker_cmd_lock = asyncio.Lock()
        self.worker_req_id = 0
        self.worker_model_loaded = False
        self.worker_model_path = None
        # Kill the worker when the AllTalk process exits. On Windows a child
        # process is not killed when its parent exits, so without this an
        # orphaned worker would keep holding the model in VRAM.
        atexit.register(self._kill_worker_sync)

    ################################################################
    # DONT CHANGE #  Print out Python, CUDA, DeepSpeed versions ####
    ################################################################
    def printout_versions(self):
        if deepspeed_available:
            print(f"[{self.branding}ENG] \033[92mDeepSpeed version :\033[93m",deepspeed.__version__,"\033[0m")
        else:
            print(f"[{self.branding}ENG] \033[92mDeepSpeed version :\033[93m Not available\033[0m")
        print(f"[{self.branding}ENG] \033[92mPython Version    :\033[93m {python_version}\033[0m")
        print(f"[{self.branding}ENG] \033[92mPyTorch Version   :\033[93m {pytorch_version}\033[0m")
        if cuda_version is None:
            print(f"[{self.branding}ENG] \033[92mCUDA Version      :\033[91m Not available\033[0m")
        else:
            print(f"[{self.branding}ENG] \033[92mCUDA Version      :\033[93m {cuda_version}\033[0m")
        print(f"[{self.branding}ENG]")
        return

    ###################################################################################
    ###################################################################################
    # OmniVoice sidecar worker helpers (spawn / command / load / unload) #
    ###################################################################################
    async def _kill_worker(self):
        if self.worker_proc is not None and self.worker_proc.returncode is None:
            try:
                self.worker_proc.terminate()
                try:
                    await asyncio.wait_for(self.worker_proc.wait(), timeout=10)
                except asyncio.TimeoutError:
                    self.worker_proc.kill()
                    await self.worker_proc.wait()
            except Exception:
                pass
        self.worker_proc = None
        self.worker_model_loaded = False
        self.worker_model_path = None

    def _kill_worker_sync(self):
        # Sync version used by the atexit handler, where the asyncio event loop
        # is no longer running, so the asyncio Process API (terminate/wait and
        # even returncode) can no longer be relied on. Liveness is checked and
        # the kill is enforced at the OS level instead.
        proc = self.worker_proc
        pid = proc.pid if proc is not None else None
        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
            except Exception:
                pass
        if pid is not None and _process_alive(pid):
            try:
                deadline = time.time() + 10
                while _process_alive(pid) and time.time() < deadline:
                    time.sleep(0.1)
                if _process_alive(pid):
                    _process_kill_hard(pid)
            except Exception:
                pass
        self.worker_proc = None
        self.worker_model_loaded = False
        self.worker_model_path = None

    @staticmethod
    def _rotate_worker_log():
        # Simple size based rotation: once the current log grows past
        # WORKER_LOG_MAX_BYTES it is moved to a single .old backup (any
        # previous backup is discarded) before a fresh log is started.
        try:
            if WORKER_LOG_FILE.is_file() and WORKER_LOG_FILE.stat().st_size > WORKER_LOG_MAX_BYTES:
                if WORKER_LOG_BACKUP.exists():
                    WORKER_LOG_BACKUP.unlink()
                WORKER_LOG_FILE.rename(WORKER_LOG_BACKUP)
        except OSError:
            pass

    async def _spawn_worker(self):
        if not OMNIVOICE_VENV_PYTHON or not os.path.isfile(OMNIVOICE_VENV_PYTHON):
            raise HTTPException(
                status_code=500,
                detail=f"OmniVoice worker interpreter not found"
                       + (f" at '{OMNIVOICE_VENV_PYTHON}'." if OMNIVOICE_VENV_PYTHON else ".")
                       + f" Set it in '{OMNIVOICE_LOCAL_CONFIG}' (copy it from omnivoice_local.json.example)"
                        " or via the OMNIVOICE_VENV_PYTHON environment variable."
            )
        if not WORKER_SCRIPT.is_file():
            raise HTTPException(status_code=500, detail=f"OmniVoice worker script not found at '{WORKER_SCRIPT}'.")

        await self._kill_worker()
        self._rotate_worker_log()
        log_fh = open(WORKER_LOG_FILE, "ab")
        env = os.environ.copy()
        env["HF_HUB_CACHE"] = str(OMNIVOICE_HUB_CACHE)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        print(f"[{self.branding}ENG] \033[94mStarting OmniVoice worker process\033[0m")
        self.worker_proc = await asyncio.create_subprocess_exec(
            OMNIVOICE_VENV_PYTHON,
            str(WORKER_SCRIPT),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=log_fh,
            env=env,
        )
        self.worker_model_loaded = False
        self.worker_model_path = None
        # Confirm the worker is alive and speaking the protocol.
        resp = await self._send_command({"type": "ping"}, timeout=WORKER_PING_TIMEOUT)
        if not (resp.get("ok") and resp.get("type") == "pong"):
            await self._kill_worker()
            raise HTTPException(status_code=500, detail="OmniVoice worker process did not respond to the startup ping.")
        print(f"[{self.branding}ENG] \033[92mOmniVoice worker process is running (worker log: {WORKER_LOG_FILE})\033[0m")

    async def _send_command(self, cmd, timeout=WORKER_GENERATE_TIMEOUT):
        if self.worker_proc is None or self.worker_proc.returncode is not None:
            raise WorkerNotRunningError("OmniVoice worker process is not running.")
        self.worker_req_id += 1
        cmd_id = self.worker_req_id
        cmd_line = json.dumps({**cmd, "id": cmd_id}, ensure_ascii=False).encode("utf-8") + b"\n"
        self.worker_proc.stdin.write(cmd_line)
        await self.worker_proc.stdin.drain()
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"OmniVoice worker timed out ({timeout}s) waiting for a response to command '{cmd.get('type')}'."
                )
            raw = await asyncio.wait_for(self.worker_proc.stdout.readline(), timeout=max(remaining, 0.1))
            if not raw:
                code = self.worker_proc.returncode
                raise WorkerDeadError(
                    f"OmniVoice worker process exited unexpectedly (exit code {code}) while handling '{cmd.get('type')}'. "
                    f"See {WORKER_LOG_FILE} for details."
                )
            text = raw.decode("utf-8", errors="replace").strip()
            if not text or not text.startswith("{"):
                # Anything that is not a JSON protocol line is treated as noise and skipped.
                if self.debug_tts and text:
                    print(f"[{self.branding}Debug] OmniVoice worker non-protocol output: {text[:200]}")
                continue
            try:
                resp = json.loads(text)
            except json.JSONDecodeError:
                continue
            if resp.get("id") == cmd_id:
                return resp

    def _model_path_for(self, model_name):
        key = f"omnivoice - {model_name}"
        if self.available_models is None:
            self.available_models = self.scan_models_folder()
        return self.available_models.get(key)

    async def _ensure_worker_loaded(self, model_name):
        model_path = self._model_path_for(model_name)
        if not model_path or model_path == "omnivoice":
            raise HTTPException(
                status_code=500,
                detail=f"OmniVoice model '{model_name}' was not found. Available: {list(self.available_models.keys())}"
            )
        if self.worker_model_loaded and self.worker_model_path == model_path:
            self.is_tts_model_loaded = True
            return
        if self.worker_proc is None or self.worker_proc.returncode is not None:
            await self._spawn_worker()
        elif self.worker_model_loaded:
            resp = await self._send_command({"type": "unload"}, timeout=WORKER_UNLOAD_TIMEOUT)
            if not resp.get("ok"):
                raise HTTPException(status_code=500, detail=f"Failed to unload previous OmniVoice model: {resp.get('detail')}")
            self.worker_model_loaded = False
            self.worker_model_path = None
        device_hint = "cuda:0" if self.cuda_is_available else "cpu"
        dtype_hint = "float16" if self.cuda_is_available else "float32"
        print(f"[{self.branding}ENG] \033[94mLoading OmniVoice model '{model_name}' in worker ({device_hint}, {dtype_hint})...\033[0m")
        load_start = time.time()
        resp = await self._send_command(
            {"type": "load", "model_path": model_path, "device": device_hint, "dtype": dtype_hint},
            timeout=WORKER_LOAD_TIMEOUT,
        )
        if resp.get("ok") and resp.get("type") == "loaded":
            self.worker_model_loaded = True
            self.worker_model_path = model_path
            self.is_tts_model_loaded = True
            load_elapsed = time.time() - load_start
            print(f"[{self.branding}ENG] \033[92mOmniVoice model loaded in {load_elapsed:.2f} seconds (sampling rate {resp.get('sampling_rate')} Hz).\033[0m")
        else:
            self.is_tts_model_loaded = False
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load OmniVoice model '{model_name}': {resp.get('detail', 'unknown error')}"
            )

    ###################################################################################
    ###################################################################################
    # CHANGE ME # Inital setup of the model and engine. Called when the script starts #
    ###################################################################################
    ###################################################################################
    async def setup(self):
        self.printout_versions()
        self.available_models = self.scan_models_folder()
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        if self.selected_model:
            tts_model = f"{self.selected_model}"
            if tts_model in self.available_models:
                ok = await self.handle_tts_method_change(tts_model)
                if ok:
                    self.current_model_loaded = tts_model
                else:
                    self.current_model_loaded = "No Models Available"
            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            # ↓↓↓ Keep everything below this line ↓↓↓
            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            else:
                self.current_model_loaded = "No Models Available"
                print(f"[{self.branding}ENG] \033[91mError\033[0m: Selected model '{self.selected_model}' not found in the models folder.")
        self.setup_has_run = True

    ##################################
    ##################################
    # CHANGE ME #  Low VRAM Swapping #
    ##################################
    ##################################
    # The model lives in the sidecar worker process, so there is nothing to swap here.
    async def handle_lowvram_change(self):
        pass

    ########################################
    ########################################
    # CHANGE ME #  DeepSpeed model loading #
    ########################################
    ########################################
    # OmniVoice does not use DeepSpeed.
    async def handle_deepspeed_change(self, value):
        return value

    ##############################################################################################################################################
    ##############################################################################################################################################
    # CHANGE ME # scan for available models/voices that are relevant to this TTS engine #
    ##############################################################################################################################################
    ##############################################################################################################################################
    # Two locations are scanned:
    #   1) <AllTalk>/models/omnivoice/<model folder>  (folder must contain config.json + model.safetensors)
    #   2) The local Hugging Face cache of the OmniVoice setup (<OMNIVOICE_MODEL_ROOT>\hub,
    #      see omnivoice_local.json) so the already-downloaded k2-fsa/OmniVoice snapshot
    #      is picked up without re-downloading.
    def scan_models_folder(self):
        self.available_models = {}
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        # 1) Local AllTalk models folder
        models_folder = self.main_dir / "models" / "omnivoice"
        if models_folder.is_dir():
            for subfolder in models_folder.iterdir():
                if subfolder.is_dir():
                    if (subfolder / "config.json").is_file() and (subfolder / "model.safetensors").is_file():
                        self.available_models[f"omnivoice - {subfolder.name}"] = str(subfolder)
                    else:
                        print(f"[{self.branding}ENG] \033[91mWarning\033[0m: Model folder '{subfolder.name}' is missing config.json or model.safetensors, skipping it.")

        # 2) Local Hugging Face cache of the OmniVoice setup (offline, no download)
        if OMNIVOICE_HUB_CACHE is None:
            print(f"[{self.branding}ENG] \033[91mWarning\033[0m: OMNIVOICE_MODEL_ROOT is not configured, skipping the local OmniVoice hub cache. "
                  f"Set it in '{OMNIVOICE_LOCAL_CONFIG}' or via the OMNIVOICE_MODEL_ROOT environment variable.")
            return self.available_models
        hub_snapshots = OMNIVOICE_HUB_CACHE / f"models--{DEFAULT_HF_MODEL_ID.replace('/', '--')}" / "snapshots"
        if hub_snapshots.is_dir():
            for snap in hub_snapshots.iterdir():
                if not snap.is_dir():
                    continue
                if (
                    (snap / "config.json").is_file()
                    and (snap / "model.safetensors").is_file()
                    and (snap / "audio_tokenizer" / "config.json").is_file()
                ):
                    if f"omnivoice - {DEFAULT_HF_MODEL_ID}" not in self.available_models:
                        self.available_models[f"omnivoice - {DEFAULT_HF_MODEL_ID}"] = str(snap)
                        print(f"[{self.branding}ENG] \033[92mFound cached OmniVoice model\033[93m {DEFAULT_HF_MODEL_ID} \033[92mat\033[93m {snap}\033[0m")

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        if not self.available_models:
            self.available_models = {'No Models Available': 'omnivoice'}
            print(f"[{self.branding}ENG] \033[91mWarning\033[0m: No OmniVoice models found. Place a model in 'models\\omnivoice' or download k2-fsa/OmniVoice into the OmniVoice hub cache.")
        return self.available_models

    #############################################################
    #############################################################
    # CHANGE ME #  POPULATE FILES LIST FROM VOICES DIRECTORY ####
    #############################################################
    #############################################################
    def voices_file_list(self):
        try:
            voices = []
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
            # ↑↑↑ Keep everything above this line ↑↑↑
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

            directory = self.main_dir / "voices"
            # Step 1: Add .wav files in the main "voices" directory to the list
            if directory.is_dir():
                voices.extend([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(".wav")])
                # Step 2: Walk through subfolders and add subfolder names if they contain .wav files
                for root, dirs, files in os.walk(directory):
                    if os.path.normpath(root) != os.path.normpath(directory):
                        if any(f.endswith(".wav") for f in files):
                            folder_name = os.path.basename(root) + "/"
                            voices.append(folder_name)
            voices = [v for v in voices if v != "voices/"]

            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            # ↓↓↓ Keep everything below this line ↓↓↓
            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            if not voices:
                return ["No Voices Found"]
            return voices
        except Exception as e:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: Voices/Voice Models not found. Cannot load a list of voices. {str(e)}")
            return ["No Voices Found"]

    #############################################################################################################
    #############################################################################################################
    # CHANGE ME # Model loading / unloading / changing (delegated to the sidecar worker) #
    #############################################################################################################
    #############################################################################################################
    async def unload_model(self):
        self.is_tts_model_loaded = False
        if not self.current_model_loaded == None:
            print(f"[{self.branding}ENG] \033[94mUnloading model \033[0m") if self.debug_tts else None
        # The worker process is kept alive (it is cheap once running); the model
        # inside it is released and CUDA cache cleared on the worker side.
        if self.worker_proc is not None and self.worker_proc.returncode is None and self.worker_model_loaded:
            try:
                resp = await self._send_command({"type": "unload"}, timeout=WORKER_UNLOAD_TIMEOUT)
                if not resp.get("ok"):
                    print(f"[{self.branding}ENG] \033[91mWarning\033[0m: Worker model unload reported: {resp.get('detail')}")
            except Exception as e:
                print(f"[{self.branding}ENG] \033[91mWarning\033[0m: Could not unload model in worker: {str(e)}")
        self.worker_model_loaded = False
        self.worker_model_path = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

    async def handle_tts_method_change(self, tts_method):
        generate_start_time = time.time()
        if "No Models Available" in self.available_models:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: No models for this TTS engine were found to load. Please download a model.")
            return False
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        if not isinstance(tts_method, str) or not tts_method.startswith("omnivoice"):
            self.current_model_loaded = None
            return False
        model_name = tts_method.split(" - ")[1]

        # Make sure the model scan is current before trying to load.
        if f"omnivoice - {model_name}" not in self.available_models:
            self.available_models = self.scan_models_folder()
        if f"omnivoice - {model_name}" not in self.available_models:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: OmniVoice model '{model_name}' not found. Available: {list(self.available_models.keys())}")
            return False

        print(f"[{self.branding}ENG]\033[94m Model/Engine :\033[93m {model_name}\033[94m loading into worker on\033[93m", self.device, "\033[0m")
        try:
            await self._ensure_worker_loaded(model_name)
        except HTTPException as e:
            self.is_tts_model_loaded = False
            self.current_model_loaded = None
            print(f"[{self.branding}ENG] \033[91mError\033[0m: {str(e)}")
            return False

        self.current_model_loaded = f"omnivoice - {model_name}"
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        generate_end_time = time.time()
        generate_elapsed_time = generate_end_time - generate_start_time
        print(f"[{self.branding}ENG] \033[94mLoad time :\033[93m {generate_elapsed_time:.2f} seconds.\033[0m")
        return True

    def _resolve_ref_audio(self, voice):
        """Resolve the 'voice' argument to a single reference .wav file path."""
        print(f"[{self.branding}Debug] Voice name sent in request is:", voice) if self.debug_tts else None
        if voice is None:
            voice = self.def_character_voice
        if voice.endswith("/") or voice.endswith("\\"):
            voice = voice.rstrip("/\\")
        candidate = self.main_dir / "voices" / voice
        if os.path.isdir(str(candidate)):
            wavs_files = sorted(glob.glob(os.path.join(str(candidate), "*.wav")))
            if not wavs_files:
                raise HTTPException(status_code=400, detail=f"Voice folder '{voice}' contains no .wav files.")
            wavs_files = wavs_files[0]
            print(f"[{self.branding}Debug] Voice folder detected. Using reference WAV:", wavs_files) if self.debug_tts else None
        elif os.path.isfile(str(candidate)):
            wavs_files = str(candidate)
            print(f"[{self.branding}Debug] Single voice sample detected. Using one WAV sample:", wavs_files) if self.debug_tts else None
        else:
            raise HTTPException(status_code=400, detail=f"Voice file '{voice}' not found in the voices folder.")
        return Path(wavs_files)

    def _find_ref_text(self, wav_path):
        """Look for a transcript of the reference WAV: <stem>.reference.txt then <stem>.txt."""
        base = str(wav_path)
        if base.lower().endswith(".wav"):
            base = base[:-4]
        for suffix in (".reference.txt", ".txt"):
            cand = Path(base + suffix)
            if cand.is_file():
                try:
                    txt = cand.read_text(encoding="utf-8", errors="replace").strip()
                    if txt and not txt.lower().startswith(("http://", "https://")):
                        print(f"[{self.branding}Debug] Using reference transcript from:", cand.name) if self.debug_tts else None
                        return txt
                    # Some voice folders store a download/source URL instead of a transcript;
                    # ignore those so the worker can auto-transcribe instead.
                except Exception:
                    pass
        print(f"[{self.branding}Debug] No reference transcript found for {wav_path.name}; worker will auto-transcribe if needed.") if self.debug_tts else None
        return None

    ##########################################################################################################################################
    ##########################################################################################################################################
    # CHANGE ME # TTS generation #
    ##########################################################################################################################################
    ##########################################################################################################################################
    # The model runs inside the sidecar worker. This method sends a 'generate' command,
    # waits for the worker to write the 16-bit PCM WAV, then either hands off the file
    # (non-streaming) or fakes streaming by yielding the WAV header followed by ~0.25s
    # chunks of the raw PCM data.
    async def generate_tts(self, text, voice, language, temperature, repetition_penalty, speed, pitch, output_file, streaming):
        print(f"[{self.branding}Debug] Entered model_engine.py generate_tts function") if self.debug_tts else None
        if not self.is_tts_model_loaded:
            # Best effort: try to (re)load the selected model once before giving up.
            try:
                if self.current_model_loaded and " - " in self.current_model_loaded:
                    await self._ensure_worker_loaded(self.current_model_loaded.split(" - ")[1])
            except Exception:
                pass
        if not self.is_tts_model_loaded:
            error_message = f"[{self.branding}ENG] \033[91mError\033[0m: You currently have no TTS model loaded."
            print(error_message)
            raise HTTPException(status_code=400, detail="You currently have no TTS model loaded.")
        self.tts_generating_lock = True
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        generate_start_time = time.time()
        try:
            try:
                ref_wav_path = self._resolve_ref_audio(voice)
            except HTTPException:
                raise
            ref_text = self._find_ref_text(ref_wav_path)

            # Language: AllTalk sends 2-letter codes or "auto"/None. OmniVoice accepts
            # ISO codes or full language names, and None for language-agnostic mode.
            lang = None
            if language is not None and str(language).strip().lower() not in ("", "auto", "none"):
                lang = str(language).strip()

            # Speed: OmniVoice uses a multiplier where 1.0 = natural speed.
            try:
                speed_val = float(speed)
            except (TypeError, ValueError):
                speed_val = 1.0
            if speed_val <= 0:
                speed_val = 1.0

            output_file = str(output_file)
            print(f"[{self.branding}Debug] Text arriving at TTS engine is:", text) if self.debug_tts else None
            print(f"[{self.branding}Debug] OmniVoice generate params: lang=%s speed=%s ref=%s" % (lang, speed_val, ref_wav_path)) if self.debug_tts_variables else None

            cmd = {
                "type": "generate",
                "text": text,
                "language": lang,
                "ref_audio": str(ref_wav_path),
                "ref_text": ref_text,
                "speed": speed_val,
                "output_file": output_file,
            }
            try:
                resp = await self._send_command(cmd, timeout=WORKER_GENERATE_TIMEOUT)
            except WorkerDeadError:
                # The worker crashed mid-generation: bring it (and the model) back,
                # then ask the user to retry this generation.
                print(f"[{self.branding}ENG] \033[93mOmniVoice worker died during generation; restarting it...\033[0m")
                try:
                    await self._ensure_worker_loaded(self.current_model_loaded.split(" - ")[1])
                except Exception as restart_err:
                    print(f"[{self.branding}ENG] \033[91mCould not restart the OmniVoice worker: {str(restart_err)}\033[0m")
                    raise HTTPException(status_code=503, detail="OmniVoice worker crashed and could not be restarted. See the AllTalk server output.")
                raise HTTPException(status_code=503, detail="OmniVoice worker restarted after a crash. Please retry the generation.")
            except WorkerNotRunningError:
                await self._ensure_worker_loaded(self.current_model_loaded.split(" - ")[1])
                raise HTTPException(status_code=503, detail="OmniVoice worker is not running. Please retry the generation.")

            if not (resp.get("ok") and resp.get("type") == "generated"):
                raise HTTPException(status_code=500, detail=f"OmniVoice generation failed: {resp.get('detail', 'unknown error')}")

            if not os.path.isfile(output_file):
                raise HTTPException(status_code=500, detail="OmniVoice reported success but the output WAV file was not written.")

            if streaming:
                # Fake streaming: emit a fresh 16-bit PCM WAV header, then ~0.25s chunks.
                print(f"[{self.branding}Debug] Streaming audio generation started") if self.debug_tts else None
                with wave.open(output_file, "rb") as w:
                    sr = w.getframerate()
                    channels = max(1, w.getnchannels())
                    with io.BytesIO() as wav_buf:
                        with wave.open(wav_buf, "wb") as vfout:
                            vfout.setnchannels(channels)
                            vfout.setsampwidth(2)
                            vfout.setframerate(sr)
                            vfout.writeframes(b"")
                        yield wav_buf.getvalue()
                    frames = w.readframes(w.getnframes())
                data = np.frombuffer(frames, dtype=np.int16)
                chunk_samples = int(sr * STREAM_CHUNK_SECONDS)
                if channels > 1:
                    data = data.reshape(-1, channels)
                for i in range(0, data.shape[0], chunk_samples):
                    if self.tts_stop_generation:
                        print(f"[{self.branding}GEN] Stopping audio generation.") if self.debug_tts else None
                        self.tts_stop_generation = False
                        break
                    chunk = data[i:i + chunk_samples]
                    yield chunk.tobytes()
                print(f"[{self.branding}Debug] Streaming audio generation completed") if self.debug_tts else None
            else:
                print(f"[{self.branding}Debug] Non-streaming audio generation", output_file) if self.debug_tts else None
                yield None
        except HTTPException:
            raise
        except Exception as e:
            print(f"[{self.branding}ENG] \033[91mGeneration error: {str(e)}\033[0m")
            raise HTTPException(status_code=500, detail=f"OmniVoice generation failed: {str(e)}")
        finally:
            generate_end_time = time.time()
            generate_elapsed_time = generate_end_time - generate_start_time
            print(f"[{self.branding}GEN] \033[94mTTS Generate: \033[93m{generate_elapsed_time:.2f} seconds. \033[94mLowVRAM: \033[33m{self.lowvram_enabled} \033[94mDeepSpeed: \033[33m{self.deepspeed_enabled}\033[0m")
            self.tts_generating_lock = False
            print(f"[{self.branding}Debug] generate_tts function completed") if self.debug_tts else None
        return
