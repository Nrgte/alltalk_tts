###############################################
# DONT CHANGE # These are base imports needed #
###############################################
import os
import sys
import json
import time
import torch
import logging
from pathlib import Path
from fastapi import (HTTPException)
logging.disable(logging.WARNING)
#################################################################
# DONT CHANGE # Get Pytorch & Python versions & setup DeepSpeed #
#################################################################
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
import soundfile as sf

# For Qwen3-TTS, we need the qwen-tts package
try:
    from qwen_tts import Qwen3TTSModel
    from huggingface_hub import snapshot_download
    qwen_tts_available = True
except ImportError:
    print("[Qwen3TTS] qwen-tts or huggingface_hub package not found. Please install them.")
    qwen_tts_available = False

#################################################################################################################################
# DONT CHANGE # Do not change the Class name from tts_class as this is what will be imported into the main tts_server.py script #
#################################################################################################################################
class tts_class:
    def __init__(self):
        ########################################################################
        # DONT CHANGE # Sets up the base variables required for any tts engine #
        ########################################################################
        self.branding = None                                                    
        self.this_dir = Path(__file__).parent.resolve()                         # Sets up self.this_dir as a variable for the folder THIS script is running in.
        self.main_dir = Path(__file__).parent.parent.parent.parent.resolve()    # Sets up self.main_dir as a variable for the folder AllTalk is running in
        self.device = "cuda" if torch.cuda.is_available() else "cpu"            # Sets up self.device to cuda if torch exists with Nvidia/CUDA, otherwise sets to cpu
        self.cuda_is_available = torch.cuda.is_available()                      # Sets up cuda_is_available as a True/False to track if Nvidia/CUDA was found on the system
        self.tts_generating_lock = False                                        # Used to lock and unlock the tts generation process at the start/end of tts generation. 
        self.tts_stop_generation = False                                        # Used in conjunction with tts_generating_lock to call for a stop to the current generation. If called (set True) it needs to be set back to False when generation has been stopped.
        self.tts_narrator_generatingtts = False                                 # Used to track if the current tts processes is narrator based. This can be used in conjunction with lowvram and device to avoid moving model between GPU(CUDA)<>RAM(CPU) each chunk of narrated text generated.
        self.model = None                                                       # If loading a model into CUDA/VRAM/RAM "model" is used as the variable name to load and interact with (see the XTTS model_engine script for examples.)
        self.is_tts_model_loaded = False                                        # Used to track if a model is actually loaded in and error/fail things like TTS generation if its False
        self.current_model_loaded = None                                        # Stores the name of the currenly loaded in model
        self.available_models = None                                            # List of available models found by "def scan_models_folder"
        self.setup_has_run = False                                              # Tracks if async def setup(self) has run, by setting to True, so that the /api/ready endpoint can provide a "Ready" status
        ##############################################################################################
        # DONT CHANGE # Load in a list of the available TTS engines and the currently set TTS engine #
        ##############################################################################################
        tts_engines_file = os.path.join(self.main_dir, "system", "tts_engines", "tts_engines.json")
        with open(tts_engines_file, "r") as f:
            tts_engines_data = json.load(f)
        self.engines_available = [engine["name"] for engine in tts_engines_data["engines_available"]]       # A list of ALL the TTS engines available to be loaded by AllTalk
        self.engine_loaded = tts_engines_data["engine_loaded"]                                              # In "tts_engines.json" what is the currently set TTS engine loading into AllTalk
        self.selected_model = tts_engines_data["selected_model"]                                            # In "tts_engines.json" what is the currently set TTS model loading into AllTalk
        ############################################################################
        # DONT CHANGE # Pull out all the settings for the currently set TTS engine #
        ############################################################################
        with open(os.path.join(self.this_dir, "model_settings.json"), "r") as f:
            tts_model_loaded = json.load(f)
        # Access the model details
        self.manufacturer_name = tts_model_loaded["model_details"]["manufacturer_name"]                     # The company/person/body that generated the TTS engine/models etc
        self.manufacturer_website = tts_model_loaded["model_details"]["manufacturer_website"]               # The website of the company/person/body where people can find more information
        # Access the features the model is capable of:
        self.audio_format = tts_model_loaded["model_capabilties"]["audio_format"]                           # This details the audio format your TTS engine is set to generate TTS in e.g. wav, mp3, flac, opus, acc, pcm. Please use only 1x format.
        self.deepspeed_capable = tts_model_loaded["model_capabilties"]["deepspeed_capable"]                 # Is your model capable of DeepSpeed
        self.deepspeed_available = 'deepspeed' in globals()                                                 # When we did the import earlier, at the top of this script, was DeepSpeed available for use
        self.generationspeed_capable = tts_model_loaded["model_capabilties"]["generationspeed_capable"]     # Does this TTS engine support changing the speed of the generated TTS
        self.languages_capable = tts_model_loaded["model_capabilties"]["languages_capable"]                 # Are the actual models themselves capable of generating in multiple languages OR is each model language specific
        self.lowvram_capable = tts_model_loaded["model_capabilties"]["lowvram_capable"]                     # Is this engine capable of using low VRAM (moving the model between CPU And GPU memory)
        self.multimodel_capable = tts_model_loaded["model_capabilties"]["multimodel_capable"]               # Is there just the one model or are there multiple models this engine supports.
        self.repetitionpenalty_capable = tts_model_loaded["model_capabilties"]["repetitionpenalty_capable"] # Is this TTS engine capable of changing the repititon penalty
        self.streaming_capable = tts_model_loaded["model_capabilties"]["streaming_capable"]                 # Is this TTS engine capabale of generating streaming audio
        self.temperature_capable = tts_model_loaded["model_capabilties"]["temperature_capable"]             # Is this TTS engine capable of changing the temperature of the models
        self.multivoice_capable = tts_model_loaded["model_capabilties"]["multivoice_capable"]               # Are the models multi-voice or single vocice models
        self.pitch_capable = tts_model_loaded["model_capabilties"]["pitch_capable"]                         # Is this TTS engine capable of changing the pitch of the genrated TTS
        # Access the current enginesettings
        self.def_character_voice = tts_model_loaded["settings"]["def_character_voice"]                      # What is the current default main/character voice that will be used if no voice specified.
        self.def_narrator_voice = tts_model_loaded["settings"]["def_narrator_voice"]                        # What is the current default narrator voice that will be used if no voice specified.
        self.deepspeed_enabled = tts_model_loaded["settings"]["deepspeed_enabled"]                          # If its available, is DeepSpeed enabled for the TTS engine
        self.engine_installed = tts_model_loaded["settings"]["engine_installed"]                            # Has the TTS engine been setup/installed (not curently used)
        self.generationspeed_set = tts_model_loaded["settings"]["generationspeed_set"]                      # What is the set/stored speed for generation.
        self.lowvram_enabled = tts_model_loaded["settings"]["lowvram_enabled"]                              # If its available, is LowVRAM enabled for the TTS engine
        # Check if someone has enabled lowvram on a system that's not CUDA enabled
        self.lowvram_enabled = False if not torch.cuda.is_available() else self.lowvram_enabled             # If LowVRAM is mistakenly set and CUDA is not available, this will force it back off
        self.repetitionpenalty_set = tts_model_loaded["settings"]["repetitionpenalty_set"]                  # What is the currenly set repitition policy of the model (If it support repetition)
        self.temperature_set = tts_model_loaded["settings"]["temperature_set"]                              # What is the currenly set temperature of the model (If it support temp)
        self.pitch_set = tts_model_loaded["settings"]["pitch_set"]                                          # What is the currenly set pitch of the model (If it support temp)
        # Gather the OpenAI API Voice Mappings
        self.openai_alloy = tts_model_loaded["openai_voices"]["alloy"]                                      # The TTS engine voice that will be mapped to Open AI Alloy voice
        self.openai_echo = tts_model_loaded["openai_voices"]["echo"]                                        # The TTS engine voice that will be mapped to Open AI Echo voice
        self.openai_fable = tts_model_loaded["openai_voices"]["fable"]                                      # The TTS engine voice that will be mapped to Open AI Fable voice
        self.openai_nova = tts_model_loaded["openai_voices"]["nova"]                                        # The TTS engine voice that will be mapped to Open AI Nova voice
        self.openai_onyx = tts_model_loaded["openai_voices"]["onyx"]                                        # The TTS engine voice that will be mapped to Open AI Onyx voice
        self.openai_shimmer = tts_model_loaded["openai_voices"]["shimmer"]                                  # The TTS engine voice that will be mapped to Open AI Shimmer voice
        ###################################################################
        # DONT CHANGE #  Load params and api_defaults from confignew.json #
        ###################################################################
        # Define the path to the confignew.json file
        configfile_path = self.main_dir / "confignew.json"
        # Load config file and get settings
        with open(configfile_path, "r") as configfile:
            configfile_data = json.load(configfile)
        self.branding = configfile_data.get("branding", "")                                                 # Sets up self.branding for outputting the name stored in the "confgnew.json" file, as used in print statements.
        self.params = configfile_data                                                                       # Loads in the curent "confgnew.json" file to self.params.
        self.debug_tts = configfile_data.get("debugging").get("debug_tts")                                  # Can be used within this script as a True/False flag for generally debugging the TTS generation process. 
        self.debug_tts_variables = configfile_data.get("debugging").get("debug_tts_variables")              # Can be used within this script as a True/False flag for generally debugging variables (if you wish).
        
        # Qwen3-TTS specific setup
        self.models_dir = self.main_dir / "models" / "qwen3tts"
        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=True)
            
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
    # CHANGE ME # Inital setup of the model and engine. Called when the script starts #
    ###################################################################################
    ###################################################################################
    async def setup(self):
        self.printout_versions()
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        print(f"[{self.branding}ENG] \033[92mInitializing Qwen3-TTS engine...\033[0m")
        
        if not qwen_tts_available:
            print(f"[{self.branding}ENG] \033[91mqwen-tts package is missing. Cannot initialize.\033[0m")
            return

        # Scan for available models
        self.available_models = self.scan_models_folder()
        
        # Determine the model path (Hugging Face ID or local path)
        model_id = self.selected_model # e.g., "qwen3tts - Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        
        # If the model_id contains the engine name prefix, strip it
        if " - " in model_id:
            model_id = model_id.split(" - ")[1]
        
        # Check if we need to download the model to the local models folder
        local_model_path = self.models_dir / model_id.split("/")[-1]
        
        if not local_model_path.exists():
            print(f"[{self.branding}ENG] \033[93mModel not found locally. Downloading {model_id} to {local_model_path}...\033[0m")
            try:
                snapshot_download(repo_id=model_id, local_dir=local_model_path)
                print(f"[{self.branding}ENG] \033[92mDownload complete.\033[0m")
            except Exception as e:
                print(f"[{self.branding}ENG] \033[91mFailed to download model: {e}\033[0m")
                return

        try:
            # Determine best attention implementation. 
            # flash_attention_2 is often problematic on Windows.
            # sdpa is built into PyTorch 2.x and is very reliable.
            attn_implementation = "sdpa"
            if self.cuda_is_available:
                try:
                    import flash_attn
                    from flash_attn import flash_attn_func
                    # Double check it's not our 'broken' bridge version
                    if getattr(flash_attn, "__version__", "") != "0.0.0-broken":
                        attn_implementation = "flash_attention_2"
                    else:
                        print(f"[{self.branding}ENG] \033[93mFlash Attention found but non-functional (missing kernels). Falling back to SDPA.\033[0m")
                except (ImportError, AttributeError):
                    print(f"[{self.branding}ENG] \033[93mFlash Attention not found or incompatible. Falling back to SDPA.\033[0m")
                    attn_implementation = "sdpa"
            
            # Load from the local path
            print(f"[{self.branding}ENG] \033[94mLoading model from: {local_model_path} using {attn_implementation}\033[0m")
            try:
                self.model = Qwen3TTSModel.from_pretrained(
                    str(local_model_path),
                    device_map=self.device,
                    dtype=torch.bfloat16 if self.cuda_is_available else torch.float32,
                    attn_implementation=attn_implementation,
                )
            except Exception as e:
                if attn_implementation == "flash_attention_2":
                    print(f"[{self.branding}ENG] \033[93mFailed to load with flash_attention_2: {e}. Retrying with sdpa...\033[0m")
                    self.model = Qwen3TTSModel.from_pretrained(
                        str(local_model_path),
                        device_map=self.device,
                        dtype=torch.bfloat16 if self.cuda_is_available else torch.float32,
                        attn_implementation="sdpa",
                    )
                else:
                    raise e
            
            self.is_tts_model_loaded = True
            self.current_model_loaded = model_id
            print(f"[{self.branding}ENG] \033[92mQwen3-TTS engine ready with model: {model_id}\033[0m")
        except Exception as e:
            print(f"[{self.branding}ENG] \033[91mFailed to load Qwen3-TTS model: {e}\033[0m")
            self.is_tts_model_loaded = False
         
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        self.setup_has_run = True # Flag that setup has run, so the /api/ready endpoint will send a "Ready" status and load the webui

    def scan_models_folder(self):
        self.available_models = {}
        if not self.models_dir.exists():
            return {"No Models Found": "No Models Found"}
            
        # Check for subdirectories in models/qwen3-tts
        for model_path in self.models_dir.iterdir():
            if model_path.is_dir():
                # We assume if the folder exists, it's a model
                # You might want to add more checks here (e.g., config.json)
                if (model_path / "config.json").exists():
                    model_name = model_path.name
                    # Map local name back to HF ID if possible, or just use local name
                    # For now, let's just use the folder name as the ID
                    self.available_models[model_name] = model_name
        
        if not self.available_models:
            self.available_models["No Models Found"] = "No Models Found"
        return self.available_models

    async def handle_tts_method_change(self, tts_method):
        if self.is_tts_model_loaded:
            await self.unload_model()
        
        self.selected_model = tts_method
        await self.setup()
        return self.is_tts_model_loaded

    def voices_file_list(self):
        try:
            voices = []
            directory = self.main_dir / "voices"
            
            # Step 1: Add .wav files in the main "voices" directory to the list
            if directory.exists():
                voices.extend([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(".wav")])
                
                # Step 2: Walk through subfolders and add subfolder names if they contain .wav files
                for root, dirs, files in os.walk(directory):
                    # Skip the root directory itself and only consider subfolders
                    if os.path.normpath(root) != os.path.normpath(directory):
                        if any(f.endswith(".wav") for f in files):
                            folder_name = os.path.basename(root) + "/"
                            voices.append(folder_name)
            
            # Remove "voices/" from the list if it somehow got added
            voices = [v for v in voices if v != "voices/"]
            
            if not voices:
                return ["No Voices Found"]
            return voices
        except Exception as e:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: Voices/Voice Models not found. {str(e)}")
            return ["No Voices Found"]

    
    ##################################
    ##################################
    # CHANGE ME #  Low VRAM Swapping #
    ##################################
    ##################################
    async def handle_lowvram_change(self):
        if self.lowvram_enabled and self.cuda_is_available:
            print(f"[{self.branding}ENG] \033[93mMoving Qwen3-TTS model to CPU\033[0m")
            # self.model.to("cpu")
        elif not self.lowvram_enabled and self.cuda_is_available:
            print(f"[{self.branding}ENG] \033[93mMoving Qwen3-TTS model to GPU\033[0m")
            # self.model.to("cuda")
        
    ########################################
    ########################################
    # CHANGE ME #  DeepSpeed model loading #
    ########################################
    ########################################
    async def handle_deepspeed_change(self, value):
        if value:
            # DeepSpeed enabled
            print(f"[{self.branding}ENG] \033[93mDeepSpeed Activating for Qwen3-TTS\033[0m")
            await self.unload_model()
            self.params["tts_method_api_local"] = True
            self.deepspeed_enabled = True
            await self.setup()

    async def unload_model(self):
        print(f"[{self.branding}ENG] \033[93mUnloading Qwen3-TTS model\033[0m")
        self.model = None
        self.is_tts_model_loaded = False
        if self.cuda_is_available:
            torch.cuda.empty_cache()

    async def generate_tts(self, text, voice, language, temperature, repetition_penalty, speed, pitch, output_file, streaming):
        if not self.is_tts_model_loaded:
            raise HTTPException(status_code=500, detail="TTS model not loaded")
        
        self.tts_generating_lock = True
        print(f"[{self.branding}ENG] Generating TTS with Qwen3-TTS: {text[:50]}...")
        
        start_time = time.time()
        try:
            # Language mapping for Qwen3-TTS
            # AllTalk uses ISO codes like 'en', 'zh', 'fr', etc.
            # Qwen3-TTS expects full names like 'english', 'chinese', 'french', etc.
            lang_map = {
                "en": "english",
                "zh": "chinese",
                "fr": "french",
                "de": "german",
                "it": "italian",
                "ja": "japanese",
                "ko": "korean",
                "pt": "portuguese",
                "ru": "russian",
                "es": "spanish",
                "tr": "turkish"
            }
            
            # Normalize language input
            if language:
                language = language.lower()
                if language in lang_map:
                    language = lang_map[language]
                elif language == "auto":
                    language = "auto"
            else:
                language = "auto"

            # Handle voice path detection similar to XTTS
            if voice.endswith("/") or voice.endswith("\\"):
                voice = voice.rstrip("/\\")
            
            ref_audio = None
            ref_audio_path = self.main_dir / "voices" / voice
            if ref_audio_path.exists():
                ref_audio = str(ref_audio_path)

            # Check which sub-model is loaded to use the correct generation method
            model_name = self.current_model_loaded.lower()
            
            if "customvoice" in model_name:
                # CustomVoice: text, language, speaker, instruct
                wavs, sr = self.model.generate_custom_voice(
                    text=text,
                    language=language if language else "Auto",
                    speaker=voice if voice else self.def_character_voice,
                    x_vector_only_mode=True
                )
            elif "voicedesign" in model_name:
                # VoiceDesign: text, language, instruct
                wavs, sr = self.model.generate_voice_design(
                    text=text,
                    language=language if language else "Auto",
                    x_vector_only_mode=True
                )
            elif "base" in model_name:
                # Base (Voice Clone): text, language, ref_audio, ref_text
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language=language if language else "Auto",
                    ref_audio=ref_audio if ref_audio else "",
                    x_vector_only_mode=True
                )
            else:
                # Fallback to generic generation if model type is unclear
                wavs, sr = self.model.generate_custom_voice(
                    text=text,
                    language=language if language else "Auto",
                    speaker=voice if voice else "Vivian",
                    x_vector_only_mode=True
                )

            # Save the first generated wav
            sf.write(output_file, wavs[0], sr)
            
            end_time = time.time()
            duration = end_time - start_time
            print(f"[{self.branding}ENG] Inference completed in {duration:.2f} seconds")
            
            # Since AllTalk expects an async generator that yields chunks (for both streaming and non-streaming exhaustion)
            # We yield a single dummy value to satisfy the 'async for _ in response' loop in tts_server.py
            yield None
            
        except Exception as e:
            print(f"[{self.branding}ENG] \033[91mGeneration error: {e}\033[0m")
            self.tts_generating_lock = False
            raise HTTPException(status_code=500, detail=f"Qwen3-TTS generation failed: {e}")
        
        self.tts_generating_lock = False
