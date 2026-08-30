import os
import json
import gradio as gr
from pathlib import Path

this_dir = Path(__file__).parent.resolve()
main_dir = Path(__file__).parent.parent.parent.parent.resolve()


def omnivoice_voices_file_list():
    voices = []
    voices_dir = main_dir / "voices"
    if voices_dir.exists():
        voices.extend(sorted(f.name for f in voices_dir.glob("*.wav")))
        for root, dirs, files in os.walk(voices_dir):
            if os.path.normpath(root) != os.path.normpath(str(voices_dir)):
                if any(f.endswith(".wav") for f in files):
                    voices.append(os.path.basename(root) + "/")
    if not voices:
        voices = ["No Voices Found"]
    return voices


def omnivoice_model_update_settings(def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, deepspeed_enabled_gr, temperature_set_gr, repetitionpenalty_set_gr, pitch_set_gr, generationspeed_set_gr, alloy_gr, echo_gr, fable_gr, nova_gr, onyx_gr, shimmer_gr):
    with open(os.path.join(this_dir, "model_settings.json"), "r") as f:
        model_config_data = json.load(f)

    model_config_data["settings"]["def_character_voice"] = def_character_voice_gr
    model_config_data["settings"]["def_narrator_voice"] = def_narrator_voice_gr
    model_config_data["openai_voices"]["alloy"] = alloy_gr
    model_config_data["openai_voices"]["echo"] = echo_gr
    model_config_data["openai_voices"]["fable"] = fable_gr
    model_config_data["openai_voices"]["nova"] = nova_gr
    model_config_data["openai_voices"]["onyx"] = onyx_gr
    model_config_data["openai_voices"]["shimmer"] = shimmer_gr
    model_config_data["settings"]["lowvram_enabled"] = lowvram_enabled_gr == "Enabled"
    model_config_data["settings"]["deepspeed_enabled"] = deepspeed_enabled_gr == "Enabled"
    model_config_data["settings"]["temperature_set"] = temperature_set_gr
    model_config_data["settings"]["repetitionpenalty_set"] = repetitionpenalty_set_gr
    model_config_data["settings"]["pitch_set"] = pitch_set_gr
    model_config_data["settings"]["generationspeed_set"] = generationspeed_set_gr

    with open(os.path.join(this_dir, "model_settings.json"), "w") as f:
        json.dump(model_config_data, f, indent=4)
    return "Settings updated successfully!"


def omnivoice_at_gradio_settings_page(model_config_data):
    features_list = model_config_data['model_capabilties']
    voice_list = omnivoice_voices_file_list()
    with gr.Tabs():
        with gr.Tab("Default Settings"):
            with gr.Row():
                lowvram_enabled_gr = gr.Radio(choices=[("Enabled", "true"), ("Disabled", "false")], label="Low VRAM" if model_config_data["model_capabilties"]["lowvram_capable"] else "Low VRAM N/A", value="true" if model_config_data["settings"]["lowvram_enabled"] else "false", interactive=model_config_data["model_capabilties"]["lowvram_capable"])
                deepspeed_enabled_gr = gr.Radio(choices=[("Enabled", "true"), ("Disabled", "false")], label="DeepSpeed Activate" if model_config_data["model_capabilties"]["deepspeed_capable"] else "DeepSpeed N/A", value="true" if model_config_data["settings"]["deepspeed_enabled"] else "false", interactive=model_config_data["model_capabilties"]["deepspeed_capable"])
                temperature_set_gr = gr.Slider(value=float(model_config_data["settings"]["temperature_set"]), minimum=0, maximum=1, step=0.05, label="Temperature" if model_config_data["model_capabilties"]["temperature_capable"] else "Temperature N/A", interactive=model_config_data["model_capabilties"]["temperature_capable"])
                repetitionpenalty_set_gr = gr.Slider(value=float(model_config_data["settings"]["repetitionpenalty_set"]), minimum=1, maximum=20, step=0.1, label="Repetition Penalty" if model_config_data["model_capabilties"]["repetitionpenalty_capable"] else "Repetition N/A", interactive=model_config_data["model_capabilties"]["repetitionpenalty_capable"])
                pitch_set_gr = gr.Slider(value=float(model_config_data["settings"]["pitch_set"]), minimum=-10, maximum=10, step=1, label="Pitch" if model_config_data["model_capabilties"]["pitch_capable"] else "Pitch N/A", interactive=model_config_data["model_capabilties"]["pitch_capable"])
                generationspeed_set_gr = gr.Slider(value=float(model_config_data["settings"]["generationspeed_set"]), minimum=0.25, maximum=2.00, step=0.25, label="Speed" if model_config_data["model_capabilties"]["generationspeed_capable"] else "Speed N/A", interactive=model_config_data["model_capabilties"]["generationspeed_capable"])
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### OpenAI Voice Mappings")
                    with gr.Group():
                        with gr.Row():
                            alloy_gr = gr.Dropdown(value=model_config_data["openai_voices"]["alloy"], label="Alloy", choices=voice_list, allow_custom_value=True)
                            echo_gr = gr.Dropdown(value=model_config_data["openai_voices"]["echo"], label="Echo", choices=voice_list, allow_custom_value=True)
                        with gr.Row():
                            fable_gr = gr.Dropdown(value=model_config_data["openai_voices"]["fable"], label="Fable", choices=voice_list, allow_custom_value=True)
                            nova_gr = gr.Dropdown(value=model_config_data["openai_voices"]["nova"], label="Nova", choices=voice_list, allow_custom_value=True)
                        with gr.Row():
                            onyx_gr = gr.Dropdown(value=model_config_data["openai_voices"]["onyx"], label="Onyx", choices=voice_list, allow_custom_value=True)
                            shimmer_gr = gr.Dropdown(value=model_config_data["openai_voices"]["shimmer"], label="Shimmer", choices=voice_list, allow_custom_value=True)
                with gr.Column():
                    gr.Markdown("### Default Voices")
                    with gr.Row():
                        def_character_voice_gr = gr.Dropdown(value=model_config_data["settings"]["def_character_voice"], label="Default/Character Voice", choices=voice_list, allow_custom_value=True)
                        def_narrator_voice_gr = gr.Dropdown(value=model_config_data["settings"]["def_narrator_voice"], label="Narrator Voice", choices=voice_list, allow_custom_value=True)
                    with gr.Group():
                        with gr.Row():
                            details_text = gr.Textbox(label="Details", show_label=False, lines=5, interactive=False, value="In this section, you can set the default settings for the OmniVoice engine. Settings that are not supported by this engine are greyed out. OmniVoice clones the voice from the reference WAV selected per request (or the default voices set here when none is specified). A transcript file named '<voice>.reference.txt' or '<voice>.txt' next to the reference WAV speeds up generation; without one, the worker auto-transcribes the reference with a local Whisper model.")
            with gr.Row():
                submit_button = gr.Button("Update Settings")
                output_message = gr.Textbox(label="Output Message", interactive=False, show_label=False)
            submit_button.click(omnivoice_model_update_settings, inputs=[def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, deepspeed_enabled_gr, temperature_set_gr, repetitionpenalty_set_gr, pitch_set_gr, generationspeed_set_gr, alloy_gr, echo_gr, fable_gr, nova_gr, onyx_gr, shimmer_gr], outputs=output_message)

        with gr.Tab("Engine Information"):
            with gr.Row():
                with gr.Group():
                    gr.Textbox(label="Manufacturer Name", value=model_config_data['model_details']['manufacturer_name'], interactive=False)
                    gr.Textbox(label="Manufacturer Website/TTS Engine Support", value=model_config_data['model_details']['manufacturer_website'], interactive=False)
                    gr.Textbox(label="Engine/Model Description", value=model_config_data['model_details']['model_description'], interactive=False, lines=13)
                with gr.Column():
                    with gr.Row():
                        gr.Textbox(label="DeepSpeed Capable", value='Yes' if features_list['deepspeed_capable'] else 'No', interactive=False)
                        gr.Textbox(label="Pitch Capable", value='Yes' if features_list['pitch_capable'] else 'No', interactive=False)
                        gr.Textbox(label="Generation Speed Capable", value='Yes' if features_list['generationspeed_capable'] else 'No', interactive=False)
                    with gr.Row():
                        gr.Textbox(label="Repetition Penalty Capable", value='Yes' if features_list['repetitionpenalty_capable'] else 'No', interactive=False)
                        gr.Textbox(label="Multi Languages Capable", value='Yes' if features_list['languages_capable'] else 'No', interactive=False)
                        gr.Textbox(label="Streaming Capable", value='Yes' if features_list['streaming_capable'] else 'No', interactive=False)
                    with gr.Row():
                        gr.Textbox(label="Low VRAM Capable", value='Yes' if features_list['lowvram_capable'] else 'No', interactive=False)
                        gr.Textbox(label="Temperature Capable", value='Yes' if features_list['temperature_capable'] else 'No', interactive=False)
                        gr.Textbox(label="Multi Model Capable Engine", value='Yes' if features_list['multimodel_capable'] else 'No', interactive=False)
                    with gr.Row():
                        gr.Textbox(label="Multi Voice Capable Models", value='Yes' if features_list['multivoice_capable'] else 'No', interactive=False)
                        gr.Textbox(label="Default Audio output format", value=model_config_data['model_capabilties']['audio_format'], interactive=False)
                        gr.Textbox(label="TTS Engine Name", value="omnivoice", interactive=False)
                    with gr.Row():
                        gr.Textbox(label="Windows Support", value='Yes' if features_list['windows_capable'] else 'No', interactive=False)
                        gr.Textbox(label="Linux Support", value='Yes' if features_list['linux_capable'] else 'No', interactive=False)
                        gr.Textbox(label="Mac Support", value='Yes' if features_list['mac_capable'] else 'No', interactive=False)
            with gr.Row():
                gr.Markdown("""
                ####  How the OmniVoice engine works
                OmniVoice requires transformers 5.x while AllTalk runs transformers 4.x, so the model cannot be imported directly. Instead, this engine spawns a persistent sidecar worker process (`omnivoice_worker.py`) using the OmniVoice venv interpreter and talks to it over a JSON protocol on stdin/stdout. The worker loads the model once, keeps it in memory between generations, and writes each generated 16-bit PCM WAV file which AllTalk then serves.
                ####  Required paths (configured per machine)
                Set these in `system/tts_engines/omnivoice/omnivoice_local.json` (copy it from `omnivoice_local.json.example` and adjust). Environment variables of the same name take precedence:
                - `OMNIVOICE_VENV_PYTHON` - the OmniVoice venv interpreter used to run the worker.
                - `OMNIVOICE_MODEL_ROOT` - root of the OmniVoice setup; its `hub` folder is the offline Hugging Face cache.
                - Worker diagnostics are written to `system/tts_engines/omnivoice/omnivoice_worker.log` (rotated to `omnivoice_worker.log.old` once it exceeds 1 MB).
                #### 🟧 Generation Speed Capable
                Generation speed refers to the rate at which the model generates speech. OmniVoice accepts a speed multiplier where 1.0 is the natural speaking rate, values above 1.0 are faster and values below 1.0 are slower.
                #### 🟧 Multi-Languages Capable
                OmniVoice can generate speech in many languages. Selecting a language per request improves results; leaving it unset runs the model in language-agnostic mode.
                #### 🟧 Multi-Voice Capable
                OmniVoice is a voice cloning model: any reference WAV sample in the voices folder (ideally with a matching transcript file) can be used to clone that voice for generation.
                #### 🟧 Multi-Model Capable Engine
                Any OmniVoice checkpoint placed in the `models/omnivoice/` folder (containing `config.json` and `model.safetensors`) is picked up, in addition to the cached `k2-fsa/OmniVoice` snapshot from the OmniVoice hub cache.
                """)

        with gr.Tab("Models/Voices Download"):
            with gr.Row():
                gr.Markdown("""
                ### OmniVoice models
                No download is needed when the local OmniVoice setup is present: the already-cached `k2-fsa/OmniVoice` snapshot (in the OmniVoice hub cache) is used directly, fully offline.
                To use a different/additional checkpoint, copy the full model folder (including the `audio_tokenizer` subfolder, `config.json`, `model.safetensors` and tokenizer files) into:
                `models/omnivoice/<model name>` and press "Refresh Settings".
                Official model (for manual download elsewhere): https://huggingface.co/k2-fsa/OmniVoice
                """)
