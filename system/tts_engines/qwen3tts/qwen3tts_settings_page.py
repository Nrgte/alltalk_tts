import os
import json
import requests
import gradio as gr
from tqdm import tqdm
from pathlib import Path

this_dir = Path(__file__).parent.resolve()
main_dir = Path(__file__).parent.parent.parent.parent.resolve()

def qwen3tts_voices_file_list():
    voices = []
    # Default Qwen3-TTS speakers for CustomVoice models
    qwen3_speakers = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]
    
    # Also check for user uploaded reference voices in the models/qwen3tts/voices folder
    voices_dir = main_dir / "models" / "qwen3tts" / "voices"
    if voices_dir.exists():
        user_voices = [f.name for f in voices_dir.glob("*.wav")]
        voices.extend(user_voices)
    
    # Merge predefined speakers with user voices
    voices.extend(qwen3_speakers)
    
    if not voices:
        voices = ["defaultvoice.wav"]
    return voices

def qwen3tts_model_update_settings(def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, deepspeed_enabled_gr, temperature_set_gr, repetitionpenalty_set_gr, pitch_set_gr, generationspeed_set_gr, alloy_gr, echo_gr, fable_gr, nova_gr, onyx_gr, shimmer_gr):
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

def qwen3tts_at_gradio_settings_page(model_config_data):
    features_list = model_config_data['model_capabilties']
    voice_list = qwen3tts_voices_file_list()
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
                            details_text = gr.Textbox(label="Details", show_label=False, lines=5, interactive=False, value="In this section, you can set the default settings for the Qwen3-TTS engine. Settings that are not supported by the current engine will be greyed out and cannot be selected. Default voices specified here will be used when no specific voice is provided in the TTS generation request. If a voice is specified in the request, it will override these default settings. When using the OpenAI API compatible API with this TTS engine, the voice mappings will be applied. As the OpenAI API has a limited set of 6 voices, these mappings ensure compatibility by mapping the OpenAI voices to the available voices in this TTS engine.")
            with gr.Row():
                submit_button = gr.Button("Update Settings")
                output_message = gr.Textbox(label="Output Message", interactive=False, show_label=False)
            submit_button.click(qwen3tts_model_update_settings, inputs=[def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, deepspeed_enabled_gr, temperature_set_gr, repetitionpenalty_set_gr, pitch_set_gr, generationspeed_set_gr, alloy_gr, echo_gr, fable_gr, nova_gr, onyx_gr, shimmer_gr], outputs=output_message)

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
                        gr.Textbox(label="TTS Engine Name", value="qwen3-tts", interactive=False)
                    with gr.Row():
                        gr.Textbox(label="Windows Support", value='Yes' if features_list['windows_capable'] else 'No', interactive=False)
                        gr.Textbox(label="Linux Support", value='Yes' if features_list['linux_capable'] else 'No', interactive=False)
                        gr.Textbox(label="Mac Support", value='Yes' if features_list['mac_capable'] else 'No', interactive=False)
            with gr.Row():
                gr.Markdown("""
                ####  🟧 DeepSpeed Capable
                DeepSpeed is a deep learning optimization library that can significantly speed up model training and inference. If a model is DeepSpeed capable, it means it can utilize DeepSpeed to accelerate the generation of text-to-speech output. This requires the model to be loaded into CUDA/VRAM on an Nvidia GPU and the model's inference method to support DeepSpeed.
                ####  🟧 Pitch Capable
                Pitch refers to the highness or lowness of a sound. If a model is pitch capable, it can adjust the pitch of the generated speech, allowing for more expressive and varied output.
                #### 🟧 Generation Speed Capable
                Generation speed refers to the rate at which the model can generate text-to-speech output. If a model is generation speed capable, it means the speed of the generated speech can be adjusted, making it faster or slower depending on the desired output.
                #### 🟧 Repetition Penalty Capable
                Repetition penalty is a technique used to discourage the model from repeating the same words or phrases multiple times in the same sounding way. If a model is repetition penalty capable, it can apply this penalty during generation to improve the diversity and naturalness of the output.
                #### 🟧 Multi-Languages Capable
                Multi-language capable models can generate speech in multiple languages. This means that the model has been trained on data from different languages and can switch between them during generation. Some models are language-specific.
                #### 🟧 Multi-Voice Capable
                Multi-voice capable models generate speech in multiple voices or speaking styles. This means that the model has been trained on data from different speakers and can mimic their voices during generation, or is a voice cloning model that can generate speech based on the input sample.
                """)
                gr.Markdown("""
                #### 🟧 Streaming Capable
                Streaming refers to the ability to generate speech output in real-time, without the need to generate the entire output before playback. If a model is streaming capable, it can generate speech on-the-fly, allowing for faster response times and more interactive applications.
                #### 🟧 Low VRAM Capable
                VRAM (Video Random Access Memory) is a type of memory used by GPUs to store and process data. If a model is low VRAM capable, it can efficiently utilize the available VRAM by moving data between CPU and GPU memory as needed, allowing for generation even on systems with limited VRAM where it may be competing with an LLM model for VRAM.
                #### 🟧 Temperature Capable
                Temperature is a hyperparameter that controls the randomness of the generated output. If a model is temperature capable, the temperature can be adjusted to make the output more or less random, affecting the creativity and variability of the generated speech.
                #### 🟧 Multi-Model Capable Engine
                If an engine is multi-model capable, it means that it can support and utilize multiple models for text-to-speech generation. This allows for greater flexibility and the ability to switch between different models depending on the desired output. Different models may be capable of different languages, specific languages, voices, etc.
                #### 🟧 Default Audio Output Format
                Specifies the file format in which the generated speech will be saved. Common audio formats include WAV, MP3, FLAC, Opus, AAC, and PCM. If you want different outputs, you can set the transcode function to change the output audio, though transcoding will add a little time to the generation and is not available for streaming generation.
                #### 🟧 Windows/Linux/Mac Support
                These indicators show whether the model and engine are compatible with Windows, Linux, or macOS. However, additional setup or requirements may be necessary to ensure full functionality on your operating system. Please note that full platform support has not been extensively tested.
                """)

        with gr.Tab("Models/Voices Download"):
            with gr.Row():
                gr.Markdown("Coming soon for Qwen3-TTS")
