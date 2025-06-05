import os  # For path joining
import random  # For fallback seeding

import torch  # Used by wgp.py, good for type hints
from transformers import AutoTokenizer  # For prompt enhancer

import wgp
from ltx_video.utils.prompt_enhance_utils import (
    generate_cinematic_prompt,
)  # For prompt enhancer
from PIL import Image

# 1. Initialize minimal state and helper functions
# A default T2V model
model_name = "ckpts/ltxv_0.9.7_13B_dev_quanto_bf16_int8.safetensors"

# Prompt enhancer configuration
ENABLE_PROMPT_ENHANCER = True  # Set to False to disable prompt enhancement
LLM_ENHANCER_MODEL_DIR = "ckpts/Llama3_2"
LLM_ENHANCER_MODEL_FILE = "ckpts/Llama3_2/Llama3_2_quanto_bf16_int8.safetensors"

# Ensure the 'ckpts' directory exists for model download if it's the first run
os.makedirs("ckpts", exist_ok=True)
# Ensure 'settings' dir exists for wgp.py to write default model settings
# Accessing wgp.args requires wgp.py to have fully initialized its args.
# Safer to ensure wgp's init path handles this or do it after wgp's setup.
if hasattr(wgp, "args") and hasattr(wgp.args, "settings"):
    os.makedirs(wgp.args.settings, exist_ok=True)
else:
    # Fallback if wgp.args.settings not yet available (e.g. wgp.py
    # not fully run)
    # This might happen if wgp.py's main execution block hasn't run.
    # In a script, wgp.py's __main__ block won't run automatically.
    # We rely on its top-level initializations.
    # A common pattern is for wgp.py to define args at the top level.
    # If wgp.args is defined at top level, this 'else' might not be needed.
    # Assume wgp.py makes `args.settings` available upon import.
    # If not, this script would need to replicate some of wgp.py's setup.
    # For simplicity, assume wgp.args.settings is available.
    # If wgp.py is imported, its top-level code runs, including _parse_args()
    # which defines `args`.
    pass


# Minimal state required by generate_video and its callees
script_state = {
    "model_filename": model_name,  # This will be used by generate_video
    "gen": {
        "file_list": [],
        "file_settings_list": [],
        "prompt_no": 0,
        "selected": 0,
        "last_selected": True,
        "queue": [],
        "loras": [],
        "loras_presets": [],
        "loras_names": [],
        "advanced": False,
        # Other gen-specific keys like 'status', 'progress_status'
        # might be set by generate_video
    },
    "loras": [],  # Top-level state['loras']
    "advanced": False,  # Top-level state['advanced']
    # wgp.py's get_default_settings might populate model-specific settings here
    # e.g., state['t2v_1.3B'] = { ... defaults ... }
    # For a direct call to generate_video, these aren't strictly needed
    # in `state` if all params are passed directly.
}


# Dummy send_cmd function
def script_send_cmd(command, data=None):
    if command == "progress":
        if data and isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], tuple) and len(data[0]) == 2:
                print(f"Progress: {data[0][0]}/{data[0][1]} - {data[1]}")
            elif len(data) >= 2:
                print(f"Status: {data[1]}")  # Assuming data[1] is the message
            else:
                print(f"Progress data: {data}")
        # if data is not None but not a list as expected for progress
        elif data:
            print(f"Progress (unexpected data format): {data}")
        else:
            print("Progress update (no data).")
    elif command == "preview":
        print("Preview generated (not shown in script mode).")
    elif command == "output":
        print("Output signal received.")  # This command doesn't use data
    elif command == "status":
        print(f"Status update: {data if data is not None else 'N/A'}")
    elif command == "info":
        print(f"Info: {data if data is not None else 'N/A'}")
    elif command == "error":
        print(f"Error: {data if data is not None else 'N/A'}")
        # Option to halt on error
        # raise Exception(f"Generation error: {data}")
    elif command == "exit":
        # This command doesn't use data
        print("Generation process finished signal.")
    else:
        print(
            f"send_cmd: Unknown command '{command}' with data: "
            f"{data if data is not None else 'N/A'}"
        )


script_task_item = {
    "id": "sample_script_task_001",
    "params": {},  # Populated by video_params for generate_video call
    "prompt": "A cute cat wearing a wizard hat",
    "repeats": 1,
    "length": 17,
    "steps": 10,
    "start_image_data_base64": None,  # For update_task_thumbnails
    "end_image_data_base64": None,  # For update_task_thumbnails
}

video_params = {
    "prompt": "Liquid-mercury contortionist flows between morphing sound sculptures, abstract geometry space, dynamic dolly zoom, surrealist style.",
    "negative_prompt": "blurry, low quality, ugly, deformed",
    "resolution": "832x480",
    "video_length": 482,  # Must be 8k+1 for LTX Video (240%8=0, 241%8=1)
    "seed": -1,
    "num_inference_steps": 20,
    "guidance_scale": 7.0,
    "audio_guidance_scale": 5.0,
    "flow_shift": 5.0,
    "embedded_guidance_scale": 6.0,
    "repeat_generation": 1,
    "multi_images_gen_type": 0,
    "tea_cache_setting": 0.0,
    "tea_cache_start_step_perc": 0,
    "activated_loras": [],
    "loras_multipliers": "",
    "image_prompt_type": "I2V",
    "image_start": [Image.open("/root/schnell-output.png")],
    "image_end": None,
    "model_mode": 0,
    "video_source": None,
    "keep_frames_video_source": "",
    "video_prompt_type": "",
    "image_refs": None,
    "video_guide": None,
    "keep_frames_video_guide": "",
    "video_mask": None,
    "audio_guide": None,
    "sliding_window_size": 257,  # Must be 8k+1 for LTX Video (129%8=1)
    "sliding_window_overlap": 9,
    "sliding_window_overlap_noise": 20,
    "sliding_window_discard_last_frames": 8,
    "remove_background_images_ref": 0,
    "temporal_upsampling": "",
    "spatial_upsampling": "",
    "RIFLEx_setting": 0,
    "slg_switch": 0,
    "slg_layers": [9],
    "slg_start_perc": 10,
    "slg_end_perc": 90,
    "cfg_star_switch": 0,
    "cfg_zero_step": -1,
    "prompt_enhancer": "",
    "model_filename": model_name,
}

script_task_item["prompt"] = video_params["prompt"]
script_task_item["length"] = video_params["video_length"]
script_task_item["steps"] = video_params["num_inference_steps"]
# script_task_item["params"] is not directly used if we pass
# video_params to generate_video

if __name__ == "__main__":
    print(
        f"Wan2GP version: "
        f"{wgp.WanGP_version if hasattr(wgp, 'WanGP_version') else 'Unknown'}"
    )
    print(f"Using model: {model_name}")

    # wgp.py initializes args, server_config, etc. at its top level.
    # These should be available after `import wgp`.
    # Specifically, wgp.save_path should be set.
    # And wgp.py should handle creation of wgp_config.json
    # and default model settings.

    # Ensure settings directory from wgp.args is created if not
    # already by wgp.py
    # This is a bit redundant if wgp.py already does it, but safe.
    if hasattr(wgp, "args") and hasattr(wgp.args, "settings") and wgp.args.settings:
        if not os.path.exists(wgp.args.settings):
            os.makedirs(wgp.args.settings, exist_ok=True)
            print(f"Ensured settings directory exists: {wgp.args.settings}")

    # Ensure save_path directory exists
    if hasattr(wgp, "save_path") and wgp.save_path:
        if not os.path.exists(wgp.save_path):
            os.makedirs(wgp.save_path, exist_ok=True)
        print(f"Output will be saved to: {wgp.save_path}")
    else:
        # Fallback if wgp.save_path isn't set for some reason
        fallback_save_path = "outputs_script"
        os.makedirs(fallback_save_path, exist_ok=True)
        # Make it available to generate_video
        wgp.save_path = fallback_save_path
        print(f"wgp.save_path not found, using fallback: {fallback_save_path}")

    if hasattr(wgp, "args") and wgp.args.gpu and torch.cuda.is_available():
        torch.set_default_device(wgp.args.gpu)
        print(f"Set default torch device to: {wgp.args.gpu}")
    elif torch.cuda.is_available():
        # Default to cuda if available and not specified
        torch.set_default_device("cuda")
        print("Set default torch device to: cuda")
    else:
        print(
            "CUDA not available. This script will likely fail or run on CPU "
            "if the model supports it."
        )

    # Download prompt enhancer models if needed
    def download_prompt_enhancer_models():
        """Download Llama3_2 model for prompt enhancement if missing."""
        try:
            from huggingface_hub import hf_hub_download

            # Check if model files exist
            required_files = [
                "config.json",
                "generation_config.json",
                "Llama3_2_quanto_bf16_int8.safetensors",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ]

            missing_files = []
            for filename in required_files:
                file_path = os.path.join(LLM_ENHANCER_MODEL_DIR, filename)
                if not os.path.isfile(file_path):
                    missing_files.append(filename)

            if not missing_files:
                print("✅ All Llama3_2 model files already present")
                return True

            print(
                f"📥 Downloading {len(missing_files)} missing Llama3_2 model files..."
            )

            # Create directory if needed
            os.makedirs(LLM_ENHANCER_MODEL_DIR, exist_ok=True)

            # Download missing files
            repo_id = "DeepBeepMeep/LTX_Video"
            subfolder = "Llama3_2"

            for filename in missing_files:
                print(f"⬇️  Downloading {filename}...")
                try:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir="ckpts",
                        subfolder=subfolder,
                    )
                    print(f"✅ {filename} downloaded successfully")
                except Exception as e:
                    print(f"❌ Failed to download {filename}: {e}")
                    return False

            print("🎉 All Llama3_2 model files downloaded successfully!")
            return True

        except ImportError:
            print(
                "❌ huggingface_hub not available. Please install: pip install huggingface_hub"
            )
            return False
        except Exception as e:
            print(f"❌ Error downloading models: {e}")
            return False

    # Load prompt enhancer models if enabled
    loaded_enhancer_llm_model = None
    loaded_enhancer_llm_tokenizer = None

    if ENABLE_PROMPT_ENHANCER:
        # Download models if missing
        if not download_prompt_enhancer_models():
            print("Failed to download prompt enhancer models. Disabling enhancement.")
            ENABLE_PROMPT_ENHANCER = False

    if ENABLE_PROMPT_ENHANCER:
        print("Prompt enhancer enabled. Attempting to load LLM model and tokenizer...")
        try:
            # Load tokenizer
            loaded_enhancer_llm_tokenizer = AutoTokenizer.from_pretrained(
                LLM_ENHANCER_MODEL_DIR
            )
            print(f"Loaded LLM enhancer tokenizer from {LLM_ENHANCER_MODEL_DIR}")

            # Load model - try wgp.offload first, fallback to standard loading
            if hasattr(wgp, "offload") and hasattr(
                wgp.offload, "fast_load_transformers_model"
            ):
                print(
                    f"Loading LLM model using wgp.offload from {LLM_ENHANCER_MODEL_FILE}"
                )
                loaded_enhancer_llm_model = wgp.offload.fast_load_transformers_model(
                    LLM_ENHANCER_MODEL_FILE
                )

                # Move to correct device
                if hasattr(wgp, "args") and wgp.args.gpu and torch.cuda.is_available():
                    loaded_enhancer_llm_model = loaded_enhancer_llm_model.to(
                        wgp.args.gpu
                    )
                elif torch.cuda.is_available():
                    loaded_enhancer_llm_model = loaded_enhancer_llm_model.to("cuda")
                else:
                    loaded_enhancer_llm_model = loaded_enhancer_llm_model.to("cpu")
                print("Loaded LLM enhancer model using wgp.offload")
            else:
                print("WARN: wgp.offload.fast_load_transformers_model not available.")
                print(
                    "Standard loading for quantized models would require additional setup."
                )
                loaded_enhancer_llm_model = None

            if not loaded_enhancer_llm_model:
                print("LLM enhancer model could not be loaded. Skipping enhancement.")

        except Exception as e:
            print(
                f"Error loading LLM enhancer model/tokenizer: {e}. Skipping enhancement."
            )
            loaded_enhancer_llm_model = None
            loaded_enhancer_llm_tokenizer = None

    try:
        # Apply prompt enhancement if enabled and models are loaded
        if (
            ENABLE_PROMPT_ENHANCER
            and loaded_enhancer_llm_model
            and loaded_enhancer_llm_tokenizer
        ):
            original_prompt_text = video_params["prompt"]
            print(f"Original prompt: {original_prompt_text}")

            # Seed before enhancement (similar to wgp.py)
            current_seed = video_params.get("seed", -1)
            if hasattr(wgp, "seed_everything"):
                wgp.seed_everything(current_seed)
            else:
                # Fallback seeding
                seed_val = (
                    current_seed if current_seed != -1 else random.randint(0, 2**32 - 1)
                )
                torch.manual_seed(seed_val)
                print(f"Seeded with: {seed_val} (using fallback seeding)")

            # Call prompt enhancer (T2V mode - no images)
            enhanced_prompts_list = generate_cinematic_prompt(
                image_caption_model=None,
                image_caption_processor=None,
                prompt_enhancer_model=loaded_enhancer_llm_model,
                prompt_enhancer_tokenizer=loaded_enhancer_llm_tokenizer,
                prompt=original_prompt_text,
                images=None,  # T2V enhancement
                max_new_tokens=256,
            )

            if (
                enhanced_prompts_list
                and isinstance(enhanced_prompts_list, list)
                and len(enhanced_prompts_list) > 0
            ):
                enhanced_prompt_text = enhanced_prompts_list[0]
                video_params["prompt"] = enhanced_prompt_text
                script_task_item["prompt"] = "!enhanced!\n" + enhanced_prompt_text
                print(f"Enhanced prompt: {enhanced_prompt_text}")
            else:
                print(
                    "Prompt enhancement failed or returned empty result. Using original prompt."
                )
        elif ENABLE_PROMPT_ENHANCER:
            print(
                "Prompt enhancement enabled but models not loaded. Using original prompt."
            )

        print("Starting video generation...")
        # The `task` argument to generate_video is used for some UI updates
        # (thumbnails) and potentially for some logic within generate_video
        # if it expects certain fields.
        # The actual generation parameters are passed via **video_params.
        wgp.generate_video(
            task=script_task_item,
            send_cmd=script_send_cmd,
            state=script_state,
            **video_params,
        )
        print("Video generation script finished.")
        print(f"Check the '{wgp.save_path}' directory for the output video.")

    except Exception as e:
        print(f"An error occurred during video generation: {e}")
        import traceback

        traceback.print_exc()
