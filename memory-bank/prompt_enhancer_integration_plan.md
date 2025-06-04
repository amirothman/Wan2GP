# Plan: Integrating Prompt Enhancer in `create_sample_video.py`

This document outlines the plan to incorporate a text-to-video (T2V) prompt enhancer into the `create_sample_video.py` script, leveraging existing utilities and models found within the WanGP project.

## 1. Objective

To enhance the user-provided prompt in `create_sample_video.py` using an LLM to make it more descriptive and "cinematic" before passing it to the video generation model (`wgp.generate_video`).

## 2. Analysis of Existing Utilities

*   **`ltx_video/utils/prompt_enhance_utils.py`**:
    *   Provides `generate_cinematic_prompt()`, which can handle T2V and Image-to-Video (I2V) enhancement.
    *   Relies on an image captioning model (e.g., Florence2) for I2V and an LLM (e.g., Llama3) for the actual prompt rewriting.
    *   This utility is well-suited for the task.
*   **`hyvideo/prompt_rewrite.py`**:
    *   Formats prompts for an external "hunyuan-large rewrite model."
    *   Less self-contained and more specific to the Hunyuan model workflow.
    *   **Conclusion**: The LTX utility is preferred for `create_sample_video.py`.

## 3. Model Information (from `wgp.py` analysis)

*   **Prompt Enhancer LLM:**
    *   **Model:** Quantized Llama3
    *   **File:** `ckpts/Llama3_2/Llama3_2_quanto_bf16_int8.safetensors`
    *   **Tokenizer:** `ckpts/Llama3_2`
    *   **Loading in `wgp.py`:** Uses `offload.fast_load_transformers_model()` for the model and `AutoTokenizer.from_pretrained()` for the tokenizer.
*   **Image Captioning Model (for I2V, secondary for this T2V plan):**
    *   **Model:** Florence2
    *   **Path:** `ckpts/Florence2`
    *   **Loading in `wgp.py`:** Uses `AutoModelForCausalLM.from_pretrained()` and `AutoProcessor.from_pretrained()`.
*   **Control in `wgp.py`:**
    *   Enhancement is enabled via `server_config.get("enhancer_enabled", 0) == 1`.
    *   Global variables hold the loaded models: `prompt_enhancer_llm_model`, `prompt_enhancer_llm_tokenizer`, etc.

## 4. Integration Plan for `create_sample_video.py` (T2V Focus)

### 4.1. Control and Configuration

*   Add a global boolean variable `ENABLE_PROMPT_ENHANCER` at the top of `create_sample_video.py`.
*   Define model paths:
    ```python
    LLM_ENHANCER_MODEL_DIR = "ckpts/Llama3_2"
    LLM_ENHANCER_MODEL_FILE = "ckpts/Llama3_2/Llama3_2_quanto_bf16_int8.safetensors"
    ```

### 4.2. Model Loading

*   Conditionally (if `ENABLE_PROMPT_ENHANCER` is true) load the LLM enhancer and tokenizer:
    *   **Tokenizer:** Use `transformers.AutoTokenizer.from_pretrained(LLM_ENHANCER_MODEL_DIR)`.
    *   **Model:**
        1.  Attempt to use `wgp.offload.fast_load_transformers_model(LLM_ENHANCER_MODEL_FILE)` if `wgp.offload` is accessible. This is preferred for consistency.
        2.  If not available, a fallback to standard Hugging Face loading (e.g., `AutoModelForCausalLM.from_pretrained()`) would be needed. This might require ensuring `LLM_ENHANCER_MODEL_DIR` contains necessary config files for the `.safetensors` quantized model or specific loading instructions for such files. This is a potential complexity for true standalone operation.
    *   Move loaded models to the appropriate device (`wgp.args.gpu` or fallback to `cuda`/`cpu`).
    *   Handle errors gracefully if models cannot be loaded, and skip enhancement.

### 4.3. Enhancement Logic

*   If models are loaded successfully:
    1.  Import `generate_cinematic_prompt` from `ltx_video.utils.prompt_enhance_utils`.
    2.  Import `seed_everything` from `wan.utils.utils` (or use `wgp.seed_everything` if available, or a local fallback).
    3.  Get the original prompt from `video_params["prompt"]`.
    4.  Call `seed_everything()` with `video_params["seed"]`.
    5.  Call `generate_cinematic_prompt()`:
        ```python
        enhanced_prompts_list = generate_cinematic_prompt(
            image_caption_model=None,
            image_caption_processor=None,
            prompt_enhancer_model=loaded_llm_model,
            prompt_enhancer_tokenizer=loaded_llm_tokenizer,
            prompt=original_prompt_text,
            images=None, # For T2V
            max_new_tokens=256
        )
        ```
    6.  Update `video_params["prompt"]` with `enhanced_prompts_list[0]`.
    7.  Update `script_task_item["prompt"]` with `"!enhanced!\n" + enhanced_prompts_list[0]`.
    8.  Print original and enhanced prompts for logging.

### 4.4. Mermaid Diagram of Flow

```mermaid
graph TD
    A[Start: create_sample_video.py] --> B{Define Original Prompt in video_params};
    B --> B1{ENABLE_PROMPT_ENHANCER?};
    B1 -- Yes --> B2[Load Llama3 Enhancer LLM & Tokenizer];
    B2 -- Models Loaded --> C{Call generate_cinematic_prompt (T2V Mode)};
    C -- Enhanced Prompt --> D[Update video_params & script_task_item with Enhanced Prompt];
    D --> E[Log Original and Enhanced Prompts];
    B1 -- No --> E;
    B2 -- Load Fail/Not Available --> E;
    E --> F[Call wgp.generate_video];
    F --> G[End: Video Generated];

    subgraph ltx_video.utils.prompt_enhance_utils
        direction LR
        P1[Input: Original Prompt, Llama3 LLM, Tokenizer] --> P2{T2V_CINEMATIC_PROMPT Logic};
        P2 --> P3[Output: Enhanced Prompt List];
    end

    C --> P1;
```

## 5. Key Considerations / Challenges

*   **Standalone Model Loading:** The primary challenge for `create_sample_video.py` running independently is the loading of the `Llama3_2_quanto_bf16_int8.safetensors` model if `wgp.offload.fast_load_transformers_model` is not available or not desired as a direct dependency for this script. Standard Hugging Face methods for quantized `.safetensors` need to be confirmed.
*   **Dependencies:** `transformers`, `torch`. `wan.utils.utils.seed_everything` or an equivalent.
*   **Model Availability:** The script will depend on `ckpts/Llama3_2` being correctly populated.
*   **Error Handling:** Robust error handling for model loading and the enhancement process.

## 6. Next Steps (Post-Planning)

1.  Implement the changes in `create_sample_video.py` using `apply_diff`.
2.  Address the LLM loading mechanism carefully.
3.  Test thoroughly.
4.  Update Memory Bank `activeContext.md` and `progress.md`.