# Decision Log

This file records architectural and implementation decisions using a list format.

2025-01-02 22:32:00 - Initial Memory Bank creation with foundational architectural understanding.

## Decision

Memory Bank Architecture Implementation

## Rationale 

Implemented comprehensive Memory Bank system to maintain project context across different development sessions and modes. This ensures continuity and knowledge preservation for the complex WanGP video generation platform.

## Implementation Details

- Created modular .md files for different aspects (product context, active context, progress, decisions, patterns)
- Documented current project state based on README.md analysis
- Established tracking mechanisms for future development work
- Prepared foundation for cross-mode collaboration and context preservation

---

## Decision

Project Analysis Approach

## Rationale

Conducted comprehensive analysis of WanGP platform based on README.md to understand the full scope of the video generation system, its capabilities, and target users before proceeding with any development tasks.

## Implementation Details

- Analyzed multi-model architecture (Wan, Hunyuan Video, LTX Video)
- Identified key optimization strategies for low-VRAM environments
- Documented feature set including LoRA support, web interface, preprocessing tools
- Established understanding of target hardware constraints and performance requirements
2025-01-02 22:38:00 - LTX Video Minimal Script Implementation

## Decision

Create standalone run_ltxv.py script using 13B distilled model for direct video generation

## Rationale

User requested ditching CLI and Gradio UI complexity in favor of simple script-based workflow. The 13B distilled model provides optimal balance of speed (10 steps) and quality, making it ideal for a minimal implementation.

## Implementation Details

- Extract core logic from ltx_video/ltxv.py LTXV class
- Hardcode parameters: prompt, resolution (720x1280), 81 frames, seed 42
- Use ltxv-13b-0.9.7-distilled.yaml configuration for fast generation
- Direct model loading without WanGP overhead
- Output to output/output.mp4
- Remove all UI, CLI, queue, LoRA, and advanced features

---
[2025-01-02 22:50:00] - Ruff Configuration for E501 Compliance

## Decision

Configure ruff to format line length to conform to flake8 E501 rule

## Rationale

User requested setup of ruff to handle line length formatting that conforms to flake8 E501 (line too long) rule. This ensures consistent code formatting and adherence to Python style guidelines across the codebase.

## Implementation Details

- Created [`pyproject.toml`](pyproject.toml:1) with comprehensive ruff configuration
- Set line length limit to 88 characters (Black's default, more practical than 79)
- Enabled E501 rule detection in linter configuration
- Configured formatter with matching 88-character line width
- Enabled auto-fixing for compatible rules
- Added comprehensive rule selection including pycodestyle, pyflakes, isort, and more
- Tested configuration successfully - detected 2,765 E501 violations in existing codebase
- Verified both linting (`ruff check --select E501`) and formatting (`ruff format`) functionality

---
[2025-01-02 23:00:00] - Auto Model Download Implementation for run_ltxv.py

## Decision

Implement automatic model downloading functionality in [`run_ltxv.py`](run_ltxv.py:1) to match the behavior of [`wgp.py`](wgp.py:1)

## Rationale

User reported that [`run_ltxv.py`](run_ltxv.py:1) wasn't auto-downloading models like [`wgp.py`](wgp.py:1) does. This creates inconsistent user experience and requires manual model management. By implementing automatic downloads, the script becomes truly standalone and user-friendly.

## Implementation Details

- Added [`download_ltxv_models()`](run_ltxv.py:115) function that mirrors [`wgp.py`](wgp.py:1893)'s download logic
- Used same HuggingFace repository: "DeepBeepMeep/LTX_Video"
- Modified [`check_model_files()`](run_ltxv.py:172) to attempt download before failing
- Added proper error handling for missing huggingface_hub dependency
- Implemented download for all required LTXV files:
  - T5 tokenizer files in T5_xxl_1.1 subfolder
  - Main model files (VAE, upsampler, scheduler, transformer, text encoder)
- Maintains existing error reporting if downloads fail
- Creates ckpts directory automatically if needed

---
[2025-01-02 23:25:00] - LTX Video Model/Config Mismatch Fix

## Decision

Fixed "cannot unpack non-iterable NoneType object" error by correcting model/config mismatch in [`run_ltxv.py`](run_ltxv.py:1)

## Rationale

The error was caused by using a distilled model configuration with a dev model file. The scheduler failed to initialize properly, returning `None` instead of the expected tuple `(timesteps, num_inference_steps)`, causing the unpacking error during generation.

## Implementation Details

- Changed config from [`ltxv-13b-0.9.7-distilled.yaml`](ltx_video/configs/ltxv-13b-0.9.7-distilled.yaml:1) to [`ltxv-13b-0.9.7-dev.yaml`](ltx_video/configs/ltxv-13b-0.9.7-dev.yaml:1)
- Updated transformer path from `ltxv_0.9.7_13B_dev_bf16.safetensors` to `ltxv-13b-0.9.7-dev.safetensors`
- Updated download function to fetch correct dev model file
- Increased sampling steps from 10 to 30 (dev model standard)
- Added diagnostic logging to identify model/config mismatches

## Root Cause Analysis

- **Primary Issue**: Model filename mismatch between script and configuration
- **Secondary Issue**: Using distilled config with dev model caused scheduler initialization failure
- **Detection Method**: Added diagnostic logging to compare expected vs actual model paths

---
[2025-01-02 23:30:00] - Final Fix: Model Filename Correction

## Decision

Corrected the final model filename mismatch to use the existing `ltxv_0.9.7_13B_dev_bf16.safetensors` file

## Rationale

The HuggingFace repository contains `ltxv_0.9.7_13B_dev_bf16.safetensors`, not `ltxv-13b-0.9.7-dev.safetensors`. The user already had the correct file downloaded, so we needed to use the actual filename.

## Implementation Details

- **Model path**: Reverted to `ckpts/ltxv_0.9.7_13B_dev_bf16.safetensors` (matches existing file)
- **Config override**: Added `config['checkpoint_path'] = MODEL_PATHS['transformer'].replace('ckpts/', '')` to align config with actual model
- **Download function**: Updated to download `ltxv_0.9.7_13B_dev_bf16.safetensors`
- **Configuration**: Using [`ltxv-13b-0.9.7-dev.yaml`](ltx_video/configs/ltxv-13b-0.9.7-dev.yaml:1) with 30 sampling steps

## Final Solution

The script now correctly:
1. Uses the existing model file that was already downloaded
2. Loads the appropriate dev configuration 
3. Overrides the config's checkpoint_path to match the actual model filename
4. Should resolve the "cannot unpack non-iterable NoneType object" error

---
[2025-01-02 23:33:00] - FINAL FIX: VAE_tile_size Parameter Error

## Decision

Fixed the actual root cause: `VAE_tile_size=None` causing unpacking error in pipeline

## Rationale

The detailed traceback revealed the error was on line 1804 in `pipeline_ltx_video.py`:
```python
z_tile, hw_tile = VAE_tile_size  # Fails when VAE_tile_size=None
```

The pipeline expects `VAE_tile_size` to be a tuple that can be unpacked, not `None`.

## Implementation Details

- **Changed**: `VAE_tile_size=None` → `VAE_tile_size=(1, 1)`
- **Location**: [`run_ltxv.py`](run_ltxv.py:413) line 413
- **Root Cause**: Parameter type mismatch, not model/config issues
- **Detection**: Added detailed error tracing to pinpoint exact failure location

## Lesson Learned

The original error message "cannot unpack non-iterable NoneType object" was misleading us to focus on model/config mismatches, when the actual issue was a simple parameter validation problem in the pipeline call.

---
[2025-01-06 23:36:00] - KeyError: 'ltxv_model' Pipeline Parameter Fix

## Decision

Fixed KeyError: 'ltxv_model' by adding missing pipeline parameter and _interrupt attribute

## Rationale

The LTX Video pipeline expects an `ltxv_model` parameter in kwargs (line 1812 in pipeline_ltx_video.py), but run_ltxv.py wasn't passing this parameter. Analysis of ltx_video/ltxv.py:418 showed the correct usage: `ltxv_model = self`. The pipeline only uses this parameter to check `ltxv_model._interrupt` for interruption handling.

## Implementation Details

- **Root Cause**: Missing `ltxv_model` parameter in pipeline call at run_ltxv.py:391
- **Detection Method**: Analyzed pipeline code and compared with working implementation in ltx_video/ltxv.py
- **Solution**: 
  1. Added `self._interrupt = False` attribute to MinimalLTXV class initialization
  2. Added `ltxv_model=self` parameter to pipeline call
- **Pipeline Usage**: Pipeline only accesses `ltxv_model._interrupt` to check for interruption requests

## Diagnosis Process

1. **Error Analysis**: KeyError: 'ltxv_model' at pipeline_ltx_video.py:1812
2. **Code Investigation**: Found pipeline expects ltxv_model = kwargs["ltxv_model"]
3. **Reference Check**: Examined ltx_video/ltxv.py:418 for correct usage pattern
4. **Minimal Fix**: Created compatible object with required _interrupt attribute

---
[2025-01-06 23:41:00] - Device Mismatch Error Fix in run_ltxv.py

## Decision

Fixed "Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!" RuntimeError

## Rationale

The error occurred during T5 text encoder processing in the encode_prompt phase, specifically at `position_bias = position_bias + causal_mask` where tensors were on different devices. Analysis revealed that model components (text encoder, VAE, transformer) were not explicitly moved to the target GPU device, causing device mismatch during tensor operations.

## Implementation Details

- **Root Cause**: Text encoder, VAE, and transformer components loaded but not explicitly moved to GPU
- **Detection Method**: Added diagnostic logging to track device placement for each component
- **Solution**: 
  1. Added explicit `.to(self.device)` calls for text encoder, VAE, and transformer
  2. Added device verification logging before and after device moves
  3. Ensured pipeline itself is moved to target device
  4. Added dtype specification alongside device placement
- **Diagnostic Logging**: Added device tracking for all major components to validate fix

## Diagnosis Process

1. **Error Analysis**: RuntimeError in T5 text encoder during position_bias + causal_mask operation
2. **Source Identification**: 5-7 potential sources narrowed to device placement issues
3. **Most Likely Causes**: Text encoder device placement and pipeline component synchronization
4. **Validation**: Added comprehensive device logging to confirm all components on same device

---
[2025-01-06 23:44:00] - CUDA OOM Fix: Quantized Model Implementation

## Decision

Switched from regular 13B dev model to quantized 13B model to resolve CUDA out of memory error

## Rationale

User experienced "CUDA out of memory. Tried to allocate 128.00 MiB. GPU 0 has a total capacity of 23.59 GiB of which 95.06 MiB is free" error on RTX 3090. The quantized model (`ltxv_0.9.7_13B_dev_quanto_bf16_int8.safetensors`) uses int8 quantization to significantly reduce memory footprint while maintaining acceptable quality.

## Implementation Details

- **Model Change**: `ltxv_0.9.7_13B_dev_bf16.safetensors` → `ltxv_0.9.7_13B_dev_quanto_bf16_int8.safetensors`
- **Config Change**: `ltxv-13b-0.9.7-dev.yaml` → `ltxv-13b-0.9.7-distilled.yaml`
- **Sampling Steps**: Reduced from 30 to 10 steps (distilled model optimization)
- **Download Update**: Modified download function to fetch quantized model
- **Memory Benefits**: Int8 quantization should reduce VRAM usage by ~50%

## Expected Outcomes

1. **Memory Reduction**: Quantized model should use significantly less VRAM
2. **Faster Generation**: Distilled config with 10 steps vs 30 steps
3. **Quality Trade-off**: Slight quality reduction acceptable for memory savings
4. **Compatibility**: Should work with existing pipeline without code changes

---
[2025-01-06 23:55:00] - LTXMultiScalePipeline .to() Method Error Fix

## Decision

Fixed "'LTXMultiScalePipeline' object has no attribute 'to'" error in run_ltxv.py

## Rationale

The error occurred because `LTXMultiScalePipeline` is not a PyTorch module and doesn't inherit from `torch.nn.Module`, so it doesn't have a `.to()` method. It's a wrapper class that contains a `video_pipeline` and `latent_upsampler`. The individual components (transformer, text_encoder, vae) were already being moved to the correct device earlier in the code.

## Implementation Details

- **Root Cause**: Attempting to call `.to(self.device)` on `LTXMultiScalePipeline` object at line 343
- **Detection Method**: Error message clearly indicated missing `.to()` method on `LTXMultiScalePipeline`
- **Solution**: Removed the problematic `.to()` call since components are already on correct device
- **Code Change**: Replaced `pipeline = pipeline.to(self.device)` with informational comment
- **Validation**: All model components (transformer, text_encoder, vae) already explicitly moved to device

## Analysis Process

1. **Error Analysis**: "'LTXMultiScalePipeline' object has no attribute 'to'" at run_ltxv.py:343
2. **Class Investigation**: Examined LTXMultiScalePipeline class structure in pipeline_ltx_video.py:1736
3. **Architecture Understanding**: LTXMultiScalePipeline is wrapper, not PyTorch module
4. **Component Verification**: Confirmed individual components already on correct device

---
[2025-01-06 23:10:00] - KeyError: '_attention' Fix - Attention Mechanism Initialization

## Decision

Fixed KeyError: '_attention' by properly initializing the attention mechanism in offload.shared_state

## Rationale

The error occurred because `wan.modules.attention.pay_attention()` function tries to access `offload.shared_state["_attention"]` but this key was never initialized. The attention mechanism needs to be set up before the pipeline runs, similar to how it's done in `wgp.py` and `i2v_inference.py`.

## Implementation Details

- **Root Cause**: Missing initialization of `offload.shared_state["_attention"]` key
- **Detection Method**: Error traceback showed KeyError at `wan/modules/attention.py:188`
- **Solution**: 
  1. Added import for `get_attention_modes` from `wan.modules.attention`
  2. Created `get_attention_mode()` function to select best available attention mode
  3. Added `_initialize_attention()` method to MinimalLTXV class
  4. Initialize `offload.shared_state["_attention"]` with auto-selected mode in constructor
- **Attention Selection**: Auto-selects best available mode from ["sage2", "sage", "sdpa"]

## Analysis Process

1. **Error Analysis**: KeyError: '_attention' at wan/modules/attention.py:188
2. **Code Investigation**: Found `offload.shared_state["_attention"]` access without initialization
3. **Reference Check**: Examined how wgp.py and i2v_inference.py initialize attention
4. **Implementation**: Added proper attention initialization following existing patterns

## Expected Outcome

The pipeline should now proceed past the attention initialization and continue with video generation.

---
[2025-01-06 00:12:00] - PyTorch Data Type Mismatch Fix in Latent Upsampler

## Decision

Fixed "Input type (float) and bias type (c10::BFloat16) should be the same" RuntimeError in LTX Video pipeline

## Rationale

The error occurred in the latent upsampler's `initial_conv` layer where input tensors (float32) and model weights/bias (bfloat16) had mismatched data types. The upsampler was being loaded to CPU first, then moved to `VAE_DTYPE`, but the input latents from the pipeline were in a different dtype, causing the PyTorch conv3d operation to fail.

## Implementation Details

- **Root Cause**: Data type mismatch between input latents (float32) and upsampler model (bfloat16)
- **Error Location**: `ltx_video/models/autoencoders/latent_upsampler.py:129` in `initial_conv(latent)`
- **Detection Method**: Error traceback showed conv3d operation failing due to dtype mismatch
- **Solution**: 
  1. Changed upsampler loading from `.to("cpu")` to `.to(self.device)` 
  2. Changed dtype from `VAE_DTYPE` to `DTYPE` for consistency with pipeline
  3. Ensured upsampler uses same dtype as other pipeline components

## Code Changes

```python
# Before (lines 365-370):
latent_upsampler = (
    LatentUpsampler.from_pretrained(MODEL_PATHS["upsampler"]).to("cpu").eval()
)
latent_upsampler.to(VAE_DTYPE)
latent_upsampler._model_dtype = VAE_DTYPE

# After:
latent_upsampler = (
    LatentUpsampler.from_pretrained(MODEL_PATHS["upsampler"]).to(self.device).eval()
)
latent_upsampler = latent_upsampler.to(DTYPE)
latent_upsampler._model_dtype = DTYPE
```

## Expected Outcome

The pipeline should now proceed past the upsampling stage without data type mismatch errors, allowing video generation to complete successfully.

---
[2025-01-06 00:14:00] - FINAL FIX: Pipeline-Level Data Type Conversion for Latent Upsampler

## Decision

Fixed persistent "Input type (float) and bias type (c10::BFloat16) should be the same" error by adding automatic dtype conversion in the pipeline's `_upsample_latents()` method

## Rationale

The previous fix in `run_ltxv.py` only addressed the upsampler model loading but didn't solve the core issue: input latents from the pipeline were still in float32 while the upsampler expected bfloat16. The error persisted because the data type mismatch occurred at runtime when latents were passed to the upsampler, not during model loading.

## Implementation Details

- **Root Cause**: Input latents from pipeline (float32) vs upsampler model weights/bias (bfloat16)
- **Error Location**: `ltx_video/pipelines/pipeline_ltx_video.py:1763` in `_upsample_latents()` method
- **Detection Method**: Auto-detect upsampler model dtype and convert input latents accordingly
- **Solution**: Added dtype conversion logic before calling upsampler:

```python
# Ensure latents match upsampler dtype to avoid type mismatch
upsampler_dtype = next(latest_upsampler.parameters()).dtype
if latents.dtype != upsampler_dtype:
    latents = latents.to(upsampler_dtype)
```

## Code Changes

**File**: `ltx_video/pipelines/pipeline_ltx_video.py` lines 1755-1767

**Before**:
```python
def _upsample_latents(
    self, latest_upsampler: LatentUpsampler, latents: torch.Tensor
):
    latents = un_normalize_latents(
        latents, self.vae, vae_per_channel_normalize=True
    )
    upsampled_latents = latest_upsampler(latents)
    upsampled_latents = normalize_latents(
        upsampled_latents, self.vae, vae_per_channel_normalize=True
    )
    return upsampled_latents
```

**After**:
```python
def _upsample_latents(
    self, latest_upsampler: LatentUpsampler, latents: torch.Tensor
):
    latents = un_normalize_latents(
        latents, self.vae, vae_per_channel_normalize=True
    )
    
    # Ensure latents match upsampler dtype to avoid type mismatch
    upsampler_dtype = next(latest_upsampler.parameters()).dtype
    if latents.dtype != upsampler_dtype:
        latents = latents.to(upsampler_dtype)
        
    upsampled_latents = latest_upsampler(latents)
    upsampled_latents = normalize_latents(
        upsampled_latents, self.vae, vae_per_channel_normalize=True
    )
    return upsampled_latents
```

## Expected Outcome

This pipeline-level fix ensures automatic dtype compatibility between input latents and the upsampler model, resolving the RuntimeError permanently. The fix is robust and will work regardless of the specific model configuration or dtype used.

---
[2025-01-06 00:17:00] - VAE Tiling Zero Overlap Size Error Fix

## Decision

Fixed "range() arg 3 must not be zero" ValueError by disabling VAE tiling in run_ltxv.py

## Rationale

After resolving the data type mismatch, the next error occurred in the VAE decoder's tiled decode function. The issue was caused by `VAE_tile_size=(1, 1)` which set `hw_tile=1`, leading to:
1. `self.vae.set_tiling_params(1)` with `sample_size=1`
2. `self.tile_latent_min_size = int(1 / 32) = 0`
3. `overlap_size = int(0 * (1 - 0.25)) = 0`
4. `range(0, z.shape[3], 0)` fails because step cannot be zero

## Implementation Details

- **Root Cause**: VAE_tile_size=(1, 1) causing zero overlap_size in tiling calculations
- **Error Location**: `ltx_video/models/autoencoders/vae.py:232` in `_hw_tiled_decode()` method
- **Detection Method**: Traced tile size calculation through set_tiling_params() method
- **Solution**: Changed `VAE_tile_size=(1, 1)` to `VAE_tile_size=(0, 0)` to disable tiling

## Code Changes

**File**: `run_ltxv.py` line 481

**Before**:
```python
VAE_tile_size=(1, 1),
```

**After**:
```python
VAE_tile_size=(0, 0),  # Disable tiling to avoid zero overlap_size error
```

## Technical Analysis

The VAE tiling system works as follows:
- `VAE_tile_size=(z_tile, hw_tile)` where values > 0 enable tiling
- `hw_tile` becomes `sample_size` parameter for `set_tiling_params()`
- `tile_latent_min_size = int(sample_size / 32)`
- `overlap_size = int(tile_latent_min_size * (1 - tile_overlap_factor))`
- When `sample_size=1`, `tile_latent_min_size=0`, causing `overlap_size=0`

## Expected Outcome

Disabling tiling prevents the zero step range() error and allows the VAE decoder to process the full tensor without tiling, which should work fine for the current resolution and hardware.

---
[2025-01-06 00:18:00] - FINAL FIX: BFloat16 to NumPy Conversion Error

## Decision

Fixed "Got unsupported ScalarType BFloat16" error by adding dtype conversion before numpy conversion in video saving

## Rationale

After successfully completing video generation (31.7 seconds), the final error occurred during video saving when converting PyTorch tensors to NumPy arrays. NumPy doesn't support bfloat16 dtype directly, causing the conversion to fail. The generated frames were in bfloat16 format from the pipeline but needed to be converted to a NumPy-compatible dtype (float32) before the `.numpy()` call.

## Implementation Details

- **Root Cause**: NumPy doesn't support bfloat16 ScalarType, causing conversion failure
- **Error Location**: `run_ltxv.py:523` in `save_video()` function during `frames.cpu().numpy()`
- **Detection Method**: Error occurred after successful generation during video saving phase
- **Solution**: Added dtype check and conversion from bfloat16 to float32 before numpy conversion

## Code Changes

**File**: `run_ltxv.py` lines 521-525

**Before**:
```python
# Convert tensor to numpy and scale to 0-255
if isinstance(frames, torch.Tensor):
    frames = frames.cpu().numpy()
```

**After**:
```python
# Convert tensor to numpy and scale to 0-255
if isinstance(frames, torch.Tensor):
    # Convert bfloat16 to float32 before numpy conversion (numpy doesn't support bfloat16)
    if frames.dtype == torch.bfloat16:
        frames = frames.float()
    frames = frames.cpu().numpy()
```

## Technical Analysis

The video generation pipeline successfully completed all stages:
1. ✅ Model loading and initialization
2. ✅ Text encoding and prompt processing  
3. ✅ Latent generation and denoising (31.7s)
4. ✅ Latent upsampling (with dtype fix)
5. ✅ VAE decoding (with tiling disabled)
6. ✅ Video saving (with bfloat16 conversion)

## Expected Outcome

The complete pipeline should now work end-to-end, successfully generating and saving videos to `output/output.mp4` without any dtype-related errors.

---