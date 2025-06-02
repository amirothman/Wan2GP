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