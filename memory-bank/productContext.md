# Product Context

This file provides a high-level overview of the WanGP project and the expected product that will be created. Based on README.md and project structure analysis.

2025-01-02 22:31:00 - Initial Memory Bank creation based on README.md analysis.

## Project Goal

WanGP (Wan2GP) by DeepBeepMeep is a comprehensive video generation platform designed to make advanced AI video models accessible to users with limited GPU resources ("GPU Poor"). The platform supports multiple state-of-the-art video generation models with optimizations for low VRAM requirements (as low as 6GB).

## Key Features

- **Multi-Model Support**: Wan models (1.3B, 14B), Hunyuan Video, LTX Video
- **Low VRAM Optimization**: Runs on GPUs with as little as 6GB VRAM
- **Legacy GPU Support**: Compatible with RTX 10XX, 20XX series
- **Web Interface**: Full Gradio-based web interface for easy use
- **Auto Model Download**: Automatic download of models adapted to specific architecture
- **LoRA Support**: Customization through LoRA models for each supported model
- **Advanced Tools**: Integrated mask editor, prompt enhancer, temporal/spatial generation
- **Queuing System**: Batch video generation with different parameters
- **Control Features**: VACE ControlNet for video-to-video, inpainting, outpainting
- **Preprocessing Tools**: Integrated Matanyone, depth estimation, pose detection
- **Performance Optimizations**: Sage attention, TeaCache, PyTorch compilation

## Overall Architecture

- **Core Engine**: Python-based with modular design
- **Web Interface**: Gradio server (default port 7860)
- **Model Modules**: 
  - `wan/` - Core Wan model implementations
  - `hyvideo/` - Hunyuan Video integration
  - `ltx_video/` - LTX Video support
- **Preprocessing**: `preprocessing/` with specialized tools (Matanyone, MiDaS, DWPose)
- **Utilities**: Frame interpolation (RIFE), various optimization tools
- **Configuration**: Flexible profile system for different hardware configurations
- **Entry Point**: `wgp.py` - main application launcher

## Target Architecture

The system is designed as a unified platform that abstracts the complexity of different video generation models behind a consistent web interface, with intelligent resource management to maximize accessibility across different hardware configurations.