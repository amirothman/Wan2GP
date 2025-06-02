# Minimal LTX Video Generator

A stripped-down script to run LTX Video model directly without CLI/Gradio complexity.

## 🚀 Quick Start

```bash
python run_ltxv.py
```

That's it! The script will generate a video and save it to `output/output.mp4`.

## 📋 What It Does

- **Model**: Uses LTX Video 13B distilled model for fast generation (10 steps)
- **Prompt**: "A beautiful sunset over mountains, cinematic lighting, golden hour, dramatic clouds"
- **Output**: 5-second video (81 frames at 16fps) at 720x1280 resolution
- **Seed**: 42 (for reproducible results)

## 🔧 Requirements

### Model Files
The script expects these model files in the `ckpts/` directory:

```
ckpts/
├── ltxv_0.9.7_13B_distilled_bf16.safetensors    # Main transformer model
├── ltxv_0.9.7_VAE.safetensors                   # Video autoencoder
├── T5_xxl_1.1_enc_bf16.safetensors              # Text encoder
├── ltxv_scheduler.json                          # Scheduler config
├── ltxv_0.9.7_spatial_upscaler.safetensors      # Spatial upsampler
└── T5_xxl_1.1/                                  # Tokenizer directory
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── ...
```

### Python Dependencies
All dependencies from the main WanGP `requirements.txt` are needed:

```bash
pip install -r requirements.txt
```

### Hardware Requirements
- **GPU**: CUDA-compatible GPU with 8-12GB VRAM (recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: ~20GB for model files

## 🎛️ Customization

To modify the generation parameters, edit the hardcoded values in `run_ltxv.py`:

```python
# Generation Parameters
PROMPT = "Your custom prompt here"
NEGATIVE_PROMPT = "blurry, low quality, distorted, ugly"
OUTPUT_PATH = "output/my_video.mp4"
SEED = 42
HEIGHT = 720
WIDTH = 1280
NUM_FRAMES = 81  # 5 seconds at 16fps
FRAME_RATE = 16
SAMPLING_STEPS = 10  # Distilled model uses fewer steps
```

## 📊 Performance

**Expected generation time:**
- RTX 4090: ~1-2 minutes
- RTX 3080: ~2-3 minutes  
- RTX 2080 Ti: ~4-6 minutes

**VRAM usage:** ~8-12GB (similar to full WanGP)

## 🔍 Troubleshooting

### Missing Model Files
```
ERROR - Missing model files:
  - transformer: ckpts/ltxv_0.9.7_13B_distilled_bf16.safetensors
```
**Solution**: Download the required model files using the main WanGP application first.

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions**:
- Close other GPU applications
- Reduce `NUM_FRAMES` (e.g., to 41 for ~2.5 seconds)
- Use a smaller model variant (edit model paths)

### Generation Failed
```
ERROR - Generation failed: pipeline returned None
```
**Solutions**:
- Check model file integrity
- Verify CUDA installation
- Try reducing resolution or frame count

## 🆚 Differences from Full WanGP

| Feature | Full WanGP | run_ltxv.py |
|---------|------------|-------------|
| UI | Gradio web interface | Command line only |
| Models | Multiple (Wan, Hunyuan, LTX) | LTX Video only |
| Configuration | CLI args + web forms | Hardcoded parameters |
| LoRA Support | ✅ | ❌ |
| Prompt Enhancement | ✅ | ❌ |
| Queue System | ✅ | ❌ |
| Image Conditioning | ✅ | ❌ |
| Sliding Windows | ✅ | ❌ |
| Setup Complexity | High | Minimal |
| Generation Speed | Variable | Optimized (distilled) |

## 🎯 Use Cases

Perfect for:
- **Quick testing** of LTX Video model
- **Batch processing** with custom scripts
- **Integration** into other applications
- **Learning** the core generation pipeline
- **Debugging** without UI overhead

## 📝 Notes

- This script uses the **distilled** model configuration for faster generation
- Output quality is optimized for the 13B model size
- The script is designed to be **self-contained** and **minimal**
- For advanced features, use the full WanGP application

## 🔗 Related

- [Main WanGP Repository](https://github.com/deepbeepmeep/Wan2GP)
- [LTX Video Paper](https://arxiv.org/abs/2410.07954)
- [Discord Community](https://discord.gg/g7efUW9jGV)