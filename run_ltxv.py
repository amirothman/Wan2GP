#!/usr/bin/env python3
"""Minimal LTX Video Generation Script
==================================

A stripped-down script to run LTX Video model directly without CLI/Gradio
complexity.
Uses the 13B dev model for high-quality generation (30 steps).

Usage:
    python run_ltxv.py

Output:
    Generates video at: output/output.mp4
"""

import logging
import os
import random
import sys
import time
from pathlib import Path

import imageio
import numpy as np
import torch
import yaml

# Import mmgp for model loading
from mmgp import offload
from transformers import T5Tokenizer

# Import for model downloading
try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    hf_hub_download = None
    snapshot_download = None

# Import LTX Video components
from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.autoencoders.latent_upsampler import LatentUpsampler
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.pipelines.pipeline_ltx_video import (
    LTXMultiScalePipeline,
    LTXVideoPipeline,
)
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

# ============================================================================
# HARDCODED CONFIGURATION
# ============================================================================

# Generation Parameters
PROMPT = "A beautiful sunset over mountains, cinematic lighting, golden hour, dramatic clouds"
NEGATIVE_PROMPT = "blurry, low quality, distorted, ugly"
OUTPUT_PATH = "output/output.mp4"
SEED = 42
HEIGHT = 720
WIDTH = 1280
NUM_FRAMES = 81  # 5 seconds at 16fps
FRAME_RATE = 16
SAMPLING_STEPS = 30  # Dev model uses 30 steps for better quality

# Model Paths (based on WanGP structure)
MODEL_PATHS = {
    "transformer": "ckpts/ltxv_0.9.7_13B_dev_bf16.safetensors",
    "vae": "ckpts/ltxv_0.9.7_VAE.safetensors",
    "text_encoder": "ckpts/T5_xxl_1.1/T5_xxl_1.1_enc_bf16.safetensors",
    "tokenizer": "ckpts/T5_xxl_1.1",
    "scheduler": "ckpts/ltxv_scheduler.json",
    "upsampler": "ckpts/ltxv_0.9.7_spatial_upscaler.safetensors",
    "config": "ltx_video/configs/ltxv-13b-0.9.7-dev.yaml",
}

# Model Configuration
DTYPE = torch.bfloat16
VAE_DTYPE = torch.bfloat16
MIXED_PRECISION = False

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def download_ltxv_models(logger):
    """Download required LTXV model files from HuggingFace."""
    if hf_hub_download is None or snapshot_download is None:
        logger.error("huggingface_hub is required for automatic model download.")
        logger.error("Please install it via: pip install huggingface_hub")
        return False

    logger.info("Downloading LTXV model files...")

    try:
        # Create ckpts directory if it doesn't exist
        os.makedirs("ckpts", exist_ok=True)

        # Download T5 tokenizer files
        logger.info("  Downloading T5 tokenizer...")
        tokenizer_files = [
            "added_tokens.json",
            "special_tokens_map.json",
            "spiece.model",
            "tokenizer_config.json",
        ]

        for file in tokenizer_files:
            if not os.path.exists(f"ckpts/T5_xxl_1.1/{file}"):
                hf_hub_download(
                    repo_id="DeepBeepMeep/LTX_Video",
                    filename=file,
                    local_dir="ckpts",
                    subfolder="T5_xxl_1.1",
                )

        # Download main model files
        logger.info("  Downloading main model files...")
        main_files = [
            "ltxv_0.9.7_VAE.safetensors",
            "ltxv_0.9.7_spatial_upscaler.safetensors",
            "ltxv_scheduler.json",
            "ltxv_0.9.7_13B_dev_bf16.safetensors",
            "T5_xxl_1.1/T5_xxl_1.1_enc_bf16.safetensors",
        ]

        for file in main_files:
            if not os.path.exists(f"ckpts/{file}"):
                hf_hub_download(
                    repo_id="DeepBeepMeep/LTX_Video", filename=file, local_dir="ckpts"
                )

        logger.info("✓ All model files downloaded successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to download models: {e}")
        return False


def check_model_files(logger):
    """Check if all required model files exist."""
    missing_files = []
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")

    if missing_files:
        logger.info("Missing model files detected. Attempting to download...")
        if download_ltxv_models(logger):
            # Re-check after download
            missing_files = []
            for name, path in MODEL_PATHS.items():
                if not os.path.exists(path):
                    missing_files.append(f"{name}: {path}")

            if not missing_files:
                logger.info("✓ All model files are now available")
                return True

        logger.error("Missing model files:")
        for file in missing_files:
            logger.error(f"  - {file}")
        logger.error(
            "\nPlease ensure all model files are downloaded to the ckpts/ directory."
        )
        logger.error(
            "Refer to the WanGP documentation for model download instructions."
        )
        return False

    logger.info("✓ All model files found")
    return True


def create_output_directory():
    """Create output directory if it doesn't exist."""
    output_dir = Path(OUTPUT_PATH).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def calculate_padding(
    source_height: int, source_width: int, target_height: int, target_width: int
):
    """Calculate padding needed to reach target dimensions."""
    pad_height = target_height - source_height
    pad_width = target_width - source_width

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    return (pad_left, pad_right, pad_top, pad_bottom)


# ============================================================================
# LTXV MODEL CLASS
# ============================================================================


class MinimalLTXV:
    """Simplified LTXV model class for direct video generation."""

    def __init__(self, logger):
        self.logger = logger
        self.device = get_device()
        self.logger.info(f"Using device: {self.device}")

        # Load pipeline configuration
        self.config = self._load_config()

        # Initialize model components
        self.pipeline = self._load_pipeline()

    def _load_config(self):
        """Load the dev model configuration."""
        self.logger.info("Loading pipeline configuration...")
        with open(MODEL_PATHS["config"]) as f:
            config = yaml.safe_load(f)

        # Override checkpoint_path to match our actual model file
        config['checkpoint_path'] = MODEL_PATHS['transformer'].replace('ckpts/', '')

        # DIAGNOSTIC: Log config details
        self.logger.info(
            f"  Config checkpoint_path: {config.get('checkpoint_path', 'NOT_FOUND')}"
        )
        self.logger.info(f"  Actual transformer path: {MODEL_PATHS['transformer']}")
        self.logger.info(
            f"  Config pipeline_type: {config.get('pipeline_type', 'NOT_FOUND')}"
        )

        return config

    def _load_pipeline(self):
        """Load and initialize the LTX Video pipeline."""
        self.logger.info("Loading model components...")

        # Load VAE
        self.logger.info("  Loading VAE...")
        vae = offload.fast_load_transformers_model(
            MODEL_PATHS["vae"], modelClass=CausalVideoAutoencoder
        )
        vae = vae.to(VAE_DTYPE)
        vae._model_dtype = VAE_DTYPE

        # Load Transformer
        self.logger.info("  Loading Transformer...")
        transformer = offload.fast_load_transformers_model(
            MODEL_PATHS["transformer"], modelClass=Transformer3DModel
        )
        transformer._model_dtype = DTYPE
        if MIXED_PRECISION:
            transformer._lock_dtype = torch.float

        # Load Text Encoder
        self.logger.info("  Loading Text Encoder...")
        text_encoder = offload.fast_load_transformers_model(MODEL_PATHS["text_encoder"])

        # Load other components
        self.logger.info("  Loading additional components...")
        patchifier = SymmetricPatchifier(patch_size=1)
        tokenizer = T5Tokenizer.from_pretrained(MODEL_PATHS["tokenizer"])
        scheduler = RectifiedFlowScheduler.from_pretrained(MODEL_PATHS["scheduler"])

        # Load spatial upsampler
        latent_upsampler = (
            LatentUpsampler.from_pretrained(MODEL_PATHS["upsampler"]).to("cpu").eval()
        )
        latent_upsampler.to(VAE_DTYPE)
        latent_upsampler._model_dtype = VAE_DTYPE

        # Create pipeline
        submodel_dict = {
            "transformer": transformer,
            "patchifier": patchifier,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
            "vae": vae,
            "prompt_enhancer_image_caption_model": None,
            "prompt_enhancer_image_caption_processor": None,
            "prompt_enhancer_llm_model": None,
            "prompt_enhancer_llm_tokenizer": None,
            "allowed_inference_steps": None,
        }

        pipeline = LTXVideoPipeline(**submodel_dict)
        pipeline = LTXMultiScalePipeline(pipeline, latent_upsampler=latent_upsampler)

        self.logger.info("✓ Pipeline loaded successfully")
        return pipeline

    def generate(self):
        """Generate video with hardcoded parameters."""
        self.logger.info("Starting video generation...")
        self.logger.info(f"  Prompt: {PROMPT}")
        self.logger.info(f"  Resolution: {WIDTH}x{HEIGHT}")
        self.logger.info(
            f"  Frames: {NUM_FRAMES} ({NUM_FRAMES / FRAME_RATE:.1f}s at {FRAME_RATE}fps)"
        )
        self.logger.info(f"  Steps: {SAMPLING_STEPS}")
        self.logger.info(f"  Seed: {SEED}")

        # Set random seed
        seed_everything(SEED)
        generator = torch.Generator(device=self.device).manual_seed(SEED)

        # Adjust dimensions to be divisible by 32 and frames to be (N * 8 + 1)
        height_padded = ((HEIGHT - 1) // 32 + 1) * 32
        width_padded = ((WIDTH - 1) // 32 + 1) * 32
        num_frames_padded = ((NUM_FRAMES - 2) // 8 + 1) * 8 + 1

        padding = calculate_padding(HEIGHT, WIDTH, height_padded, width_padded)

        self.logger.info(
            f"  Padded dimensions: {width_padded}x{height_padded}x{num_frames_padded}"
        )

        # Configure skip layer strategy
        stg_mode = self.config.get("stg_mode", "attention_values")
        if stg_mode.lower() in ["stg_av", "attention_values"]:
            skip_layer_strategy = SkipLayerStrategy.AttentionValues
        elif stg_mode.lower() in ["stg_as", "attention_skip"]:
            skip_layer_strategy = SkipLayerStrategy.AttentionSkip
        elif stg_mode.lower() in ["stg_r", "residual"]:
            skip_layer_strategy = SkipLayerStrategy.Residual
        elif stg_mode.lower() in ["stg_t", "transformer_block"]:
            skip_layer_strategy = SkipLayerStrategy.TransformerBlock
        else:
            skip_layer_strategy = SkipLayerStrategy.AttentionValues

        # Prepare input sample
        sample = {
            "prompt": PROMPT,
            "prompt_attention_mask": None,
            "negative_prompt": NEGATIVE_PROMPT,
            "negative_prompt_attention_mask": None,
        }

        # Generate video
        start_time = time.time()

        try:
            # DIAGNOSTIC: Log pipeline call parameters
            self.logger.info("  Pipeline call parameters:")
            self.logger.info(f"    num_inference_steps1: {SAMPLING_STEPS}")
            self.logger.info(f"    num_inference_steps2: {SAMPLING_STEPS}")
            self.logger.info(f"    skip_layer_strategy: {skip_layer_strategy}")
            self.logger.info(
                f"    height: {height_padded}, width: {width_padded}, frames: {num_frames_padded}"
            )

            images = self.pipeline(
                **self.config,
                num_inference_steps1=SAMPLING_STEPS,
                num_inference_steps2=SAMPLING_STEPS,
                skip_layer_strategy=skip_layer_strategy,
                generator=generator,
                output_type="pt",
                callback_on_step_end=None,
                height=height_padded,
                width=width_padded,
                num_frames=num_frames_padded,
                frame_rate=FRAME_RATE,
                **sample,
                media_items=None,
                strength=1.0,
                conditioning_items=None,
                is_video=True,
                vae_per_channel_normalize=True,
                image_cond_noise_scale=0.15,
                mixed_precision=self.config.get("mixed", MIXED_PRECISION),
                callback=None,
                VAE_tile_size=None,
                device=self.device,
            )

            if images is None:
                raise RuntimeError("Generation failed - pipeline returned None")

            generation_time = time.time() - start_time
            self.logger.info(f"✓ Generation completed in {generation_time:.1f}s")

            # Crop padded images to desired resolution and frame count
            (pad_left, pad_right, pad_top, pad_bottom) = padding
            pad_bottom = -pad_bottom if pad_bottom != 0 else images.shape[3]
            pad_right = -pad_right if pad_right != 0 else images.shape[4]

            images = images[:, :, :NUM_FRAMES, pad_top:pad_bottom, pad_left:pad_right]
            images = images.sub_(0.5).mul_(2).squeeze(0)

            return images

        except Exception as e:
            self.logger.error(f"Generation failed: {e!s}")
            raise


def save_video(frames, output_path, fps=16, logger=None):
    """Save video frames to file."""
    if logger:
        logger.info(f"Saving video to {output_path}...")

    # Convert tensor to numpy and scale to 0-255
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()

    # Ensure frames are in the right format: (T, H, W, C)
    if frames.ndim == 4 and frames.shape[0] == 3:  # (C, T, H, W)
        frames = frames.transpose(1, 2, 3, 0)  # (T, H, W, C)
    elif frames.ndim == 4 and frames.shape[1] == 3:  # (T, C, H, W)
        frames = frames.transpose(0, 2, 3, 1)  # (T, H, W, C)

    # Scale to 0-255 and convert to uint8
    frames = np.clip(frames * 255, 0, 255).astype(np.uint8)

    # Save video
    imageio.mimsave(output_path, frames, fps=fps, quality=8)

    if logger:
        logger.info("✓ Video saved successfully")
        logger.info(f"  Duration: {len(frames) / fps:.1f}s")
        logger.info(f"  Resolution: {frames.shape[2]}x{frames.shape[1]}")
        logger.info(f"  Frames: {len(frames)}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main execution function."""
    print("=" * 60)
    print("🎬 Minimal LTX Video Generator")
    print("=" * 60)

    # Setup logging
    logger = setup_logging()

    try:
        # Check prerequisites
        logger.info("Checking prerequisites...")

        # Check CUDA availability
        if not torch.cuda.is_available():
            logger.warning("CUDA not available - using CPU (will be very slow)")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")

        # Check model files
        if not check_model_files(logger):
            return 1

        # Create output directory
        output_dir = create_output_directory()
        logger.info(f"Output directory: {output_dir}")

        # Initialize model
        logger.info("Initializing LTX Video model...")
        model = MinimalLTXV(logger)

        # Generate video
        frames = model.generate()

        # Save video
        save_video(frames, OUTPUT_PATH, FRAME_RATE, logger)

        logger.info("🎉 Video generation completed successfully!")
        logger.info(f"📁 Output: {OUTPUT_PATH}")

        return 0

    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error: {e!s}")
        logger.error("Check the logs above for more details")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
