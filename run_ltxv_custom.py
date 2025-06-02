#!/usr/bin/env python3
"""
Customizable LTX Video Generation Script
========================================

This is an example of how to customize the minimal LTX Video generator
with different prompts, resolutions, and parameters.

Usage:
    python run_ltxv_custom.py
"""

# Import the main script components
from run_ltxv import MinimalLTXV, save_video, setup_logging, get_device, seed_everything, check_model_files, create_output_directory
import torch
import time
import sys

# ============================================================================
# CUSTOM CONFIGURATION EXAMPLES
# ============================================================================

# Example 1: Cinematic landscape
CONFIG_LANDSCAPE = {
    "prompt": "Epic mountain landscape at golden hour, cinematic wide shot, dramatic lighting, 4k quality",
    "negative_prompt": "blurry, low quality, distorted, ugly, text, watermark",
    "output_path": "output/landscape.mp4",
    "seed": 123,
    "height": 720,
    "width": 1280,
    "num_frames": 81,  # 5 seconds
    "frame_rate": 16,
    "sampling_steps": 10
}

# Example 2: Portrait/vertical format
CONFIG_PORTRAIT = {
    "prompt": "Beautiful woman walking in a garden, soft lighting, portrait orientation, elegant movement",
    "negative_prompt": "blurry, low quality, distorted, ugly, deformed",
    "output_path": "output/portrait.mp4",
    "seed": 456,
    "height": 1280,
    "width": 720,
    "num_frames": 65,  # ~4 seconds
    "frame_rate": 16,
    "sampling_steps": 10
}

# Example 3: Short clip
CONFIG_SHORT = {
    "prompt": "Cat playing with a ball of yarn, cute and playful, high quality",
    "negative_prompt": "blurry, low quality, distorted",
    "output_path": "output/cat_short.mp4",
    "seed": 789,
    "height": 512,
    "width": 768,
    "num_frames": 41,  # ~2.5 seconds
    "frame_rate": 16,
    "sampling_steps": 8  # Even faster
}

# Example 4: Abstract/artistic
CONFIG_ABSTRACT = {
    "prompt": "Abstract flowing colors, liquid motion, artistic, vibrant, mesmerizing patterns",
    "negative_prompt": "realistic, photographic, blurry, low quality",
    "output_path": "output/abstract.mp4",
    "seed": 999,
    "height": 768,
    "width": 768,  # Square format
    "num_frames": 97,  # ~6 seconds
    "frame_rate": 16,
    "sampling_steps": 12
}

# ============================================================================
# CUSTOM GENERATION FUNCTION
# ============================================================================

def generate_custom_video(config, logger):
    """Generate a video with custom configuration."""
    logger.info(f"Generating: {config['output_path']}")
    logger.info(f"Prompt: {config['prompt']}")
    logger.info(f"Resolution: {config['width']}x{config['height']}")
    logger.info(f"Frames: {config['num_frames']} ({config['num_frames']/config['frame_rate']:.1f}s)")
    
    # Temporarily modify the global constants in run_ltxv
    import run_ltxv
    original_values = {}
    
    # Save original values
    for key in config:
        if hasattr(run_ltxv, key.upper()):
            original_values[key] = getattr(run_ltxv, key.upper())
            setattr(run_ltxv, key.upper(), config[key])
    
    try:
        # Initialize model (reuse if already loaded)
        if not hasattr(generate_custom_video, 'model'):
            generate_custom_video.model = MinimalLTXV(logger)
        
        model = generate_custom_video.model
        
        # Set random seed
        seed_everything(config['seed'])
        device = get_device()
        generator = torch.Generator(device=device).manual_seed(config['seed'])
        
        # Calculate padded dimensions
        height_padded = ((config['height'] - 1) // 32 + 1) * 32
        width_padded = ((config['width'] - 1) // 32 + 1) * 32
        num_frames_padded = ((config['num_frames'] - 2) // 8 + 1) * 8 + 1
        
        # Calculate padding
        pad_height = height_padded - config['height']
        pad_width = width_padded - config['width']
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        
        # Prepare sample
        sample = {
            "prompt": config['prompt'],
            "prompt_attention_mask": None,
            "negative_prompt": config['negative_prompt'],
            "negative_prompt_attention_mask": None,
        }
        
        # Generate
        start_time = time.time()
        
        images = model.pipeline(
            **model.config,
            num_inference_steps1=config['sampling_steps'],
            num_inference_steps2=config['sampling_steps'],
            skip_layer_strategy=model.pipeline.pipeline.skip_layer_strategy if hasattr(model.pipeline.pipeline, 'skip_layer_strategy') else None,
            generator=generator,
            output_type="pt",
            height=height_padded,
            width=width_padded,
            num_frames=num_frames_padded,
            frame_rate=config['frame_rate'],
            **sample,
            media_items=None,
            strength=1.0,
            conditioning_items=None,
            is_video=True,
            vae_per_channel_normalize=True,
            image_cond_noise_scale=0.15,
            mixed_precision=model.config.get("mixed", False),
            device=device,
        )
        
        if images is None:
            raise RuntimeError("Generation failed - pipeline returned None")
        
        generation_time = time.time() - start_time
        logger.info(f"Generation completed in {generation_time:.1f}s")
        
        # Crop to desired dimensions
        (pad_left, pad_right, pad_top, pad_bottom) = padding
        pad_bottom = -pad_bottom if pad_bottom != 0 else images.shape[3]
        pad_right = -pad_right if pad_right != 0 else images.shape[4]
        
        images = images[:, :, :config['num_frames'], pad_top:pad_bottom, pad_left:pad_right]
        images = images.sub_(0.5).mul_(2).squeeze(0)
        
        # Save video
        save_video(images, config['output_path'], config['frame_rate'], logger)
        
        return True
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        return False
    
    finally:
        # Restore original values
        for key, value in original_values.items():
            setattr(run_ltxv, key.upper(), value)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 60)
    print("🎬 Custom LTX Video Generator")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Check prerequisites
        logger.info("Checking prerequisites...")
        
        if not check_model_files(logger):
            return 1
        
        create_output_directory()
        
        # Available configurations
        configs = {
            "1": ("Cinematic Landscape", CONFIG_LANDSCAPE),
            "2": ("Portrait Format", CONFIG_PORTRAIT), 
            "3": ("Short Clip", CONFIG_SHORT),
            "4": ("Abstract Art", CONFIG_ABSTRACT),
            "5": ("Generate All", None)
        }
        
        # Show menu
        print("\nAvailable configurations:")
        for key, (name, _) in configs.items():
            print(f"  {key}. {name}")
        
        # Get user choice
        choice = input("\nSelect configuration (1-5): ").strip()
        
        if choice not in configs:
            logger.error("Invalid choice")
            return 1
        
        if choice == "5":
            # Generate all
            success_count = 0
            for key, (name, config) in configs.items():
                if config is None:  # Skip "Generate All" option
                    continue
                logger.info(f"\n{'='*40}")
                logger.info(f"Generating {name}...")
                logger.info(f"{'='*40}")
                if generate_custom_video(config, logger):
                    success_count += 1
                    logger.info(f"✓ {name} completed")
                else:
                    logger.error(f"✗ {name} failed")
            
            logger.info(f"\n🎉 Generated {success_count}/{len(configs)-1} videos successfully!")
        else:
            # Generate single video
            name, config = configs[choice]
            logger.info(f"Generating {name}...")
            if generate_custom_video(config, logger):
                logger.info(f"🎉 {name} generated successfully!")
            else:
                logger.error(f"❌ {name} generation failed")
                return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)