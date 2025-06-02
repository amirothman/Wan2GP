#!/usr/bin/env python3
"""Setup Validation Script for run_ltxv.py
=======================================

This script checks if all requirements are met to run the minimal LTX Video generator.
"""

import os
import sys
from pathlib import Path


def check_python_packages():
    """Check if required Python packages are installed."""
    required_packages = [
        'torch',
        'torchvision',
        'transformers',
        'diffusers',
        'imageio',
        'numpy',
        'PIL',
        'yaml',
        'mmgp'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'yaml':
                import yaml
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} - MISSING")

    return missing_packages

def check_model_files():
    """Check if required model files exist."""
    model_paths = {
        "Transformer Model": "ckpts/ltxv_0.9.7_13B_distilled_bf16.safetensors",
        "VAE Model": "ckpts/ltxv_0.9.7_VAE.safetensors",
        "Text Encoder": "ckpts/T5_xxl_1.1_enc_bf16.safetensors",
        "Tokenizer": "ckpts/T5_xxl_1.1",
        "Scheduler": "ckpts/ltxv_scheduler.json",
        "Spatial Upsampler": "ckpts/ltxv_0.9.7_spatial_upscaler.safetensors",
        "Config File": "ltx_video/configs/ltxv-13b-0.9.7-distilled.yaml"
    }

    missing_files = []

    for name, path in model_paths.items():
        if os.path.exists(path):
            if os.path.isfile(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"✓ {name}: {path} ({size_mb:.1f} MB)")
            else:
                print(f"✓ {name}: {path} (directory)")
        else:
            missing_files.append((name, path))
            print(f"✗ {name}: {path} - MISSING")

    return missing_files

def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print("✓ CUDA Available")
            print(f"  GPU Count: {gpu_count}")
            print(f"  GPU 0: {gpu_name}")
            print(f"  GPU Memory: {gpu_memory:.1f} GB")

            if gpu_memory < 8:
                print(f"⚠️  Warning: GPU has only {gpu_memory:.1f}GB VRAM. 8GB+ recommended.")

            return True
        print("✗ CUDA Not Available - will use CPU (very slow)")
        return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def check_output_directory():
    """Check if output directory exists and is writable."""
    output_dir = Path("output")

    if output_dir.exists():
        if output_dir.is_dir():
            # Test write permissions
            test_file = output_dir / "test_write.tmp"
            try:
                test_file.write_text("test")
                test_file.unlink()
                print(f"✓ Output directory: {output_dir} (writable)")
                return True
            except PermissionError:
                print(f"✗ Output directory: {output_dir} (not writable)")
                return False
        else:
            print(f"✗ Output path exists but is not a directory: {output_dir}")
            return False
    else:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Output directory created: {output_dir}")
            return True
        except PermissionError:
            print(f"✗ Cannot create output directory: {output_dir}")
            return False

def main():
    """Main validation function."""
    print("=" * 60)
    print("🔍 LTX Video Setup Validation")
    print("=" * 60)

    all_good = True

    # Check Python packages
    print("\n📦 Checking Python packages...")
    missing_packages = check_python_packages()
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        all_good = False
    else:
        print("✅ All required packages installed")

    # Check CUDA
    print("\n🖥️  Checking CUDA...")
    cuda_available = check_cuda()

    # Check model files
    print("\n📁 Checking model files...")
    missing_files = check_model_files()
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} model files:")
        for name, path in missing_files:
            print(f"   - {name}: {path}")
        print("\nDownload models using the main WanGP application first.")
        all_good = False
    else:
        print("✅ All model files found")

    # Check output directory
    print("\n📂 Checking output directory...")
    output_ok = check_output_directory()
    if not output_ok:
        all_good = False

    # Final summary
    print("\n" + "=" * 60)
    if all_good:
        print("🎉 Setup validation PASSED!")
        print("You can now run: python run_ltxv.py")
        if not cuda_available:
            print("\n⚠️  Note: CUDA not available - generation will be very slow on CPU")
    else:
        print("❌ Setup validation FAILED!")
        print("Please fix the issues above before running run_ltxv.py")
    print("=" * 60)

    return 0 if all_good else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
