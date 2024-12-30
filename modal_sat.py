import modal
import os
from pathlib import Path

# Define the Modal app
app = modal.App("sat-medical-segmentation-nano")

# Create persistent volume for model weights
volume = modal.Volume.from_name("sat-model-weights", create_if_missing=True)

# Create secret for HF token
secret = modal.Secret.from_dict({"HF_TOKEN": "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"})

# Create a custom image with all dependencies
image = (
    modal.Image.from_registry("pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel")
    .apt_install("git", "wget")
    .pip_install(
        "numpy>=1.24.0",
        "monai==0.9.1",
        "transformers>=4.21.3",
        "nibabel>=4.0.2",
        "einops>=0.6.0",
        "positional_encodings>=6.0.1",
        "huggingface_hub",
        "pandas",
        "scipy",
        "openpyxl"
    )
    .run_commands(
        # Clone SAT repository
        "git clone https://github.com/zhaoziheng/SAT.git /root/SAT",
        # Install dynamic-network-architectures
        "cd /root/SAT/model && pip install -e dynamic-network-architectures-main",
        # Install mamba-ssm with CUDA support
        "pip install mamba-ssm --no-build-isolation"
    )
)

@app.function(
    image=image,
    volumes={"/root/checkpoints": volume},
    secrets=[secret]
)
def download_models():
    """CPU-only function to download model weights."""
    from huggingface_hub import hf_hub_download
    import os
    
    results = {
        "success": True,
        "messages": []
    }
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs("/root/checkpoints", exist_ok=True)
    
    # Get token from environment
    hf_token = os.environ["HF_TOKEN"]
    
    # Download models from HuggingFace - Updated for Nano
    models = {
        "nano.pth": "Nano/nano.pth",
        "nano_text_encoder.pth": "Nano/nano_text_encoder.pth"
    }
    
    for local_name, hf_path in models.items():
        if not os.path.exists(f"/root/checkpoints/{local_name}"):
            try:
                results["messages"].append(f"Downloading {local_name}...")
                file_path = hf_hub_download(
                    repo_id="zzh99/SAT",
                    filename=hf_path,
                    local_dir="/root/checkpoints",
                    token=hf_token
                )
                results["messages"].append(f"Downloaded to: {file_path}")
                
                # Create target directory if needed
                target_dir = os.path.dirname(f"/root/checkpoints/{local_name}")
                os.makedirs(target_dir, exist_ok=True)
                
                # Move file to correct location
                target_path = f"/root/checkpoints/{local_name}"
                os.rename(file_path, target_path)
                results["messages"].append(f"Moved to: {target_path}")
                
            except Exception as e:
                results["success"] = False
                results["messages"].append(f"Error downloading {local_name}: {str(e)}")
    
    results["messages"].append("Model download complete!")
    return results

@app.function(
    image=image,
    gpu="T4",
    volumes={"/root/checkpoints": volume},
    mounts=[
        modal.Mount.from_local_dir(
            "data",
            remote_path="/root/SAT/data"
        ),
        modal.Mount.from_local_dir(
            "results",
            remote_path="/root/SAT/results"
        )
    ],
    timeout=3600,
)
async def run_sat_inference(
    input_jsonl: str = "/root/SAT/data/inference_demo/demo.jsonl",
    output_dir: str = "results",
    model_type: str = "nano",
):
    """GPU function that runs the actual inference."""
    import subprocess
    import os
    from pathlib import Path
    
    # Set memory management environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # Convert all paths to strings to avoid mixing Path and str objects
    base_path = str(Path("/root/SAT"))
    checkpoint_path = str(Path("/root/checkpoints"))
    
    # Update paths to match download location
    vision_model_path = str(Path(checkpoint_path) / "nano.pth")
    text_encoder_path = str(Path(checkpoint_path) / "nano_text_encoder.pth")
    output_path = str(Path("/root/SAT/results"))
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Add debug prints
    print("Contents of checkpoints directory:")
    for file in Path(checkpoint_path).glob("**/*"):
        print(f"  {file}")
    print(f"Checking if vision model exists: {os.path.exists(vision_model_path)}")
    print(f"Checking if text encoder exists: {os.path.exists(text_encoder_path)}")
    print(f"Checking if input_jsonl exists: {os.path.exists(input_jsonl)}")
    
    # Construct command with all string paths
    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        "--master_port", "1234",
        str(Path(base_path) / "inference.py"),
        "--rcd_dir", output_path,
        "--datasets_jsonl", input_jsonl,
        "--vision_backbone", "UNET",
        "--checkpoint", vision_model_path,
        "--text_encoder", "ours",
        "--text_encoder_checkpoint", text_encoder_path,
        "--max_queries", "128",
        "--batchsize_3d", "1",
    ]
    
    print("Running command:", " ".join(cmd))
    
    try:
        process = subprocess.run(
            cmd,
            cwd=base_path,
            capture_output=True,
            text=True
        )
        print("STDOUT:", process.stdout)
        print("STDERR:", process.stderr)
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, cmd, 
                output=process.stdout,
                stderr=process.stderr
            )
            
        if process.returncode == 0:
            # Debug: List contents of results directory
            print("\nContents of results directory after inference:")
            for file in Path(output_path).glob("**/*"):
                print(f"  {file}")

        return {
            "output_dir": output_path,
            "stdout": process.stdout,
            "stderr": process.stderr
        }
        
    except subprocess.CalledProcessError as e:
        print("Error running inference:")
        print("STDOUT:", e.output)
        print("STDERR:", e.stderr)
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise

@app.local_entrypoint()
def main():
    # First check if models are downloaded
    print("Checking models...")
    download_result = download_models.remote()
    for message in download_result["messages"]:
        print(message)
    if not download_result["success"]:
        print("Failed to download models.")
        return
    
    # Run inference
    print("\nRunning inference with Nano model...")
    try:
        result = run_sat_inference.remote()
        print("\nInference completed successfully!")
        print(f"Results saved to: {result['output_dir']}")
        if result["stdout"]:
            print("\nOutput log:")
            print(result["stdout"])
    except Exception as e:
        print(f"\nError during inference: {str(e)}")
        if hasattr(e, 'stderr'):
            print("\nError details:")
            print(e.stderr)