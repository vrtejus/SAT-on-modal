import modal  # type: ignore
import os
import subprocess
from pathlib import Path

# Define the Modal app
app = modal.App("sat-medical-segmentation-nano")

# Create persistent volume for model weights
model_weights_volume = modal.Volume.from_name("sat-model-weights", create_if_missing=True)
results_volume = modal.Volume.from_name("sat-results-volume", create_if_missing=True)

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
    volumes={"/root/checkpoints": model_weights_volume},
    secrets=[secret]
)
def download_models():
    """CPU-only function to download model weights."""
    from huggingface_hub import hf_hub_download  # type: ignore
    
    results = {"success": True, "messages": []}
    os.makedirs("/root/checkpoints", exist_ok=True)
    hf_token = os.environ["HF_TOKEN"]

    models = {
        "nano.pth": "Nano/nano.pth",
        "nano_text_encoder.pth": "Nano/nano_text_encoder.pth"
    }
    
    for local_name, hf_path in models.items():
        full_ckpt_path = f"/root/checkpoints/{local_name}"
        if not os.path.exists(full_ckpt_path):
            try:
                results["messages"].append(f"Downloading {local_name}...")
                file_path = hf_hub_download(
                    repo_id="zzh99/SAT",
                    filename=hf_path,
                    local_dir="/root/checkpoints",
                    token=hf_token
                )
                results["messages"].append(f"Downloaded to: {file_path}")

                os.makedirs(os.path.dirname(full_ckpt_path), exist_ok=True)
                os.rename(file_path, full_ckpt_path)
                results["messages"].append(f"Moved to: {full_ckpt_path}")
            except Exception as e:
                results["success"] = False
                results["messages"].append(f"Error downloading {local_name}: {str(e)}")

    results["messages"].append("Model download complete!")
    return results

@app.function(
    image=image,
    gpu="T4",
    # Here we only mount model weights by default;
    # We'll dynamically attach the results volume below.
    volumes={
        "/root/checkpoints": model_weights_volume,
        "/root/SAT/results": results_volume,
    },
    mounts=[
        modal.Mount.from_local_dir("data", remote_path="/root/SAT/data")
    ],
    timeout=3600,
)
async def run_sat_inference(
    input_jsonl: str = "/root/SAT/data/inference_demo/demo.jsonl",
    output_dir: str = "results",
    model_type: str = "nano",
):
    """GPU function that runs the actual inference."""
    from modal import Volume  # type: ignore

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    base_path = "/root/SAT"
    checkpoint_path = "/root/checkpoints"
    vision_model_path = f"{checkpoint_path}/nano.pth"
    text_encoder_path = f"{checkpoint_path}/nano_text_encoder.pth"
    output_path = "/root/SAT/results"

    os.makedirs(output_path, exist_ok=True)

    print("Contents of checkpoints directory:")
    for file_path in Path(checkpoint_path).glob("**/*"):
        print(f"  {file_path}")
    print(f"Checking if vision model exists: {os.path.exists(vision_model_path)}")
    print(f"Checking if text encoder exists: {os.path.exists(text_encoder_path)}")
    print(f"Checking if input_jsonl exists: {os.path.exists(input_jsonl)}")

    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        "--master_port", "1234",
        f"{base_path}/inference.py",
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
    process = subprocess.run(cmd, cwd=base_path, capture_output=True, text=True)
    print("STDOUT:", process.stdout)
    print("STDERR:", process.stderr)

    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode, cmd, 
            output=process.stdout,
            stderr=process.stderr
        )

    # Debug: list what got written
    print("\nContents of /root/SAT/results after inference:")
    for file_path in Path(output_path).glob("**/*"):
        print("  ", file_path)

    return {
        "output_dir": output_path,
        "stdout": process.stdout,
        "stderr": process.stderr
    }

@app.local_entrypoint()
def main():
    # 1. Download Models
    print("Checking models...")
    dl_result = download_models.remote()
    for msg in dl_result["messages"]:
        print(msg)
    if not dl_result["success"]:
        print("Failed to download models.")
        return

    # 2. Run Inference
    print("\nRunning inference with Nano model...")
    try:
        result = run_sat_inference.remote()
        print("\nInference completed successfully!")
        print(f"Results saved (remotely) to: {result['output_dir']}")

        # 3. Download from the volume’s root "/" to local
        # 'sat-results-volume' -> local 'results_downloaded/'
        local_folder = "results_downloaded"
        print(f"\nPulling results from volume '{RESULTS_VOLUME_NAME}' to ./{local_folder}/ ...")
        subprocess.run(
            ["modal", "volume", "get", RESULTS_VOLUME_NAME, "/", local_folder],
            check=True
        )
        print(f"Results downloaded into: ./{local_folder}/")

        # Print logs if needed
        if result["stdout"]:
            print("\n=== Inference STDOUT ===\n")
            print(result["stdout"])

    except Exception as e:
        print(f"\nError during inference: {str(e)}")
        if hasattr(e, 'stderr'):
            print("\nError details:")
            print(e.stderr)

