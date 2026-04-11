from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import modal


APP_NAME = "cs4248-meme-reply-training"

LOCAL_TRAINING_DIR = Path(__file__).resolve().parent
LOCAL_IMAGE_DIR = LOCAL_TRAINING_DIR / "data" / "non-annotation-dataset" / "images"

REMOTE_PROJECT_ROOT = "/root/project"
REMOTE_TRAINING_DIR = f"{REMOTE_PROJECT_ROOT}/training"
REMOTE_IMAGE_DIR = f"{REMOTE_TRAINING_DIR}/data/non-annotation-dataset/images"
REMOTE_CHECKPOINT_ROOT = f"{REMOTE_TRAINING_DIR}/checkpoints"
REMOTE_HF_CACHE = "/root/.cache/huggingface"

IMAGES_VOLUME_NAME = "cs4248-meme-images"
CHECKPOINTS_VOLUME_NAME = "cs4248-meme-checkpoints"
HF_CACHE_VOLUME_NAME = "cs4248-hf-cache"

images_volume = modal.Volume.from_name(IMAGES_VOLUME_NAME, create_if_missing=True)
checkpoints_volume = modal.Volume.from_name(CHECKPOINTS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

code_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(LOCAL_TRAINING_DIR / "requirements.txt"))
    .pip_install("requests")
    .workdir(REMOTE_PROJECT_ROOT)
    .add_local_dir(
        str(LOCAL_TRAINING_DIR),
        remote_path=REMOTE_TRAINING_DIR,
        ignore=[
            "__pycache__",
            "**/__pycache__",
            "*.pyc",
            "**/*.pyc",
            "modal_checkpoints",
            "modal_checkpoints/**",
            "checkpoints",
            "checkpoints/**",
            "tmp_test_ckpts",
            "tmp_test_ckpts/**",
            "data/non-annotation-dataset/images",
            "data/non-annotation-dataset/images/**",
        ],
    )
)

shared_volumes = {
    REMOTE_IMAGE_DIR: images_volume,
    REMOTE_CHECKPOINT_ROOT: checkpoints_volume,
    REMOTE_HF_CACHE: hf_cache_volume,
}

app = modal.App(APP_NAME)


def _parse_pipeline_csv(pipelines: str) -> list[str]:
    items = [item.strip() for item in pipelines.split(",") if item.strip()]
    valid = {"text", "image", "multimodal"}
    unknown = [item for item in items if item not in valid]
    if unknown:
        raise ValueError(
            f"Invalid pipelines: {', '.join(unknown)}. Use a comma-separated subset of text,image,multimodal."
        )
    return items


def _run_all_command(
    *,
    mode: str,
    pipelines: str,
    num_epochs: int,
    text_batch_size: int,
    qwen_batch_size: int,
    freeze_encoder: bool,
    save_root: str,
    image_dir: str,
) -> list[str]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    if not save_root.startswith("training/checkpoints"):
        raise ValueError(
            "save_root must stay under 'training/checkpoints' so checkpoints persist in the Modal volume. "
            f"Got: {save_root}"
        )

    cmd = [
        "bash",
        "training/run_all_pipelines.sh",
        f"--{mode}",
        "--device",
        "cuda",
        "--only",
        pipelines,
        "--save-root",
        save_root,
        "--image-dir",
        image_dir,
        "--text-batch-size",
        str(text_batch_size),
        "--qwen-batch-size",
        str(qwen_batch_size),
    ]
    if num_epochs > 0:
        cmd.extend(["--num-epochs", str(num_epochs)])
    if not freeze_encoder:
        cmd.append("--no-freeze-encoder")
    return cmd


def _run_subprocess(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=REMOTE_PROJECT_ROOT, check=True)


def _sync_volume_state() -> None:
    images_volume.reload()
    checkpoints_volume.reload()
    hf_cache_volume.reload()


def _commit_training_state() -> None:
    checkpoints_volume.commit()
    hf_cache_volume.commit()


@app.function(
    image=code_image,
    gpu="T4",
    timeout=24 * 60 * 60,
    startup_timeout=60 * 60,
    volumes=shared_volumes,
)
def train_text_remote(
    *,
    mode: str = "full",
    num_epochs: int = 0,
    text_batch_size: int = 16,
    save_root: str = "training/checkpoints",
) -> str:
    _sync_volume_state()
    cmd = _run_all_command(
        mode=mode,
        pipelines="text",
        num_epochs=num_epochs,
        text_batch_size=text_batch_size,
        qwen_batch_size=1,
        freeze_encoder=True,
        save_root=save_root,
        image_dir="training/data/non-annotation-dataset/images",
    )
    try:
        _run_subprocess(cmd)
    finally:
        _commit_training_state()
    return "text complete"


@app.function(
    image=code_image,
    gpu="A100-40GB",
    timeout=24 * 60 * 60,
    startup_timeout=60 * 60,
    volumes=shared_volumes,
)
def train_qwen_remote(
    *,
    pipelines: str = "image,multimodal",
    mode: str = "full",
    num_epochs: int = 0,
    qwen_batch_size: int = 1,
    freeze_encoder: bool = True,
    save_root: str = "training/checkpoints",
) -> str:
    selected = [item for item in _parse_pipeline_csv(pipelines) if item in {"image", "multimodal"}]
    if not selected:
        return "no qwen pipelines selected"

    _sync_volume_state()
    cmd = _run_all_command(
        mode=mode,
        pipelines=",".join(selected),
        num_epochs=num_epochs,
        text_batch_size=16,
        qwen_batch_size=qwen_batch_size,
        freeze_encoder=freeze_encoder,
        save_root=save_root,
        image_dir="training/data/non-annotation-dataset/images",
    )
    try:
        _run_subprocess(cmd)
    finally:
        _commit_training_state()
    return ",".join(selected) + " complete"


@app.local_entrypoint()
def train(*cli_args: str) -> None:
    parser = argparse.ArgumentParser(description="Run the training pipelines on Modal.")
    parser.add_argument(
        "--pipelines",
        default="text,image,multimodal",
        help="Comma-separated subset of text,image,multimodal.",
    )
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=0,
        help="Total target epochs for each run. Use 0 to keep training defaults.",
    )
    parser.add_argument("--text-batch-size", type=int, default=16)
    parser.add_argument("--qwen-batch-size", type=int, default=1)
    parser.add_argument("--no-freeze-encoder", action="store_true")
    parser.add_argument("--save-root", default="training/checkpoints")
    args = parser.parse_args(list(cli_args))

    selected = _parse_pipeline_csv(args.pipelines)
    if "text" in selected:
        print(train_text_remote.remote(
            mode=args.mode,
            num_epochs=args.num_epochs,
            text_batch_size=args.text_batch_size,
            save_root=args.save_root,
        ))

    qwen_selected = [item for item in selected if item in {"image", "multimodal"}]
    if qwen_selected:
        print(train_qwen_remote.remote(
            pipelines=",".join(qwen_selected),
            mode=args.mode,
            num_epochs=args.num_epochs,
            qwen_batch_size=args.qwen_batch_size,
            freeze_encoder=not args.no_freeze_encoder,
            save_root=args.save_root,
        ))


@app.local_entrypoint()
def sync_images(*cli_args: str) -> None:
    parser = argparse.ArgumentParser(description="Upload the local image directory into the Modal image volume.")
    parser.add_argument("--local-dir", default=str(LOCAL_IMAGE_DIR))
    parser.add_argument("--force", action="store_true", help="Overwrite existing volume files if needed.")
    args = parser.parse_args(list(cli_args))

    local_dir = Path(args.local_dir).expanduser().resolve()
    if not local_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {local_dir}")
    if not local_dir.is_dir():
        raise NotADirectoryError(f"Image path is not a directory: {local_dir}")

    image_files = [
        path for path in local_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    print(f"Uploading {len(image_files)} image files from {local_dir} to Modal volume {IMAGES_VOLUME_NAME}.")
    with images_volume.batch_upload(force=args.force) as batch:
        batch.put_directory(f"{local_dir}/", "/")
    print("Upload complete.")


@app.local_entrypoint()
def volume_info() -> None:
    print(f"images volume: {IMAGES_VOLUME_NAME}")
    print(f"checkpoints volume: {CHECKPOINTS_VOLUME_NAME}")
    print(f"huggingface cache volume: {HF_CACHE_VOLUME_NAME}")
