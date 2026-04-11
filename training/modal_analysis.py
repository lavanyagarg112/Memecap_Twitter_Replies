from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "cs4248-meme-analysis"

LOCAL_TRAINING_DIR = Path(__file__).resolve().parent
LOCAL_MODAL_CHECKPOINT_DIR = LOCAL_TRAINING_DIR / "modal_checkpoints"

REMOTE_PROJECT_ROOT = "/root/project"
REMOTE_TRAINING_DIR = f"{REMOTE_PROJECT_ROOT}/training"
REMOTE_IMAGE_DIR = f"{REMOTE_TRAINING_DIR}/data/non-annotation-dataset/images"
REMOTE_CHECKPOINT_ROOT = f"{REMOTE_TRAINING_DIR}/checkpoints"
REMOTE_HF_CACHE = "/root/.cache/huggingface"

IMAGES_VOLUME_NAME = "cs4248-meme-images"
CHECKPOINTS_VOLUME_NAME = "cs4248-meme-checkpoints"
HF_CACHE_VOLUME_NAME = "cs4248-hf-cache"

DEFAULT_LOCAL_TEXT_CHECKPOINT = LOCAL_MODAL_CHECKPOINT_DIR / "text_hf_best.pt"
DEFAULT_LOCAL_IMAGE_CHECKPOINT = LOCAL_MODAL_CHECKPOINT_DIR / "image_qwen_best.pt"
DEFAULT_REMOTE_MULTIMODAL_CHECKPOINT = "training/checkpoints/multimodal_qwen_clean/best.pt"
DEFAULT_REMOTE_UPLOAD_DIR = "manual_eval"
DEFAULT_ANALYSIS_NAME = "final_analysis"

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


def _normalize_remote_checkpoint(path: str) -> str:
    cleaned = path.strip()
    if not cleaned:
        raise ValueError("Checkpoint path must not be empty.")
    if cleaned.startswith("training/checkpoints/"):
        return cleaned
    if cleaned.startswith("/"):
        return f"training/checkpoints{cleaned}"
    return cleaned


def _remote_checkpoint_path(remote_dir: str, filename: str) -> str:
    cleaned = remote_dir.strip("/")
    if not cleaned:
        raise ValueError("remote_dir must not be empty.")
    return f"training/checkpoints/{cleaned}/{filename}"


def _volume_upload_path(remote_dir: str, filename: str) -> str:
    cleaned = remote_dir.strip("/")
    if not cleaned:
        raise ValueError("remote_dir must not be empty.")
    return f"/{cleaned}/{filename}"


def _remote_fs_path(path: str) -> Path:
    path = _normalize_remote_checkpoint(path)
    return Path(REMOTE_PROJECT_ROOT) / path


def _require_local_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"{label} checkpoint not found: {resolved}")
    return resolved


def _require_remote_file(path: str) -> None:
    resolved = _remote_fs_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Remote checkpoint not found: {path} ({resolved})")


def _sync_volume_state() -> None:
    images_volume.reload()
    checkpoints_volume.reload()
    hf_cache_volume.reload()


def _commit_state() -> None:
    checkpoints_volume.commit()
    hf_cache_volume.commit()


def _sync_local_checkpoints(
    *,
    text_checkpoint: Path,
    image_checkpoint: Path,
    remote_dir: str,
    force: bool,
) -> tuple[str, str]:
    local_text = _require_local_file(text_checkpoint, "text")
    local_image = _require_local_file(image_checkpoint, "image")

    remote_text = _remote_checkpoint_path(remote_dir, local_text.name)
    remote_image = _remote_checkpoint_path(remote_dir, local_image.name)

    print(f"Uploading text checkpoint: {local_text} -> {remote_text}")
    print(f"Uploading image checkpoint: {local_image} -> {remote_image}")
    with checkpoints_volume.batch_upload(force=force) as batch:
        batch.put_file(str(local_text), _volume_upload_path(remote_dir, local_text.name))
        batch.put_file(str(local_image), _volume_upload_path(remote_dir, local_image.name))
    print("Checkpoint upload complete.")
    return remote_text, remote_image


def _run_subprocess(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    subprocess.run(cmd, cwd=REMOTE_PROJECT_ROOT, check=True, env=env)


@app.function(
    image=code_image,
    gpu="A100-40GB",
    timeout=12 * 60 * 60,
    startup_timeout=60 * 60,
    volumes=shared_volumes,
)
def analyze_remote(
    *,
    text_checkpoint: str,
    image_checkpoint: str,
    multimodal_checkpoint: str,
    splits: str = "val,test",
    analysis_name: str = DEFAULT_ANALYSIS_NAME,
    top_k: int = 3,
    audit_examples: int = 5,
    qwen_pair_chunk_size: int = 1,
) -> str:
    _sync_volume_state()
    text_checkpoint = _normalize_remote_checkpoint(text_checkpoint)
    image_checkpoint = _normalize_remote_checkpoint(image_checkpoint)
    multimodal_checkpoint = _normalize_remote_checkpoint(multimodal_checkpoint)

    for checkpoint in [text_checkpoint, image_checkpoint, multimodal_checkpoint]:
        _require_remote_file(checkpoint)

    output_dir = f"training/checkpoints/analysis/{analysis_name}"
    split_list = [item.strip() for item in splits.split(",") if item.strip()]

    try:
        cmd = [
            "python",
            "training/analyze_predictions.py",
            "--run",
            f"text={text_checkpoint}",
            "--run",
            f"image={image_checkpoint}",
            "--run",
            f"multimodal={multimodal_checkpoint}",
            "--output_dir",
            output_dir,
            "--device",
            "cuda",
            "--num_workers",
            "0",
            "--top_k",
            str(top_k),
            "--audit_examples",
            str(audit_examples),
            "--qwen_pair_chunk_size",
            str(qwen_pair_chunk_size),
        ]
        for split in split_list:
            cmd.extend(["--split", split])
        _run_subprocess(cmd)
    finally:
        _commit_state()

    return output_dir


@app.local_entrypoint()
def run(*cli_args: str) -> None:
    parser = argparse.ArgumentParser(
        description="Run prediction/error analysis for text, image, and multimodal checkpoints on Modal."
    )
    parser.add_argument("--analysis-name", default=DEFAULT_ANALYSIS_NAME)
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--audit-examples", type=int, default=5)
    parser.add_argument("--qwen-pair-chunk-size", type=int, default=1)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--force-upload", action="store_true")
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_UPLOAD_DIR)
    parser.add_argument("--text-checkpoint", default=str(DEFAULT_LOCAL_TEXT_CHECKPOINT))
    parser.add_argument("--image-checkpoint", default=str(DEFAULT_LOCAL_IMAGE_CHECKPOINT))
    parser.add_argument("--remote-text-checkpoint", default="")
    parser.add_argument("--remote-image-checkpoint", default="")
    parser.add_argument(
        "--multimodal-checkpoint",
        default=DEFAULT_REMOTE_MULTIMODAL_CHECKPOINT,
    )
    args = parser.parse_args(list(cli_args))

    remote_text = args.remote_text_checkpoint or _remote_checkpoint_path(
        args.remote_dir,
        DEFAULT_LOCAL_TEXT_CHECKPOINT.name,
    )
    remote_image = args.remote_image_checkpoint or _remote_checkpoint_path(
        args.remote_dir,
        DEFAULT_LOCAL_IMAGE_CHECKPOINT.name,
    )

    if not args.skip_upload:
        uploaded_text, uploaded_image = _sync_local_checkpoints(
            text_checkpoint=Path(args.text_checkpoint),
            image_checkpoint=Path(args.image_checkpoint),
            remote_dir=args.remote_dir,
            force=args.force_upload,
        )
        if not args.remote_text_checkpoint:
            remote_text = uploaded_text
        if not args.remote_image_checkpoint:
            remote_image = uploaded_image

    output_dir = analyze_remote.remote(
        text_checkpoint=remote_text,
        image_checkpoint=remote_image,
        multimodal_checkpoint=args.multimodal_checkpoint,
        splits=args.splits,
        analysis_name=args.analysis_name,
        top_k=args.top_k,
        audit_examples=args.audit_examples,
        qwen_pair_chunk_size=args.qwen_pair_chunk_size,
    )
    print(f"Analysis artifacts written to: {output_dir}")
    print(
        "Download them with: "
        f"modal volume get {CHECKPOINTS_VOLUME_NAME} /analysis/{args.analysis_name} training/analysis/{args.analysis_name}"
    )
