from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "cs4248-meme-reply-eval"

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

DEFAULT_REMOTE_UPLOAD_DIR = "manual_eval"
DEFAULT_REMOTE_TEXT_CHECKPOINT = (
    f"training/checkpoints/{DEFAULT_REMOTE_UPLOAD_DIR}/text_hf_best.pt"
)
DEFAULT_REMOTE_IMAGE_CHECKPOINT = (
    f"training/checkpoints/{DEFAULT_REMOTE_UPLOAD_DIR}/image_qwen_best.pt"
)
DEFAULT_REMOTE_MULTIMODAL_CHECKPOINT = "training/checkpoints/multimodal_qwen_clean/best.pt"

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


def _parse_pipelines(pipelines: str) -> list[str]:
    items = [item.strip() for item in pipelines.split(",") if item.strip()]
    valid = {"text", "image", "multimodal"}
    unknown = [item for item in items if item not in valid]
    if unknown:
        raise ValueError(
            f"Invalid pipelines: {', '.join(unknown)}. Use a comma-separated subset of text,image,multimodal."
        )
    return items


def _parse_splits(splits: str) -> list[str]:
    items = [item.strip() for item in splits.split(",") if item.strip()]
    valid = {"val", "test"}
    unknown = [item for item in items if item not in valid]
    if unknown:
        raise ValueError(
            f"Invalid splits: {', '.join(unknown)}. Use a comma-separated subset of val,test."
        )
    return items


def _volume_upload_path(remote_dir: str, filename: str) -> str:
    cleaned = remote_dir.strip("/")
    if not cleaned:
        raise ValueError("remote_dir must not be empty.")
    return f"/{cleaned}/{filename}"


def _remote_checkpoint_path(remote_dir: str, filename: str) -> str:
    cleaned = remote_dir.strip("/")
    if not cleaned:
        raise ValueError("remote_dir must not be empty.")
    return f"training/checkpoints/{cleaned}/{filename}"


def _normalize_remote_checkpoint_arg(path: str) -> str:
    cleaned = path.strip()
    if not cleaned:
        raise ValueError("Checkpoint path must not be empty.")
    if cleaned.startswith("training/checkpoints/"):
        return cleaned
    if cleaned.startswith("/"):
        return f"training/checkpoints{cleaned}"
    return cleaned


def _remote_fs_path(path: str) -> Path:
    p = Path(_normalize_remote_checkpoint_arg(path))
    if p.is_absolute():
        return p
    return Path(REMOTE_PROJECT_ROOT) / p


def _require_local_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} checkpoint not found: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} checkpoint is not a file: {resolved}")
    return resolved


def _require_remote_file(path: str) -> None:
    resolved = _remote_fs_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found on Modal volume: {path} ({resolved})")


def _run_subprocess(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    subprocess.run(cmd, cwd=REMOTE_PROJECT_ROOT, check=True, env=env)


def _sync_volume_state() -> None:
    images_volume.reload()
    checkpoints_volume.reload()
    hf_cache_volume.reload()


def _commit_eval_state() -> None:
    checkpoints_volume.commit()
    hf_cache_volume.commit()


def _sync_eval_checkpoints(
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


def _eval_command(
    *,
    checkpoint: str,
    split: str,
    batch_size: int = 0,
    qwen_pair_chunk_size: int = 0,
) -> list[str]:
    normalized_checkpoint = _normalize_remote_checkpoint_arg(checkpoint)
    cmd = [
        "python",
        "training/eval.py",
        "--checkpoint",
        normalized_checkpoint,
        "--split",
        split,
        "--device",
        "cuda",
        "--image_dir",
        "training/data/non-annotation-dataset/images",
        "--num_workers",
        "0",
    ]
    if batch_size > 0:
        cmd.extend(["--batch_size", str(batch_size)])
    if qwen_pair_chunk_size > 0:
        cmd.extend(["--qwen_pair_chunk_size", str(qwen_pair_chunk_size)])
    return cmd


@app.function(
    image=code_image,
    gpu="T4",
    timeout=12 * 60 * 60,
    startup_timeout=60 * 60,
    volumes=shared_volumes,
)
def eval_text_remote(
    *,
    checkpoint: str = DEFAULT_REMOTE_TEXT_CHECKPOINT,
    splits: str = "val,test",
    batch_size: int = 0,
) -> str:
    checkpoint = _normalize_remote_checkpoint_arg(checkpoint)
    selected_splits = _parse_splits(splits)
    _sync_volume_state()
    _require_remote_file(checkpoint)

    try:
        for split in selected_splits:
            print(f"Evaluating pipeline=text split={split}")
            _run_subprocess(
                _eval_command(
                    checkpoint=checkpoint,
                    split=split,
                    batch_size=batch_size,
                )
            )
    finally:
        _commit_eval_state()

    return f"text eval complete for splits={','.join(selected_splits)}"


@app.function(
    image=code_image,
    gpu="A100-40GB",
    timeout=12 * 60 * 60,
    startup_timeout=60 * 60,
    volumes=shared_volumes,
)
def eval_qwen_remote(
    *,
    pipelines: str = "image,multimodal",
    image_checkpoint: str = DEFAULT_REMOTE_IMAGE_CHECKPOINT,
    multimodal_checkpoint: str = DEFAULT_REMOTE_MULTIMODAL_CHECKPOINT,
    splits: str = "val,test",
    batch_size: int = 1,
    qwen_pair_chunk_size: int = 1,
) -> str:
    selected_pipelines = [
        item for item in _parse_pipelines(pipelines) if item in {"image", "multimodal"}
    ]
    if not selected_pipelines:
        return "no qwen pipelines selected"

    checkpoint_map = {
        "image": _normalize_remote_checkpoint_arg(image_checkpoint),
        "multimodal": _normalize_remote_checkpoint_arg(multimodal_checkpoint),
    }
    selected_splits = _parse_splits(splits)
    _sync_volume_state()
    for pipeline in selected_pipelines:
        _require_remote_file(checkpoint_map[pipeline])

    try:
        for pipeline in selected_pipelines:
            checkpoint = checkpoint_map[pipeline]
            for split in selected_splits:
                print(f"Evaluating pipeline={pipeline} split={split}")
                _run_subprocess(
                    _eval_command(
                        checkpoint=checkpoint,
                        split=split,
                        batch_size=batch_size,
                        qwen_pair_chunk_size=qwen_pair_chunk_size,
                    )
                )
    finally:
        _commit_eval_state()

    return f"qwen eval complete for pipelines={','.join(selected_pipelines)} splits={','.join(selected_splits)}"


@app.local_entrypoint()
def sync_checkpoints(*cli_args: str) -> None:
    parser = argparse.ArgumentParser(description="Upload local text/image eval checkpoints into the Modal checkpoints volume.")
    parser.add_argument("--text-checkpoint", default=str(DEFAULT_LOCAL_TEXT_CHECKPOINT))
    parser.add_argument("--image-checkpoint", default=str(DEFAULT_LOCAL_IMAGE_CHECKPOINT))
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_UPLOAD_DIR)
    parser.add_argument("--force", action="store_true", help="Overwrite existing uploaded checkpoint files.")
    args = parser.parse_args(list(cli_args))

    remote_text, remote_image = _sync_eval_checkpoints(
        text_checkpoint=Path(args.text_checkpoint),
        image_checkpoint=Path(args.image_checkpoint),
        remote_dir=args.remote_dir,
        force=args.force,
    )
    print(f"text checkpoint uploaded to: {remote_text}")
    print(f"image checkpoint uploaded to: {remote_image}")


@app.local_entrypoint()
def run(*cli_args: str) -> None:
    parser = argparse.ArgumentParser(
        description="Upload local text/image checkpoints and run val/test evaluation on Modal using training/eval.py."
    )
    parser.add_argument(
        "--pipelines",
        default="text,image,multimodal",
        help="Comma-separated subset of text,image,multimodal.",
    )
    parser.add_argument(
        "--splits",
        default="val,test",
        help="Comma-separated subset of val,test.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Reuse the text/image checkpoints already present in the Modal checkpoints volume.",
    )
    parser.add_argument(
        "--force-upload",
        action="store_true",
        help="Overwrite uploaded text/image checkpoint files if they already exist.",
    )
    parser.add_argument(
        "--remote-dir",
        default=DEFAULT_REMOTE_UPLOAD_DIR,
        help="Checkpoint-volume subdirectory used for uploaded text/image checkpoints.",
    )
    parser.add_argument("--text-checkpoint", default=str(DEFAULT_LOCAL_TEXT_CHECKPOINT))
    parser.add_argument("--image-checkpoint", default=str(DEFAULT_LOCAL_IMAGE_CHECKPOINT))
    parser.add_argument(
        "--remote-text-checkpoint",
        default="",
        help="Remote checkpoint path to use for text evaluation. Defaults to training/checkpoints/<remote-dir>/text_hf_best.pt.",
    )
    parser.add_argument(
        "--remote-image-checkpoint",
        default="",
        help="Remote checkpoint path to use for image evaluation. Defaults to training/checkpoints/<remote-dir>/image_qwen_best.pt.",
    )
    parser.add_argument(
        "--multimodal-checkpoint",
        default=DEFAULT_REMOTE_MULTIMODAL_CHECKPOINT,
        help="Remote checkpoint path already present in the Modal checkpoints volume for multimodal evaluation.",
    )
    parser.add_argument("--text-batch-size", type=int, default=0)
    parser.add_argument("--qwen-batch-size", type=int, default=1)
    parser.add_argument("--qwen-pair-chunk-size", type=int, default=1)
    args = parser.parse_args(list(cli_args))

    selected_pipelines = _parse_pipelines(args.pipelines)
    selected_splits = ",".join(_parse_splits(args.splits))

    remote_text = args.remote_text_checkpoint or _remote_checkpoint_path(
        args.remote_dir,
        DEFAULT_LOCAL_TEXT_CHECKPOINT.name,
    )
    remote_image = args.remote_image_checkpoint or _remote_checkpoint_path(
        args.remote_dir,
        DEFAULT_LOCAL_IMAGE_CHECKPOINT.name,
    )

    if not args.skip_upload and any(item in {"text", "image"} for item in selected_pipelines):
        uploaded_text, uploaded_image = _sync_eval_checkpoints(
            text_checkpoint=Path(args.text_checkpoint),
            image_checkpoint=Path(args.image_checkpoint),
            remote_dir=args.remote_dir,
            force=args.force_upload,
        )
        if not args.remote_text_checkpoint:
            remote_text = uploaded_text
        if not args.remote_image_checkpoint:
            remote_image = uploaded_image
    elif not args.skip_upload:
        print("Skipping checkpoint upload because only multimodal evaluation was requested.")

    if "text" in selected_pipelines:
        print(eval_text_remote.remote(
            checkpoint=_normalize_remote_checkpoint_arg(remote_text),
            splits=selected_splits,
            batch_size=args.text_batch_size,
        ))

    qwen_pipelines = [item for item in selected_pipelines if item in {"image", "multimodal"}]
    if qwen_pipelines:
        print(eval_qwen_remote.remote(
            pipelines=",".join(qwen_pipelines),
            image_checkpoint=_normalize_remote_checkpoint_arg(remote_image),
            multimodal_checkpoint=_normalize_remote_checkpoint_arg(args.multimodal_checkpoint),
            splits=selected_splits,
            batch_size=args.qwen_batch_size,
            qwen_pair_chunk_size=args.qwen_pair_chunk_size,
        ))


@app.local_entrypoint()
def volume_info() -> None:
    print(f"images volume: {IMAGES_VOLUME_NAME}")
    print(f"checkpoints volume: {CHECKPOINTS_VOLUME_NAME}")
    print(f"huggingface cache volume: {HF_CACHE_VOLUME_NAME}")
    print(f"default uploaded text checkpoint: {DEFAULT_REMOTE_TEXT_CHECKPOINT}")
    print(f"default uploaded image checkpoint: {DEFAULT_REMOTE_IMAGE_CHECKPOINT}")
    print(f"default multimodal checkpoint: {DEFAULT_REMOTE_MULTIMODAL_CHECKPOINT}")
