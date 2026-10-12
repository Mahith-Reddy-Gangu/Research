"""MRI-only self-supervised pretraining for the S-RegNET velocity field.

Each step samples a smooth, band-limited stationary velocity field, warps an
unlabelled MRI with its exponential, and regresses the known velocity directly.
No segmentation labels or Dice loss are used in this stage.
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from git_provenance import write_git_sha
from losses import bending_energy_loss, jacobian_det, jacobian_det_loss
from model import SegRegistrationNet, SpatialTransformer


class UnlabelledMRIDataset(Dataset):
    """Load one normalized brain MRI per subject directory; labels are unused."""

    def __init__(self, list_path, mri_filename, target_size, excluded_subject_dir=None):
        if not list_path:
            raise ValueError("Both data.train_txt and data.val_txt are required")
        list_path = Path(list_path).expanduser()
        if not list_path.is_file():
            raise FileNotFoundError(f"MRI subject list does not exist: {list_path}")

        entries = [line.strip() for line in list_path.read_text().splitlines()
                   if line.strip() and not line.lstrip().startswith("#")]
        if not entries:
            raise ValueError(f"MRI subject list is empty: {list_path}")

        self.paths = []
        excluded = (Path(excluded_subject_dir).expanduser().resolve()
                    if excluded_subject_dir else None)
        for entry in entries:
            subject_dir = Path(entry).expanduser().parent
            path = subject_dir / mri_filename
            if not path.is_file():
                raise FileNotFoundError(f"MRI volume does not exist: {path}")
            if excluded is not None and subject_dir.resolve() == excluded:
                raise ValueError(
                    f"Template subject must be excluded from SSL lists: {subject_dir}")
            self.paths.append(path)
        if len(set(self.paths)) != len(self.paths):
            raise ValueError(f"MRI subject list contains duplicate subjects: {list_path}")
        self.target_size = tuple(int(n) for n in target_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        volume = np.load(path, mmap_mode="r")
        if volume.ndim != 3:
            raise ValueError(f"Expected a 3-D MRI at {path}, got shape {volume.shape}")
        if not np.isfinite(volume).all():
            raise ValueError(f"MRI contains non-finite values: {path}")

        image = torch.from_numpy(np.array(volume, dtype=np.float32, copy=True))[None, None]
        image_min, image_max = image.amin(), image.amax()
        if not torch.isfinite(image_min) or not torch.isfinite(image_max):
            raise ValueError(f"MRI has invalid intensity range: {path}")
        if (image_max - image_min).item() <= 1e-8:
            raise ValueError(f"MRI is constant and cannot provide registration signal: {path}")
        image = (image - image_min) / (image_max - image_min)
        image = F.interpolate(image, size=self.target_size, mode="trilinear",
                              align_corners=False)
        if image.median() > 0.5:
            image = 1.0 - image
        return image.squeeze(0)


def integrate_velocity(velocity, stn, steps=7):
    """Scaling-and-squaring integration matching SegRegistrationNet.forward."""
    flow = velocity / (2 ** steps)
    for _ in range(steps):
        flow = flow + stn(flow, flow)
    return flow


def generate_synthetic_batch(images, stn, control_points=16, lowpass_size=96,
                             velocity_std_voxels=1.5, min_jacobian_det=0.05,
                             max_safety_retries=8, generator=None):
    """Return a warped MRI and its known, model-band-limited SVF.

    The random control lattice is expressed in voxel units, converted to the
    STN's normalized (x, y, z) coordinates, then passed through the exact
    lowpass down/up operator used by BandLimitedHead's base branch.
    """
    if images.ndim != 5 or images.shape[1] != 1:
        raise ValueError(f"Expected MRI batch (B,1,D,H,W), got {tuple(images.shape)}")
    _, _, depth, height, width = images.shape
    if len({depth, height, width}) != 1:
        raise ValueError("BandLimitedHead currently requires a cubic target grid")
    if not 2 <= control_points < depth:
        raise ValueError("control_points must be >=2 and smaller than the MRI grid")
    if not 1 <= lowpass_size <= depth:
        raise ValueError("lowpass_size must be between 1 and the MRI grid size")
    if velocity_std_voxels <= 0:
        raise ValueError("velocity_std_voxels must be positive")

    lattice = torch.randn(
        images.shape[0], 3, control_points, control_points, control_points,
        device=images.device, dtype=torch.float32,
        generator=generator,
    ) * velocity_std_voxels
    voxel_field = F.interpolate(lattice, size=(depth, height, width),
                                mode="trilinear", align_corners=False)
    norm_scale = voxel_field.new_tensor(
        [2.0 / width, 2.0 / height, 2.0 / depth]
    ).view(1, 3, 1, 1, 1)
    velocity = voxel_field * norm_scale
    velocity = F.interpolate(
        F.interpolate(velocity, size=(lowpass_size,) * 3,
                      mode="trilinear", align_corners=False),
        size=(depth, height, width), mode="trilinear", align_corners=False,
    )
    det_ref = float(np.prod([2.0 / n for n in (depth, height, width)]))
    for attempt in range(max_safety_retries + 1):
        target_flow = integrate_velocity(velocity, stn)
        det_interior = jacobian_det(target_flow) / det_ref
        min_det = det_interior[..., :-1, :-1, :-1].amin()
        if torch.isfinite(min_det) and min_det.item() >= min_jacobian_det:
            break
        if attempt == max_safety_retries:
            raise RuntimeError(
                "Could not generate a fold-free synthetic target after "
                f"{max_safety_retries} velocity reductions "
                f"(minimum normalized Jacobian {min_det.item():.4g})"
            )
        velocity = velocity * 0.5
    with torch.no_grad():
        warped = stn(images.float(), target_flow.float())
    return warped, velocity, target_flow, min_det


def _brain_mae(prediction, target):
    mask = target > 0.02
    if not mask.any():
        raise ValueError("Synthetic target has no non-background MRI voxels")
    return (prediction - target).abs()[mask].mean()


def _make_logger(log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sregnet_ssl")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(log_dir / "pretrain.log"),
                    logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _run_epoch(model, stn, loader, optimizer, scaler, device, cfg, training, epoch):
    model.train(training)
    totals = {"loss": 0.0, "velocity_mse": 0.0, "bending": 0.0,
              "jacobian": 0.0, "image_mae": 0.0, "identity_mae": 0.0,
              "folding_pct": 0.0, "target_folding_pct": 0.0}
    target_size = tuple(cfg["model"]["target_size"])
    det_ref = float(np.prod([2.0 / n for n in target_size]))
    amp_enabled = bool(cfg["training"].get("use_amp", True)) and device.type == "cuda"
    generator = torch.Generator(device=device)
    generator.manual_seed(
        int(cfg.get("seed", 42)) + (int(epoch) if training else 1_000_003))

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in tqdm(loader, desc="train" if training else "val",
                          leave=False, ncols=100):
            fixed = batch.to(device, non_blocking=True)
            synthetic, target_velocity, target_flow, target_min_det = generate_synthetic_batch(
                fixed, stn,
                control_points=int(cfg["synthetic"]["control_points"]),
                lowpass_size=int(cfg["model"]["lowpass_size"]),
                velocity_std_voxels=float(cfg["synthetic"]["velocity_std_voxels"]),
                min_jacobian_det=float(cfg["synthetic"].get("min_jacobian_det", 0.05)),
                max_safety_retries=int(cfg["synthetic"].get("max_safety_retries", 8)),
                generator=generator,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                flow_fw, _, _, _, predicted_velocity = model(
                    fixed, synthetic, return_velocity=True)

            with torch.autocast(device_type=device.type, enabled=False):
                predicted_velocity = predicted_velocity.float()
                flow_fw = flow_fw.float()
                velocity_mse = F.mse_loss(predicted_velocity, target_velocity.float())
                bending = bending_energy_loss(predicted_velocity)
                jacobian = jacobian_det_loss(flow_fw)
                loss = (velocity_mse
                        + float(cfg["loss"]["bending"]) * bending
                        + float(cfg["loss"]["jacobian"]) * jacobian)

                with torch.no_grad():
                    predicted_image = stn(fixed.float(), flow_fw)
                    identity_mae = _brain_mae(fixed.float(), synthetic.float())
                    image_mae = _brain_mae(predicted_image, synthetic.float())
                    det_pred = jacobian_det(flow_fw) / det_ref
                    folding_pct = (det_pred < 0).float().mean() * 100.0
                    target_folding_pct = (target_min_det < 0).float() * 100.0

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite SSL loss; refusing to continue")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(cfg["training"].get("max_grad_norm", 1.0)),
                )
                scaler.step(optimizer)
                scaler.update()

            values = {
                "loss": loss, "velocity_mse": velocity_mse, "bending": bending,
                "jacobian": jacobian, "image_mae": image_mae,
                "identity_mae": identity_mae, "folding_pct": folding_pct,
                "target_folding_pct": target_folding_pct,
            }
            for name, value in values.items():
                totals[name] += float(value.detach().item())

    return {name: value / len(loader) for name, value in totals.items()}


def _atomic_save(checkpoint, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description="S-RegNET MRI self-supervised pretraining")
    parser.add_argument("--config", default=str(Path(__file__).with_name(
        "config_pretrain.yaml")))
    parser.add_argument("--resume", default=None,
                        help="resume from an SSL checkpoint (model, optimizer, and epoch)")
    args = parser.parse_args()
    config_path = Path(args.config).expanduser()
    with config_path.open() as stream:
        cfg = yaml.safe_load(stream)

    target_size = tuple(int(n) for n in cfg["model"]["target_size"])
    if len(target_size) != 3 or len(set(target_size)) != 1:
        raise ValueError("model.target_size must be a cubic three-dimensional grid")
    lowpass_size = int(cfg["model"].get(
        "lowpass_size", target_size[0] * 3 // 4))
    cfg["model"]["lowpass_size"] = lowpass_size
    if not 1 <= lowpass_size <= target_size[0]:
        raise ValueError("model.lowpass_size is outside model.target_size")

    train_data = UnlabelledMRIDataset(
        cfg["data"].get("train_txt"), cfg["data"].get("mri_filename", "brain.npy"),
        target_size, cfg["data"].get("template_subject_dir"))
    val_data = UnlabelledMRIDataset(
        cfg["data"].get("val_txt"), cfg["data"].get("mri_filename", "brain.npy"),
        target_size, cfg["data"].get("template_subject_dir"))
    train_subjects, val_subjects = set(train_data.paths), set(val_data.paths)
    overlap = train_subjects & val_subjects
    if overlap:
        raise ValueError(
            f"Train/validation MRI lists overlap ({len(overlap)} subjects); "
            f"first duplicate: {next(iter(overlap))}")

    device_name = cfg.get("device", {}).get("gpu", "cuda:0")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = bool(
        cfg.get("device", {}).get("cudnn_benchmark", False))

    batch_size = int(cfg["training"].get("batch_size", 1))
    loader_args = {
        "batch_size": batch_size,
        "num_workers": int(cfg["training"].get("num_workers", 0)),
        "pin_memory": bool(cfg["training"].get("pin_memory", True))
                      and device.type == "cuda",
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg["output"]["base_dir"]).expanduser() / timestamp
    checkpoint_dir = output_dir / cfg["output"].get("checkpoint_subdir", "checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = _make_logger(output_dir / cfg["output"].get("log_subdir", "logs"))
    with (output_dir / "config.yaml").open("w") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    with (output_dir / "config.json").open("w") as stream:
        json.dump(cfg, stream, indent=2)
    write_git_sha(str(output_dir), logger)

    model = SegRegistrationNet(
        target_size=target_size, seg_channels=1, use_affine=False,
        head="bandlimited", lowpass_size=lowpass_size,
    ).to(device)
    stn = model.stn
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-5)),
    )
    amp_enabled = bool(cfg["training"].get("use_amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    epochs = int(cfg["training"]["num_epochs"])
    patience = int(cfg["training"].get("patience", epochs))
    warmup = int(cfg["training"].get("warmup_epochs", 0))
    base_lr = float(cfg["training"]["learning_rate"])
    min_lr = float(cfg["training"].get("min_learning_rate", 1e-6))
    best_val_mse = float("inf")
    best_epoch = 0
    stale_epochs = 0
    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume).expanduser()
        if not resume_path.is_file():
            raise FileNotFoundError(f"SSL resume checkpoint does not exist: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if not checkpoint.get("ssl_pretraining", False):
            raise ValueError(f"Not an MRI-SSL checkpoint: {resume_path}")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_mse = float(checkpoint["best_val_velocity_mse"])
        best_epoch = int(checkpoint.get("best_epoch", checkpoint["epoch"]))
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        logger.info(f"Resumed SSL at epoch {start_epoch - 1}: {resume_path}")

    logger.info(
        f"MRI-only SSL | train={len(train_data)} val={len(val_data)} "
        f"grid={target_size} lowpass={lowpass_size} device={device}"
    )
    logger.info(
        "Primary objective: direct GT velocity MSE; "
        f"bending={cfg['loss']['bending']} jacobian={cfg['loss']['jacobian']}. "
        "No Dice or segmentation labels are used in pretraining."
    )

    for epoch in range(start_epoch, epochs + 1):
        if epoch <= warmup:
            lr = base_lr * epoch / max(warmup, 1)
        else:
            progress = (epoch - warmup) / max(epochs - warmup, 1)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr

        train_metrics = _run_epoch(
            model, stn, train_loader, optimizer, scaler, device, cfg, True, epoch)
        val_metrics = _run_epoch(
            model, stn, val_loader, optimizer, scaler, device, cfg, False, epoch)
        logger.info(
            f"epoch={epoch:03d}/{epochs} lr={lr:.2e} "
            f"train_v_mse={train_metrics['velocity_mse']:.6g} "
            f"val_v_mse={val_metrics['velocity_mse']:.6g} "
            f"val_image_mae={val_metrics['image_mae']:.5f} "
            f"(identity={val_metrics['identity_mae']:.5f}) "
            f"val_folding={val_metrics['folding_pct']:.5f}% "
            f"target_folding={val_metrics['target_folding_pct']:.5f}%"
        )
        with (output_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({
                "epoch": epoch, "learning_rate": lr,
                "train": train_metrics, "validation": val_metrics,
            }) + "\n")

        if val_metrics["velocity_mse"] < best_val_mse:
            best_val_mse = val_metrics["velocity_mse"]
            best_epoch = epoch
            stale_epochs = 0
            _atomic_save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_velocity_mse": best_val_mse,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "validation_metrics": val_metrics,
                "ssl_pretraining": True,
                "pretraining_config": cfg,
            }, checkpoint_dir / "best_pretrain_model.pth")
        else:
            stale_epochs += 1

        if epoch % int(cfg["training"].get("save_every", 10)) == 0:
            _atomic_save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_velocity_mse": best_val_mse,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "ssl_pretraining": True, "pretraining_config": cfg,
            }, checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth")
        if stale_epochs >= patience:
            logger.info(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}")
            break

    logger.info(
        f"Pretraining complete: best epoch={best_epoch}, "
        f"held-out velocity MSE={best_val_mse:.6g}, "
        f"checkpoint={checkpoint_dir / 'best_pretrain_model.pth'}"
    )


if __name__ == "__main__":
    main()
