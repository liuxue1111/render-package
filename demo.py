#!/usr/bin/env python3
"""Render an editable Gaussian scene into a RadarSplat RA image."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import Tensor

try:
    from gsplat import _radar_rasterization
    from gsplat.rendering import azimuth_antenna_gain_projection, spectral_leakage
except (ImportError, OSError) as error:
    raise RuntimeError(
        "RadarSplat CUDA backend is unavailable. Follow README.md to install it."
    ) from error


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class GaussianRadarField:
    means: Tensor
    quats: Tensor
    log_scales: Tensor
    opacity_logits: Tensor
    noise_logits: Tensor
    reflectance: Tensor


@dataclass(frozen=True)
class SensorConfig:
    range_resolution_m: float
    range_bins: int
    intermediate_azimuth_resolution_deg: float
    output_azimuth_resolution_deg: float
    azimuth_beamwidth_deg: float
    spectral_leakage_enabled: bool
    sinc_width_m: float
    near_plane: float
    far_plane: float

    @property
    def intermediate_azimuth_bins(self) -> int:
        return int(round(360.0 / self.intermediate_azimuth_resolution_deg))

    def validate(self) -> None:
        positive = (
            self.range_resolution_m,
            self.intermediate_azimuth_resolution_deg,
            self.output_azimuth_resolution_deg,
            self.azimuth_beamwidth_deg,
            self.sinc_width_m,
            self.far_plane,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("sensor resolutions, beamwidth, sinc width and far plane must be positive")
        if self.range_bins <= 0:
            raise ValueError("range_bins must be positive")
        intermediate_bins = 360.0 / self.intermediate_azimuth_resolution_deg
        output_bins = 360.0 / self.output_azimuth_resolution_deg
        ratio = self.output_azimuth_resolution_deg / self.intermediate_azimuth_resolution_deg
        if not math.isclose(intermediate_bins, round(intermediate_bins), abs_tol=1e-9):
            raise ValueError("360 / intermediate azimuth resolution must be an integer")
        if not math.isclose(output_bins, round(output_bins), abs_tol=1e-9):
            raise ValueError("360 / output azimuth resolution must be an integer")
        if ratio < 1.0 or not math.isclose(ratio, round(ratio), abs_tol=1e-9):
            raise ValueError("output/intermediate azimuth resolution must be an integer")
        if self.spectral_leakage_enabled:
            sigma = int(((self.sinc_width_m / 2.0) / self.range_resolution_m) / 3.0)
            if sigma < 1:
                raise ValueError("sinc_width_m is too small for spectral leakage")
        if not math.isfinite(self.near_plane) or self.near_plane >= self.far_plane:
            raise ValueError("near_plane must be finite and smaller than far_plane")

    def intrinsics(self, device: torch.device) -> Tensor:
        return torch.tensor(
            [
                [1.0 / self.range_resolution_m, 0.0, 0.0],
                [0.0, 1.0 / math.radians(self.intermediate_azimuth_resolution_deg), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )


def _vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("configuration must use schema_version: 1")
    if not isinstance(config.get("scene"), list) or not config["scene"]:
        raise ValueError("scene must be a non-empty list")
    _vector(config["radar"]["position_m"], 3, "radar.position_m")
    float(config["radar"]["yaw_deg"])
    for index, target in enumerate(config["scene"]):
        _vector(target["position_m"], 3, f"scene[{index}].position_m")
        scale = _vector(target["scale_m"], 3, f"scene[{index}].scale_m")
        if any(value <= 0.0 for value in scale):
            raise ValueError(f"scene[{index}].scale_m values must be positive")
        quat = _vector(
            target.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]),
            4,
            f"scene[{index}].quaternion_wxyz",
        )
        if sum(value * value for value in quat) < 1e-12:
            raise ValueError(f"scene[{index}].quaternion_wxyz cannot be zero")
        opacity = float(target["opacity"])
        reflectance = float(target["reflectance"])
        if not 0.0 < opacity < 1.0:
            raise ValueError(f"scene[{index}].opacity must be between 0 and 1")
        if not math.isfinite(reflectance) or reflectance < 0.0:
            raise ValueError(f"scene[{index}].reflectance must be nonnegative")
        target.setdefault("name", f"target_{index}")
    return config


def make_sensor(config: dict[str, Any]) -> SensorConfig:
    values = config["sensor"]
    sensor = SensorConfig(
        range_resolution_m=float(values["range_resolution_m"]),
        range_bins=int(values["range_bins"]),
        intermediate_azimuth_resolution_deg=float(values["intermediate_azimuth_resolution_deg"]),
        output_azimuth_resolution_deg=float(values["output_azimuth_resolution_deg"]),
        azimuth_beamwidth_deg=float(values["azimuth_beamwidth_deg"]),
        spectral_leakage_enabled=bool(values.get("spectral_leakage_enabled", True)),
        sinc_width_m=float(values.get("sinc_width_m", 1.0)),
        near_plane=float(values.get("near_plane", -10.0)),
        far_plane=float(values.get("far_plane", 1e10)),
    )
    sensor.validate()
    return sensor


def make_scene(config: dict[str, Any], device: torch.device) -> tuple[GaussianRadarField, Tensor]:
    targets = config["scene"]
    means = torch.tensor([item["position_m"] for item in targets], dtype=torch.float32, device=device)
    scales = torch.tensor([item["scale_m"] for item in targets], dtype=torch.float32, device=device)
    quats = torch.tensor(
        [item.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]) for item in targets],
        dtype=torch.float32,
        device=device,
    )
    quats = quats / torch.linalg.vector_norm(quats, dim=-1, keepdim=True)
    opacity = torch.tensor([item["opacity"] for item in targets], dtype=torch.float32, device=device)
    reflectance = torch.tensor(
        [item["reflectance"] for item in targets], dtype=torch.float32, device=device
    )
    field = GaussianRadarField(
        means=means,
        quats=quats,
        log_scales=torch.log(scales),
        opacity_logits=torch.logit(opacity),
        noise_logits=torch.zeros(len(targets), dtype=torch.float32, device=device),
        reflectance=reflectance,
    )

    radar = config["radar"]
    x, y, z = (float(value) for value in radar["position_m"])
    yaw = math.radians(float(radar["yaw_deg"]))
    cosine, sine = math.cos(yaw), math.sin(yaw)
    T_WR = torch.tensor(
        [
            [cosine, -sine, 0.0, x],
            [sine, cosine, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    return field, torch.linalg.inv(T_WR)


def render(field: GaussianRadarField, T_RW: Tensor, sensor: SensorConfig) -> dict[str, Tensor]:
    if not torch.cuda.is_available() or field.means.device.type != "cuda":
        raise RuntimeError("the official RadarSplat renderer requires an NVIDIA CUDA GPU")
    scales = torch.exp(field.log_scales)
    opacities = torch.sigmoid(field.opacity_logits)
    noise_probs = torch.sigmoid(field.noise_logits) * 0.0
    zeros = torch.zeros_like(field.reflectance)
    colors = torch.stack((field.reflectance, zeros, zeros), dim=-1)
    with torch.cuda.device(field.means.device):
        outputs = _radar_rasterization(
            means=field.means,
            quats=field.quats,
            scales=scales,
            opacities=opacities,
            noise_probs=noise_probs,
            colors=colors,
            viewmats=torch.linalg.inv(T_RW).unsqueeze(0),
            Ks=sensor.intrinsics(field.means.device).unsqueeze(0),
            width=sensor.range_bins,
            height=sensor.intermediate_azimuth_bins,
            packed=False,
            camera_model="ortho",
            sph=True,
            sh_degree=None,
            near_plane=sensor.near_plane,
            far_plane=sensor.far_plane,
        )
        raw_power, raw_occupancy = outputs[0][0], outputs[1][0]
        power, occupancy = raw_power, raw_occupancy
        if sensor.spectral_leakage_enabled:
            power = spectral_leakage(power, sensor.range_resolution_m, sinc_width=sensor.sinc_width_m)
            occupancy = spectral_leakage(
                occupancy, sensor.range_resolution_m, sinc_width=sensor.sinc_width_m
            )
        power = azimuth_antenna_gain_projection(
            power,
            new_resolution=sensor.output_azimuth_resolution_deg,
            beamwidth=sensor.azimuth_beamwidth_deg,
        )
        occupancy = azimuth_antenna_gain_projection(
            occupancy,
            new_resolution=sensor.output_azimuth_resolution_deg,
            beamwidth=sensor.azimuth_beamwidth_deg,
        )
    radii = outputs[-1]["radii"][0]
    return {
        "power": power,
        "occupancy": occupancy,
        "visible_ids": torch.nonzero(radii > 0, as_tuple=False).squeeze(-1),
    }


def _image_array(tensor: Tensor) -> np.ndarray:
    array = tensor.detach().float().cpu().numpy()
    return array[..., 0] if array.ndim == 3 and array.shape[-1] == 1 else array


def save_outputs(
    config: dict[str, Any], sensor: SensorConfig, result: dict[str, Tensor], output_dir: Path
) -> dict[str, Any]:
    power = np.maximum(_image_array(result["power"]), 0.0)
    occupancy = np.maximum(_image_array(result["occupancy"]), 0.0)
    angles = np.arange(power.shape[0], dtype=np.float32) * sensor.output_azimuth_resolution_deg
    angles = (angles + 180.0) % 360.0 - 180.0
    order = np.argsort(angles)
    angles, power, occupancy = angles[order], power[order], occupancy[order]
    ranges = (np.arange(sensor.range_bins, dtype=np.float32) + 0.5) * sensor.range_resolution_m
    dynamic_range = float(config.get("visualization", {}).get("dynamic_range_db", 45.0))
    peak = max(float(power.max()), np.finfo(np.float32).tiny)
    power_db = 10.0 * np.log10(np.maximum(power / peak, 10.0 ** (-dynamic_range / 10.0)))
    occupancy_norm = occupancy / max(float(occupancy.max()), np.finfo(np.float32).tiny)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    radar_x, radar_y, _ = config["radar"]["position_m"]
    target_x = np.asarray([item["position_m"][0] for item in config["scene"]])
    target_y = np.asarray([item["position_m"][1] for item in config["scene"]])
    sizes = 80.0 + 160.0 * np.asarray([item["reflectance"] for item in config["scene"]])
    axes[0].scatter(target_x, target_y, s=sizes, c="tab:orange", edgecolor="black")
    for item in config["scene"]:
        axes[0].annotate(item["name"], item["position_m"][:2], xytext=(5, 5), textcoords="offset points")
    axes[0].scatter([radar_x], [radar_y], marker="*", s=260, c="tab:blue", edgecolor="black")
    yaw = math.radians(float(config["radar"]["yaw_deg"]))
    axes[0].arrow(radar_x, radar_y, 1.2 * math.cos(yaw), 1.2 * math.sin(yaw), width=0.035,
                  head_width=0.22, color="tab:blue", length_includes_head=True)
    all_x, all_y = np.r_[target_x, radar_x], np.r_[target_y, radar_y]
    margin = max(1.5, 0.12 * max(float(np.ptp(all_x)), float(np.ptp(all_y)), 1.0))
    axes[0].set_xlim(float(all_x.min() - margin), float(all_x.max() + margin))
    axes[0].set_ylim(float(all_y.min() - margin), float(all_y.max() + margin))
    axes[0].set_aspect("equal")
    axes[0].set(xlabel="world x (m)", ylabel="world y (m)", title="Editable Gaussian scene")
    axes[0].grid(alpha=0.25)

    extent = [float(ranges[0]), float(ranges[-1]), float(angles[0]), float(angles[-1])]
    image = axes[1].imshow(power_db, origin="lower", aspect="auto", extent=extent,
                           cmap="magma", vmin=-dynamic_range, vmax=0.0)
    axes[1].set(xlabel="range (m)", ylabel="azimuth (deg)", title="RA power (relative dB)")
    figure.colorbar(image, ax=axes[1], label="dB")
    image = axes[2].imshow(occupancy_norm, origin="lower", aspect="auto", extent=extent,
                           cmap="viridis", vmin=0.0, vmax=1.0)
    axes[2].set(xlabel="range (m)", ylabel="azimuth (deg)", title="RA occupancy (normalized)")
    figure.colorbar(image, ax=axes[2], label="normalized value")
    visible = result["visible_ids"].detach().cpu().tolist()
    figure.suptitle(f"RadarSplat demo | {len(config['scene'])} targets | visible: {len(visible)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "render.png", dpi=170)
    plt.close(figure)
    np.savez_compressed(
        output_dir / "render_arrays.npz",
        power_ra=power,
        occupancy_ra=occupancy,
        azimuth_deg=angles,
        range_m=ranges,
    )
    summary = {
        "target_count": len(config["scene"]),
        "visible_target_indices": visible,
        "visible_target_names": [config["scene"][index]["name"] for index in visible],
        "power_shape": list(power.shape),
        "power_max": float(power.max()),
        "occupancy_max": float(occupancy.max()),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/example.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/example")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        sensor = make_sensor(config)
        field, T_RW = make_scene(config, torch.device(args.device))
        with torch.no_grad():
            result = render(field, T_RW, sensor)
        summary = save_outputs(config, sensor, result, args.output)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {(args.output / 'render.png').resolve()}")


if __name__ == "__main__":
    main()

