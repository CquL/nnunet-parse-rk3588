from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

try:
    from scipy.ndimage import binary_fill_holes, gaussian_filter, map_coordinates
except Exception:
    binary_fill_holes = None
    gaussian_filter = None
    map_coordinates = None

try:
    from skimage.transform import resize as skimage_resize
except Exception:
    skimage_resize = None


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "model_config_parse_current_32x64x64.json"
ANISO_THRESHOLD = 3.0


def nii_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def file_sha256(path: Path, max_bytes: int = 16 * 1024 * 1024) -> str:
    # Avoid hashing huge RKNN files on slow eMMC unless explicitly small enough.
    if not path.exists() or path.stat().st_size > max_bytes:
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_proc_status_kb() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value = line.split(":", 1)
                parts = value.strip().split()
                if parts:
                    result[key] = int(parts[0])
    except Exception:
        pass
    return result


def mem_available_kb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except Exception:
        pass
    return -1


def disk_free_mb(path: Path) -> int:
    try:
        usage = shutil.disk_usage(str(path if path.exists() else path.parent))
        return int(usage.free / (1024 * 1024))
    except Exception:
        return -1


def read_rknpu_load_text() -> str:
    paths = [Path("/sys/kernel/debug/rknpu/load")]
    paths.extend(Path("/sys/kernel/debug").glob("rknpu*/load") if Path("/sys/kernel/debug").exists() else [])
    texts = []
    for path in paths:
        try:
            if path.exists():
                texts.append(f"{path}: {path.read_text(encoding='utf-8', errors='ignore').strip()}")
        except Exception:
            continue
    return "\n".join(texts)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = [int(x) for x in value.lower().replace(",", "x").split("x")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Shape must look like 96x160x160")
    return tuple(parts)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def spacing_zyx_to_xyz(spacing_zyx: tuple[float, float, float]) -> tuple[float, float, float]:
    z, y, x = spacing_zyx
    return (float(x), float(y), float(z))


def compute_new_shape(
    old_shape: tuple[int, int, int],
    old_spacing: tuple[float, float, float],
    new_spacing: tuple[float, float, float],
) -> tuple[int, int, int]:
    return tuple(int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape))


def get_do_separate_z(spacing: tuple[float, ...]) -> bool:
    return (max(spacing) / min(spacing)) > ANISO_THRESHOLD


def get_lowres_axis(spacing: tuple[float, ...]) -> list[int]:
    spacing_arr = np.asarray(spacing, dtype=np.float64)
    return np.where(max(spacing_arr) / spacing_arr == 1)[0].tolist()


def determine_do_sep_z_and_axis(
    force_separate_z: Any,
    current_spacing: tuple[float, ...],
    new_spacing: tuple[float, ...],
) -> tuple[bool, Any]:
    if force_separate_z is not None:
        do_separate_z = bool(force_separate_z)
        axis = get_lowres_axis(current_spacing) if do_separate_z else None
    elif get_do_separate_z(current_spacing):
        do_separate_z = True
        axis = get_lowres_axis(current_spacing)
    elif get_do_separate_z(new_spacing):
        do_separate_z = True
        axis = get_lowres_axis(new_spacing)
    else:
        do_separate_z = False
        axis = None

    if axis is not None:
        if len(axis) in (2, 3):
            return False, None
        return do_separate_z, int(axis[0])
    return do_separate_z, None


def resize_with_coordinates(image: np.ndarray, new_shape: tuple[int, ...], order: int) -> np.ndarray:
    if map_coordinates is None:
        raise RuntimeError("scipy is required for coordinate-map resampling")
    old_shape = image.shape
    if tuple(old_shape) == tuple(new_shape):
        return image
    coord_maps = np.meshgrid(*[np.arange(i, dtype=np.float32) for i in new_shape], indexing="ij")
    for axis, coord in enumerate(coord_maps):
        scale = float(old_shape[axis]) / float(new_shape[axis])
        coord_maps[axis] = scale * (coord + 0.5) - 0.5
    return map_coordinates(image.astype(float, copy=False), np.asarray(coord_maps), order=order, mode="nearest")


def resize_data_like_nnunet(image: np.ndarray, new_shape: tuple[int, ...], order: int) -> np.ndarray:
    if skimage_resize is not None:
        return skimage_resize(image, new_shape, order=order, mode="edge", anti_aliasing=False)
    return resize_with_coordinates(image, new_shape, order)


def resample_data_or_seg_to_shape_official_like(
    data: np.ndarray,
    new_shape: tuple[int, int, int],
    current_spacing: tuple[float, float, float],
    new_spacing: tuple[float, float, float],
    order: int,
    order_z: int,
    force_separate_z: Any,
) -> np.ndarray:
    # Local lightweight implementation of nnUNetv2.preprocessing.resampling.default_resampling.
    # Prefer the same skimage/scipy primitives as nnUNetv2; fall back to coordinate-map resize if needed.
    if data.ndim != 4:
        raise ValueError(f"Expected data shape (c, z, y, x), got {data.shape}")
    shape = tuple(int(i) for i in data.shape[1:])
    new_shape = tuple(int(i) for i in new_shape)
    if shape == new_shape:
        return data

    do_separate_z, axis = determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing)
    dtype_out = data.dtype
    data_float = data.astype(float, copy=False)
    out = np.zeros((data.shape[0], *new_shape), dtype=dtype_out)

    if not do_separate_z:
        for c in range(data.shape[0]):
            out[c] = resize_data_like_nnunet(data_float[c], new_shape, order)
        return out

    if axis is None:
        raise RuntimeError("Separate-z resampling requested without a low-resolution axis")
    if map_coordinates is None:
        raise RuntimeError("scipy is required for separate-z nnUNetv2-like resampling")
    if axis == 0:
        new_shape_2d = (new_shape[1], new_shape[2])
    elif axis == 1:
        new_shape_2d = (new_shape[0], new_shape[2])
    else:
        new_shape_2d = (new_shape[0], new_shape[1])

    for c in range(data.shape[0]):
        tmp_shape = list(new_shape)
        tmp_shape[axis] = shape[axis]
        reshaped_here = np.zeros(tmp_shape, dtype=np.float64)
        for slice_id in range(shape[axis]):
            if axis == 0:
                reshaped_here[slice_id] = resize_data_like_nnunet(data_float[c, slice_id], new_shape_2d, order)
            elif axis == 1:
                reshaped_here[:, slice_id] = resize_data_like_nnunet(data_float[c, :, slice_id], new_shape_2d, order)
            else:
                reshaped_here[:, :, slice_id] = resize_data_like_nnunet(data_float[c, :, :, slice_id], new_shape_2d, order)
        out[c] = resize_with_coordinates(reshaped_here, new_shape, order_z)
    return out


def bounding_box_to_slices(bbox: list[list[int]]) -> tuple[slice, slice, slice]:
    return tuple(slice(int(axis_bbox[0]), int(axis_bbox[1])) for axis_bbox in bbox)


def crop_to_nonzero_official_like(data: np.ndarray) -> tuple[np.ndarray, list[list[int]], tuple[int, int, int]]:
    # Mirrors nnUNetv2 crop_to_nonzero for inference data. For CT this is often the full image,
    # but keeping it here makes bbox restoration match the official export path when zeros are present.
    if binary_fill_holes is None:
        full_bbox = [[0, int(i)] for i in data.shape[1:]]
        return data, full_bbox, tuple(int(i) for i in data.shape[1:])
    nonzero_mask = data[0] != 0
    for c in range(1, data.shape[0]):
        nonzero_mask |= data[c] != 0
    nonzero_mask = binary_fill_holes(nonzero_mask)
    coords = np.argwhere(nonzero_mask)
    if coords.size == 0:
        full_bbox = [[0, int(i)] for i in data.shape[1:]]
        return data, full_bbox, tuple(int(i) for i in data.shape[1:])
    bbox = [[int(coords[:, axis].min()), int(coords[:, axis].max()) + 1] for axis in range(coords.shape[1])]
    slicer = (slice(None), *bounding_box_to_slices(bbox))
    return data[slicer], bbox, tuple(int(i) for i in data.shape[1:])


def resample_image_to_spacing(image: sitk.Image, target_spacing_zyx: tuple[float, float, float]) -> sitk.Image:
    target_spacing_xyz = spacing_zyx_to_xyz(target_spacing_zyx)
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    target_size = [
        max(1, int(round(original_size[i] * original_spacing[i] / target_spacing_xyz[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing_xyz)
    resampler.SetSize(target_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0.0)
    return resampler.Execute(image)


def normalize_ct(image_zyx: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    norm = cfg["normalization"]
    image = image_zyx.astype(np.float32, copy=False)
    image = np.clip(image, float(norm["clip_low"]), float(norm["clip_high"]))
    image = (image - float(norm["mean"])) / float(norm["std"])
    return image.astype(np.float32, copy=False)


def preprocess_image_official_like(
    image: sitk.Image,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    original_spacing_zyx = tuple(float(x) for x in image.GetSpacing()[::-1])
    target_spacing_zyx = tuple(float(x) for x in cfg["target_spacing_zyx"])
    data = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)[None]
    data, bbox, shape_before_cropping = crop_to_nonzero_official_like(data)
    shape_after_cropping = tuple(int(i) for i in data.shape[1:])

    # nnUNetv2 DefaultPreprocessor normalizes before resampling.
    data[0] = normalize_ct(data[0], cfg)
    new_shape = compute_new_shape(shape_after_cropping, original_spacing_zyx, target_spacing_zyx)

    if skimage_resize is None and map_coordinates is None:
        preprocessed_image = resample_image_to_spacing(image, target_spacing_zyx)
        image_zyx = normalize_ct(sitk.GetArrayFromImage(preprocessed_image), cfg)
        return image_zyx, {
            "mode": "simpleitk_fallback",
            "preprocessed_image": preprocessed_image,
            "target_spacing_zyx": target_spacing_zyx,
        }

    data = resample_data_or_seg_to_shape_official_like(
        data,
        new_shape,
        original_spacing_zyx,
        target_spacing_zyx,
        order=3,
        order_z=0,
        force_separate_z=None,
    )
    return data[0].astype(np.float32, copy=False), {
        "mode": "official_like",
        "shape_before_cropping": shape_before_cropping,
        "shape_after_cropping_and_before_resampling": shape_after_cropping,
        "bbox_used_for_cropping": bbox,
        "original_spacing_zyx": original_spacing_zyx,
        "target_spacing_zyx": target_spacing_zyx,
        "resampling_backend": "skimage" if skimage_resize is not None else "scipy_map_coordinates",
    }


def pad_to_minimum(image: np.ndarray, patch_size: tuple[int, int, int]) -> tuple[np.ndarray, tuple[slice, slice, slice]]:
    pads = []
    crop_slices = []
    for axis, (size, patch) in enumerate(zip(image.shape, patch_size)):
        total = max(0, patch - size)
        before = total // 2
        after = total - before
        pads.append((before, after))
        crop_slices.append(slice(before, before + image.shape[axis]))
    return np.pad(image, pads, mode="constant", constant_values=0), tuple(crop_slices)


def compute_steps_for_axis(image_size: int, tile_size: int, tile_step_size: float) -> list[int]:
    if image_size <= tile_size:
        return [0]
    target_step = tile_size * tile_step_size
    num_steps = int(math.ceil((image_size - tile_size) / target_step)) + 1
    actual_step = (image_size - tile_size) / (num_steps - 1)
    return [int(round(actual_step * i)) for i in range(num_steps)]


def compute_sliding_starts(
    image_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    tile_step_size: float,
) -> list[tuple[int, int, int]]:
    per_axis = [
        compute_steps_for_axis(image_shape[i], patch_size[i], tile_step_size)
        for i in range(3)
    ]
    return [tuple(int(x) for x in start) for start in itertools.product(*per_axis)]


def compute_gaussian(
    patch_size: tuple[int, int, int],
    sigma_scale: float = 1.0 / 8.0,
    value_scaling_factor: float = 1.0,
) -> np.ndarray:
    if gaussian_filter is not None:
        tmp = np.zeros(patch_size, dtype=np.float32)
        center_coords = tuple(int(i // 2) for i in patch_size)
        sigmas = [float(i) * sigma_scale for i in patch_size]
        tmp[center_coords] = 1.0
        weight = gaussian_filter(tmp, sigmas, 0, mode="constant", cval=0)
    else:
        axes = []
        for dim in patch_size:
            center = dim // 2
            sigma = max(dim * sigma_scale, 1e-6)
            coords = np.arange(dim, dtype=np.float32)
            axes.append(np.exp(-0.5 * ((coords - center) / sigma) ** 2))
        weight = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    weight = weight.astype(np.float32, copy=False)
    weight /= float(weight.max() / value_scaling_factor)
    nonzero = weight[weight > 0]
    if nonzero.size:
        weight[weight == 0] = float(nonzero.min())
    return weight


def read_device_compatible() -> str:
    try:
        return (
            Path("/proc/device-tree/compatible")
            .read_bytes()
            .replace(b"\x00", b"\n")
            .decode("utf-8", errors="ignore")
            .lower()
        )
    except Exception:
        return ""


def select_npu_core_mask(rknn_lite_cls: Any, requested: str) -> tuple[Any, str]:
    requested = requested.lower().replace("-", "_")
    explicit = {
        "auto": "NPU_CORE_AUTO",
        "all": "NPU_CORE_ALL",
        "0": "NPU_CORE_0",
        "1": "NPU_CORE_1",
        "2": "NPU_CORE_2",
        "0_1": "NPU_CORE_0_1",
        "0_1_2": "NPU_CORE_0_1_2",
    }
    if requested != "auto_detect":
        attr = explicit.get(requested)
        if attr is None:
            raise ValueError(f"Unsupported --core-mask {requested!r}; use auto-detect, auto, all, 0, 1, 2, 0_1, or 0_1_2")
        if not hasattr(rknn_lite_cls, attr):
            raise ValueError(f"RKNNLite on this device does not expose {attr}")
        return int(getattr(rknn_lite_cls, attr)), attr

    compatible = read_device_compatible()
    if "rk3588" in compatible:
        candidates = ["NPU_CORE_0_1_2", "NPU_CORE_ALL", "NPU_CORE_AUTO"]
    elif "rk356" in compatible:
        candidates = ["NPU_CORE_0", "NPU_CORE_AUTO"]
    else:
        candidates = ["NPU_CORE_AUTO", "NPU_CORE_ALL", "NPU_CORE_0_1_2", "NPU_CORE_0"]

    for attr in candidates:
        if hasattr(rknn_lite_cls, attr):
            return int(getattr(rknn_lite_cls, attr)), attr
    return None, "default"


def load_rknn(model_path: Path, core_mask: str = "auto-detect"):
    try:
        from rknnlite.api import RKNNLite
    except Exception as exc:
        raise RuntimeError(
            "Cannot import rknnlite.api.RKNNLite. Install rknn-toolkit-lite2 "
            "and a matching librknnrt.so on the RK3588 device first."
        ) from exc

    rknn = RKNNLite()
    load_start = time.perf_counter()
    ret = rknn.load_rknn(str(model_path))
    load_ms = (time.perf_counter() - load_start) * 1000.0
    if ret != 0:
        raise RuntimeError(f"load_rknn failed, ret={ret}")

    selected_mask, selected_name = select_npu_core_mask(RKNNLite, core_mask)
    init_start = time.perf_counter()
    if selected_mask is None:
        print("NPU core mask: default init_runtime()", flush=True)
        ret = rknn.init_runtime()
    else:
        print(f"NPU core mask: {selected_name}={selected_mask}", flush=True)
        ret = rknn.init_runtime(core_mask=selected_mask)
        if ret != 0 and selected_name not in ("NPU_CORE_AUTO", "default") and hasattr(RKNNLite, "NPU_CORE_AUTO"):
            print(f"init_runtime with {selected_name} failed, retrying NPU_CORE_AUTO", flush=True)
            ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
            selected_name = "NPU_CORE_AUTO"
            selected_mask = int(RKNNLite.NPU_CORE_AUTO)
    init_ms = (time.perf_counter() - init_start) * 1000.0
    if ret != 0:
        raise RuntimeError(f"init_runtime failed, ret={ret}")
    return rknn, {
        "core_mask_requested": core_mask,
        "core_mask_selected_name": selected_name,
        "core_mask_selected_value": selected_mask,
        "load_rknn_ms": round(load_ms, 3),
        "init_runtime_ms": round(init_ms, 3),
    }


def output_to_logits(output: np.ndarray, patch_size: tuple[int, int, int], num_classes: int) -> np.ndarray:
    expected = (1, num_classes, *patch_size)
    output = np.asarray(output)
    if output.shape == expected:
        return output[0].astype(np.float32, copy=False)
    if output.size == int(np.prod(expected)):
        return output.reshape(expected)[0].astype(np.float32, copy=False)
    raise ValueError(f"Unexpected RKNN output shape: {output.shape}, expected {expected}")


def infer_patch(
    rknn: Any,
    patch_zyx: np.ndarray,
    patch_size: tuple[int, int, int],
    num_classes: int,
) -> np.ndarray:
    input_tensor = patch_zyx[None, None].astype(np.float32, copy=False)
    outputs = rknn.inference(inputs=[input_tensor])
    if outputs is None or len(outputs) == 0:
        raise RuntimeError("RKNN inference returned no outputs")
    return output_to_logits(outputs[0], patch_size, num_classes)


def tta_flip_axes(enabled: bool) -> list[tuple[int, ...]]:
    if not enabled:
        return [()]
    axes = (0, 1, 2)
    combos: list[tuple[int, ...]] = [()]
    for length in range(1, len(axes) + 1):
        combos.extend(itertools.combinations(axes, length))
    return combos


def infer_patch_with_tta(
    rknn: Any,
    patch_zyx: np.ndarray,
    patch_size: tuple[int, int, int],
    num_classes: int,
    use_tta: bool,
) -> np.ndarray:
    logits_sum = None
    axes_combos = tta_flip_axes(use_tta)
    for axes in axes_combos:
        patch = np.flip(patch_zyx, axis=axes).copy() if axes else patch_zyx
        logits = infer_patch(rknn, patch, patch_size, num_classes)
        if axes:
            logits = np.flip(logits, axis=tuple(axis + 1 for axis in axes)).copy()
        logits_sum = logits if logits_sum is None else logits_sum + logits
    return logits_sum / float(len(axes_combos))


def run_sliding_window(
    rknn: Any,
    image_zyx: np.ndarray,
    patch_size: tuple[int, int, int],
    num_classes: int,
    tile_step_size: float,
    use_gaussian: bool,
    use_tta: bool,
) -> np.ndarray:
    padded, crop_slices = pad_to_minimum(image_zyx, patch_size)
    starts = compute_sliding_starts(padded.shape, patch_size, tile_step_size)
    weight = compute_gaussian(patch_size) if use_gaussian else np.ones(patch_size, dtype=np.float32)

    score_sum = np.zeros((num_classes, *padded.shape), dtype=np.float32)
    weight_sum = np.zeros(padded.shape, dtype=np.float32)
    dz, dy, dx = patch_size

    print(
        f"Preprocessed shape zyx={image_zyx.shape}, padded zyx={padded.shape}, "
        f"patches={len(starts)}, patch={patch_size}, step={tile_step_size}, "
        f"gaussian={use_gaussian}, tta={use_tta}",
        flush=True,
    )

    patch_times_ms: list[float] = []
    for index, (z, y, x) in enumerate(starts, start=1):
        patch = padded[z : z + dz, y : y + dy, x : x + dx]
        patch_start = time.perf_counter()
        logits = infer_patch_with_tta(rknn, patch, patch_size, num_classes, use_tta)
        patch_times_ms.append((time.perf_counter() - patch_start) * 1000.0)
        score_sum[:, z : z + dz, y : y + dy, x : x + dx] += logits * weight[None]
        weight_sum[z : z + dz, y : y + dy, x : x + dx] += weight
        if index == 1 or index % 25 == 0 or index == len(starts):
            print(f"Processed patches: {index}/{len(starts)}", flush=True)

    score_sum /= np.maximum(weight_sum, 1e-6)[None]
    times = np.asarray(patch_times_ms, dtype=np.float64)
    elapsed_ms = float(times.sum()) if times.size else 0.0
    stats = {
        "preprocessed_shape_zyx": list(image_zyx.shape),
        "padded_shape_zyx": list(padded.shape),
        "patch_size_zyx": list(patch_size),
        "num_patches": len(starts),
        "tta_multiplier": len(tta_flip_axes(use_tta)),
        "patch_windows": len(starts),
        "npu_forward_calls_estimated": len(starts) * len(tta_flip_axes(use_tta)),
        "patch_infer_ms_total": round(elapsed_ms, 3),
        "patch_infer_ms_mean": round(float(times.mean()), 3) if times.size else 0.0,
        "patch_infer_ms_p50": round(float(np.percentile(times, 50)), 3) if times.size else 0.0,
        "patch_infer_ms_p95": round(float(np.percentile(times, 95)), 3) if times.size else 0.0,
        "patch_throughput_patches_per_sec": round(len(starts) / (elapsed_ms / 1000.0), 6) if elapsed_ms > 0 else 0.0,
    }
    return score_sum[(slice(None), *crop_slices)], stats


def logits_to_original_mask(
    logits_czyx: np.ndarray,
    preprocess_info: dict[str, Any],
    reference_image: sitk.Image,
    resample_logits: bool,
) -> np.ndarray:
    if preprocess_info.get("mode") == "official_like":
        bbox = preprocess_info["bbox_used_for_cropping"]
        shape_after_cropping = tuple(int(i) for i in preprocess_info["shape_after_cropping_and_before_resampling"])
        shape_before_cropping = tuple(int(i) for i in preprocess_info["shape_before_cropping"])
        target_spacing = tuple(float(i) for i in preprocess_info["target_spacing_zyx"])
        original_spacing = tuple(float(i) for i in preprocess_info["original_spacing_zyx"])

        if resample_logits:
            logits_czyx = resample_data_or_seg_to_shape_official_like(
                logits_czyx.astype(np.float32, copy=False),
                shape_after_cropping,
                target_spacing,
                original_spacing,
                order=1,
                order_z=0,
                force_separate_z=None,
            )
            cropped_mask = np.argmax(logits_czyx, axis=0).astype(np.uint8)
        else:
            cropped_mask = np.argmax(logits_czyx, axis=0).astype(np.uint8)
            cropped_mask = resample_data_or_seg_to_shape_official_like(
                cropped_mask[None],
                shape_after_cropping,
                target_spacing,
                original_spacing,
                order=0,
                order_z=0,
                force_separate_z=None,
            )[0].astype(np.uint8)

        mask = np.zeros(shape_before_cropping, dtype=np.uint8)
        mask[bounding_box_to_slices(bbox)] = cropped_mask
        return mask

    preprocessed_image = preprocess_info["preprocessed_image"]
    if not resample_logits:
        mask = np.argmax(logits_czyx, axis=0).astype(np.uint8)
        mask_image = sitk.GetImageFromArray(mask)
        mask_image.CopyInformation(preprocessed_image)
        mask_image = sitk.Resample(
            mask_image,
            reference_image,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        return sitk.GetArrayFromImage(mask_image).astype(np.uint8, copy=False)

    resampled_logits = []
    for class_index in range(logits_czyx.shape[0]):
        logit_image = sitk.GetImageFromArray(logits_czyx[class_index].astype(np.float32, copy=False))
        logit_image.CopyInformation(preprocessed_image)
        resampled = sitk.Resample(
            logit_image,
            reference_image,
            sitk.Transform(),
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32,
        )
        resampled_logits.append(sitk.GetArrayFromImage(resampled))
    return np.argmax(np.stack(resampled_logits, axis=0), axis=0).astype(np.uint8)


def write_mask(mask_zyx: np.ndarray, reference: sitk.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_image = sitk.GetImageFromArray(mask_zyx.astype(np.uint8, copy=False))
    mask_image.CopyInformation(reference)
    sitk.WriteImage(mask_image, str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RKNN inference with nnUNetv2-like resampling, CT normalization, Gaussian sliding window, and optional TTA."
    )
    parser.add_argument("-i", "--input", required=True, help="Input .nii.gz CT volume")
    parser.add_argument("-o", "--output", required=True, help="Output .nii.gz segmentation mask")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Deployment JSON config")
    parser.add_argument("--model", default=None, help="Override RKNN model path")
    parser.add_argument("--patch-size", type=parse_shape, default=None, help="Override RKNN patch size, e.g. 96x160x160")
    parser.add_argument("--tile-step-size", type=float, default=None, help="Override tile step fraction; nnUNetv2 default is 0.5")
    parser.add_argument("--no-gaussian", action="store_true", help="Disable Gaussian sliding-window weighting")
    parser.add_argument("--tta", action="store_true", help="Enable mirror TTA over z/y/x axes; much slower")
    parser.add_argument("--no-tta", action="store_true", help="Disable mirror TTA even if the config enables it")
    parser.add_argument("--core-mask", default="auto-detect", help="NPU core mask: auto-detect, auto, all, 0, 1, 2, 0_1, or 0_1_2")
    parser.add_argument("--argmax-before-resample", action="store_true", help="Resample mask instead of class logits")
    parser.add_argument("--metrics-jsonl", default=None, help="Append one structured inference metrics row to this JSONL file")
    args = parser.parse_args()
    total_start = time.perf_counter()
    started_at = now_iso()
    metrics: dict[str, Any] = {
        "timestamp_start_iso": started_at,
        "status": "running",
        "input_path": args.input,
        "output_path": args.output,
        "rss_kb_start": read_proc_status_kb().get("VmRSS", -1),
        "mem_available_kb_start": mem_available_kb(),
        "disk_free_mb_start": disk_free_mb(Path(args.output)),
        "rknpu_load_start": read_rknpu_load_text(),
    }

    config_path = Path(args.config)
    cfg = load_config(config_path)
    patch_size = args.patch_size or tuple(int(x) for x in cfg["rknn_patch_size_zyx"])
    target_spacing = tuple(float(x) for x in cfg["target_spacing_zyx"])
    tile_step_size = float(args.tile_step_size if args.tile_step_size is not None else cfg.get("tile_step_size", 0.5))
    use_gaussian = bool(cfg.get("use_gaussian", True)) and not args.no_gaussian
    use_tta = (bool(cfg.get("tta", False)) or args.tta) and not args.no_tta
    resample_logits = bool(cfg.get("resample_logits_to_original", True)) and not args.argmax_before_resample
    num_classes = int(cfg["num_classes"])

    model_path = Path(args.model) if args.model else config_path.parent / str(cfg["model_file"])
    if not model_path.exists():
        raise FileNotFoundError(f"RKNN model not found: {model_path}")
    metrics.update(
        {
            "case_id": nii_stem(Path(args.output)),
            "model_path": str(model_path),
            "model_size_bytes": model_path.stat().st_size,
            "model_sha256_if_small": file_sha256(model_path),
            "config_path": str(config_path),
            "config_sha256": file_sha256(config_path),
            "tile_step_size": tile_step_size,
            "gaussian_enabled": use_gaussian,
            "tta_enabled": use_tta,
            "resample_logits_to_original": resample_logits,
            "num_classes": num_classes,
        }
    )

    preprocess_start = time.perf_counter()
    reference_image = sitk.ReadImage(args.input)
    image_zyx, preprocess_info = preprocess_image_official_like(reference_image, cfg)
    metrics["preprocess_ms"] = round((time.perf_counter() - preprocess_start) * 1000.0, 3)
    metrics.update(
        {
            "input_size_xyz": list(reference_image.GetSize()),
            "spacing_xyz": [float(i) for i in reference_image.GetSpacing()],
            "input_shape_zyx": list(sitk.GetArrayViewFromImage(reference_image).shape),
            "target_spacing_zyx": list(target_spacing),
            "preprocess_mode": preprocess_info.get("mode"),
            "resampling_backend": preprocess_info.get("resampling_backend", ""),
        }
    )

    print(f"Input size xyz={reference_image.GetSize()}, spacing xyz={reference_image.GetSpacing()}")
    print(f"Target shape zyx={image_zyx.shape}, spacing zyx={target_spacing}")
    print(f"Preprocess mode={preprocess_info.get('mode')}")
    if "resampling_backend" in preprocess_info:
        print(f"Resampling backend={preprocess_info['resampling_backend']}")
    print(f"RKNN model={model_path}")
    print(f"Config={config_path}")

    rknn, rknn_metrics = load_rknn(model_path, args.core_mask)
    metrics.update(rknn_metrics)
    try:
        sliding_start = time.perf_counter()
        logits, sliding_metrics = run_sliding_window(
            rknn,
            image_zyx,
            patch_size,
            num_classes,
            tile_step_size,
            use_gaussian,
            use_tta,
        )
        metrics["sliding_window_ms"] = round((time.perf_counter() - sliding_start) * 1000.0, 3)
        metrics.update(sliding_metrics)
    finally:
        rknn.release()

    postprocess_start = time.perf_counter()
    mask = logits_to_original_mask(logits, preprocess_info, reference_image, resample_logits)
    metrics["postprocess_ms"] = round((time.perf_counter() - postprocess_start) * 1000.0, 3)
    write_start = time.perf_counter()
    write_mask(mask, reference_image, Path(args.output))
    metrics["write_output_ms"] = round((time.perf_counter() - write_start) * 1000.0, 3)
    labels, counts = np.unique(mask, return_counts=True)
    label_counts = dict(zip([str(i) for i in labels.tolist()], [int(i) for i in counts.tolist()]))
    print(f"Output labels: {label_counts}")
    print(f"Output written: {args.output}")
    proc_status = read_proc_status_kb()
    metrics.update(
        {
            "timestamp_end_iso": now_iso(),
            "status": "ok",
            "total_ms": round((time.perf_counter() - total_start) * 1000.0, 3),
            "rss_kb_end": proc_status.get("VmRSS", -1),
            "rss_kb_peak": proc_status.get("VmHWM", -1),
            "mem_available_kb_end": mem_available_kb(),
            "disk_free_mb_end": disk_free_mb(Path(args.output)),
            "rknpu_load_end": read_rknpu_load_text(),
            "output_label_counts": label_counts,
        }
    )
    if args.metrics_jsonl:
        append_jsonl(Path(args.metrics_jsonl), metrics)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        raise
