#!/usr/bin/env python3
"""Export the six reference-point panels for a Spiideo SoccerNet SynLoc paper.

Run from the Player Location project, with its saved YOLO, SAM and Halpe26
pose caches. Use the same source frames as the caches and the original
SynLoc split JSON. The second annotated keypoint is the GT ground point.

    python -m pip install numpy scipy matplotlib pillow
    python generate_synloc_fig2.py --dataset-root /path/to/spiideo

The three panels within each row share a crop. The rows are magnified
independently, so panel (a) reports each box height in source-image pixels.
Use --help for custom paths, explicit annotation IDs and SVG export.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


HIPS = (11, 12)
FEET = (20, 21, 22, 23, 24, 25)  # Halpe26 big toes, small toes and heels.
JOINT_LINKS = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16))
BOX_COLOR = "#6ee7ff"
MASK_COLOR = "#38bdec"
BAND_COLOR = "#ffbe51"
GT_COLOR = "#ff80c5"
INK = "#152022"
PANEL_SIZE = (2.20, 2.80)
PHOTO_RECT = (0.035, 0.145, 0.93, 0.780)


@dataclass
class MaskDetail:
    origin: tuple[int, int]
    cropped: np.ndarray
    component: np.ndarray
    band: np.ndarray
    point: tuple[float, float]


@dataclass
class Sample:
    image_id: int
    annotation_id: int
    detection_index: int
    image_path: Path
    image_size: tuple[int, int]  # width, height
    box: np.ndarray
    score: float
    iou: float
    distance: float
    gt: np.ndarray
    mask: MaskDetail
    joints: np.ndarray
    visible: np.ndarray

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--annotations", type=Path, help="Override the original SynLoc split JSON path.")
    parser.add_argument("--images-dir", type=Path, help="Override the split image directory.")
    parser.add_argument("--yolo-cache-root", type=Path, default=Path("output/synloc_yolo_cache"))
    parser.add_argument("--sam-cache-root", type=Path, default=Path("output/synloc_sam_cache"))
    parser.add_argument("--pose-cache-root", type=Path, default=Path("output/synloc_pose_cache"))
    parser.add_argument("--pose-model", choices=("halpe26_m", "halpe26_x_384"), default="halpe26_m")
    parser.add_argument("--min-large-height", type=float, default=70.0, help="Source-image pixels, inclusive.")
    parser.add_argument("--max-small-height", type=float, default=40.0, help="Source-image pixels, inclusive.")
    parser.add_argument("--min-iou", type=float, default=0.50)
    parser.add_argument("--min-score", type=float, default=0.4302865565)
    parser.add_argument("--keypoint-conf", type=float, default=0.30)
    parser.add_argument("--large-ann-id", type=int, help="SynLoc annotation ID for row 1. Supply both IDs.")
    parser.add_argument("--small-ann-id", type=int, help="SynLoc annotation ID for row 2. Supply both IDs.")
    parser.add_argument("--output-dir", type=Path, default=Path("output/figures/synloc_fig2"))
    parser.add_argument("--format", choices=("png", "svg"), default="png")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if (args.large_ann_id is None) != (args.small_ann_id is None):
        parser.error("Supply --large-ann-id and --small-ann-id together.")
    if args.large_ann_id is not None and args.large_ann_id == args.small_ann_id:
        parser.error("The two rows require different annotation IDs.")
    if not 0 < args.max_small_height < args.min_large_height < float("inf"):
        parser.error("Require 0 < --max-small-height < --min-large-height.")
    if not all(0 <= value <= 1 for value in (args.min_iou, args.min_score, args.keypoint_conf)):
        parser.error("IoU and confidence thresholds must be in [0, 1].")
    if args.dpi <= 0:
        parser.error("--dpi must be positive.")
    return args


def dataset_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    root = args.dataset_root.expanduser()
    roots = (root, root / "SpiideoSynLoc")
    splits = ("val", "valid", "validation") if args.split == "val" else (args.split,)
    json_candidates = [r / "annotations" / f"{s}.json" for r in roots for s in splits]
    image_candidates = [p for r in roots for s in splits for p in (r / "images" / s, r / s)]
    annotation_path = args.annotations or next((p for p in json_candidates if p.is_file()), None)
    image_dir = args.images_dir or next((p for p in image_candidates if p.is_dir()), None)
    if annotation_path is None or not annotation_path.expanduser().is_file():
        raise FileNotFoundError("SynLoc split JSON missing. Pass its path with --annotations.")
    if image_dir is None or not image_dir.expanduser().is_dir():
        raise FileNotFoundError("SynLoc image directory missing. Pass its path with --images-dir.")
    return annotation_path.expanduser(), image_dir.expanduser()


def cache_filename(info: dict) -> str:
    return f"{int(info['id']):06d}_{Path(info['file_name']).stem}.npz"


def bbox_matches(boxes: np.ndarray, annotations: list[dict], scale: np.ndarray) -> list[tuple[int, int, float]]:
    """One-to-one assignment by IoU in source-image coordinates."""
    if len(boxes) == 0 or len(annotations) == 0:
        return []
    truth = np.array([ann["bbox"] for ann in annotations], dtype=np.float64)
    truth[:, 2:] += truth[:, :2]  # COCO xywh -> xyxy.
    truth *= np.tile(scale, 2)
    top_left = np.maximum(boxes[:, None, :2], truth[None, :, :2])
    bottom_right = np.minimum(boxes[:, None, 2:], truth[None, :, 2:])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0.0), axis=2)
    box_area = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0.0), axis=1)
    gt_area = np.prod(np.maximum(truth[:, 2:] - truth[:, :2], 0.0), axis=1)
    union = box_area[:, None] + gt_area[None, :] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    det_indices, gt_indices = linear_sum_assignment(1.0 - iou)
    return [(int(d), int(g), float(iou[d, g])) for d, g in zip(det_indices, gt_indices)]


def extract_mask(mask: np.ndarray, box: np.ndarray) -> MaskDetail | None:
    x1, y1, x2, y2 = map(float, box)
    height, width = mask.shape
    ix1 = max(0, min(width - 1, int(np.floor(x1))))
    iy1 = max(0, min(height - 1, int(np.floor(y1))))
    ix2 = max(ix1 + 1, min(width, int(np.ceil(x2))))
    iy2 = max(iy1 + 1, min(height, int(np.ceil(y2))))
    raw = mask[iy1:iy2, ix1:ix2] > 0
    labels, count = ndimage.label(raw, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return None
    sizes = np.bincount(labels[raw], minlength=count + 1)
    if np.count_nonzero(sizes[1:] == sizes[1:].max()) > 1:
        # Use the evaluator's OpenCV label order for the equal-area case.
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("Install opencv-python for a mask with tied largest components.") from error
        _, labels, stats, _ = cv2.connectedComponentsWithStats(raw.astype(np.uint8), connectivity=8)
        sizes = np.r_[0, stats[1:, cv2.CC_STAT_AREA]]
    component = labels == int(np.argmax(sizes))
    rows, columns = np.nonzero(component)
    v_s = float(np.percentile(rows + iy1, 98.0))
    thickness = max(0.05 * (y2 - y1), 1.0)
    in_band = rows + iy1 >= v_s - thickness
    band = np.zeros_like(component)
    band[rows[in_band], columns[in_band]] = True
    u_s = float(np.median(columns[in_band] + ix1))
    return MaskDetail((ix1, iy1), raw, component, band, (u_s, v_s))


def verify_cache(cache: np.lib.npyio.NpzFile, path: Path, image_id: int, boxes: np.ndarray, scores: np.ndarray) -> None:
    if (
        int(cache["image_id"].item()) != image_id
        or not np.array_equal(np.asarray(cache["bboxes"], dtype=np.float32), boxes)
        or not np.array_equal(np.asarray(cache["detection_scores"], dtype=np.float32), scores)
    ):
        raise ValueError(f"Cache detection order or values differ from the YOLO cache for {path}.")


def frame_samples(
    info: dict,
    annotations: list[dict],
    image_dir: Path,
    cache_dirs: tuple[Path, Path, Path],
    args: argparse.Namespace,
    pinned_ids: set[int] | None = None,
) -> list[Sample]:
    name = cache_filename(info)
    yolo_path, sam_path, pose_path = (p / name for p in cache_dirs)
    image_id = int(info["id"])
    with np.load(yolo_path, allow_pickle=False) as cache:
        boxes = np.asarray(cache["bboxes"], dtype=np.float32)
        scores = np.asarray(cache["detection_scores"], dtype=np.float32)
        width, height = (int(cache[k].item()) for k in ("actual_width", "actual_height"))
        ann_width, ann_height = (int(cache[k].item()) for k in ("annotation_width", "annotation_height"))
        file_name = str(cache["file_name"].item())
        if int(cache["image_id"].item()) != image_id or file_name != info["file_name"]:
            raise ValueError(f"YOLO cache does not describe the annotation frame for {yolo_path}.")
    if min(width, height, ann_width, ann_height) <= 0 or (ann_width, ann_height) != (info["width"], info["height"]):
        raise ValueError(f"YOLO cache and annotation dimensions disagree for {yolo_path}.")
    if scores.ndim != 1 or boxes.shape != (len(scores), 4):
        raise ValueError(f"Unexpected YOLO arrays for {yolo_path}.")
    if not (np.isfinite(boxes).all() and np.isfinite(scores).all()):
        raise ValueError(f"Non-finite box coordinates or scores in {yolo_path}.")

    scale = np.array([width / ann_width, height / ann_height])
    eligible = []
    for det_index, gt_index, iou in bbox_matches(boxes, annotations, scale):
        ann = annotations[gt_index]
        x1, y1, x2, y2 = map(float, boxes[det_index])
        box_height = y2 - y1
        if iou < args.min_iou or scores[det_index] < args.min_score or min(x2 - x1, box_height) <= 0:
            continue
        if pinned_ids is not None:
            if int(ann["id"]) not in pinned_ids:
                continue
        elif args.max_small_height < box_height < args.min_large_height:
            continue
        # The original SynLoc annotation has [pelvis, projected pelvis].
        keypoints = np.asarray(ann.get("keypoints", []), dtype=np.float64).reshape(-1, 3)
        if keypoints.shape != (2, 3) or keypoints[1, 2] <= 0 or not np.isfinite(keypoints[1]).all():
            continue
        eligible.append((det_index, ann, iou, keypoints[1, :2] * scale))
    if not eligible:
        return []

    with np.load(pose_path, allow_pickle=False) as cache:
        verify_cache(cache, pose_path, image_id, boxes, scores)
        joints = np.asarray(cache["keypoints"], dtype=np.float32)
        joint_scores = np.asarray(cache["keypoint_scores"], dtype=np.float32)
    if joints.shape != (len(boxes), 26, 2) or joint_scores.shape != (len(boxes), 26):
        raise ValueError(f"Expected Halpe26 keypoints and scores in {pose_path}.")
    visible = (joint_scores >= args.keypoint_conf) & np.isfinite(joint_scores) & np.isfinite(joints).all(axis=2)
    eligible = [item for item in eligible if visible[item[0], HIPS].all() and visible[item[0], FEET].any()]
    if not eligible:
        return []

    with np.load(sam_path, allow_pickle=False) as cache:
        verify_cache(cache, sam_path, image_id, boxes, scores)
        masks = cache["masks"]
        mask_valid = np.asarray(cache["mask_valid"], dtype=bool) if "mask_valid" in cache else np.ones(len(boxes), dtype=bool)
    if masks.shape != (len(boxes), height, width) or mask_valid.shape != (len(boxes),):
        raise ValueError(f"Mask cache dimensions disagree with the YOLO cache for {sam_path}.")

    image_path = image_dir / file_name
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing source image {image_path}. Check --images-dir.")
    camera = np.asarray(info["camera_matrix"], dtype=np.float64)
    camera_center = np.linalg.solve(camera[:, :3], -camera[:, 3])
    samples = []
    for det_index, ann, iou, gt in eligible:
        if not mask_valid[det_index]:
            continue
        mask = extract_mask(masks[det_index], boxes[det_index])
        if mask is None:
            continue
        pitch_point = np.r_[ann["position_on_pitch"][:2], 0.0]
        samples.append(Sample(
            image_id=image_id, annotation_id=int(ann["id"]), detection_index=det_index,
            image_path=image_path, image_size=(width, height), box=boxes[det_index],
            score=float(scores[det_index]), iou=iou, distance=float(np.linalg.norm(pitch_point - camera_center)),
            gt=gt, mask=mask, joints=joints[det_index], visible=visible[det_index],
        ))
    return samples


def select_samples(dataset: dict, image_dir: Path, args: argparse.Namespace) -> tuple[Sample, Sample]:
    cache_dirs = (
        args.yolo_cache_root.expanduser() / args.split / "yolo_raw",
        args.sam_cache_root.expanduser() / args.split / "sam_raw",
        args.pose_cache_root.expanduser() / args.split / args.pose_model / "pose_raw",
    )
    for directory in cache_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(f"Cache directory missing: {directory}")
    common = set.intersection(*({p.name for p in directory.glob("*.npz")} for directory in cache_dirs))
    if not common:
        raise ValueError("No frame has all three caches. Check the split, pose model and cache roots.")
    annotations = defaultdict(list)
    annotation_frames = {}
    for ann in dataset["annotations"]:
        if int(ann.get("category_id", 1)) == 1 and not ann.get("iscrowd", 0):
            annotations[int(ann["image_id"])].append(ann)
            annotation_frames[int(ann["id"])] = int(ann["image_id"])
    frames = {int(info["id"]): info for info in dataset["images"]}

    if args.large_ann_id is not None:
        requested = (args.large_ann_id, args.small_ann_id)
        missing = set(requested) - annotation_frames.keys()
        if missing:
            raise ValueError(f"Player annotation IDs absent from the split: {sorted(missing)}")
        selected = {}
        for image_id in sorted({annotation_frames[ann_id] for ann_id in requested}):
            info = frames[image_id]
            if cache_filename(info) not in common:
                raise ValueError(f"Frame {image_id} is missing a YOLO, SAM or {args.pose_model} cache.")
            for sample in frame_samples(info, annotations[image_id], image_dir, cache_dirs, args, set(requested)):
                selected[sample.annotation_id] = sample
        missing = set(requested) - selected.keys()
        if missing:
            raise ValueError(f"IDs {sorted(missing)} lack an IoU-matched detection, valid GT or SAM mask, or confident hips/feet.")
        return selected[requested[0]], selected[requested[1]]

    nearest_large = farthest_small = None
    for image_id, info in sorted(frames.items()):
        if cache_filename(info) not in common:
            continue
        samples = frame_samples(info, annotations[image_id], image_dir, cache_dirs, args)
        large = min((s for s in samples if s.height >= args.min_large_height), key=lambda s: s.distance, default=None)
        small = max((s for s in samples if s.height <= args.max_small_height), key=lambda s: s.distance, default=None)
        if large is not None and small is not None and small.distance > large.distance:
            return large, small
        if large is not None and (nearest_large is None or large.distance < nearest_large.distance):
            nearest_large = large
        if small is not None and (farthest_small is None or small.distance > farthest_small.distance):
            farthest_small = small
        if nearest_large is not None and farthest_small is not None and farthest_small.distance > nearest_large.distance:
            return nearest_large, farthest_small
    raise ValueError(
        "No large/small pair with valid masks, pose and GT was found in the common caches. "
        "Adjust --min-large-height and --max-small-height or supply both annotation IDs."
    )


def crop_for_row(sample: Sample) -> tuple[np.ndarray, tuple[int, int], tuple[float, float, float, float]]:
    x1, y1, x2, y2 = sample.box
    shown = np.array(HIPS + FEET + (13, 14, 15, 16))
    points = np.vstack(([x1, y1], [x2, y2], sample.gt, sample.joints[shown[sample.visible[shown]]]))
    pad = max(5.0, 0.17 * sample.height)
    left, top = points.min(axis=0) - pad
    right, bottom = points.max(axis=0) + pad
    ratio = (PANEL_SIZE[0] * PHOTO_RECT[2]) / (PANEL_SIZE[1] * PHOTO_RECT[3])
    extra_x = max(0.0, (bottom - top) * ratio - (right - left)) / 2
    extra_y = max(0.0, (right - left) / ratio - (bottom - top)) / 2
    bounds = left - extra_x, top - extra_y, right + extra_x, bottom + extra_y
    width, height = sample.image_size
    ix1, iy1 = max(0, int(np.floor(bounds[0]))), max(0, int(np.floor(bounds[1])))
    ix2, iy2 = min(width, int(np.ceil(bounds[2])) + 1), min(height, int(np.ceil(bounds[3])) + 1)
    with Image.open(sample.image_path) as image:
        if image.size != sample.image_size:
            raise ValueError(f"{sample.image_path} is {image.size}, but the caches use {sample.image_size}. Use the matching image resolution.")
        pixels = np.asarray(image.crop((ix1, iy1, ix2, iy2)).convert("RGB"))
    return pixels, (ix1, iy1), bounds


def pixel_extent(origin: tuple[int, int], shape: tuple[int, ...]) -> tuple[float, float, float, float]:
    x, y = origin
    height, width = shape[:2]
    return x - 0.5, x + width - 0.5, y + height - 0.5, y - 0.5


def point_label(ax, text: str, point: tuple[float, float], offset: tuple[float, float], **kwargs) -> None:
    ax.annotate(
        text, xy=point, xytext=offset, textcoords="offset points", fontsize=8.2, color="white",
        path_effects=[effects.withStroke(linewidth=2, foreground=INK)], zorder=12, **kwargs,
    )


def draw_box(ax, sample: Sample, faint: bool = False) -> None:
    x1, y1, x2, y2 = map(float, sample.box)
    patch = Rectangle(
        (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white" if faint else BOX_COLOR,
        linewidth=0.65 if faint else 1.25, alpha=0.55 if faint else 1.0, zorder=6,
    )
    if not faint:
        patch.set_path_effects([effects.withStroke(linewidth=2.25, foreground=INK)])
    ax.add_patch(patch)


def ring(ax, point: tuple[float, float], color: str) -> None:
    ax.plot(*point, marker="o", ms=8.0, mfc="none", mec=color, mew=1.35, ls="none", zorder=9,
            path_effects=[effects.withStroke(linewidth=2.1, foreground=INK)])


def panel_box(ax, sample: Sample) -> list:
    x1, _, x2, y2 = map(float, sample.box)
    p_b = ((x1 + x2) / 2, y2)
    draw_box(ax, sample)
    ring(ax, p_b, BOX_COLOR)
    point_label(ax, r"$(u_B,v_B)$", p_b, (0, -8), ha="center", va="top")
    return []


def panel_mask(ax, sample: Sample) -> list:
    detail = sample.mask
    x, y = detail.origin
    height, width = detail.cropped.shape
    overlay = np.zeros((height, width, 4))
    overlay[detail.component] = matplotlib.colors.to_rgba(MASK_COLOR, 0.55)
    overlay[detail.band] = matplotlib.colors.to_rgba(BAND_COLOR, 0.94)
    ax.imshow(overlay, extent=pixel_extent(detail.origin, overlay.shape), interpolation="nearest", zorder=3)
    ax.contour(
        x + np.arange(-1, width + 1), y + np.arange(-1, height + 1), np.pad(detail.cropped, 1),
        levels=[0.5], colors="white", linewidths=0.62, linestyles="dashed", zorder=5,
    )
    draw_box(ax, sample, faint=True)
    u_s, v_s = detail.point
    x1, _, x2, y2 = map(float, sample.box)
    span = max(3.0, 0.08 * sample.height)
    ax.hlines(v_s, x1 - span, x2 + span, colors=BAND_COLOR, linestyles="dashed", lw=0.9, zorder=7)
    ax.vlines(u_s, v_s - 2 * span, max(v_s, y2) + 0.55 * span, colors=BAND_COLOR, linestyles="dashed", lw=0.9, zorder=7)
    ring(ax, detail.point, BAND_COLOR)
    point_label(ax, r"$v_S$", (x2 + span, v_s), (0, 3), ha="right", va="bottom")
    point_label(ax, r"$u_S$", (u_s, max(v_s, y2) + 0.55 * span), (0, -3), ha="center", va="top")
    return [
        Line2D([], [], ls="--", lw=0.9, color="#46545a", label="Cropped mask"),
        Patch(facecolor=MASK_COLOR, edgecolor="#28677c", linewidth=0.4, label="Largest component"),
        Patch(facecolor=BAND_COLOR, edgecolor="#a16c12", linewidth=0.4, label=r"Bottom band $\mathcal{P}_b$"),
    ]


def panel_pose(ax, sample: Sample) -> list:
    joints, visible = sample.joints, sample.visible
    for a, b in JOINT_LINKS:
        if visible[a] and visible[b]:
            ax.plot(joints[[a, b], 0], joints[[a, b], 1], color="white", alpha=0.78, lw=0.8, zorder=5)
    hip_points = joints[list(HIPS)]
    x1, _, x2, y2 = map(float, sample.box)
    u_p = float(np.clip(sum(map(float, hip_points[:, 0])) / 2.0, x1, x2))
    hip_y = float(hip_points[:, 1].mean())
    ax.vlines(u_p, hip_y, y2, colors=BOX_COLOR, linestyles="dashed", lw=1.15, zorder=6)
    feet = joints[[i for i in FEET if visible[i]]]
    ax.scatter(*hip_points.T, marker="o", c=BOX_COLOR, edgecolors=INK, linewidths=0.55, s=15, zorder=8)
    ax.scatter(*feet.T, marker="s", c=BAND_COLOR, edgecolors=INK, linewidths=0.42, s=9, zorder=8)
    point_label(ax, r"$u_P$", (u_p, y2), (0, -7), ha="center", va="top")
    return [
        Line2D([], [], ls="none", marker="o", ms=4.4, mfc=BOX_COLOR, mec=INK, mew=0.5, label="Hips"),
        Line2D([], [], ls="none", marker="s", ms=3.8, mfc=BAND_COLOR, mec=INK, mew=0.5, label="Feet"),
    ]


def render_row(sample: Sample, row: str, output_dir: Path, args: argparse.Namespace) -> list[Path]:
    pixels, origin, (left, top, right, bottom) = crop_for_row(sample)
    paths = []
    panels = (
        ("a", "Bounding box", "bbox", panel_box),
        ("b", "MobileSAM", "mobilesam", panel_mask),
        ("c", "RTMPose", "pose", panel_pose),
    )
    for letter, title, suffix, draw in panels:
        fig = plt.figure(figsize=PANEL_SIZE, facecolor="white")
        try:
            ax = fig.add_axes(PHOTO_RECT, facecolor="#273c33")
            ax.imshow(pixels, extent=pixel_extent(origin, pixels.shape), interpolation="nearest", zorder=0)
            handles = draw(ax, sample)
            ax.plot(*sample.gt, marker="x", ms=5.1, mew=1.25, color=GT_COLOR, ls="none", zorder=11,
                    path_effects=[effects.withStroke(linewidth=2.1, foreground=INK)])
            handles.append(Line2D([], [], ls="none", marker="x", ms=4.5, mew=1.15, color="#a61969", label="GT ground point"))
            ax.set(xlim=(left, right), ylim=(bottom, top), aspect="equal", xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_color("#45514d")
                spine.set_linewidth(0.35)
            fig.text(0.035, 0.938, f"({letter})  {title}", ha="left", va="bottom", fontsize=8.8, weight="bold", color=INK)
            if letter == "a":
                fig.text(0.965, 0.938, rf"$h={sample.height:.1f}$ px", ha="right", va="bottom", fontsize=7.7, color=INK)
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.016), ncol=2 if letter == "b" else len(handles),
                       frameon=False, fontsize=6.6, handlelength=1.4, handletextpad=0.45, columnspacing=1.15, labelspacing=0.52)
            path = output_dir / f"{row}_{letter}_{suffix}.{args.format}"
            fig.savefig(path, dpi=args.dpi, facecolor="white")
            paths.append(path)
        finally:
            plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    annotation_path, image_dir = dataset_paths(args)
    with annotation_path.open(encoding="utf-8") as handle:
        dataset = json.load(handle)
    large, small = select_samples(dataset, image_dir, args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = (("large", large), ("small", small))
    with tempfile.TemporaryDirectory(dir=output_dir) as staging:
        paths = [path for row, sample in rows for path in render_row(sample, row, Path(staging), args)]
        for path in paths:
            path.replace(output_dir / path.name)
    for row, sample in rows:
        print(f"{row}: image {sample.image_id} ({sample.image_path.name}), annotation {sample.annotation_id}, "
              f"YOLO index {sample.detection_index}, score {sample.score:.3f}, IoU {sample.iou:.3f}, "
              f"box height {sample.height:.1f} px, camera distance {sample.distance:.1f} m")
    print(f"Wrote six {args.format.upper()} panels to {output_dir}")
    print(f"To pin these examples: --large-ann-id {large.annotation_id} --small-ann-id {small.annotation_id}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error
