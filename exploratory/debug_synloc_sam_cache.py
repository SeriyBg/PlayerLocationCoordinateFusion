from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from sskit import image_to_ground
from tqdm import tqdm


# OpenCV uses BGR.
GT_COLOR = (0, 255, 0)       # green
BBOX_COLOR = (255, 0, 0)     # blue
METHOD_COLOR = (0, 0, 255)   # red
BOX_COLOR = (220, 220, 220)
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (0, 0, 0)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Debug cached SynLoc YOLO+SAM predictions. For every detection, "
            "match it to a GT player by bbox IoU, compare bbox and SAM "
            "ground-point errors in pitch-space metres, draw GT/bbox/SAM "
            "points, and save a per-detection CSV."
        )
    )
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help=(
            "Root passed to cache_synloc_sam.py, e.g. "
            "output/synloc_sam_cache"
        ),
    )
    p.add_argument(
        "--split",
        choices=["train", "val", "test", "challenge"],
        default="val",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../output/synloc_sam_debug"),
    )
    p.add_argument(
        "--sam-method",
        choices=["lowest", "robust"],
        default="robust",
        help="Which cached-mask ground-point rule to visualize/compare.",
    )
    p.add_argument(
        "--bottom-band-height",
        type=int,
        default=3,
        help="Bottom band for the original 'lowest' method.",
    )
    p.add_argument(
        "--bottom-percentile",
        type=float,
        default=98.0,
        help="Robust SAM bottom percentile.",
    )
    p.add_argument(
        "--band-fraction",
        type=float,
        default=0.05,
        help="Robust bottom-band height as a fraction of bbox height.",
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=0.05,
        help="Ignore cached YOLO detections below this confidence.",
    )
    p.add_argument(
        "--match-iou",
        type=float,
        default=0.30,
        help="Minimum bbox IoU for assigning a detection to GT.",
    )
    p.add_argument(
        "--better-margin-m",
        type=float,
        default=0.0,
        help=(
            "SAM is called BETTER only if its error is smaller than bbox "
            "error by more than this many metres."
        ),
    )
    p.add_argument(
        "--correct-threshold-m",
        type=float,
        default=0.25,
        help="Threshold used for bbox/method correctness transition columns.",
    )
    p.add_argument("--point-radius", type=int, default=2)
    p.add_argument(
        "--draw-mask-outline",
        action="store_true",
        help="Also draw the selected SAM mask contour in red.",
    )
    p.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Overwrite already-rendered debug images.",
    )
    return p.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_image_directory(root: Path, split: str) -> Path:
    aliases = {
        "train": ["train"],
        "val": ["val", "valid", "validation"],
        "test": ["test"],
        "challenge": ["challenge"],
    }
    tried: list[Path] = []
    for name in aliases[split]:
        for candidate in (root / "images" / name, root / name):
            tried.append(candidate)
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError(f"Could not find image directory. Tried: {tried}")


def resolve_annotation_file(root: Path, split: str) -> Path:
    aliases = {
        "train": ["train"],
        "val": ["val", "valid", "validation"],
        "test": ["test"],
        "challenge": ["challenge"],
    }
    tried: list[Path] = []
    for name in aliases[split]:
        candidate = root / "annotations" / f"{name}.json"
        tried.append(candidate)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find annotation file. Tried: {tried}")


def extract_gt_ground_point(annotation: dict[str, Any]) -> tuple[float, float]:
    """
    SynLoc GT keypoint 1 is the image-space ground-projected point.
    Supports both [[x,y,v], [x,y,v]] and flat COCO-style arrays.
    """
    keypoints = np.asarray(annotation["keypoints"], dtype=np.float32)

    if keypoints.ndim == 1:
        keypoints = keypoints.reshape(-1, 3)

    if keypoints.shape[0] < 2:
        raise ValueError(
            f"Annotation {annotation.get('id')} has fewer than 2 keypoints."
        )

    return float(keypoints[1, 0]), float(keypoints[1, 1])


def annotation_bbox_to_actual(
    bbox_xywh: list[float],
    ann_width: int,
    ann_height: int,
    actual_width: int,
    actual_height: int,
) -> np.ndarray:
    x, y, w, h = map(float, bbox_xywh)
    sx = actual_width / float(ann_width)
    sy = actual_height / float(ann_height)
    return np.asarray(
        [x * sx, y * sy, (x + w) * sx, (y + h) * sy],
        dtype=np.float32,
    )


def annotation_point_to_actual(
    point: tuple[float, float],
    ann_width: int,
    ann_height: int,
    actual_width: int,
    actual_height: int,
) -> tuple[float, float]:
    sx = actual_width / float(ann_width)
    sy = actual_height / float(ann_height)
    return point[0] * sx, point[1] * sy


def actual_point_to_annotation(
    point: tuple[float, float],
    actual_width: int,
    actual_height: int,
    ann_width: int,
    ann_height: int,
) -> tuple[float, float]:
    sx = ann_width / float(actual_width)
    sy = ann_height / float(actual_height)
    return point[0] * sx, point[1] * sy


def bbox_ground_point(box: np.ndarray) -> tuple[float, float]:
    x1, _, x2, y2 = map(float, box)
    return (x1 + x2) / 2.0, y2


def find_mask_ground_point_lowest(
    mask: np.ndarray,
    bottom_band_height: int = 3,
) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    lowest_y = int(ys.max())
    selected = ys >= lowest_y - bottom_band_height

    if not np.any(selected):
        return None

    return (
        float(np.median(xs[selected])),
        float(np.median(ys[selected])),
    )


def find_mask_ground_point_robust(
    mask: np.ndarray,
    bbox: np.ndarray,
    bottom_percentile: float = 98.0,
    band_fraction: float = 0.05,
) -> tuple[float, float] | None:
    """
    Robust cached-mask estimator:
    - crop to the YOLO bbox,
    - retain the largest connected component,
    - use a percentile rather than the absolute bottom pixel,
    - estimate x from a player-scale bottom band.
    """
    image_height, image_width = mask.shape
    x1, y1, x2, y2 = map(float, bbox)

    bx1 = max(0, int(np.floor(x1)))
    by1 = max(0, int(np.floor(y1)))
    bx2 = min(image_width - 1, int(np.ceil(x2)))
    by2 = min(image_height - 1, int(np.ceil(y2)))

    if bx2 <= bx1 or by2 <= by1:
        return None

    cropped = np.zeros_like(mask, dtype=np.uint8)
    cropped[by1:by2 + 1, bx1:bx2 + 1] = (
        mask[by1:by2 + 1, bx1:bx2 + 1] > 0
    ).astype(np.uint8)

    if int(cropped.sum()) == 0:
        return None

    number_of_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cropped,
        connectivity=8,
    )
    if number_of_labels <= 1:
        return None

    largest_label = 1 + int(
        np.argmax(stats[1:, cv2.CC_STAT_AREA])
    )
    clean = labels == largest_label

    ys, xs = np.nonzero(clean)
    if len(xs) == 0:
        return None

    robust_bottom_y = float(
        np.percentile(ys, bottom_percentile)
    )

    bbox_height = max(y2 - y1, 1.0)
    band_height = max(2.0, bbox_height * band_fraction)

    selected = (
        (ys >= robust_bottom_y - band_height)
        & (ys <= robust_bottom_y)
    )
    if not np.any(selected):
        return None

    return (
        float(np.median(xs[selected])),
        robust_bottom_y,
    )


def mask_quality_features(
    mask: np.ndarray | None,
    bbox: np.ndarray,
    method_point: tuple[float, float] | None,
) -> dict[str, float | int | None]:
    if mask is None:
        return {
            "mask_area_px": 0,
            "connected_components": 0,
            "largest_component_ratio": None,
            "mask_inside_bbox_ratio": None,
            "mask_bbox_area_ratio": None,
            "relative_ground_y": None,
            "relative_ground_x_offset": None,
            "relative_bottom_width": None,
        }

    binary = (mask > 0).astype(np.uint8)
    mask_area = int(binary.sum())

    if mask_area == 0:
        return {
            "mask_area_px": 0,
            "connected_components": 0,
            "largest_component_ratio": None,
            "mask_inside_bbox_ratio": None,
            "mask_bbox_area_ratio": None,
            "relative_ground_y": None,
            "relative_ground_x_offset": None,
            "relative_bottom_width": None,
        }

    h, w = binary.shape
    x1, y1, x2, y2 = map(float, bbox)
    bbox_width = max(x2 - x1, 1.0)
    bbox_height = max(y2 - y1, 1.0)
    bbox_area = bbox_width * bbox_height

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    component_count = max(0, num_labels - 1)
    if component_count:
        largest_area = int(stats[1:, cv2.CC_STAT_AREA].max())
        largest_ratio = largest_area / float(mask_area)
    else:
        largest_ratio = None

    bx1 = max(0, int(np.floor(x1)))
    by1 = max(0, int(np.floor(y1)))
    bx2 = min(w, int(np.ceil(x2)))
    by2 = min(h, int(np.ceil(y2)))

    inside_area = int(binary[by1:by2, bx1:bx2].sum())
    inside_ratio = inside_area / float(mask_area)
    mask_bbox_ratio = inside_area / float(bbox_area)

    relative_y = None
    relative_x_offset = None
    relative_bottom_width = None

    if method_point is not None:
        gx, gy = method_point
        relative_y = (gy - y1) / bbox_height
        relative_x_offset = (
            abs(gx - (x1 + x2) / 2.0) / bbox_width
        )

        ys, xs = np.nonzero(binary)
        band_h = max(2.0, bbox_height * 0.05)
        selected = (ys >= gy - band_h) & (ys <= gy)
        if np.any(selected):
            bottom_width = (
                float(xs[selected].max() - xs[selected].min() + 1)
            )
            relative_bottom_width = bottom_width / bbox_width

    return {
        "mask_area_px": mask_area,
        "connected_components": component_count,
        "largest_component_ratio": largest_ratio,
        "mask_inside_bbox_ratio": inside_ratio,
        "mask_bbox_area_ratio": mask_bbox_ratio,
        "relative_ground_y": relative_y,
        "relative_ground_x_offset": relative_x_offset,
        "relative_bottom_width": relative_bottom_width,
    }


def pairwise_iou(
    detections: np.ndarray,
    gt_boxes: np.ndarray,
) -> np.ndarray:
    if len(detections) == 0 or len(gt_boxes) == 0:
        return np.empty((len(detections), len(gt_boxes)), dtype=np.float32)

    a = detections[:, None, :]
    b = gt_boxes[None, :, :]

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, a[..., 2] - a[..., 0]) * np.maximum(
        0.0, a[..., 3] - a[..., 1]
    )
    area_b = np.maximum(0.0, b[..., 2] - b[..., 0]) * np.maximum(
        0.0, b[..., 3] - b[..., 1]
    )

    union = area_a + area_b - inter
    return np.divide(
        inter,
        union,
        out=np.zeros_like(inter, dtype=np.float32),
        where=union > 0,
    )


def match_by_iou(
    detections: np.ndarray,
    gt_boxes: np.ndarray,
    minimum_iou: float,
) -> dict[int, tuple[int, float]]:
    """
    One-to-one Hungarian assignment based only on bbox IoU.
    This deliberately avoids using ground-point error to choose the GT match.
    """
    if len(detections) == 0 or len(gt_boxes) == 0:
        return {}

    ious = pairwise_iou(detections, gt_boxes)
    row_indices, col_indices = linear_sum_assignment(1.0 - ious)

    matches: dict[int, tuple[int, float]] = {}
    for det_idx, gt_idx in zip(row_indices, col_indices):
        iou = float(ious[det_idx, gt_idx])
        if iou >= minimum_iou:
            matches[int(det_idx)] = (int(gt_idx), iou)

    return matches


def project_actual_point_to_pitch(
    point_actual: tuple[float, float],
    image_info: dict[str, Any],
    actual_width: int,
    actual_height: int,
) -> np.ndarray:
    ann_width = int(image_info["width"])
    ann_height = int(image_info["height"])

    x, y = actual_point_to_annotation(
        point_actual,
        actual_width,
        actual_height,
        ann_width,
        ann_height,
    )

    normalized = np.asarray(
        [[
            (x - (ann_width - 1.0) / 2.0) / ann_width,
            (y - (ann_height - 1.0) / 2.0) / ann_width,
        ]],
        dtype=np.float32,
    )

    pitch = image_to_ground(
        image_info["camera_matrix"],
        image_info["undist_poly"],
        normalized,
    )

    if hasattr(pitch, "detach"):
        pitch = pitch.detach().cpu().numpy()
    else:
        pitch = np.asarray(pitch)

    return np.asarray(pitch, dtype=np.float32)[0, :2]


def classify_comparison(
    bbox_error: float,
    method_error: float,
    margin: float,
) -> str:
    improvement = bbox_error - method_error
    if improvement > margin:
        return "better"
    if improvement < -margin:
        return "worse"
    return "equal"


def correctness_transition(
    bbox_error: float,
    method_error: float,
    threshold: float,
) -> str:
    bbox_ok = bbox_error <= threshold
    method_ok = method_error <= threshold

    if bbox_ok and method_ok:
        return "both_correct"
    if bbox_ok and not method_ok:
        return "bbox_only_correct"
    if not bbox_ok and method_ok:
        return "method_only_correct"
    return "both_wrong"


def draw_point(
    image: np.ndarray,
    point: tuple[float, float],
    color: tuple[int, int, int],
    radius: int,
) -> None:
    center = (int(round(point[0])), int(round(point[1])))
    # cv2.circle(image, center, radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)


def draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.43
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    x = max(0, min(x, image.shape[1] - tw - 4))
    y = max(th + 4, min(y, image.shape[0] - baseline - 4))

    cv2.rectangle(
        image,
        (x, y - th - 4),
        (x + tw + 4, y + baseline + 2),
        TEXT_BG,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 2, y - 2),
        font,
        scale,
        TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )


def draw_legend(image: np.ndarray, method_name: str) -> None:
    entries = [
        ("GT", GT_COLOR),
        ("bbox", BBOX_COLOR),
        (method_name, METHOD_COLOR),
    ]
    x = 15
    y = 25
    for text, color in entries:
        cv2.circle(image, (x + 6, y - 5), 5, color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            text,
            (x + 18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        x += 95


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve() / args.split
    images_output = output_root / "images"
    images_output.mkdir(parents=True, exist_ok=True)

    image_dir = resolve_image_directory(dataset_root, args.split)
    annotation_path = resolve_annotation_file(dataset_root, args.split)
    dataset = load_json(annotation_path)

    images_by_id = {
        int(item["id"]): item for item in dataset.get("images", [])
    }
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in dataset.get("annotations", []):
        if int(annotation.get("category_id", 1)) == 1:
            annotations_by_image[int(annotation["image_id"])].append(annotation)

    cache_dir = cache_root / args.split / "sam_raw"
    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"SAM cache directory not found: {cache_dir}"
        )

    cache_files = sorted(cache_dir.glob("*.npz"))
    if not cache_files:
        raise RuntimeError(f"No .npz SAM cache files found in {cache_dir}")

    csv_path = output_root / "sam_detection_debug.csv"
    summary_path = output_root / "sam_debug_summary.json"

    fieldnames = [
        "image_id", "file_name", "detection_index",
        "yolo_confidence",
        "matched_gt", "gt_annotation_id", "bbox_match_iou",
        "comparison", "improvement_m", "transition_0_25m",
        "bbox_error_m", "sam_error_m",
        "bbox_correct_0_25m", "sam_correct_0_25m",
        "sam_available", "sam_method",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "gt_image_x", "gt_image_y",
        "bbox_image_x", "bbox_image_y",
        "sam_image_x", "sam_image_y",
        "gt_pitch_x", "gt_pitch_y",
        "bbox_pitch_x", "bbox_pitch_y",
        "sam_pitch_x", "sam_pitch_y",
        "mask_area_px", "connected_components",
        "largest_component_ratio",
        "mask_inside_bbox_ratio",
        "mask_bbox_area_ratio",
        "relative_ground_y",
        "relative_ground_x_offset",
        "relative_bottom_width",
    ]

    counts = Counter()
    bbox_errors: list[float] = []
    sam_errors: list[float] = []
    improvements: list[float] = []

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for cache_path in tqdm(cache_files, desc="Rendering SAM debug"):
            with np.load(cache_path, allow_pickle=False) as data:
                image_id = int(data["image_id"])
                file_name = str(data["file_name"].item())
                actual_width = int(data["actual_width"])
                actual_height = int(data["actual_height"])
                boxes = np.asarray(data["bboxes"], dtype=np.float32)
                scores = np.asarray(data["detection_scores"], dtype=np.float32)
                masks = np.asarray(data["masks"], dtype=np.uint8)

            if image_id not in images_by_id:
                print(
                    f"WARNING: image_id={image_id} from {cache_path.name} "
                    "not present in annotations; skipping."
                )
                continue

            image_info = images_by_id[image_id]
            image_path = image_dir / file_name
            if not image_path.is_file():
                image_path = image_dir / Path(file_name).name

            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Could not read image: {image_path}")

            h, w = image.shape[:2]
            if (w, h) != (actual_width, actual_height):
                raise RuntimeError(
                    f"Cache/image dimension mismatch for {file_name}: "
                    f"cache={(actual_width, actual_height)}, image={(w, h)}"
                )

            gt_annotations = annotations_by_image.get(image_id, [])
            ann_width = int(image_info["width"])
            ann_height = int(image_info["height"])

            gt_boxes = np.asarray(
                [
                    annotation_bbox_to_actual(
                        gt["bbox"],
                        ann_width,
                        ann_height,
                        w,
                        h,
                    )
                    for gt in gt_annotations
                ],
                dtype=np.float32,
            ).reshape(-1, 4)

            keep_indices = np.flatnonzero(scores >= args.min_score)
            kept_boxes = boxes[keep_indices] if len(keep_indices) else np.empty(
                (0, 4), dtype=np.float32
            )

            local_matches = match_by_iou(
                kept_boxes,
                gt_boxes,
                args.match_iou,
            )
            matches = {
                int(keep_indices[local_det_idx]): match
                for local_det_idx, match in local_matches.items()
            }

            draw_legend(image, f"SAM-{args.sam_method}")

            for det_idx in keep_indices:
                det_idx = int(det_idx)
                box = boxes[det_idx]
                score = float(scores[det_idx])

                x1, y1, x2, y2 = map(float, box)
                cv2.rectangle(
                    image,
                    (int(round(x1)), int(round(y1))),
                    (int(round(x2)), int(round(y2))),
                    BOX_COLOR,
                    1,
                    cv2.LINE_AA,
                )

                bbox_point = bbox_ground_point(box)
                draw_point(image, bbox_point, BBOX_COLOR, args.point_radius)

                mask = (
                    masks[det_idx]
                    if det_idx < len(masks)
                    else None
                )

                if mask is not None and int(mask.sum()) > 0:
                    if args.sam_method == "lowest":
                        sam_point = find_mask_ground_point_lowest(
                            mask,
                            args.bottom_band_height,
                        )
                    else:
                        sam_point = find_mask_ground_point_robust(
                            mask,
                            box,
                            args.bottom_percentile,
                            args.band_fraction,
                        )
                else:
                    sam_point = None

                if sam_point is not None:
                    draw_point(
                        image,
                        sam_point,
                        METHOD_COLOR,
                        args.point_radius,
                    )

                    if args.draw_mask_outline and mask is not None:
                        contours, _ = cv2.findContours(
                            (mask > 0).astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        cv2.drawContours(
                            image,
                            contours,
                            -1,
                            METHOD_COLOR,
                            1,
                            cv2.LINE_AA,
                        )

                quality = mask_quality_features(mask, box, sam_point)

                row: dict[str, Any] = {
                    "image_id": image_id,
                    "file_name": file_name,
                    "detection_index": det_idx,
                    "yolo_confidence": score,
                    "matched_gt": False,
                    "gt_annotation_id": None,
                    "bbox_match_iou": None,
                    "comparison": "unmatched",
                    "improvement_m": None,
                    "transition_0_25m": "unmatched",
                    "bbox_error_m": None,
                    "sam_error_m": None,
                    "bbox_correct_0_25m": None,
                    "sam_correct_0_25m": None,
                    "sam_available": sam_point is not None,
                    "sam_method": args.sam_method,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "gt_image_x": None,
                    "gt_image_y": None,
                    "bbox_image_x": bbox_point[0],
                    "bbox_image_y": bbox_point[1],
                    "sam_image_x": sam_point[0] if sam_point else None,
                    "sam_image_y": sam_point[1] if sam_point else None,
                    "gt_pitch_x": None,
                    "gt_pitch_y": None,
                    "bbox_pitch_x": None,
                    "bbox_pitch_y": None,
                    "sam_pitch_x": None,
                    "sam_pitch_y": None,
                    **quality,
                }

                match = matches.get(det_idx)
                if match is None:
                    counts["unmatched_detection"] += 1
                    label = (
                        f"D{det_idx} conf={score:.2f} UNMATCHED"
                        if sam_point is not None
                        else f"D{det_idx} conf={score:.2f} NO_MASK UNMATCHED"
                    )
                    # draw_label(
                    #     image,
                    #     label,
                    #     int(round(x1)),
                    #     int(round(y1)) - 3,
                    # )
                    writer.writerow(row)
                    continue

                gt_idx, match_iou = match
                gt = gt_annotations[gt_idx]

                gt_point_ann = extract_gt_ground_point(gt)
                gt_point = annotation_point_to_actual(
                    gt_point_ann,
                    ann_width,
                    ann_height,
                    w,
                    h,
                )
                gt_pitch = np.asarray(
                    gt["position_on_pitch"][:2],
                    dtype=np.float32,
                )

                draw_point(
                    image,
                    gt_point,
                    GT_COLOR,
                    args.point_radius,
                )

                bbox_pitch = project_actual_point_to_pitch(
                    bbox_point,
                    image_info,
                    w,
                    h,
                )
                bbox_error = float(
                    np.linalg.norm(bbox_pitch - gt_pitch)
                )

                row.update({
                    "matched_gt": True,
                    "gt_annotation_id": int(gt["id"]),
                    "bbox_match_iou": match_iou,
                    "gt_image_x": gt_point[0],
                    "gt_image_y": gt_point[1],
                    "gt_pitch_x": float(gt_pitch[0]),
                    "gt_pitch_y": float(gt_pitch[1]),
                    "bbox_pitch_x": float(bbox_pitch[0]),
                    "bbox_pitch_y": float(bbox_pitch[1]),
                    "bbox_error_m": bbox_error,
                    "bbox_correct_0_25m": (
                        bbox_error <= args.correct_threshold_m
                    ),
                })

                bbox_errors.append(bbox_error)

                if sam_point is None:
                    counts["sam_unavailable"] += 1
                    row["comparison"] = "sam_unavailable"
                    row["transition_0_25m"] = "sam_unavailable"
                    label = (
                        f"D{det_idx} B={bbox_error:.2f}m "
                        "SAM=NA"
                    )
                    draw_label(
                        image,
                        label,
                        int(round(x1)),
                        int(round(y1)) - 3,
                    )
                    writer.writerow(row)
                    continue

                sam_pitch = project_actual_point_to_pitch(
                    sam_point,
                    image_info,
                    w,
                    h,
                )
                sam_error = float(
                    np.linalg.norm(sam_pitch - gt_pitch)
                )
                improvement = bbox_error - sam_error

                comparison = classify_comparison(
                    bbox_error,
                    sam_error,
                    args.better_margin_m,
                )
                transition = correctness_transition(
                    bbox_error,
                    sam_error,
                    args.correct_threshold_m,
                )

                row.update({
                    "comparison": comparison,
                    "improvement_m": improvement,
                    "transition_0_25m": transition,
                    "sam_error_m": sam_error,
                    "sam_correct_0_25m": (
                        sam_error <= args.correct_threshold_m
                    ),
                    "sam_pitch_x": float(sam_pitch[0]),
                    "sam_pitch_y": float(sam_pitch[1]),
                })

                counts[comparison] += 1
                counts[transition] += 1
                sam_errors.append(sam_error)
                improvements.append(improvement)

                sign = "+" if improvement >= 0 else ""
                label = (
                    f"D{det_idx} B={bbox_error:.2f} "
                    f"S={sam_error:.2f}m "
                    f"{comparison.upper()} {sign}{improvement:.2f}"
                )
                if comparison == "worse":
                    draw_label(
                        image,
                        label,
                        int(round(x1)),
                        int(round(y1)) - 3,
                    )

                writer.writerow(row)

            # Draw unmatched GT points as green crosses so missed players are visible.
            matched_gt_indices = {
                gt_idx for gt_idx, _ in matches.values()
            }
            for gt_idx, gt in enumerate(gt_annotations):
                if gt_idx in matched_gt_indices:
                    continue
                gt_point_ann = extract_gt_ground_point(gt)
                gt_point = annotation_point_to_actual(
                    gt_point_ann,
                    ann_width,
                    ann_height,
                    w,
                    h,
                )
                gx, gy = int(round(gt_point[0])), int(round(gt_point[1]))
                cv2.drawMarker(
                    image,
                    (gx, gy),
                    GT_COLOR,
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=12,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )

            output_path = images_output / (
                Path(file_name).stem + "_sam_debug.jpg"
            )
            if args.overwrite_images or not output_path.exists():
                if not cv2.imwrite(str(output_path), image):
                    raise RuntimeError(
                        f"Failed to save image: {output_path}"
                    )

    summary = {
        "split": args.split,
        "sam_method": args.sam_method,
        "minimum_detection_score": args.min_score,
        "bbox_gt_match_iou_threshold": args.match_iou,
        "better_margin_m": args.better_margin_m,
        "correct_threshold_m": args.correct_threshold_m,
        "counts": dict(counts),
        "mean_bbox_error_m": (
            float(np.mean(bbox_errors)) if bbox_errors else None
        ),
        "mean_sam_error_m": (
            float(np.mean(sam_errors)) if sam_errors else None
        ),
        "mean_improvement_m_bbox_minus_sam": (
            float(np.mean(improvements)) if improvements else None
        ),
        "positive_improvement_means": "SAM is more accurate than bbox",
        "csv": str(csv_path),
        "images": str(images_output),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nFinished SAM debugging.")
    print(f"Images:  {images_output}")
    print(f"CSV:     {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Better:  {counts['better']}")
    print(f"Worse:   {counts['worse']}")
    print(f"Equal:   {counts['equal']}")
    print(f"No SAM:  {counts['sam_unavailable']}")
    print(f"Unmatched detections: {counts['unmatched_detection']}")


if __name__ == "__main__":
    main()
