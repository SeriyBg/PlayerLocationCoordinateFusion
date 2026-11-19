#!/usr/bin/env python3
"""Per-detection player localization / bounding-box-size diagnostic.

MODE is set to "pose" or "sam" when creating the two standalone scripts.
Uses the canonical frozen YOLO boxes to match GT exactly as in the preceding
SynLoc analyses. Pixel coordinates always refer to the actual source images.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from sskit import image_to_ground

MODE = "pose"

SCHEMAS = {
    "coco17": dict(left_hip=11, right_hip=12, left_foot=[], right_foot=[]),
    "halpe26": dict(left_hip=11, right_hip=12, left_foot=[20, 22, 24], right_foot=[21, 23, 25]),
    "coco25": dict(left_hip=11, right_hip=12, left_foot=[19, 20, 21], right_foot=[22, 23, 24]),
    "coco133": dict(left_hip=11, right_hip=12, left_foot=[17, 18, 19], right_foot=[20, 21, 22]),
}
HEIGHT_EDGES = (0, 20, 40, 60, 90, 140, float("inf"))
AREA_EDGES = (0, 250, 500, 1000, 2000, 4000, 8000, 16000, float("inf"))

FIELDS = [
    "method", "split", "image_id", "file_name", "detection_index", "detection_score",
    "match_status", "gt_annotation_id", "matched_iou", "image_width_px", "image_height_px",
    "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px",
    "bbox_width_px", "bbox_height_px", "bbox_area_px2", "bbox_area_fraction",
    "bbox_aspect_ratio", "bbox_bottom_y_fraction", "bbox_height_bin", "bbox_area_bin",
    "bbox_ground_x_px", "bbox_ground_y_px", "prediction_source",
    "pred_x_px", "pred_y_px", "raw_x_px", "raw_y_px",
    "raw_signed_dx_px", "raw_signed_dy_px", "raw_abs_dx_px", "raw_abs_dy_px",
    "gt_x_px", "gt_y_px", "gt_pitch_x_m", "gt_pitch_y_m",
    "signed_dx_px", "signed_dy_px", "abs_dx_px", "abs_dy_px",
    "abs_dx_over_bbox_width", "abs_dy_over_bbox_height", "dominant_pixel_error_axis",
    "bbox_error_m", "prediction_error_m", "raw_error_m", "improvement_over_bbox_m",
    "raw_improvement_over_bbox_m", "prediction_beats_bbox", "raw_beats_bbox",
    "correct_within_025m", "correct_within_050m", "correct_within_100m",
    "pitch_signed_dx_m", "pitch_signed_dy_m", "pitch_error_x_only_m",
    "pitch_error_y_only_m", "dominant_projection_error_axis", "gt_projection_disagreement_m",
    "pose_pelvis_x_px", "pose_pelvis_source", "pose_feet_x_px", "pose_feet_y_px", "pose_feet_source",
    "pose_feet_ground_error_m", "pose_feet_signed_dy_px",
    "pelvis_feet_abs_dx_px", "pose_left_hip_conf", "pose_right_hip_conf",
    "sam_mask_x_px", "sam_mask_y_px", "sam_bbox_bottom_minus_mask_bottom_px",
]


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Write per-detection {MODE.upper()} bbox-size/error analysis CSVs.")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--yolo-cache-root", type=Path, default=Path("output/synloc_yolo_cache"))
    if MODE == "pose":
        p.add_argument("--pose-cache-root", type=Path, default=Path("output/synloc_pose_cache"))
        p.add_argument("--pose-model", default="halpe26_m")
        p.add_argument("--pose-schema", choices=sorted(SCHEMAS), default="halpe26")
        p.add_argument("--keypoint-conf", type=float, default=0.30)
        p.add_argument("--pelvis-feet-threshold-px", type=float, default=3.0)
    else:
        p.add_argument("--sam-cache-root", type=Path, default=Path("output/synloc_sam_cache"))
        p.add_argument("--bottom-band-frac", type=float, default=0.05)
        p.add_argument("--bottom-y-percentile", type=float, default=98.0)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--score-threshold", type=float, default=0.4302865565)
    p.add_argument("--match-iou", type=float, default=0.50)
    p.add_argument("--output-dir", type=Path, default=Path("output/synloc_bbox_size_analysis"))
    p.add_argument("--max-images", type=int, default=0, help="0 = all images; e.g. 100 for a quick test")
    return p.parse_args()


def annotation_path(root: Path, split: str) -> Path:
    for n in (("val", "valid", "validation") if split == "val" else ("test",)):
        p = root / "annotations" / f"{n}.json"
        if p.is_file():
            return p
    raise FileNotFoundError(f"Cannot find {split} annotations in {root / 'annotations'}")


def load_cache(path: Path, fields: tuple[str, ...]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {field: np.asarray(z[field]) for field in fields}


def num(z: np.ndarray) -> int:
    return int(z.item())


def txt(z: np.ndarray) -> str:
    return str(z.item())


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    l = np.maximum(a[:, None, :2], b[None, :, :2])
    r = np.minimum(a[:, None, 2:], b[None, :, 2:])
    span = np.maximum(r - l, 0)
    intersection = span[..., 0] * span[..., 1]
    area_a = np.prod(np.maximum(a[:, 2:] - a[:, :2], 0), axis=1)[:, None]
    area_b = np.prod(np.maximum(b[:, 2:] - b[:, :2], 0), axis=1)[None, :]
    union = area_a + area_b - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def hungarian_matches(det_boxes: np.ndarray, gt_boxes: np.ndarray, threshold: float) -> dict[int, tuple[int, float]]:
    """Same one-to-one bbox IoU matching as the earlier pose/SAM diagnostics."""
    if len(det_boxes) == 0 or len(gt_boxes) == 0:
        return {}
    ious = iou_matrix(det_boxes, gt_boxes)
    d, g = linear_sum_assignment(1.0 - ious)
    return {int(di): (int(gi), float(ious[di, gi]))
            for di, gi in zip(d, g) if float(ious[di, gi]) >= threshold}


def gt_ground_annotation(annotation: dict[str, Any]) -> tuple[float, float] | None:
    raw = annotation.get("keypoints")
    if raw is None:
        return None
    try:
        k = np.asarray(raw, dtype=float)
        if k.ndim == 1 and k.size >= 6:
            x, y = float(k[3]), float(k[4])
        elif k.ndim >= 2 and k.shape[0] >= 2 and k.shape[1] >= 2:
            x, y = float(k[1, 0]), float(k[1, 1])
        else:
            return None
        return (x, y) if np.isfinite([x, y]).all() else None
    except (TypeError, ValueError):
        return None


def projection(points_actual: np.ndarray, image: dict[str, Any],
               actual_width: int, actual_height: int, ann_width: int, ann_height: int) -> np.ndarray:
    """Project many source-image points in one call; mirror previous SynLoc normalization."""
    p = np.asarray(points_actual, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    ann = p * np.array([ann_width / actual_width, ann_height / actual_height], dtype=np.float64)
    w, h = float(image["width"]), float(image["height"])
    normalized = np.column_stack(((ann[:, 0] - (w - 1.0) / 2.0) / w,
                                  (ann[:, 1] - (h - 1.0) / 2.0) / w)).astype(np.float32)
    result = image_to_ground(image["camera_matrix"], image["undist_poly"], normalized)
    if hasattr(result, "detach"):
        result = result.detach().cpu().numpy()
    return np.asarray(result, dtype=np.float64).reshape(-1, np.asarray(result).shape[-1])[:, :2]


def feet_center(keypoints: np.ndarray, scores: np.ndarray, schema: str,
                threshold: float) -> tuple[tuple[float, float] | None, str]:
    definition = SCHEMAS[schema]
    feet: list[np.ndarray] = []
    for side in ("left_foot", "right_foot"):
        pts = [np.asarray(keypoints[i], dtype=float) for i in definition[side]
               if i < len(scores) and float(scores[i]) >= threshold
               and np.isfinite(keypoints[i]).all()]
        if pts:
            feet.append(np.mean(pts, axis=0))
    label = ("both_feet" if len(feet) == 2 else "one_foot" if len(feet) == 1 else "unavailable")
    return tuple(map(float, np.mean(feet, axis=0))) if feet else None, label


def pelvis_center_x(keypoints: np.ndarray, scores: np.ndarray, schema: str,
                    threshold: float) -> tuple[float | None, str, float, float]:
    d = SCHEMAS[schema]
    ls, rs = float(scores[d["left_hip"]]), float(scores[d["right_hip"]])
    good_l = ls >= threshold and np.isfinite(keypoints[d["left_hip"]]).all()
    good_r = rs >= threshold and np.isfinite(keypoints[d["right_hip"]]).all()
    if good_l and good_r:
        return ((float(keypoints[d["left_hip"], 0]) + float(keypoints[d["right_hip"], 0])) / 2.0,
                "both_hips", ls, rs)
    if good_l:
        return (float(keypoints[d["left_hip"], 0]), "left_hip_only", ls, rs)
    if good_r:
        return (float(keypoints[d["right_hip"], 0]), "right_hip_only", ls, rs)
    return None, "unavailable", ls, rs


def sam_point(mask: np.ndarray, bbox: np.ndarray,
              bottom_band_frac: float, bottom_y_percentile: float) -> tuple[float, float] | None:
    """Exact crop-only largest-component / percentile+band estimator from Hybrid A."""
    x1, y1, x2, y2 = map(float, bbox)
    height, width = mask.shape[:2]
    if height < 1 or width < 1:
        return None
    ix1 = max(0, min(width - 1, int(np.floor(x1))))
    iy1 = max(0, min(height - 1, int(np.floor(y1))))
    ix2 = max(ix1 + 1, min(width, int(np.ceil(x2))))
    iy2 = max(iy1 + 1, min(height, int(np.ceil(y2))))
    crop = np.ascontiguousarray(mask[iy1:iy2, ix1:ix2] > 0, dtype=np.uint8)
    if not np.any(crop):
        return None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    if n <= 1:
        largest = crop.astype(bool)
    else:
        largest = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    local_ys, local_xs = np.where(largest)
    if len(local_xs) == 0:
        return None
    ys, xs = local_ys + iy1, local_xs + ix1
    y = float(np.percentile(ys.astype(np.float64), bottom_y_percentile))
    band = ys >= y - max((y2 - y1) * bottom_band_frac, 1.0)
    if not np.any(band):
        return None
    return float(np.median(xs[band])), y


def bin_label(value: float, edges: tuple[float, ...]) -> str:
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= value < hi:
            return f"{lo:g}+" if math.isinf(hi) else f"{lo:g}–{hi:g}"
    return "unknown"


def finite(x: float | None) -> float | str:
    return float(x) if x is not None and np.isfinite(x) else ""


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def axis_of_larger_error(ax: float, ay: float) -> str:
    if not np.isfinite([ax, ay]).all():
        return ""
    if abs(ax - ay) < 1e-9:
        return "equal"
    return "x" if ax > ay else "y"


class GroupStats:
    """Collect bounded-size columns by bin, rather than the entire detection CSV."""
    def __init__(self) -> None:
        self.total = 0
        self.matched = 0
        self.accurate = 0
        self.raw_accurate = 0
        self.raw_available = 0
        self.method_errors: list[float] = []
        self.baseline_errors: list[float] = []
        self.raw_errors: list[float] = []
        self.feet_errors: list[float] = []
        self.raw_abs_x: list[float] = []
        self.raw_abs_y: list[float] = []
        self.improvements: list[float] = []
        self.abs_x: list[float] = []
        self.abs_y: list[float] = []
        self.x_only: list[float] = []
        self.y_only: list[float] = []
        self.axis_x = 0
        self.axis_y = 0
        self.n_valid = 0

    def add(self, row: dict[str, Any]) -> None:
        self.total += 1
        if row["match_status"] != "matched":
            return
        self.matched += 1
        if row["raw_x_px"] != "":
            self.raw_available += 1
        if row["raw_error_m"] != "":
            self.raw_errors.append(float(row["raw_error_m"]))
            self.raw_accurate += int(float(row["raw_error_m"]) <= 0.25)
        if row["pose_feet_ground_error_m"] != "":
            self.feet_errors.append(float(row["pose_feet_ground_error_m"]))
        if row["raw_abs_dx_px"] != "":
            self.raw_abs_x.append(float(row["raw_abs_dx_px"]))
        if row["raw_abs_dy_px"] != "":
            self.raw_abs_y.append(float(row["raw_abs_dy_px"]))
        if row["prediction_error_m"] == "":
            return
        self.n_valid += 1
        error = float(row["prediction_error_m"])
        self.method_errors.append(error)
        self.accurate += int(error <= 0.25)
        for key, target in (("bbox_error_m", self.baseline_errors),
                            ("improvement_over_bbox_m", self.improvements),
                            ("abs_dx_px", self.abs_x), ("abs_dy_px", self.abs_y),
                            ("pitch_error_x_only_m", self.x_only),
                            ("pitch_error_y_only_m", self.y_only)):
            if row[key] != "":
                target.append(float(row[key]))
        if row["dominant_projection_error_axis"] == "x":
            self.axis_x += 1
        if row["dominant_projection_error_axis"] == "y":
            self.axis_y += 1

    def output(self, label: str) -> dict[str, Any]:
        def avg(v: list[float]) -> float | str:
            return float(np.mean(v)) if v else ""
        return dict(size_bin=label, detections=self.total, bbox_matched=self.matched,
                    evaluated=self.n_valid, raw_method_available=self.raw_available,
                    matched_within_025m=self.accurate,
                    within_025m_rate_among_evaluated=(self.accurate / self.n_valid if self.n_valid else ""),
                    mean_error_m=avg(self.method_errors),
                    median_error_m=(float(np.median(self.method_errors)) if self.method_errors else ""),
                    mean_baseline_error_m=avg(self.baseline_errors),
                    mean_raw_error_m=avg(self.raw_errors),
                    mean_pose_feet_error_m=avg(self.feet_errors),
                    raw_within_025m_rate_among_evaluated=(self.raw_accurate / len(self.raw_errors) if self.raw_errors else ""),
                    mean_raw_abs_dx_px=avg(self.raw_abs_x), mean_raw_abs_dy_px=avg(self.raw_abs_y),
                    mean_improvement_vs_bbox_m=avg(self.improvements),
                    mean_abs_dx_px=avg(self.abs_x), mean_abs_dy_px=avg(self.abs_y),
                    mean_pitch_x_only_m=avg(self.x_only), mean_pitch_y_only_m=avg(self.y_only),
                    image_x_dominant_count=self.axis_x, image_y_dominant_count=self.axis_y)


def prepare_detection(row: dict[str, Any], method: str, data: dict[str, Any],
                      index: int, original: int, gt_list: list[dict[str, Any]],
                      match: tuple[int, float] | None, args: argparse.Namespace) -> dict[str, Any]:
    box = data["bboxes"][original]
    x1, y1, x2, y2 = map(float, box)
    w, h = x2 - x1, y2 - y1
    aw, ah = data["width"], data["height"]
    center = (x1 + x2) / 2.0
    bbox_pt = (center, y2)
    row.update(method=method, split=args.split, image_id=data["image_id"],
               file_name=data["file_name"], detection_index=original,
               detection_score=float(data["detection_scores"][original]),
               match_status="unmatched_bbox_iou", image_width_px=aw, image_height_px=ah,
               bbox_x1_px=x1, bbox_y1_px=y1, bbox_x2_px=x2, bbox_y2_px=y2,
               bbox_width_px=w, bbox_height_px=h, bbox_area_px2=w * h,
               bbox_area_fraction=w * h / (aw * ah),
               bbox_aspect_ratio=(w / h if h > 0 else ""),
               bbox_bottom_y_fraction=y2 / ah,
               bbox_height_bin=bin_label(h, HEIGHT_EDGES),
               bbox_area_bin=bin_label(w * h, AREA_EDGES),
               bbox_ground_x_px=center, bbox_ground_y_px=y2)
    raw_pt: tuple[float, float] | None = None
    pred_pt = bbox_pt
    if MODE == "pose":
        k = data["keypoints"][original]
        s = data["keypoint_scores"][original]
        hip_x, hip_label, left_conf, right_conf = pelvis_center_x(k, s, args.pose_schema, args.keypoint_conf)
        feet_pt, feet_label = feet_center(k, s, args.pose_schema, args.keypoint_conf)
        feet_x = feet_pt[0] if feet_pt is not None else None
        diff = abs(hip_x - feet_x) if hip_x is not None and feet_x is not None else None
        row.update(pose_pelvis_x_px=finite(hip_x), pose_pelvis_source=hip_label,
                   pose_feet_x_px=finite(feet_x),
                   pose_feet_y_px=finite(feet_pt[1] if feet_pt is not None else None),
                   pose_feet_source=feet_label,
                   pelvis_feet_abs_dx_px=finite(diff),
                   pose_left_hip_conf=finite(left_conf), pose_right_hip_conf=finite(right_conf))
        if hip_x is not None:
            raw_pt = (float(np.clip(hip_x, x1, x2)), y2)
        if raw_pt is not None and diff is not None and diff <= args.pelvis_feet_threshold_px:
            pred_pt = raw_pt
            source = "pelvis_accepted"
        else:
            source = ("bbox_no_hips" if hip_x is None else
                      "bbox_no_feet" if feet_x is None else "bbox_pelvis_feet_disagree")
    else:
        if data["mask_valid"][original]:
            raw_pt = sam_point(data["masks"][original], box,
                               args.bottom_band_frac, args.bottom_y_percentile)
        if raw_pt is not None:
            pred_pt = (raw_pt[0], y2)
            source = "sam_x_bbox_y"
            row["sam_mask_x_px"] = raw_pt[0]
            row["sam_mask_y_px"] = raw_pt[1]
            row["sam_bbox_bottom_minus_mask_bottom_px"] = y2 - raw_pt[1]
        else:
            source = "bbox_fallback_invalid_sam"
    row.update(prediction_source=source, pred_x_px=pred_pt[0], pred_y_px=pred_pt[1],
               raw_x_px=(raw_pt[0] if raw_pt is not None else ""),
               raw_y_px=(raw_pt[1] if raw_pt is not None else ""))
    if MODE == "pose":
        row["_feet_pt"] = feet_pt
    row["_bbox_pt"] = bbox_pt
    row["_pred_pt"] = pred_pt
    row["_raw_pt"] = raw_pt
    if match is not None:
        gt_index, iou = match
        gt = gt_list[gt_index]
        row["match_status"] = "matched"
        row["gt_annotation_id"] = gt.get("id", "")
        row["matched_iou"] = iou
        if "position_on_pitch" in gt and len(gt["position_on_pitch"]) >= 2:
            gp = np.asarray(gt["position_on_pitch"][:2], dtype=float)
            if np.isfinite(gp).all():
                row["gt_pitch_x_m"], row["gt_pitch_y_m"] = map(float, gp)
                row["_gt_pitch"] = gp
        g = gt_ground_annotation(gt)
        if g is not None:
            g = (g[0] * aw / data["ann_width"], g[1] * ah / data["ann_height"])
            row["gt_x_px"], row["gt_y_px"] = g
            row["_gt_point"] = g
            dx = pred_pt[0] - g[0]
            dy = pred_pt[1] - g[1]
            row.update(signed_dx_px=dx, signed_dy_px=dy, abs_dx_px=abs(dx), abs_dy_px=abs(dy),
                       abs_dx_over_bbox_width=(abs(dx) / w if w > 0 else ""),
                       abs_dy_over_bbox_height=(abs(dy) / h if h > 0 else ""),
                       dominant_pixel_error_axis=axis_of_larger_error(abs(dx), abs(dy)))
            if raw_pt is not None:
                rdx = raw_pt[0] - g[0]
                rdy = raw_pt[1] - g[1]
                row.update(raw_signed_dx_px=rdx, raw_signed_dy_px=rdy,
                           raw_abs_dx_px=abs(rdx), raw_abs_dy_px=abs(rdy))
            if MODE == "pose" and feet_pt is not None:
                row["pose_feet_signed_dy_px"] = feet_pt[1] - g[1]
        if "_gt_pitch" not in row:
            row["match_status"] = "matched_no_gt_pitch"
        elif "_gt_point" not in row:
            row["match_status"] = "matched_no_gt_image_point"
    return row


def add_projected_errors(rows: list[dict[str, Any]], image: dict[str, Any],
                         aw: int, ah: int, gw: int, gh: int) -> None:
    """Batch projections for all matched detections in this single image."""
    points: list[tuple[float, float]] = []
    jobs: list[tuple[dict[str, Any], dict[str, int]]] = []
    for row in rows:
        if "_gt_pitch" not in row:
            continue
        markers: dict[str, int] = {}
        for name, point in (("bbox", row["_bbox_pt"]), ("pred", row["_pred_pt"]),
                            ("raw", row["_raw_pt"]), ("feet", row.get("_feet_pt")),
                            ("gt", row.get("_gt_point"))):
            if point is not None:
                markers[name] = len(points)
                points.append(point)
        g = row.get("_gt_point")
        if g is not None:
            markers["x_only"] = len(points)
            points.append((row["_pred_pt"][0], g[1]))
            markers["y_only"] = len(points)
            points.append((g[0], row["_pred_pt"][1]))
        jobs.append((row, markers))
    if not points:
        return
    projected = projection(np.asarray(points, dtype=np.float32), image, aw, ah, gw, gh)
    if len(projected) != len(points):
        raise RuntimeError("Camera projection returned an unexpected number of points")
    for row, positions in jobs:
        gt_pitch = row["_gt_pitch"]
        def pp(key: str) -> np.ndarray | None:
            if key not in positions:
                return None
            xy = projected[positions[key]]
            return xy if np.isfinite(xy).all() else None
        baseline, prediction, raw, feet, g, xonly, yonly = (
            pp("bbox"), pp("pred"), pp("raw"), pp("feet"), pp("gt"), pp("x_only"), pp("y_only"))
        if prediction is None or baseline is None:
            continue
        b_error = distance(baseline, gt_pitch)
        p_error = distance(prediction, gt_pitch)
        row["bbox_error_m"] = b_error
        row["prediction_error_m"] = p_error
        row["improvement_over_bbox_m"] = b_error - p_error
        row["prediction_beats_bbox"] = bool(p_error < b_error)
        row["correct_within_025m"] = p_error <= 0.25
        row["correct_within_050m"] = p_error <= 0.50
        row["correct_within_100m"] = p_error <= 1.00
        row["pitch_signed_dx_m"] = float(prediction[0] - gt_pitch[0])
        row["pitch_signed_dy_m"] = float(prediction[1] - gt_pitch[1])
        if raw is not None:
            raw_error = distance(raw, gt_pitch)
            row["raw_error_m"] = raw_error
            row["raw_improvement_over_bbox_m"] = b_error - raw_error
            row["raw_beats_bbox"] = bool(raw_error < b_error)
        if feet is not None:
            row["pose_feet_ground_error_m"] = distance(feet, gt_pitch)
        if g is not None:
            row["gt_projection_disagreement_m"] = distance(g, gt_pitch)
            if xonly is not None and yonly is not None:
                x_m, y_m = distance(xonly, g), distance(yonly, g)
                row["pitch_error_x_only_m"] = x_m
                row["pitch_error_y_only_m"] = y_m
                row["dominant_projection_error_axis"] = axis_of_larger_error(x_m, y_m)


def main() -> None:
    args = arguments()
    if MODE == "sam":
        # Per-player crops are small; avoid OpenCV thread oversubscription.
        cv2.setNumThreads(1)
    if not (0 <= args.match_iou <= 1):
        raise ValueError("--match-iou must be in [0, 1]")
    if args.score_threshold < 0:
        raise ValueError("--score-threshold must be >= 0")
    if MODE == "sam" and not (0 <= args.bottom_y_percentile <= 100 and args.bottom_band_frac > 0):
        raise ValueError("SAM percentile/band settings invalid")
    if MODE == "pose" and args.pelvis_feet_threshold_px < 0:
        raise ValueError("--pelvis-feet-threshold-px must be >= 0")

    root = args.dataset_root.expanduser().resolve()
    yolo_dir = args.yolo_cache_root.expanduser().resolve() / args.split / "yolo_raw"
    if MODE == "pose":
        cache_dir = args.pose_cache_root.expanduser().resolve() / args.split / args.pose_model / "pose_raw"
        run_label = f"{args.pose_model}_hybrid_{args.pelvis_feet_threshold_px:g}px"
    else:
        cache_dir = args.sam_cache_root.expanduser().resolve() / args.split / "sam_raw"
        run_label = f"sam_hybrid_a_band{args.bottom_band_frac:g}_p{args.bottom_y_percentile:g}"
    out = args.output_dir.expanduser().resolve() / args.split / run_label
    out.mkdir(parents=True, exist_ok=True)
    for p in (yolo_dir, cache_dir):
        if not p.is_dir():
            raise FileNotFoundError(p)
    yolo_files = sorted(yolo_dir.glob("*.npz"))
    if not yolo_files:
        raise FileNotFoundError(f"No YOLO npz files in {yolo_dir}")
    if args.max_images:
        yolo_files = yolo_files[:args.max_images]

    with annotation_path(root, args.split).open("r", encoding="utf-8") as f:
        dataset = json.load(f)
    image_map = {int(im["id"]): im for im in dataset["images"]}
    gt_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in dataset.get("annotations", []):
        if int(ann.get("category_id", 1)) == 1 and "bbox" in ann:
            gt_map[int(ann["image_id"])].append(ann)

    output_csv = out / ("pose_predictions_by_bbox_size.csv" if MODE == "pose" else "sam_predictions_by_bbox_size.csv")
    groups_height: dict[str, GroupStats] = defaultdict(GroupStats)
    groups_area: dict[str, GroupStats] = defaultdict(GroupStats)
    counts: Counter[str] = Counter()
    print(f"Mode: {MODE}, source-image bbox sizes, score >= {args.score_threshold}", flush=True)
    print(f"Cache: {cache_dir}", flush=True)
    print(f"CSV:   {output_csv}", flush=True)
    yolo_fields = ("image_id", "file_name", "actual_width", "actual_height",
                   "annotation_width", "annotation_height", "bboxes", "detection_scores")
    method_fields = ("bboxes", "detection_scores", "keypoints", "keypoint_scores") if MODE == "pose" else (
        "bboxes", "detection_scores", "masks")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for yp in tqdm(yolo_files, desc=f"Analyzing {MODE}", unit="image"):
            y = load_cache(yp, yolo_fields)
            image_id = num(y["image_id"])
            if image_id not in image_map:
                raise RuntimeError(f"Missing image {image_id} in annotations")
            method_path = cache_dir / yp.name
            if not method_path.is_file():
                raise FileNotFoundError(f"Missing {MODE} cache: {method_path}")
            m = load_cache(method_path, method_fields) if MODE == "pose" else None
            if MODE == "sam":
                with np.load(method_path, allow_pickle=False) as z:
                    m = {name: np.asarray(z[name]) for name in method_fields}
                    m["mask_valid"] = (np.asarray(z["mask_valid"], dtype=bool) if "mask_valid" in z.files
                                       else np.asarray([np.any(v) for v in m["masks"]], dtype=bool))
            assert m is not None
            if not np.array_equal(y["bboxes"], m["bboxes"]):
                raise RuntimeError(f"YOLO/{MODE.upper()} bbox cache mismatch: {yp.name}; regenerate {MODE} cache")
            if not np.array_equal(y["detection_scores"], m["detection_scores"]):
                raise RuntimeError(f"YOLO/{MODE.upper()} score cache mismatch: {yp.name}")
            if MODE == "pose" and (len(m["keypoints"]) != len(y["bboxes"]) or
                                    len(m["keypoint_scores"]) != len(y["bboxes"])):
                raise RuntimeError(f"Pose counts do not match: {yp.name}")
            if MODE == "sam" and (len(m["masks"]) != len(y["bboxes"]) or len(m["mask_valid"]) != len(y["bboxes"])):
                raise RuntimeError(f"SAM counts do not match: {yp.name}")
            aw, ah = num(y["actual_width"]), num(y["actual_height"])
            gw, gh = num(y["annotation_width"]), num(y["annotation_height"])
            if MODE == "sam" and len(y["bboxes"]) and m["masks"].shape[1:] != (ah, aw):
                raise RuntimeError(f"SAM mask shapes do not match source image dimensions: {yp.name}")
            data = dict(image_id=image_id, file_name=txt(y["file_name"]), width=aw, height=ah,
                        ann_width=gw, ann_height=gh, bboxes=y["bboxes"], detection_scores=y["detection_scores"], **{
                            k: v for k, v in m.items() if k not in ("bboxes", "detection_scores")})
            selected = np.flatnonzero(y["detection_scores"] >= args.score_threshold)
            boxes = y["bboxes"][selected].astype(np.float64)
            boxes_ann = (boxes * np.asarray([gw / aw, gh / ah, gw / aw, gh / ah]))
            gt_list = gt_map.get(image_id, [])
            gt_boxes = np.asarray([np.array([a["bbox"][0], a["bbox"][1],
                                                   a["bbox"][0] + a["bbox"][2],
                                                   a["bbox"][1] + a["bbox"][3]], dtype=float)
                                   for a in gt_list], dtype=float).reshape(-1, 4)
            matches = hungarian_matches(boxes_ann, gt_boxes, args.match_iou)
            records = []
            for j, orig in enumerate(selected):
                row: dict[str, Any] = {name: "" for name in FIELDS}
                records.append(prepare_detection(row, MODE, data, j, int(orig), gt_list, matches.get(j), args))
            add_projected_errors(records, image_map[image_id], aw, ah, gw, gh)
            for record in records:
                writer.writerow({name: record[name] for name in FIELDS})
                groups_height[record["bbox_height_bin"]].add(record)
                groups_area[record["bbox_area_bin"]].add(record)
                counts[record["match_status"]] += 1
                counts[record["prediction_source"]] += 1
    def emit_groups(path: Path, groups: dict[str, GroupStats], edges: tuple[float, ...]) -> None:
        labels = [bin_label(lo, edges) for lo in edges[:-1]]
        results = [groups[label].output(label) for label in labels]
        with path.open("w", newline="", encoding="utf-8") as f:
            out_writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            out_writer.writeheader()
            out_writer.writerows(results)

    height_file = out / "summary_by_bbox_height.csv"
    area_file = out / "summary_by_bbox_area.csv"
    emit_groups(height_file, groups_height, HEIGHT_EDGES)
    emit_groups(area_file, groups_area, AREA_EDGES)
    summary_file = out / "run_summary.json"
    summary = dict(method=MODE, split=args.split, score_threshold=args.score_threshold,
                   match_iou=args.match_iou, source_image_pixel_space=True,
                   output_csv=str(output_csv), image_files=len(yolo_files),
                   counts=dict(counts),
                   notes=["Each row is one frozen YOLO detection ABOVE the selected confidence threshold.",
                          "Hungarian bbox-IoU matching is independent of predicted ground position.",
                          "prediction_error_m compares predicted pitch point against annotated position_on_pitch.",
                       "raw_* columns describe pelvis-X+bbox-Y for pose, mask-X+mask-Y for SAM; pose_feet_* columns measure pose-estimated feet independent of the hybrid.",
                          "X/Y counterfactual displacements are computed relative to the CAMERA-PROJECTED GT image contact point, not annotated pitch GT; gt_projection_disagreement_m reports their mismatch.",
                          "This is a per-detection diagnostic, not official mAP-LocSim or 0.25m pitch-distance assignment precision/recall."])
    if MODE == "pose":
        summary.update(pose_model=args.pose_model, keypoint_conf=args.keypoint_conf,
                       pelvis_feet_threshold_px=args.pelvis_feet_threshold_px,
                       primary_prediction="pose pelvis X when pelvis–feet agreement passes threshold, otherwise bbox X; bbox-bottom Y",
                       raw_prediction="pose pelvis X + bbox-bottom Y")
    else:
        summary.update(bottom_band_frac=args.bottom_band_frac, bottom_y_percentile=args.bottom_y_percentile,
                       primary_prediction="SAM bottom-band median X + bbox-bottom Y",
                       raw_prediction="SAM bottom-band median X + SAM mask 98th-percentile Y")
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFinished. Records: {sum(s.total for s in groups_height.values())}")
    print(f"Per-prediction CSV: {output_csv}")
    print(f"Height summary:    {height_file}")
    print(f"Area summary:      {area_file}")
    print(f"Run metadata:      {summary_file}")


if __name__ == "__main__":
    main()
