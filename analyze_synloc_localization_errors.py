from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from sskit import image_to_ground


SCHEMAS = {
    "coco17": {"left_hip": 11, "right_hip": 12},
    "halpe26": {"left_hip": 11, "right_hip": 12},
    "coco25": {"left_hip": 11, "right_hip": 12},
    "coco133": {"left_hip": 11, "right_hip": 12},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Measure localization error only for YOLO detections that can be "
            "matched to a GT player by bbox IoU. Uses the canonical YOLO cache "
            "and pose cache; it does not rerun inference."
        )
    )
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--yolo-cache-root", type=Path, required=True)
    p.add_argument("--pose-cache-root", type=Path, required=True)
    p.add_argument("--pose-model", default="halpe26_m")
    p.add_argument("--pose-schema", choices=sorted(SCHEMAS), default="halpe26")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/synloc_localization_debug"),
    )
    p.add_argument(
        "--score-threshold",
        type=float,
        default=0.4302865565,
        help="Common YOLO confidence threshold used for the controlled comparison.",
    )
    p.add_argument(
        "--match-iou",
        type=float,
        default=0.50,
        help="Minimum bbox IoU used to say YOLO detected the GT player.",
    )
    p.add_argument("--keypoint-conf", type=float, default=0.30)
    return p.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_annotations(root: Path, split: str) -> Path:
    names = [split] if split != "val" else ["val", "valid", "validation"]
    for name in names:
        path = root / "annotations" / f"{name}.json"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find {split} annotations under {root / 'annotations'}")


def load_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as d:
        out = {
            "image_id": int(d["image_id"]),
            "file_name": str(d["file_name"].item()),
            "actual_width": int(d["actual_width"]),
            "actual_height": int(d["actual_height"]),
            "annotation_width": int(d["annotation_width"]),
            "annotation_height": int(d["annotation_height"]),
            "bboxes": np.asarray(d["bboxes"], dtype=np.float32),
            "detection_scores": np.asarray(d["detection_scores"], dtype=np.float32),
        }
        if "keypoints" in d:
            out["keypoints"] = np.asarray(d["keypoints"], dtype=np.float32)
        if "keypoint_scores" in d:
            out["keypoint_scores"] = np.asarray(d["keypoint_scores"], dtype=np.float32)
        return out


def xywh_to_xyxy(box: list[float]) -> np.ndarray:
    x, y, w, h = map(float, box)
    return np.asarray([x, y, x + w, y + h], dtype=np.float32)


def scale_box_actual_to_ann(box: np.ndarray, aw: int, ah: int, iw: int, ih: int) -> np.ndarray:
    sx, sy = iw / aw, ih / ah
    x1, y1, x2, y2 = map(float, box)
    return np.asarray([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=np.float32)


def ann_to_actual(point: tuple[float, float], aw: int, ah: int, iw: int, ih: int) -> tuple[float, float]:
    return point[0] * aw / iw, point[1] * ah / ih


def actual_to_ann(point: tuple[float, float], aw: int, ah: int, iw: int, ih: int) -> tuple[float, float]:
    return point[0] * iw / aw, point[1] * ih / ah


def pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    aa, bb = a[:, None, :], b[None, :, :]
    x1 = np.maximum(aa[..., 0], bb[..., 0])
    y1 = np.maximum(aa[..., 1], bb[..., 1])
    x2 = np.minimum(aa[..., 2], bb[..., 2])
    y2 = np.minimum(aa[..., 3], bb[..., 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, aa[..., 2] - aa[..., 0]) * np.maximum(0.0, aa[..., 3] - aa[..., 1])
    area_b = np.maximum(0.0, bb[..., 2] - bb[..., 0]) * np.maximum(0.0, bb[..., 3] - bb[..., 1])
    union = area_a + area_b - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_boxes(det_boxes: np.ndarray, gt_boxes: np.ndarray, min_iou: float) -> list[tuple[int, int, float]]:
    if len(det_boxes) == 0 or len(gt_boxes) == 0:
        return []
    ious = pairwise_iou(det_boxes, gt_boxes)
    di, gi = linear_sum_assignment(1.0 - ious)
    return [
        (int(d), int(g), float(ious[d, g]))
        for d, g in zip(di, gi)
        if float(ious[d, g]) >= min_iou
    ]


def bbox_bottom_center(box: np.ndarray) -> tuple[float, float]:
    x1, _, x2, y2 = map(float, box)
    return (x1 + x2) / 2.0, y2


def pelvis_bbox_bottom(
    keypoints: np.ndarray,
    scores: np.ndarray,
    box: np.ndarray,
    schema: str,
    threshold: float,
) -> tuple[float, float, str]:
    li = SCHEMAS[schema]["left_hip"]
    ri = SCHEMAS[schema]["right_hip"]
    x1, _, x2, y2 = map(float, box)
    left_ok = li < len(scores) and float(scores[li]) >= threshold
    right_ok = ri < len(scores) and float(scores[ri]) >= threshold

    if left_ok and right_ok:
        x = (float(keypoints[li, 0]) + float(keypoints[ri, 0])) / 2.0
        source = "both_hips"
    elif left_ok:
        x = float(keypoints[li, 0])
        source = "left_hip"
    elif right_ok:
        x = float(keypoints[ri, 0])
        source = "right_hip"
    else:
        x = (x1 + x2) / 2.0
        source = "bbox_center_fallback"

    return float(np.clip(x, x1, x2)), float(y2), source


def project_to_pitch(point_ann: tuple[float, float], image_info: dict[str, Any]) -> np.ndarray:
    x, y = point_ann
    width = float(image_info["width"])
    height = float(image_info["height"])
    normalized = np.asarray(
        [[(x - (width - 1.0) / 2.0) / width, (y - (height - 1.0) / 2.0) / width]],
        dtype=np.float32,
    )
    pitch = image_to_ground(
        image_info["camera_matrix"],
        image_info["undist_poly"],
        normalized,
    )
    if hasattr(pitch, "detach"):
        pitch = pitch[:, :2].detach().cpu().numpy()
    else:
        pitch = np.asarray(pitch, dtype=np.float32)[:, :2]
    return np.asarray(pitch[0], dtype=np.float32)


def stats(values: list[float]) -> dict[str, float | int | None]:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None}
    return {
        "count": int(len(a)),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
    }


def plot_hist(values: list[float], title: str, xlabel: str, path: Path, zero_line: bool = False) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=80)
    if zero_line:
        plt.axvline(0.0, linewidth=1.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()



def get_gt_ground_point_annotation(
    annotation: dict[str, Any],
) -> tuple[float, float] | None:
    """
    Return SynLoc GT ground keypoint (keypoint index 1) in annotation-image
    coordinates when it is available.

    Supported representations:
      * flat COCO-style: [x0, y0, v0, x1, y1, v1, ...]
      * nested: [[x0, y0, ...], [x1, y1, ...], ...]

    Some SynLoc annotations do not contain the second image-space keypoint.
    In that case return None rather than guessing.
    """
    raw = annotation.get("keypoints")

    if raw is None:
        return None

    try:
        keypoints = np.asarray(
            raw,
            dtype=np.float64,
        )
    except (TypeError, ValueError):
        return None

    # COCO-style flattened keypoints.
    if keypoints.ndim == 1:
        if keypoints.size < 6:
            return None

        x = float(keypoints[3])
        y = float(keypoints[4])

        if not (
            np.isfinite(x)
            and np.isfinite(y)
        ):
            return None

        return x, y

    # Nested keypoint representation.
    if (
        keypoints.ndim >= 2
        and keypoints.shape[0] >= 2
        and keypoints.shape[1] >= 2
    ):
        x = float(keypoints[1, 0])
        y = float(keypoints[1, 1])

        if not (
            np.isfinite(x)
            and np.isfinite(y)
        ):
            return None

        return x, y

    return None

def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    yolo_root = args.yolo_cache_root.expanduser().resolve()
    pose_root = args.pose_cache_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve() / args.split / args.pose_model
    output.mkdir(parents=True, exist_ok=True)

    dataset = load_json(resolve_annotations(dataset_root, args.split))
    images = {int(x["id"]): x for x in dataset.get("images", [])}
    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in dataset.get("annotations", []):
        if (
            int(ann.get("category_id", 1)) == 1
            and "bbox" in ann
            and "position_on_pitch" in ann
        ):
            gt_by_image[int(ann["image_id"])].append(ann)

    yolo_dir = yolo_root / args.split / "yolo_raw"
    pose_dir = pose_root / args.split / args.pose_model / "pose_raw"
    if not yolo_dir.is_dir():
        raise FileNotFoundError(yolo_dir)
    if not pose_dir.is_dir():
        raise FileNotFoundError(pose_dir)

    rows: list[dict[str, Any]] = []
    total_dets = total_gt = total_matches = 0
    matched_with_gt_ground_keypoint = 0
    matched_without_gt_ground_keypoint = 0

    for yolo_path in sorted(yolo_dir.glob("*.npz")):
        yolo = load_cache(yolo_path)
        image_id = yolo["image_id"]
        if image_id not in images:
            continue
        pose_path = pose_dir / yolo_path.name
        if not pose_path.is_file():
            raise FileNotFoundError(f"Missing pose cache: {pose_path}")
        pose = load_cache(pose_path)
        if not np.array_equal(yolo["bboxes"], pose["bboxes"]) or not np.array_equal(yolo["detection_scores"], pose["detection_scores"]):
            raise RuntimeError(f"YOLO/Pose cache mismatch: {yolo_path.name}")

        keep = np.flatnonzero(yolo["detection_scores"] >= args.score_threshold)
        boxes_actual = yolo["bboxes"][keep]
        scores = yolo["detection_scores"][keep]
        kpts = pose["keypoints"][keep]
        kpt_scores = pose["keypoint_scores"][keep]

        aw, ah = yolo["actual_width"], yolo["actual_height"]
        iw, ih = yolo["annotation_width"], yolo["annotation_height"]

        det_boxes_ann = np.asarray(
            [scale_box_actual_to_ann(b, aw, ah, iw, ih) for b in boxes_actual],
            dtype=np.float32,
        ).reshape(-1, 4)
        gt_anns = gt_by_image.get(image_id, [])
        gt_boxes_ann = np.asarray(
            [xywh_to_xyxy(a["bbox"]) for a in gt_anns], dtype=np.float32
        ).reshape(-1, 4)

        total_dets += len(boxes_actual)
        total_gt += len(gt_anns)
        matches = match_boxes(det_boxes_ann, gt_boxes_ann, args.match_iou)
        total_matches += len(matches)
        image_info = images[image_id]

        for local_idx, gt_idx, iou in matches:
            original_idx = int(keep[local_idx])
            box = boxes_actual[local_idx]
            gt = gt_anns[gt_idx]

            # SynLoc ground keypoint is keypoint index 1 when available.
            # Some annotations do not contain it, so do not assume that
            # gt["keypoints"] has at least 6 flattened values.
            gt_ground_ann = get_gt_ground_point_annotation(gt)

            if gt_ground_ann is None:
                matched_without_gt_ground_keypoint += 1
                continue

            matched_with_gt_ground_keypoint += 1

            gt_actual = ann_to_actual(
                gt_ground_ann,
                aw,
                ah,
                iw,
                ih,
            )
            gt_pitch = np.asarray(
                gt["position_on_pitch"][:2],
                dtype=np.float32,
            )

            bbox_pt = bbox_bottom_center(box)
            pose_x, pose_y, source = pelvis_bbox_bottom(
                kpts[local_idx], kpt_scores[local_idx], box, args.pose_schema, args.keypoint_conf
            )
            pose_pt = (pose_x, pose_y)

            bbox_pitch = project_to_pitch(actual_to_ann(bbox_pt, aw, ah, iw, ih), image_info)
            pose_pitch = project_to_pitch(actual_to_ann(pose_pt, aw, ah, iw, ih), image_info)

            bbox_error = float(np.linalg.norm(bbox_pitch - gt_pitch))
            pose_error = float(np.linalg.norm(pose_pitch - gt_pitch))
            bbox_dx = float(bbox_pt[0] - gt_actual[0])
            bbox_dy = float(bbox_pt[1] - gt_actual[1])
            pose_dx = float(pose_pt[0] - gt_actual[0])
            pose_dy = float(pose_pt[1] - gt_actual[1])

            x1, y1, x2, y2 = map(float, box)
            bw = max(x2 - x1, 1e-6)
            bh = max(y2 - y1, 1e-6)

            rows.append({
                "image_id": image_id,
                "file_name": yolo["file_name"],
                "detection_index": original_idx,
                "gt_annotation_id": int(gt.get("id", -1)),
                "yolo_score": float(scores[local_idx]),
                "bbox_match_iou": iou,
                "bbox_width_px": bw,
                "bbox_height_px": bh,
                "gt_ground_x_px": float(gt_actual[0]),
                "gt_ground_y_px": float(gt_actual[1]),
                "bbox_ground_x_px": float(bbox_pt[0]),
                "bbox_ground_y_px": float(bbox_pt[1]),
                "bbox_dx_px": bbox_dx,
                "bbox_dy_px": bbox_dy,
                "bbox_error_m": bbox_error,
                "pose_source": source,
                "pose_ground_x_px": pose_x,
                "pose_ground_y_px": pose_y,
                "pose_dx_px": pose_dx,
                "pose_dy_px": pose_dy,
                "pose_dx_bbox_fraction": pose_dx / bw,
                "pose_dy_bbox_fraction": pose_dy / bh,
                "pose_error_m": pose_error,
                "pose_better_than_bbox": pose_error < bbox_error,
                "improvement_m": bbox_error - pose_error,
                "pose_error_band": (
                    "<=0.10m" if pose_error <= 0.10 else
                    "0.10-0.25m" if pose_error <= 0.25 else
                    "0.25-0.50m" if pose_error <= 0.50 else
                    "0.50-0.75m" if pose_error <= 0.75 else
                    "0.75-1.00m" if pose_error <= 1.00 else
                    ">1.00m"
                ),
            })

    csv_path = output / "matched_localization_errors.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def subset(lo: float | None, hi: float | None) -> list[dict[str, Any]]:
        out = []
        for r in rows:
            e = float(r["pose_error_m"])
            if lo is not None and e <= lo:
                continue
            if hi is not None and e > hi:
                continue
            out.append(r)
        return out

    within_025 = subset(None, 0.25)
    borderline = subset(0.25, 0.50)
    within_050 = subset(None, 0.50)
    over_050 = subset(0.50, None)

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(group),
            "pose_error_m": stats([r["pose_error_m"] for r in group]),
            "signed_dx_px": stats([r["pose_dx_px"] for r in group]),
            "signed_dy_px": stats([r["pose_dy_px"] for r in group]),
            "abs_dx_px": stats([abs(r["pose_dx_px"]) for r in group]),
            "abs_dy_px": stats([abs(r["pose_dy_px"]) for r in group]),
        }

    summary = {
        "configuration": {
            "score_threshold": args.score_threshold,
            "bbox_match_iou": args.match_iou,
            "pose_model": args.pose_model,
            "pose_method": "pelvis_bbox_bottom",
            "signed_error_definition": (
                "prediction - GT; +dx = right of GT; +dy = lower/below GT in image"
            ),
        },
        "detection_matching": {
            "gt_players": total_gt,
            "yolo_detections_above_threshold": total_dets,
            "bbox_iou_matched_pairs": total_matches,
            "matched_pairs_with_gt_ground_keypoint": (
                matched_with_gt_ground_keypoint
            ),
            "matched_pairs_missing_gt_ground_keypoint": (
                matched_without_gt_ground_keypoint
            ),
            "unmatched_yolo": total_dets - total_matches,
            "unmatched_gt": total_gt - total_matches,
        },
        "all_bbox_matched_players": summarize(rows),
        "pose_within_0_25m": summarize(within_025),
        "pose_0_25_to_0_50m": summarize(borderline),
        "pose_within_0_50m": summarize(within_050),
        "pose_over_0_50m": summarize(over_050),
        "pose_better_than_bbox": {
            "count": sum(bool(r["pose_better_than_bbox"]) for r in rows),
            "rate": (
                sum(bool(r["pose_better_than_bbox"]) for r in rows) / len(rows)
                if rows else None
            ),
        },
    }

    summary_path = output / "localization_error_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if rows:
        plot_hist(
            [r["pose_error_m"] for r in rows],
            "Pose localization error on bbox-matched detections",
            "Pitch-space localization error (m)",
            output / "pose_error_m.png",
        )
    if within_050:
        plot_hist(
            [r["pose_dx_px"] for r in within_050],
            "Signed X error for pose predictions within 50 cm",
            "X error (px): positive = right of GT",
            output / "pose_dx_within_050m.png",
            zero_line=True,
        )
        plot_hist(
            [r["pose_dy_px"] for r in within_050],
            "Signed Y error for pose predictions within 50 cm",
            "Y error (px): positive = below GT",
            output / "pose_dy_within_050m.png",
            zero_line=True,
        )
    if borderline:
        plot_hist(
            [r["pose_dx_px"] for r in borderline],
            "Signed X error for 25-50 cm pose cases",
            "X error (px): positive = right of GT",
            output / "pose_dx_025_to_050m.png",
            zero_line=True,
        )
        plot_hist(
            [r["pose_dy_px"] for r in borderline],
            "Signed Y error for 25-50 cm pose cases",
            "Y error (px): positive = below GT",
            output / "pose_dy_025_to_050m.png",
            zero_line=True,
        )

    print("\n=== Detection-conditioned localization analysis ===")
    print(f"GT players:                    {total_gt}")
    print(f"YOLO detections >= threshold:  {total_dets}")
    print(f"BBox-IoU matched pairs:        {total_matches}")
    print(
        "Matched with GT ground point: "
        f"{matched_with_gt_ground_keypoint}"
    )
    print(
        "Matched missing GT ground point: "
        f"{matched_without_gt_ground_keypoint}"
    )
    print(f"Unmatched YOLO:                {total_dets - total_matches}")
    print(f"Unmatched GT:                  {total_gt - total_matches}")
    print(f"Pose <= 0.25 m:                {len(within_025)}")
    print(f"Pose 0.25-0.50 m:             {len(borderline)}")
    print(f"Pose <= 0.50 m:                {len(within_050)}")

    if borderline:
        dx = [r["pose_dx_px"] for r in borderline]
        dy = [r["pose_dy_px"] for r in borderline]
        print("\n=== Borderline 25-50 cm cases ===")
        print(f"Mean signed dx:   {np.mean(dx):.3f} px")
        print(f"Median signed dx: {np.median(dx):.3f} px")
        print(f"Mean signed dy:   {np.mean(dy):.3f} px")
        print(f"Median signed dy: {np.median(dy):.3f} px")

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {summary_path}")
    print(f"  plots under {output}")


if __name__ == "__main__":
    main()
