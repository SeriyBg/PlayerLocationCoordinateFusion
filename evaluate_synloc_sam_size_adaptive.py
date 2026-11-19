#!/usr/bin/env python3
"""Evaluate size-adaptive SAM ground-contact localization on Spiideo SynLoc.

Uses the EXACT cached YOLO detections and SAM masks from the original
Hybrid A evaluator (no inference or new model training).

BBox baseline:    x = YOLO bbox center; y = YOLO bbox bottom
Hybrid A:         x = SAM mask bottom-band median X; y = bbox bottom
Size adaptive:    x = SAM mask bottom-band median X, always when valid;
                  y = bbox bottom if bbox HEIGHT < threshold;
                  y = SAM mask bottom percentile if bbox HEIGHT >= threshold.
                  Invalid SAM masks fall back to bbox bottom-center.

IMPORTANT: bbox height thresholds are in the *actual source image* pixels,
not the potentially higher-resolution COCO annotation coordinate system.

Supports multiple thresholds without reading/decompressing SAM masks again.
Official LocSim evaluation uses exactly the same evaluator/settings as the
pre-existing evaluate_synloc_sam_hybrid_a.py script.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm
from xtcocotools.coco import COCO
from sskit import image_to_ground, match
from sskit.coco import LocSimCOCOeval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--yolo-cache-root', type=Path,
                   default=Path('output/synloc_yolo_cache'))
    p.add_argument('--sam-cache-root', type=Path,
                   default=Path('output/synloc_sam_cache'))
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--output-dir', type=Path,
                   default=Path('output/synloc_sam_size_adaptive_eval'))
    p.add_argument('--height-thresholds', nargs='+', type=float, default=[60.0],
                   metavar='PX',
                   help='Switch from bbox-bottom Y to SAM mask Y at these '                        'source-image bbox heights (default: 60). '                        'Example: 50 60 70 80 90')
    p.add_argument('--bottom-band-frac', type=float, default=0.05)
    p.add_argument('--bottom-y-percentile', type=float, default=98.0)
    p.add_argument('--score-threshold', type=float, default=0.4302865565,
                   help='Same fixed LocSim/baseline-style score cutoff for '                        'all compared methods (default: 0.4302865565).')
    p.add_argument('--checkpoint-every', type=int, default=500,
                   help='Save intermediate extracted points every N images; '                        '0 saves only at completion.')
    p.add_argument('--restart', action='store_true',
                   help='Discard existing extracted-points checkpoint.')
    p.add_argument('--save-per-detection-csv', action='store_true',
                   help='Save an extra per-detection CSV containing every source '                        'point and selected Y for each threshold.')
    a = p.parse_args()
    if not a.height_thresholds or any(not np.isfinite(v) or v < 0
                                      for v in a.height_thresholds):
        p.error('--height-thresholds must contain finite nonnegative pixel heights')
    a.height_thresholds = sorted(set(a.height_thresholds))
    if not (0 <= a.bottom_y_percentile <= 100):
        p.error('--bottom-y-percentile must be between 0 and 100')
    if not (0 < a.bottom_band_frac <= 1):
        p.error('--bottom-band-frac must be in (0, 1]')
    if not (0 <= a.score_threshold <= 1):
        p.error('--score-threshold must be in [0,1]')
    if a.checkpoint_every < 0:
        p.error('--checkpoint-every must be >= 0')
    return a


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def resolve_annotation_path(root: Path, split: str) -> Path:
    roots = (root, root / 'SpiideoSynLoc')
    aliases = ('val', 'valid', 'validation') if split == 'val' else ('test',)
    candidates = [r / 'annotations' / (name + '.json')
                  for r in roots for name in aliases]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError('Cannot find annotations. Tried:\n  ' +
                            '\n  '.join(map(str, candidates)))


def scalar(array: Any) -> Any:
    return np.asarray(array).item()


def largest_component(binary_crop: np.ndarray) -> np.ndarray:
    crop = np.ascontiguousarray(binary_crop > 0, dtype=np.uint8)
    if not np.any(crop):
        return np.zeros_like(crop, dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    if n <= 1:
        return crop.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def sam_bottom_band_point(mask: np.ndarray,
                          bbox: np.ndarray,
                          bottom_band_frac: float,
                          bottom_y_percentile: float,
                          ) -> tuple[float, float] | None:
    """Same crop-only algorithm as the fast SAM error diagnostic."""
    x1, y1, x2, y2 = map(float, bbox)
    height, width = mask.shape[:2]
    if height < 1 or width < 1:
        return None

    ix1 = max(0, min(width - 1, int(np.floor(x1))))
    iy1 = max(0, min(height - 1, int(np.floor(y1))))
    ix2 = max(ix1 + 1, min(width, int(np.ceil(x2))))
    iy2 = max(iy1 + 1, min(height, int(np.ceil(y2))))

    component = largest_component(mask[iy1:iy2, ix1:ix2])
    local_ys, local_xs = np.where(component)
    if len(local_xs) == 0:
        return None
    ys = local_ys + iy1
    xs = local_xs + ix1
    mask_bottom_y = float(np.percentile(ys.astype(np.float64),
                                        bottom_y_percentile))
    band_height = max((y2 - y1) * bottom_band_frac, 1.0)
    bottom_band = ys >= (mask_bottom_y - band_height)
    if not np.any(bottom_band):
        return None
    return float(np.median(xs[bottom_band])), mask_bottom_y


def scale_box(bbox: np.ndarray, aw: int, ah: int, gw: int, gh: int) -> list[float]:
    x1, y1, x2, y2 = map(float, bbox)
    sx, sy = gw / aw, gh / ah
    return [x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy]


def scale_point(point: tuple[float, float],
                aw: int, ah: int, gw: int, gh: int) -> tuple[float, float]:
    return point[0] * gw / aw, point[1] * gh / ah


def make_detection(det_id: int, image_id: int, score: float,
                   box_xywh: list[float], point: tuple[float, float]) -> dict[str, Any]:
    """Format compatible with the previously used SynLoc LocSimCOCOeval."""
    x, y, w, h = box_xywh
    return {
        'id': int(det_id),
        'image_id': int(image_id),
        'category_id': 1,
        'score': float(score),
        'bbox': [float(x), float(y), float(w), float(h)],
        'area': float(max(w, 0.0) * max(h, 0.0)),
        'keypoints': [float(x + w / 2), float(y + h / 2), 1.0,
                      float(point[0]), float(point[1]), 1.0],
    }


def get_input_data(yolo_path: Path, sam_path: Path) -> dict[str, Any]:
    """Load and *strictly verify* one-to-one canonical detection identity."""
    with np.load(yolo_path, allow_pickle=False) as y:
        boxes = np.asarray(y['bboxes'], dtype=np.float32)
        scores = np.asarray(y['detection_scores'], dtype=np.float32)
        image_id = int(scalar(y['image_id']))
        filename = str(scalar(y['file_name']))
        aw, ah = int(scalar(y['actual_width'])), int(scalar(y['actual_height']))
        gw, gh = int(scalar(y['annotation_width'])), int(scalar(y['annotation_height']))

    with np.load(sam_path, allow_pickle=False) as s:
        other_boxes = np.asarray(s['bboxes'], dtype=np.float32)
        other_scores = np.asarray(s['detection_scores'], dtype=np.float32)
        if not np.array_equal(boxes, other_boxes):
            raise RuntimeError(f'YOLO/SAM bbox cache mismatch: {yolo_path.name}. '
                               'Regenerate SAM cache from current YOLO cache.')
        if not np.array_equal(scores, other_scores):
            raise RuntimeError(f'YOLO/SAM score cache mismatch: {yolo_path.name}')
        if int(scalar(s['image_id'])) != image_id:
            raise RuntimeError(f'Image ID mismatch: {yolo_path.name}')
        masks = np.asarray(s['masks'])
        if 'mask_valid' in s.files:
            valid = np.asarray(s['mask_valid'], dtype=bool)
        else:
            valid = np.asarray([bool(np.any(m)) for m in masks], dtype=bool)

    if boxes.shape != (len(scores), 4):
        raise RuntimeError(f'Invalid YOLO bbox/score shapes for {filename}')
    if len(masks) != len(boxes) or len(valid) != len(boxes):
        raise RuntimeError(f'Mask/detection count mismatch in {sam_path.name}')
    if len(boxes) and masks.shape[1:] != (ah, aw):
        raise RuntimeError(f'Mask image dimensions do not match in {sam_path.name}: '
                           f'{masks.shape[1:]} != {(ah, aw)}')
    if min(aw, ah, gw, gh) <= 0:
        raise RuntimeError(f'Invalid dimensions in {yolo_path.name}')
    return dict(image_id=image_id, filename=filename, aw=aw, ah=ah, gw=gw, gh=gh,
                boxes=boxes, scores=scores, masks=masks, valid=valid)


def evaluate_locsim(annotation_path: Path, result_path: Path,
                    score_threshold: float) -> dict[str, float]:
    gt = COCO(str(annotation_path))
    det = gt.loadRes(str(result_path))
    evaluator = LocSimCOCOeval(gt, det, 'bbox', [0.089, 0.089], True)
    evaluator.params.useSegm = None
    evaluator.params.position_from_keypoint_index = 1
    evaluator.params.score_threshold = float(score_threshold)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        'mAP_LocSim_50_95': float(evaluator.stats[0]),
        'score_threshold': float(evaluator.stats[15]),
        'frame_accuracy_LocSim_50': float(evaluator.stats[16]),
    }


def project_result_to_pitch(result: dict[str, Any],
                            info: dict[str, Any]) -> np.ndarray:
    pts = np.asarray(result['keypoints'], dtype=np.float32).reshape(-1, 3)
    x, y = pts[1, :2]
    width, height = float(info['width']), float(info['height'])
    normalized = np.asarray([[(x - (width - 1.0) / 2.0) / width,
                              (y - (height - 1.0) / 2.0) / width]], dtype=np.float32)
    pitch = image_to_ground(info['camera_matrix'], info['undist_poly'], normalized)
    if hasattr(pitch, 'detach'):
        return pitch[:, :2].detach().cpu().numpy()[0]
    return np.asarray(pitch, dtype=np.float32)[0, :2]


def compute_baseline_style_stats(annotation_path: Path,
                                 results_path: Path, score_threshold: float,
                                 distance_threshold_m: float = 0.25,
                                 ) -> dict[str, Any]:
    """Same physical-distance matching as in the pose hybrid evaluator."""
    dataset = read_json(annotation_path)
    infos = {int(im['id']): im for im in dataset['images']}
    gt_by_image: dict[int, list[np.ndarray]] = defaultdict(list)
    for annotation in dataset.get('annotations', []):
        if int(annotation.get('category_id', 1)) != 1:
            continue
        gt_by_image[int(annotation['image_id'])].append(
            np.asarray(annotation['position_on_pitch'][:2], dtype=np.float32))

    est_by_image: dict[int, list[np.ndarray]] = defaultdict(list)
    for result in read_json(results_path):
        if float(result['score']) < score_threshold:
            continue
        image_id = int(result['image_id'])
        est_by_image[image_id].append(project_result_to_pitch(result, infos[image_id]))

    detected = missed = extra = 0
    distances: list[float] = []
    for image_id in infos:
        gt = np.asarray(gt_by_image.get(image_id, []), dtype=np.float32).reshape(-1, 2)
        est = np.asarray(est_by_image.get(image_id, []), dtype=np.float32).reshape(-1, 2)
        if len(gt) == 0:
            extra += len(est)
        elif len(est) == 0:
            missed += len(gt)
        else:
            n_detected, n_missed, n_extra, found_distances, _ = match(
                gt, est, th=distance_threshold_m)
            detected += int(n_detected)
            missed += int(n_missed)
            extra += int(n_extra)
            distances.extend(map(float, found_distances))

    p = detected / (detected + extra) if detected + extra else 0.0
    r = detected / (detected + missed) if detected + missed else 0.0
    return {
        'distance_threshold_m': distance_threshold_m,
        'detected': detected, 'missed': missed, 'extra': extra,
        'precision': p, 'recall': r,
        'mean_matched_localization_error_m':
            float(np.mean(distances)) if distances else None,
    }


def print_metrics(title: str, metrics: dict[str, Any]) -> None:
    print('\n' + title)
    print('-' * len(title))
    locsim = metrics['locsim']
    physical = metrics['baseline_style']
    print(f"mAP-LocSim:       {locsim['mAP_LocSim_50_95']:.6f}")
    print(f"Frame accuracy:  {locsim['frame_accuracy_LocSim_50']:.6f}")
    print(f"Score threshold: {locsim['score_threshold']:.10f}")
    print(f"@0.25m detected: {physical['detected']}")
    print(f"@0.25m missed:   {physical['missed']}")
    print(f"@0.25m extra:    {physical['extra']}")
    print(f"@0.25m precision:{physical['precision']:.6f}")
    print(f"@0.25m recall:   {physical['recall']:.6f}")
    print(f"Mean error (m):  {physical['mean_matched_localization_error_m']}")



# One row per cached YOLO detection; points and bbox are stored in annotation
# coordinates for exact reuse by LocSim. Height remains in source-image pixels.
POINT_DTYPE = np.dtype([
    ('image_id', np.int64), ('score', np.float32),
    ('box_x', np.float64), ('box_y', np.float64),
    ('box_w', np.float64), ('box_h', np.float64),
    ('bbox_x', np.float64), ('bbox_y', np.float64),
    ('sam_x', np.float64), ('sam_y', np.float64),
    ('height_px', np.float64), ('source_code', np.uint8),
])
# 0: mask absent/invalid, 1: mask present but unusable, 2: usable SAM point.
SOURCE_NAMES = {0: 'bbox_fallback_no_mask',
                1: 'bbox_fallback_empty_mask',
                2: 'sam_usable'}


def extracted_points_checkpoint(path: Path, points: np.ndarray,
                                processed_ids: set[int], settings: dict) -> None:
    """Small atomic checkpoint, NOT repeated enormous detection JSON files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, points=points,
                            processed_image_ids=np.array(sorted(processed_ids),
                                                         dtype=np.int64),
                            settings_json=np.asarray(json.dumps(settings, sort_keys=True)))
    tmp.replace(path)


def get_points(yolo_files: list[Path], yolo_dir: Path, sam_dir: Path,
               image_infos: dict[int, dict[str, Any]],
               checkpoint_path: Path, settings: dict[str, Any],
               restart: bool, checkpoint_every: int,
               band_frac: float, bottom_percentile: float) -> np.ndarray:
    processed: set[int] = set()
    records: list[tuple[Any, ...]] = []
    if checkpoint_path.exists() and not restart:
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            old_settings = json.loads(str(scalar(archive['settings_json'])))
            if old_settings != settings:
                raise RuntimeError('Existing point checkpoint was generated with different '
                                   'cache locations or extraction settings. '
                                   'Use --restart to recompute.')
            stored = np.asarray(archive['points'])
            if stored.dtype != POINT_DTYPE:
                raise RuntimeError('Incompatible checkpoint schema. Use --restart.')
            records = [tuple(row) for row in stored.tolist()]
            processed = set(map(int, archive['processed_image_ids']))
        if len(processed) != len(set(processed)):
            raise RuntimeError('Duplicate processed image IDs in checkpoint')
        print(f'Reusing cached SAM ground points: {len(processed)}/{len(yolo_files)} images')

    n_new = 0
    for yolo_path in tqdm(yolo_files, desc='Extract SAM contact points', unit='image'):
        with np.load(yolo_path, allow_pickle=False) as archive:
            image_id = int(scalar(archive['image_id']))
        if image_id in processed:
            continue
        sam_path = sam_dir / yolo_path.name
        if not sam_path.is_file():
            raise FileNotFoundError(f'Missing SAM cache: {sam_path}')
        data = get_input_data(yolo_path, sam_path)  # strict bbox/score equality
        if image_id not in image_infos:
            raise RuntimeError(f'Cached image ID {image_id} absent from ground truth')
        info = image_infos[image_id]
        aw, ah, gw, gh = data['aw'], data['ah'], data['gw'], data['gh']
        if gw != int(info['width']) or gh != int(info['height']):
            raise RuntimeError(f'Annotation dimensions differ for image {image_id}')

        for box, score, mask, valid in zip(data['boxes'], data['scores'],
                                          data['masks'], data['valid']):
            x1, y1, x2, y2 = map(float, box)
            bbox_x = (x1 + x2) / 2.0
            source_code = 0
            sam_x, sam_y = bbox_x, y2
            if valid:
                mask_point = sam_bottom_band_point(mask, box, band_frac,
                                                   bottom_percentile)
                if mask_point is not None and np.all(np.isfinite(mask_point)):
                    sam_x, sam_y = map(float, mask_point)
                    source_code = 2
                else:
                    source_code = 1
            ann_box = scale_box(box, aw, ah, gw, gh)
            ann_base = scale_point((bbox_x, y2), aw, ah, gw, gh)
            ann_sam = scale_point((sam_x, sam_y), aw, ah, gw, gh)
            records.append((image_id, float(score), *ann_box,
                            *ann_base, *ann_sam, y2 - y1, source_code))
        if image_id in processed:
            raise RuntimeError(f'Duplicate image ID {image_id} across YOLO cache files')
        processed.add(image_id)
        n_new += 1
        if checkpoint_every and n_new % checkpoint_every == 0:
            extracted_points_checkpoint(checkpoint_path,
                                        np.array(records, dtype=POINT_DTYPE),
                                        processed, settings)

    if processed != set(image_infos):
        missing = sorted(set(image_infos) - processed)
        extra = sorted(processed - set(image_infos))
        raise RuntimeError(f'Incomplete validation split: missing IDs {missing[:10]}, '
                           f'extra IDs {extra[:10]}. Refusing incomplete evaluation.')
    if len(processed) != len(yolo_files):
        raise RuntimeError(f'{len(yolo_files)} YOLO cache files for '
                           f'{len(processed)} unique images; duplicate cache IDs?')
    result = np.array(records, dtype=POINT_DTYPE)
    extracted_points_checkpoint(checkpoint_path, result, processed, settings)
    return result


def build_results(points: np.ndarray, method: str, threshold: float = 60.0,
                  ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Ground point selection is identical across bbox/Hybrid A/adaptive runs."""
    output: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for i, r in enumerate(points):
        valid = int(r['source_code']) == 2
        if method == 'bbox':
            x, y = float(r['bbox_x']), float(r['bbox_y'])
            source = 'bbox_center_bbox_bottom'
        elif method == 'hybrid_a':
            x = float(r['sam_x']) if valid else float(r['bbox_x'])
            y = float(r['bbox_y'])
            source = 'sam_x_bbox_bottom' if valid else SOURCE_NAMES[int(r['source_code'])]
        elif method == 'adaptive':
            if valid and float(r['height_px']) >= threshold:
                x, y = float(r['sam_x']), float(r['sam_y'])
                source = 'sam_x_sam_y'
            elif valid:
                x, y = float(r['sam_x']), float(r['bbox_y'])
                source = 'sam_x_bbox_bottom'
            else:
                x, y = float(r['bbox_x']), float(r['bbox_y'])
                source = SOURCE_NAMES[int(r['source_code'])]
        else:
            raise ValueError(f'Unknown method: {method}')
        box = [float(r[k]) for k in ('box_x', 'box_y', 'box_w', 'box_h')]
        output.append(make_detection(i + 1, int(r['image_id']), float(r['score']),
                                     box, (x, y)))
        counts[source] += 1
    return output, dict(counts)


def eval_results(annotation_path: Path, results_path: Path,
                 score_threshold: float) -> dict[str, Any]:
    locsim = evaluate_locsim(annotation_path, results_path, score_threshold)
    physical = compute_baseline_style_stats(annotation_path, results_path,
                                             score_threshold)
    return {'results_file': str(results_path),
            'locsim': locsim, 'baseline_style': physical}


def metric_delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float | int]:
    """a - b; positive is better, except mean error where negative is better."""
    return {
        'mAP_LocSim_50_95': a['locsim']['mAP_LocSim_50_95'] - b['locsim']['mAP_LocSim_50_95'],
        'frame_accuracy_LocSim_50': a['locsim']['frame_accuracy_LocSim_50'] - b['locsim']['frame_accuracy_LocSim_50'],
        'precision_0_25m': a['baseline_style']['precision'] - b['baseline_style']['precision'],
        'recall_0_25m': a['baseline_style']['recall'] - b['baseline_style']['recall'],
        'detected_0_25m': a['baseline_style']['detected'] - b['baseline_style']['detected'],
        'mean_matched_error_0_25m':
            (a['baseline_style']['mean_matched_localization_error_m'] -
             b['baseline_style']['mean_matched_localization_error_m'])
            if (a['baseline_style']['mean_matched_localization_error_m'] is not None
                and b['baseline_style']['mean_matched_localization_error_m'] is not None)
            else None,
    }


def threshold_slug(v: float) -> str:
    return (f'{v:g}').replace('.', 'p')


def write_per_detection_csv(path: Path, points: np.ndarray, threshold: float) -> None:
    with path.open('w', newline='', encoding='utf-8') as file:
        w = csv.writer(file)
        w.writerow(['image_id', 'detection_score', 'bbox_height_px', 'source',
                    'bbox_x_annotation_px', 'bbox_bottom_y_annotation_px',
                    'sam_x_annotation_px', 'sam_y_annotation_px',
                    'adaptive_x_annotation_px', 'adaptive_y_annotation_px'])
        for r in points:
            valid = int(r['source_code']) == 2
            if not valid:
                label = SOURCE_NAMES[int(r['source_code'])]
            elif float(r['height_px']) >= threshold:
                label = 'sam_x_sam_y'
            else:
                label = 'sam_x_bbox_bottom'
            w.writerow([int(r['image_id']), float(r['score']), float(r['height_px']),
                        label, float(r['bbox_x']), float(r['bbox_y']),
                        float(r['sam_x']) if valid else '',
                        float(r['sam_y']) if valid else '',
                        float(r['sam_x']) if valid else float(r['bbox_x']),
                        (float(r['sam_y']) if valid and float(r['height_px']) >= threshold
                         else float(r['bbox_y']))])


def main() -> None:
    args = parse_args()
    annotation_path = resolve_annotation_path(args.dataset_root.expanduser().resolve(),
                                              args.split)
    yolo_root = args.yolo_cache_root.expanduser().resolve()
    sam_root = args.sam_cache_root.expanduser().resolve()
    yolo_dir = yolo_root / args.split / 'yolo_raw'
    sam_dir = sam_root / args.split / 'sam_raw'
    if not yolo_dir.is_dir():
        raise FileNotFoundError(f'YOLO cache not found: {yolo_dir}')
    if not sam_dir.is_dir():
        raise FileNotFoundError(f'SAM cache not found: {sam_dir}')
    yolo_files = sorted(yolo_dir.glob('*.npz'))
    if not yolo_files:
        raise FileNotFoundError(f'No YOLO .npz files in {yolo_dir}')

    root = (args.output_dir.expanduser().resolve() / args.split /
            f'band{args.bottom_band_frac:g}_p{args.bottom_y_percentile:g}')
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / 'extracted_sam_points.npz'
    extraction_settings = {
        'schema_version': 1,
        'annotation_path': str(annotation_path),
        'yolo_cache_root': str(yolo_root),
        'sam_cache_root': str(sam_root),
        'split': args.split,
        'bottom_band_frac': args.bottom_band_frac,
        'bottom_y_percentile': args.bottom_y_percentile,
    }
    dataset = read_json(annotation_path)
    image_infos = {int(im['id']): im for im in dataset['images']}
    if not image_infos:
        raise RuntimeError('No annotated images')

    print(f'GT:     {annotation_path}', flush=True)
    print(f'YOLO:   {yolo_dir}', flush=True)
    print(f'SAM:    {sam_dir}', flush=True)
    print(f'Output: {root}', flush=True)
    print(f'Height thresholds in source-image pixels: {args.height_thresholds}', flush=True)

    points = get_points(yolo_files, yolo_dir, sam_dir, image_infos,
                        checkpoint_path, extraction_settings, args.restart,
                        args.checkpoint_every, args.bottom_band_frac,
                        args.bottom_y_percentile)
    print(f'Cached YOLO detections: {len(points)}', flush=True)
    print(f'Usable SAM masks: {int(np.count_nonzero(points["source_code"] == 2))}',
          flush=True)

    common = {}
    for method, filename in [('bbox', 'yolo_bbox_results.json'),
                             ('hybrid_a', 'yolo_sam_hybrid_a_results.json')]:
        print(f'\nEvaluating {method} ...', flush=True)
        out_path = root / filename
        detections, counts = build_results(points, method)
        save_json(out_path, detections)
        common[method] = eval_results(annotation_path, out_path, args.score_threshold)
        common[method]['ground_point_sources'] = counts
        print_metrics(method, common[method])
        del detections

    summary_rows = []
    for threshold in args.height_thresholds:
        folder = root / f'height_{threshold_slug(threshold)}px'
        folder.mkdir(parents=True, exist_ok=True)
        result_path = folder / 'yolo_sam_size_adaptive_results.json'
        summary_path = folder / 'evaluation_summary.json'
        print(f'\nEvaluating size-adaptive SAM at {threshold:g} px ...', flush=True)
        detections, counts = build_results(points, 'adaptive', threshold)
        save_json(result_path, detections)
        adaptive = eval_results(annotation_path, result_path, args.score_threshold)
        adaptive['ground_point_sources'] = counts
        del detections
        summary = {
            'split': args.split,
            'ground_point_mode': 'size_adaptive_sam_y',
            'height_threshold_px': threshold,
            'height_reference': 'source-image bbox height, not annotation pixels',
            'rule': ('SAM bottom-band X + bbox-bottom Y when height < threshold; '
                     'SAM bottom-band X + SAM bottom-percentile Y when height >= threshold; '
                     'bbox bottom-center fallback when SAM unavailable'),
            'bottom_band_frac': args.bottom_band_frac,
            'bottom_y_percentile': args.bottom_y_percentile,
            'score_threshold': args.score_threshold,
            'images': len(image_infos),
            'detections': len(points),
            'annotation_file': str(annotation_path),
            'yolo_cache_root': str(yolo_root),
            'sam_cache_root': str(sam_root),
            'bbox': common['bbox'],
            'hybrid_a': common['hybrid_a'],
            'adaptive': adaptive,
            'delta_vs_bbox': metric_delta(adaptive, common['bbox']),
            'delta_vs_hybrid_a': metric_delta(adaptive, common['hybrid_a']),
        }
        save_json(summary_path, summary)
        print_metrics(f'SIZE ADAPTIVE: {threshold:g} px', adaptive)
        print(f"Delta vs Hybrid A mAP-LocSim: "
              f"{summary['delta_vs_hybrid_a']['mAP_LocSim_50_95']:+.6f}", flush=True)
        if args.save_per_detection_csv:
            write_per_detection_csv(folder / 'ground_points.csv', points, threshold)
        summary_rows.append({
            'height_threshold_px': threshold,
            'mAP_LocSim_50_95': adaptive['locsim']['mAP_LocSim_50_95'],
            'delta_mAP_vs_bbox': summary['delta_vs_bbox']['mAP_LocSim_50_95'],
            'delta_mAP_vs_hybrid_a': summary['delta_vs_hybrid_a']['mAP_LocSim_50_95'],
            'frame_accuracy_LocSim_50': adaptive['locsim']['frame_accuracy_LocSim_50'],
            'precision_0_25m': adaptive['baseline_style']['precision'],
            'recall_0_25m': adaptive['baseline_style']['recall'],
            'detected_0_25m': adaptive['baseline_style']['detected'],
            'mean_matched_error_m_0_25m': adaptive['baseline_style']['mean_matched_localization_error_m'],
            'n_sam_x_sam_y': counts.get('sam_x_sam_y', 0),
            'n_sam_x_bbox_y': counts.get('sam_x_bbox_bottom', 0),
            'n_bbox_fallback': (counts.get('bbox_fallback_no_mask', 0) +
                                counts.get('bbox_fallback_empty_mask', 0)),
        })

    comparison_csv = root / 'threshold_comparison.csv'
    with comparison_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f'\nThreshold comparison: {comparison_csv}')
    print(f'Per-threshold summaries: {root}/height_<threshold>px/evaluation_summary.json')
    print('NOTE: threshold selection on validation data needs an independent test set.')


if __name__ == '__main__':
    main()
