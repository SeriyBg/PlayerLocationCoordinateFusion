#!/usr/bin/env python3
"""Evaluate SynLoc Hybrid A: SAM bottom-band X + YOLO bbox-bottom Y.

Uses only the *canonical* YOLO/SAM .npz caches. Never runs YOLO or SAM.
Both approaches use identical YOLO detections, scores, and boxes.

    BBox baseline: ((x1 + x2)/2, y2)
    Hybrid A:      (SAM bottom-band X, y2) if SAM mask usable;
                   bbox baseline otherwise.

The SAM X definition matches analyze_synloc_sam_localization_errors_fast.py:
- bbox crop, largest 8-connected component,
- 98th-percentile mask Y,
- median X of mask pixels within the bottom 5%-of-bbox-height band
  (including mask pixels below the percentile).
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
    p.add_argument('--dataset-root', type=Path, required=True,
                   help='SpiideoSynLoc dataset directory, or its parent.')
    p.add_argument('--yolo-cache-root', type=Path,
                   default=Path('output/synloc_yolo_cache'))
    p.add_argument('--sam-cache-root', type=Path,
                   default=Path('output/synloc_sam_cache'))
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--output-dir', type=Path,
                   default=Path('output/synloc_sam_hybrid_eval'))
    p.add_argument('--bottom-band-frac', type=float, default=0.05)
    p.add_argument('--bottom-y-percentile', type=float, default=98.0)
    p.add_argument('--bbox-score-threshold', type=float,
                   default=0.4302865565,
                   help='Fixed score cutoff from canonical bbox val evaluation.')
    p.add_argument('--hybrid-score-threshold', type=float,
                   default=0.4302865565,
                   help='Fixed score cutoff for Hybrid A, same as bbox by default.')
    p.add_argument('--checkpoint-every', type=int, default=500,
                   help='Save resumable progress every N images; 0 disables checkpoints.')
    p.add_argument('--restart', action='store_true',
                   help='Ignore any existing progress checkpoint and start over.')
    p.add_argument('--save-per-detection-csv', action='store_true',
                   help='Save source X/Y and selection per detection in an extra CSV.')
    a = p.parse_args()
    if not 0.0 <= a.bottom_y_percentile <= 100.0:
        p.error('--bottom-y-percentile must be between 0 and 100')
    if not 0.0 < a.bottom_band_frac <= 1.0:
        p.error('--bottom-band-frac must be between 0 and 1')
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
        raise FileNotFoundError(f'No .npz files in {yolo_dir}')

    run_dir = (args.output_dir.expanduser().resolve() / args.split /
               f"hybrid_a_sam_x_bbox_y_band{args.bottom_band_frac:g}_p{args.bottom_y_percentile:g}")
    run_dir.mkdir(parents=True, exist_ok=True)
    bbox_result_path = run_dir / 'yolo_bbox_results.json'
    hybrid_result_path = run_dir / 'yolo_sam_hybrid_a_results.json'
    summary_path = run_dir / 'evaluation_summary.json'
    checkpoint_path = run_dir / 'progress.json'
    csv_path = run_dir / 'ground_points.csv'

    settings = dict(split=args.split,
                    bottom_band_frac=args.bottom_band_frac,
                    bottom_y_percentile=args.bottom_y_percentile,
                    yolo_cache_root=str(yolo_root), sam_cache_root=str(sam_root),
                    bbox_score_threshold=args.bbox_score_threshold,
                    hybrid_score_threshold=args.hybrid_score_threshold,
                    save_per_detection_csv=args.save_per_detection_csv)
    if checkpoint_path.exists() and not args.restart:
        progress = read_json(checkpoint_path)
        if progress.get('settings') != settings:
            raise RuntimeError('Existing progress checkpoint has different settings. '
                               'Choose --restart or another --output-dir.')
        processed = set(map(int, progress['completed_image_ids']))
        bbox_results = progress['bbox_results']
        hybrid_results = progress['hybrid_results']
        sources = Counter(progress['source_counts'])
        next_id = int(progress['next_id'])
        point_rows = progress.get('point_rows', [])
        print(f'Resuming {len(processed)}/{len(yolo_files)} images')
    else:
        processed: set[int] = set()
        bbox_results: list[dict[str, Any]] = []
        hybrid_results: list[dict[str, Any]] = []
        sources = Counter()
        next_id = 1
        point_rows: list[dict[str, Any]] = []

    def save_progress() -> None:
        save_json(checkpoint_path, dict(settings=settings,
                    completed_image_ids=sorted(processed),
                    bbox_results=bbox_results, hybrid_results=hybrid_results,
                    source_counts=dict(sources), next_id=next_id,
                    point_rows=point_rows if args.save_per_detection_csv else []))

    print(f'Annotations: {annotation_path}')
    print(f'YOLO cache:  {yolo_dir}')
    print(f'SAM cache:   {sam_dir}')
    print(f'Output:      {run_dir}')
    print(f'Cached images: {len(yolo_files)}')
    print('Ground point: SAM bottom-band X + YOLO bbox-bottom Y')

    dataset = read_json(annotation_path)
    image_info_by_id = {int(im['id']): im for im in dataset.get('images', [])}
    if not image_info_by_id:
        raise RuntimeError('No images in annotations')
    new_count = 0

    for yolo_path in tqdm(yolo_files, desc='Hybrid A masks', unit='image'):
        sam_path = sam_dir / yolo_path.name
        if not sam_path.is_file():
            raise FileNotFoundError(f'Missing SAM cache: {sam_path}')
        # Read scalar image ID cheaply so resumed runs can skip decompressing masks.
        with np.load(yolo_path, allow_pickle=False) as y:
            image_id = int(scalar(y['image_id']))
        if image_id in processed:
            continue
        data = get_input_data(yolo_path, sam_path)
        if image_id not in image_info_by_id:
            raise RuntimeError(f'Cached image {image_id} missing in annotations')
        info = image_info_by_id[image_id]
        aw, ah, gw, gh = data['aw'], data['ah'], data['gw'], data['gh']
        if gw != int(info['width']) or gh != int(info['height']):
            raise RuntimeError(f'Annotation dimensions disagree with canonical YOLO '
                               f'cache for image {image_id}')

        for det_index, (box, score) in enumerate(zip(data['boxes'], data['scores'])):
            x1, _, x2, y2 = map(float, box)
            bbox_x = (x1 + x2) / 2.0
            baseline_point = (bbox_x, y2)
            hybrid_x = bbox_x
            sam_bottom_y = None
            source = 'bbox_fallback_no_mask'
            if data['valid'][det_index]:
                mask_point = sam_bottom_band_point(
                    data['masks'][det_index], box,
                    args.bottom_band_frac, args.bottom_y_percentile)
                if mask_point is not None and np.isfinite(mask_point[0]):
                    hybrid_x = float(mask_point[0])
                    sam_bottom_y = float(mask_point[1])
                    source = 'sam_x_bbox_y'
                else:
                    source = 'bbox_fallback_empty_mask'
            sources[source] += 1
            hybrid_point = (hybrid_x, y2)
            box_ann = scale_box(box, aw, ah, gw, gh)
            bbox_results.append(make_detection(next_id, image_id, score, box_ann,
                                               scale_point(baseline_point, aw, ah, gw, gh)))
            hybrid_results.append(make_detection(next_id, image_id, score, box_ann,
                                                 scale_point(hybrid_point, aw, ah, gw, gh)))
            if args.save_per_detection_csv:
                point_rows.append({
                    'image_id': image_id, 'file_name': data['filename'],
                    'detection_index': det_index, 'detection_score': float(score),
                    'source': source, 'bbox_x_px': bbox_x, 'bbox_bottom_y_px': y2,
                    'sam_x_px': hybrid_x if source == 'sam_x_bbox_y' else '',
                    'sam_bottom_y_px': sam_bottom_y if sam_bottom_y is not None else '',
                    'hybrid_x_px': hybrid_x, 'hybrid_y_px': y2,
                    'hybrid_minus_bbox_x_px': hybrid_x - bbox_x,
                })
            next_id += 1

        processed.add(image_id)
        new_count += 1
        if args.checkpoint_every and new_count % args.checkpoint_every == 0:
            save_progress()

    if len(processed) != len(image_info_by_id):
        missing = sorted(set(image_info_by_id) - processed)
        raise RuntimeError(
            f'Incomplete cache coverage: processed {len(processed)} of '
            f'{len(image_info_by_id)} annotated images; '
            f'first missing IDs: {missing[:10]}. Not running an invalid full-split evaluation.'
        )

    save_json(bbox_result_path, bbox_results)
    save_json(hybrid_result_path, hybrid_results)
    save_progress()
    if args.save_per_detection_csv:
        with csv_path.open('w', encoding='utf-8', newline='') as f:
            if point_rows:
                writer = csv.DictWriter(f, fieldnames=list(point_rows[0]))
                writer.writeheader()
                writer.writerows(point_rows)

    print(f'\nCanonical detections: {len(bbox_results)}')
    print(f'Ground-point sources: {dict(sources)}')
    print('\nEvaluating bbox bottom-center with LocSim ...')
    bbox_locsim = evaluate_locsim(annotation_path, bbox_result_path,
                                 args.bbox_score_threshold)
    bbox_physical = compute_baseline_style_stats(annotation_path, bbox_result_path,
                                                bbox_locsim['score_threshold'])
    print('\nEvaluating Hybrid A (SAM X + bbox Y) with LocSim ...')
    hybrid_locsim = evaluate_locsim(annotation_path, hybrid_result_path,
                                   args.hybrid_score_threshold)
    hybrid_physical = compute_baseline_style_stats(annotation_path, hybrid_result_path,
                                                  hybrid_locsim['score_threshold'])
    summary = {
        'split': args.split,
        'ground_point_mode': 'hybrid_a_sam_x_bbox_y',
        'sam_x_method': 'largest-component bottom-percentile band median X',
        'bottom_band_frac': args.bottom_band_frac,
        'bottom_y_percentile': args.bottom_y_percentile,
        'yolo_cache_root': str(yolo_root),
        'sam_cache_root': str(sam_root),
        'annotation_file': str(annotation_path),
        'detections': len(bbox_results),
        'images': len(processed),
        'ground_point_sources': dict(sources),
        'bbox': {'results_file': str(bbox_result_path),
                 'locsim': bbox_locsim, 'baseline_style': bbox_physical},
        'hybrid_a': {'results_file': str(hybrid_result_path),
                     'locsim': hybrid_locsim, 'baseline_style': hybrid_physical},
        'delta': {
            'mAP_LocSim_50_95': hybrid_locsim['mAP_LocSim_50_95'] - bbox_locsim['mAP_LocSim_50_95'],
            'frame_accuracy_LocSim_50': hybrid_locsim['frame_accuracy_LocSim_50'] - bbox_locsim['frame_accuracy_LocSim_50'],
            'precision_0_25m': hybrid_physical['precision'] - bbox_physical['precision'],
            'recall_0_25m': hybrid_physical['recall'] - bbox_physical['recall'],
            'detected_0_25m': hybrid_physical['detected'] - bbox_physical['detected'],
        },
    }
    save_json(summary_path, summary)
    print_metrics('YOLO bbox bottom-center', summary['bbox'])
    print_metrics('HYBRID A: SAM X + bbox-bottom Y', summary['hybrid_a'])
    print(f"\nDelta mAP-LocSim: {summary['delta']['mAP_LocSim_50_95']:+.6f}")
    print(f'Summary: {summary_path}')
    if args.save_per_detection_csv:
        print(f'Point diagnostics: {csv_path}')


if __name__ == '__main__':
    main()
