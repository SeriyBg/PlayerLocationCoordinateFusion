#!/usr/bin/env python3
"""Evaluate continuous X fusion with the proven SynLoc soft-Y model.

Preserves the previous combined hard-gate method (proposed Y / X-area 400 px²)
as an EVALUATED REFERENCE. Tries continuous X blending, optionally with
linear or smoothstep transitions, using the exact same extracted SAM points,
YOLO boxes, confidence threshold, ground truth, and official mAP-LocSim setup.

    beta(A) = clamp((A - start) / (end - start), 0, 1)  [linear]
    x = (1 - beta) * bbox_center_x + beta * sam_x

    y = (1 - alpha(h)) * bbox_bottom_y + alpha(h) * sam_y

Sizes are SOURCE-image pixels / px². Coordinates being blended are in the
same annotation-image coordinate system. No neural inference is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
from xtcocotools.coco import COCO
from sskit import image_to_ground, match
from sskit.coco import LocSimCOCOeval


POINT_DTYPE = np.dtype([
    ('image_id', np.int64), ('score', np.float32),
    ('box_x', np.float64), ('box_y', np.float64),
    ('box_w', np.float64), ('box_h', np.float64),
    ('bbox_x', np.float64), ('bbox_y', np.float64),
    ('sam_x', np.float64), ('sam_y', np.float64),
    ('height_px', np.float64), ('source_code', np.uint8),
])

# No ground truth labels or oracle quantities influence inference or tuning rules.
# 'proposed' follows the candidate weights from the earlier error analysis.
Y_PROFILES = {
    'proposed': 'piecewise: h<40:0; 40-50:.2; 50-60:.4; 60-70:.6; '
                '70-140:.8; >=140:1',
    'linear_30_90': 'clip((h-30)/60,0,1)',
    'linear_40_80': 'clip((h-40)/40,0,1)',
    'linear_40_90': 'clip((h-40)/50,0,1)',
    'linear_40_110': 'clip((h-40)/70,0,1)',
    'linear_50_100': 'clip((h-50)/50,0,1)',
    'linear_50_110': 'clip((h-50)/60,0,1)',
    'step70': '0 if h<70 else 1 (current reference model)',
    'bbox_y_only': 'alpha=0',
    'sam_y_only': 'alpha=1',
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--yolo-cache-root', type=Path,
                   default=Path('output/synloc_yolo_cache'))
    p.add_argument('--points-cache', type=Path, default=None,
                   help='extracted_sam_points.npz from size-adaptive SAM evaluator')
    p.add_argument('--split', choices=('val', 'test'), default='val')
    p.add_argument('--score-threshold', type=float, default=0.4302865565)
    p.add_argument('--reference-height-px', type=float, default=70.0)
    p.add_argument('--hard-x-reference-px2', type=float, default=400.0,
                   help='Hard X gate reference: bbox-center if A < value, SAM otherwise')
    p.add_argument('--output-dir', type=Path,
                   default=Path('output/synloc_sam_continuous_x_experiments'))
    p.add_argument('--soft-y-profiles', nargs='+', choices=sorted(Y_PROFILES),
                   default=['proposed', 'linear_40_90'])
    p.add_argument('--x-transitions', nargs='+',
                   default=['250:500', '300:500', '350:450', '300:600',
                            '250:650', '300:700', '350:800'],
                   metavar='START:END',
                   help='Bounding-box area (source px²) transition intervals')
    p.add_argument('--x-shapes', nargs='+', choices=('linear', 'smoothstep'),
                   default=['linear'],
                   help='Shape of continuous X weight in the transition interval')
    p.add_argument('--keep-all-predictions', action='store_true',
                   help='Keep candidate COCO prediction JSON files (large)')
    p.add_argument('--skip-yolo-cache-verification', action='store_true',
                   help='Skip strict bbox verification (not recommended)')
    a = p.parse_args()
    if not math.isfinite(a.reference_height_px) or a.reference_height_px < 0:
        p.error('--reference-height-px must be nonnegative and finite')
    if not math.isfinite(a.hard_x_reference_px2) or a.hard_x_reference_px2 < 0:
        p.error('--hard-x-reference-px2 must be finite and nonnegative')
    if not math.isfinite(a.score_threshold) or not 0 <= a.score_threshold <= 1:
        p.error('--score-threshold must be in [0,1]')
    intervals = []
    for token in a.x_transitions:
        try:
            if token.count(':') != 1:
                raise ValueError
            low, high = map(float, token.split(':'))
            if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high):
                raise ValueError
        except ValueError:
            p.error(f'Invalid --x-transitions entry {token!r}; use START:END with 0 <= START < END')
        if (low,high) not in intervals:
            intervals.append((low,high))
    a.x_transitions = intervals
    a.soft_y_profiles = list(dict.fromkeys(a.soft_y_profiles))
    a.x_shapes = list(dict.fromkeys(a.x_shapes))
    if a.points_cache is None:
        a.points_cache = (Path('output/synloc_sam_size_adaptive_eval') / a.split /
                          'band0.05_p98' / 'extracted_sam_points.npz')
    return a


def read_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def annotation_path(dataset_root: Path, split: str) -> Path:
    names = ['val', 'valid', 'validation'] if split == 'val' else ['test']
    attempts = [r / 'annotations' / (name + '.json')
                for r in (dataset_root, dataset_root / 'SpiideoSynLoc')
                for name in names]
    for path in attempts:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError('Cannot find annotations: ' + repr(list(map(str, attempts))))


def load_points(path: Path, annotation: Path, yolo_root: Path,
                split: str, infos: dict[int, dict[str, Any]]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f'No extracted points cache at {path}. Run '
            'evaluate_synloc_sam_size_adaptive.py first to create it.')
    with np.load(path, allow_pickle=False) as z:
        fields = set(z.files)
        expected = {'points', 'processed_image_ids', 'settings_json'}
        if not expected.issubset(fields):
            raise RuntimeError('Unexpected extracted-point cache format: ' + str(path))
        points = np.asarray(z['points'])
        settings = json.loads(str(np.asarray(z['settings_json']).item()))
        done = np.asarray(z['processed_image_ids'], dtype=np.int64)
    if points.dtype != POINT_DTYPE:
        raise RuntimeError('Point dtype differs from size-adaptive SAM evaluator')
    if settings.get('split') != split:
        raise RuntimeError(f'Point checkpoint split {settings.get("split")} != {split}')
    if Path(settings['annotation_path']).resolve() != annotation:
        raise RuntimeError('Point checkpoint references different ground-truth annotations')
    if Path(settings['yolo_cache_root']).resolve() != yolo_root:
        raise RuntimeError('Point checkpoint references another YOLO cache root')
    if settings.get('bottom_band_frac') != 0.05 or settings.get('bottom_y_percentile') != 98.0:
        print('NOTE: cached SAM point extraction differs from band0.05 / percentile98')
    ids = set(map(int, done))
    if len(done) != len(ids) or ids != set(infos):
        raise RuntimeError('Point cache incomplete, duplicate image IDs, or unrelated annotations')
    if len(points) == 0 or not np.all(np.isfinite(points['height_px'])):
        raise RuntimeError('Empty/nonfinite extracted points')
    if not set(map(int, np.unique(points['image_id']))).issubset(ids):
        raise RuntimeError('Point rows reference unknown images')
    if np.any(points['source_code'] > 2):
        raise RuntimeError('Unknown SAM source code')
    print(f'Reusing {len(points):,} frozen YOLO/SAM points from {path}', flush=True)
    return points


def verify_yolo_and_source_areas(points: np.ndarray, yolo_root: Path,
                                  split: str, infos: dict[int, dict[str, Any]],
                                  skip: bool) -> np.ndarray:
    """Return SOURCE-image bbox area, strictly checking row identity by image.

    Source area is NOT the 4K annotation area. The source-image height already
    appears as height_px in the checkpoint. This avoids applying an area gate
    to different-resolution coordinate systems.
    """
    # Point rows are grouped by image ID for alignment with each raw YOLO file.
    by_image: dict[int, list[int]] = defaultdict(list)
    for i, image_id in enumerate(points['image_id']):
        by_image[int(image_id)].append(i)
    directory = yolo_root / split / 'yolo_raw'
    files = sorted(directory.glob('*.npz'))
    if not files:
        raise FileNotFoundError(f'No YOLO .npz caches at {directory}')
    areas = np.empty(len(points), dtype=np.float64)
    visited: set[int] = set()
    for path in tqdm(files, desc='Verify frozen YOLO cache / compute source area',
                     unit='image', disable=False):
        with np.load(path, allow_pickle=False) as z:
            image_id = int(np.asarray(z['image_id']).item())
            boxes = np.asarray(z['bboxes'], dtype=np.float32)
            scores = np.asarray(z['detection_scores'], dtype=np.float32)
            aw, ah = int(z['actual_width']), int(z['actual_height'])
            gw, gh = int(z['annotation_width']), int(z['annotation_height'])
        if image_id in visited:
            raise RuntimeError(f'Duplicate YOLO image ID {image_id}: {path}')
        visited.add(image_id)
        if image_id not in infos or image_id not in by_image:
            # Empty images may correctly have no detections.
            if image_id not in infos or len(boxes):
                raise RuntimeError(f'Unexpected image ID in {path}')
            continue
        idx = by_image[image_id]
        if len(idx) != len(boxes) or len(boxes) != len(scores):
            raise RuntimeError(f'Point/bbox count mismatch for {path.name}')
        if (gw, gh) != (int(infos[image_id]['width']), int(infos[image_id]['height'])):
            raise RuntimeError(f'Annotation-image dimensions mismatch for {path.name}')
        if min(aw, ah, gw, gh) <= 0:
            raise RuntimeError(f'Invalid image dimensions for {path.name}')
        if boxes.shape != (len(scores), 4):
            raise RuntimeError(f'Invalid YOLO bbox shape at {path}')
        # IMPORTANT: get_points() in evaluate_synloc_sam_size_adaptive.py
        # converts each float32 box element to a Python float (float64)
        # BEFORE multiplying by the annotation/source dimension ratio.
        # Computing boxes[:, 0] * gw / aw while the array is float32
        # introduces rounding at 4K coordinates (as much as ~0.00025 px),
        # which incorrectly triggers the identity check at atol=2e-4.
        # Reconstruct the checkpoint using exactly its float64 arithmetic.
        boxes64 = boxes.astype(np.float64)
        x1, y1, x2, y2 = (boxes64[:, i] for i in range(4))
        sx, sy = gw / aw, gh / ah
        source_width = x2 - x1
        source_height = y2 - y1
        areas[idx] = np.maximum(0.0, source_width) * np.maximum(0.0, source_height)
        if not skip:
            extracted = points[idx]
            x = x1 * sx
            y = y1 * sy
            w = source_width * sx
            h = source_height * sy
            bbox_center_x = ((x1 + x2) / 2.0) * sx
            bbox_bottom_y = y2 * sy
            checks = [(extracted['score'], scores, 'confidence score'),
                      (extracted['box_x'], x, 'bbox X'),
                      (extracted['box_y'], y, 'bbox Y'),
                      (extracted['box_w'], w, 'bbox width'),
                      (extracted['box_h'], h, 'bbox height'),
                      (extracted['bbox_x'], bbox_center_x, 'bbox center X'),
                      (extracted['bbox_y'], bbox_bottom_y, 'bbox bottom Y'),
                      (extracted['height_px'], source_height, 'source bbox height')]
            for saved, current, name in checks:
                if not np.allclose(saved, current, rtol=0, atol=2e-4):
                    differences = np.abs(np.asarray(saved, dtype=np.float64)
                                         - np.asarray(current, dtype=np.float64))
                    worst = int(np.argmax(differences))
                    raise RuntimeError(
                        f'Saved point / YOLO cache {name} mismatch: {path.name}, '
                        f'detection #{worst}; saved={saved[worst]!r}, '
                        f'recomputed={current[worst]!r}, '
                        f'absolute difference={differences[worst]:.9g}. '
                        'The point checkpoint may genuinely belong to a different '
                        'YOLO cache; do not skip verification unless you have '
                        'independently checked cache identity.'
                    )
    if visited != set(infos) or set(by_image) - visited:
        raise RuntimeError('YOLO cache incomplete or contains different image IDs')
    if np.any(~np.isfinite(areas)):
        raise RuntimeError('Nonfinite computed bbox areas')
    return areas


def alpha_for_height(height_px: float, profile: str, step_at: float) -> float:
    if profile == 'bbox_y_only':
        return 0.
    if profile == 'sam_y_only':
        return 1.
    if profile == 'step70':
        return float(height_px >= step_at)
    if profile == 'proposed':
        if height_px < 40: return 0.
        if height_px < 50: return 0.2
        if height_px < 60: return 0.4
        if height_px < 70: return 0.6
        if height_px < 140: return 0.8
        return 1.
    if not profile.startswith('linear_'):
        raise ValueError('Unknown Y profile: ' + profile)
    start, end = map(float, profile.removeprefix('linear_').split('_'))
    return float(np.clip((height_px - start) / (end - start), 0., 1.))


def beta_for_area(area_px2: float, start: float, end: float,
                  shape: str = 'linear') -> float:
    """SAM-X mixing weight; continuous across both transition boundaries."""
    u = float(np.clip((area_px2 - start) / (end - start), 0.0, 1.0))
    if shape == 'linear':
        return u
    if shape == 'smoothstep':
        return u * u * (3.0 - 2.0 * u)
    raise ValueError(f'Unknown X-fusion shape: {shape}')


def candidate_settings(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = []
    for profile in args.soft_y_profiles:
        for start, end in args.x_transitions:
            for shape in args.x_shapes:
                slug = (f'y_{profile}_x_{start:g}_{end:g}_{shape}').replace('.', 'p')
                candidates.append(dict(slug=slug, soft_y_profile=profile,
                                       x_rule='continuous',
                                       start=start, end=end, shape=shape))
    return candidates


def choose_point(row: np.void, area_source_px2: float, profile: str,
                 step_at: float, x_rule: str,
                 hard_gate_px2: float = 400.,
                 start: float | None = None, end: float | None = None,
                 shape: str = 'linear',
                 ) -> tuple[float, float, str, float, float]:
    """Return X/Y, source category, Y weight alpha, and X weight beta.

    Exact hard reference matches evaluate_synloc_sam_fusion_strategies_fixed.py:
    beta=0 => bbox-center X, beta=1 => SAM X, same source coordinates.
    """
    bx, by = float(row['bbox_x']), float(row['bbox_y'])
    if int(row['source_code']) != 2:
        return bx, by, 'bbox_fallback', 0.0, 0.0
    sx, sy = float(row['sam_x']), float(row['sam_y'])
    if x_rule == 'bbox':
        beta = 0.0
    elif x_rule == 'sam':
        beta = 1.0
    elif x_rule == 'hard':
        beta = float(area_source_px2 >= hard_gate_px2)
    elif x_rule == 'continuous':
        if start is None or end is None:
            raise ValueError('Continuous X fusion requires start and end')
        beta = beta_for_area(area_source_px2, start, end, shape)
    else:
        raise ValueError(f'Unknown X rule: {x_rule}')
    # Endpoint branches preserve the previous evaluator's exact arithmetic.
    if beta == 0.0:
        x, category = bx, 'bbox_center_x'
    elif beta == 1.0:
        x, category = sx, 'sam_x'
    else:
        x, category = (1.0 - beta) * bx + beta * sx, 'blended_x'
    alpha = alpha_for_height(float(row['height_px']), profile, step_at)
    y = (1.0 - alpha) * by + alpha * sy
    return x, y, category, alpha, beta


def make_detection(det_id: int, row: np.void, point: tuple[float, float]) -> dict[str, Any]:
    """Equivalent to previous size-adaptive evaluator's prediction record."""
    x, y, w, h = (float(row[k]) for k in ('box_x','box_y','box_w','box_h'))
    return dict(id=int(det_id), image_id=int(row['image_id']), category_id=1,
                score=float(row['score']), bbox=[x,y,w,h],
                area=max(w,0.) * max(h,0.),
                keypoints=[x+w/2., y+h/2., 1., float(point[0]),float(point[1]),1.])


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





def write_predictions(path: Path, points: np.ndarray, areas: np.ndarray,
                      settings: dict[str, Any], step_at: float,
                      hard_gate: float) -> dict[str, int]:
    """Stream all predictions as in the old evaluator, without filtering scores."""
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with path.open('w', encoding='utf-8') as f:
        f.write('[')
        for i, (r, area) in enumerate(zip(points, areas, strict=True)):
            px, py, source, alpha, beta = choose_point(
                r, float(area), settings['soft_y_profile'], step_at,
                settings['x_rule'], hard_gate,
                settings.get('start'), settings.get('end'), settings.get('shape','linear'))
            counts[source] += 1
            if i:
                f.write(',\n')
            json.dump(make_detection(i + 1, r, (px, py)), f)
        f.write(']')
    return dict(counts)


def evaluate_one(gt: Path, pred_path: Path, score: float) -> dict[str, Any]:
    locsim = evaluate_locsim(gt, pred_path, score)
    physical = compute_baseline_style_stats(gt, pred_path, score)
    return {'locsim': locsim, 'baseline_style': physical}


def print_one(name: str, data: dict[str, Any]) -> None:
    loc = data['locsim']
    ph = data['baseline_style']
    print(f'{name}: mAP-LocSim={loc["mAP_LocSim_50_95"]:.6f}; '
          f'P@25cm={ph["precision"]:.4f}; R@25cm={ph["recall"]:.4f}; '
          f'correct@25cm={ph["detected"]}', flush=True)


def delta(data: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    a, b = data['locsim'], ref['locsim']
    pa, pb = data['baseline_style'], ref['baseline_style']
    return {'mAP_LocSim_50_95': a['mAP_LocSim_50_95'] - b['mAP_LocSim_50_95'],
            'frame_accuracy_LocSim_50': a['frame_accuracy_LocSim_50'] - b['frame_accuracy_LocSim_50'],
            'precision_0_25m': pa['precision'] - pb['precision'],
            'recall_0_25m': pa['recall'] - pb['recall'],
            'detected_0_25m': pa['detected'] - pb['detected']}


def best_ground_point_csv(path: Path, points: np.ndarray, areas: np.ndarray,
                          best: dict[str, Any], step_at: float,
                          hard_gate: float) -> None:
    """One row per cached prediction; no GT/oracle used for coordinates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'detection_index', 'detection_score',
                         'bbox_height_source_px', 'bbox_area_source_px2',
                         'bbox_center_x_annotation_px', 'bbox_bottom_y_annotation_px',
                         'sam_x_annotation_px', 'sam_y_annotation_px',
                         'chosen_x_annotation_px', 'chosen_y_annotation_px',
                         'sam_x_weight_beta', 'sam_y_weight_alpha', 'x_source'])
        prev_image = None
        detection_index = -1
        for r, area in zip(points, areas, strict=True):
            iid = int(r['image_id'])
            if iid != prev_image:
                prev_image = iid
                detection_index = 0
            else:
                detection_index += 1
            px, py, src, alpha, beta = choose_point(
                r, float(area), best['soft_y_profile'], step_at,
                best['x_rule'], hard_gate,
                best.get('start'), best.get('end'), best.get('shape', 'linear'))
            valid = int(r['source_code']) == 2
            writer.writerow([iid, detection_index, float(r['score']),
                             float(r['height_px']), float(area),
                             float(r['bbox_x']), float(r['bbox_y']),
                             float(r['sam_x']) if valid else '',
                             float(r['sam_y']) if valid else '',
                             px, py, beta, alpha, src])


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    yolo_root = args.yolo_cache_root.expanduser().resolve()
    cache_path = args.points_cache.expanduser().resolve()
    gt = annotation_path(dataset_root, args.split)
    dataset = read_json(gt)
    infos = {int(img['id']): img for img in dataset['images']}
    if not infos:
        raise RuntimeError('Dataset annotations have no images')
    points = load_points(cache_path, gt, yolo_root, args.split, infos)
    areas = verify_yolo_and_source_areas(points, yolo_root, args.split, infos,
                                          args.skip_yolo_cache_verification)
    candidates = candidate_settings(args)
    out = args.output_dir.expanduser().resolve() / args.split / 'continuous_x'
    out.mkdir(parents=True, exist_ok=True)
    print(f'{len(candidates)} continuous-X candidates, score threshold {args.score_threshold}', flush=True)
    print('Output:', out, flush=True)

    # Reference ground points come from exactly the prior evaluation logic.
    reference_configs = {
        'bbox': dict(x_rule='bbox', soft_y_profile='bbox_y_only'),
        'hybrid_a': dict(x_rule='sam', soft_y_profile='bbox_y_only'),
        'adaptive_70px': dict(x_rule='sam', soft_y_profile='step70'),
        'combined_hard_proposed_area_400': dict(
            x_rule='hard', soft_y_profile='proposed'),
        'combined_hard_linear_40_90_area_400': dict(
            x_rule='hard', soft_y_profile='linear_40_90'),
    }
    references = {}
    for name, config in reference_configs.items():
        path = out / f'_reference_{name}.json'
        print(f'\nEvaluating reference {name}...', flush=True)
        counts = write_predictions(path, points, areas, config,
                                   args.reference_height_px, args.hard_x_reference_px2)
        stats = evaluate_one(gt, path, args.score_threshold)
        stats['ground_point_sources_all_cached_detections'] = counts
        references[name] = stats
        print_one(name, stats)
        path.unlink()

    hard_ref = references['combined_hard_proposed_area_400']
    # CSV includes a hard-X row: comparison charts can display the incumbent.
    incumbent = {
        'candidate': f'REFERENCE_hard_x_{args.hard_x_reference_px2:g}_proposed_y',
        'kind': 'hard_reference', 'soft_y_profile': 'proposed',
        'x_transition_start_px2': None, 'x_transition_end_px2': None,
        'x_shape': 'hard',
        'mAP_LocSim_50_95': hard_ref['locsim']['mAP_LocSim_50_95'],
        'delta_mAP_vs_hard_proposed_400': 0.0,
        'delta_mAP_vs_bbox': delta(hard_ref, references['bbox'])['mAP_LocSim_50_95'],
        'delta_mAP_vs_adaptive_70px': delta(hard_ref, references['adaptive_70px'])['mAP_LocSim_50_95'],
        'frame_accuracy_LocSim_50': hard_ref['locsim']['frame_accuracy_LocSim_50'],
        'precision_0_25m': hard_ref['baseline_style']['precision'],
        'recall_0_25m': hard_ref['baseline_style']['recall'],
        'detected_0_25m': hard_ref['baseline_style']['detected'],
        'mean_matched_error_m_0_25m': hard_ref['baseline_style']['mean_matched_localization_error_m'],
        'n_source_bbox_fallback': hard_ref['ground_point_sources_all_cached_detections'].get('bbox_fallback', 0),
        'n_source_bbox_center_x': hard_ref['ground_point_sources_all_cached_detections'].get('bbox_center_x', 0),
        'n_source_blended_x': 0,
        'n_source_sam_x': hard_ref['ground_point_sources_all_cached_detections'].get('sam_x', 0),
    }
    comparison = [incumbent]
    best = None
    overall_best = dict(incumbent)
    overall_best_settings = dict(reference_configs['combined_hard_proposed_area_400'])
    for i, candidate in enumerate(candidates, 1):
        print(f'\nCandidate {i}/{len(candidates)}: {candidate["slug"]}', flush=True)
        pred_file = out / f'_candidate_{candidate["slug"]}.json'
        counts = write_predictions(pred_file, points, areas, candidate,
                                   args.reference_height_px, args.hard_x_reference_px2)
        result = evaluate_one(gt, pred_file, args.score_threshold)
        print_one(candidate['slug'], result)
        row = {
            'candidate': candidate['slug'], 'kind': 'continuous_x',
            'soft_y_profile': candidate['soft_y_profile'],
            'x_transition_start_px2': candidate['start'],
            'x_transition_end_px2': candidate['end'],
            'x_shape': candidate['shape'],
            'mAP_LocSim_50_95': result['locsim']['mAP_LocSim_50_95'],
            'delta_mAP_vs_hard_proposed_400': delta(result, hard_ref)['mAP_LocSim_50_95'],
            'delta_mAP_vs_bbox': delta(result, references['bbox'])['mAP_LocSim_50_95'],
            'delta_mAP_vs_adaptive_70px': delta(result, references['adaptive_70px'])['mAP_LocSim_50_95'],
            'frame_accuracy_LocSim_50': result['locsim']['frame_accuracy_LocSim_50'],
            'precision_0_25m': result['baseline_style']['precision'],
            'recall_0_25m': result['baseline_style']['recall'],
            'detected_0_25m': result['baseline_style']['detected'],
            'mean_matched_error_m_0_25m': result['baseline_style']['mean_matched_localization_error_m'],
            'n_source_bbox_fallback': counts.get('bbox_fallback', 0),
            'n_source_bbox_center_x': counts.get('bbox_center_x', 0),
            'n_source_blended_x': counts.get('blended_x', 0),
            'n_source_sam_x': counts.get('sam_x', 0),
        }
        comparison.append(row)
        save_json(out / 'variants' / candidate['slug'] / 'evaluation_summary.json', {
            'split': args.split, 'annotation_path': str(gt),
            'points_cache': str(cache_path),
            'score_threshold': args.score_threshold,
            'soft_y_profile': candidate['soft_y_profile'],
            'soft_y_formula': Y_PROFILES[candidate['soft_y_profile']],
            'x_transition_source_px2': [candidate['start'], candidate['end']],
            'x_shape': candidate['shape'],
            'comparison': row, 'method': result,
            'ground_point_sources_all_cached_detections': counts,
        })
        if best is None or row['mAP_LocSim_50_95'] > best['mAP_LocSim_50_95']:
            best = dict(row)
        if row['mAP_LocSim_50_95'] > overall_best['mAP_LocSim_50_95']:
            overall_best = dict(row)
            overall_best_settings = dict(candidate)
            shutil.copyfile(pred_file, out / 'best_predictions.json')
        if args.keep_all_predictions:
            keep_path = out / 'variants' / candidate['slug'] / 'predictions.json'
            keep_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(pred_file), str(keep_path))
        else:
            pred_file.unlink()

    # If hard reference remains best, recreate and preserve its predictions.
    if overall_best['kind'] == 'hard_reference':
        write_predictions(out / 'best_predictions.json', points, areas,
                          overall_best_settings, args.reference_height_px,
                          args.hard_x_reference_px2)
    csv_path = out / 'comparison.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    save_json(out / 'overall_summary.json', {
        'split': args.split, 'score_threshold': args.score_threshold,
        'reference_height_px': args.reference_height_px,
        'hard_x_reference_px2': args.hard_x_reference_px2,
        'points_cache': str(cache_path), 'annotation_path': str(gt),
        'n_images': len(infos), 'n_cached_detections': len(points),
        'source_image_bbox_area': True,
        'references': references,
        'best_continuous_candidate': best,
        'overall_best_including_hard_reference': overall_best,
        'best_beats_hard_reference': (
            bool(best and best['mAP_LocSim_50_95'] > incumbent['mAP_LocSim_50_95'])),
        'y_profiles_tested': args.soft_y_profiles,
        'x_transition_ranges_source_px2': args.x_transitions,
        'x_shapes_tested': args.x_shapes,
        'selection_note': 'Tune on validation only; verify the frozen rule on independent match sequences.',
    })
    best_ground_point_csv(out / 'best_ground_points.csv', points, areas,
                          overall_best_settings, args.reference_height_px,
                          args.hard_x_reference_px2)
    print('\nBest continuous candidate:', best['candidate'],
          f'mAP={best["mAP_LocSim_50_95"]:.6f}', flush=True)
    print('Best including hard reference:', overall_best['candidate'],
          f'mAP={overall_best["mAP_LocSim_50_95"]:.6f}', flush=True)
    print('Incumbent hard 400 / proposed Y:',
          f'{incumbent["mAP_LocSim_50_95"]:.6f}', flush=True)
    print('Share:', csv_path, 'and', out / 'overall_summary.json', flush=True)



if __name__ == '__main__':
    main()
