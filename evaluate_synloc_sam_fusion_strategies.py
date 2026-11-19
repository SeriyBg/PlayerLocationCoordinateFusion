#!/usr/bin/env python3
"""Evaluate three SAM fusion strategies on frozen SoccerNet SynLoc detections.

Modes selected using --strategy:
  soft_y    : SAM X + weighted blend of bbox-bottom Y and SAM Y.
  area_x    : bbox-center X for small boxes; SAM X otherwise; 70px Y switch.
  combined  : area-dependent X choice + size-dependent weighted Y.
  all       : test all three with a shared set of reference evaluations.

No neural inference is performed. Reads extracted_sam_points.npz written by
  evaluate_synloc_sam_size_adaptive.py
and validates alignment with canonical YOLO cache. Areas/heights are SOURCE-
IMAGE pixels (1920x1080), not the 4K annotation coordinate system.

Each candidate is scored using the same official mAP-LocSim configuration
and 0.25m distance-match evaluation as the previous size-adaptive script.
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

STRATEGIES = ("soft_y", "area_x", "combined", "all")

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
    p.add_argument('--strategy', choices=STRATEGIES, required=True,
                   help='Select soft Y, area-gated X, combined, or all three')
    p.add_argument('--dataset-root', type=Path, required=True,
                   help='Dataset containing annotations/val.json (or test.json)')
    p.add_argument('--yolo-cache-root', type=Path,
                   default=Path('output/synloc_yolo_cache'))
    p.add_argument('--points-cache', type=Path, default=None,
                   help='extracted_sam_points.npz from size-adaptive SAM evaluator')
    p.add_argument('--split', choices=('val', 'test'), default='val')
    p.add_argument('--score-threshold', type=float, default=0.4302865565)
    p.add_argument('--reference-height-px', type=float, default=70.,
                   help='Fixed size-adaptive 70px model used as main comparison')
    p.add_argument('--output-dir', type=Path,
                   default=Path('output/synloc_sam_fusion_experiments'))
    p.add_argument('--soft-y-profiles', nargs='+', choices=sorted(Y_PROFILES),
                   default=['proposed', 'linear_40_90', 'linear_50_100'])
    p.add_argument('--area-thresholds', type=float, nargs='+',
                   default=[300., 350., 400., 450., 500.],
                   metavar='PX2',
                   help='Use bbox-center X below threshold; SAM X otherwise')
    p.add_argument('--keep-all-predictions', action='store_true',
                   help='Retain a large COCO JSON for every candidate; otherwise keep only best')
    p.add_argument('--skip-yolo-cache-verification', action='store_true',
                   help='Skip checking all cached boxes/scores match the extracted-point checkpoint')
    a = p.parse_args()
    if not math.isfinite(a.reference_height_px) or a.reference_height_px < 0:
        p.error('--reference-height-px must be nonnegative and finite')
    if not math.isfinite(a.score_threshold) or not 0 <= a.score_threshold <= 1:
        p.error('--score-threshold must be in [0,1]')
    if any(not math.isfinite(v) or v < 0 for v in a.area_thresholds):
        p.error('All --area-thresholds must be finite and nonnegative')
    a.area_thresholds = sorted(set(a.area_thresholds))
    a.soft_y_profiles = list(dict.fromkeys(a.soft_y_profiles))
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


def candidate_settings(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Candidates never use ground truth labels to select coordinates."""
    modes = ('soft_y', 'area_x', 'combined') if args.strategy == 'all' else (args.strategy,)
    candidates: list[dict[str, Any]] = []
    for mode in modes:
        if mode == 'soft_y':
            candidates.extend({
                'strategy': mode,
                'profile': profile,
                'area_threshold': None,
                'slug': 'y_' + profile,
            } for profile in args.soft_y_profiles)
        elif mode == 'area_x':
            candidates.extend({
                'strategy': mode,
                'profile': 'step70',
                'area_threshold': area,
                'slug': f'area_{area:g}'.replace('.', 'p'),
            } for area in args.area_thresholds)
        elif mode == 'combined':
            candidates.extend({
                'strategy': mode,
                'profile': profile,
                'area_threshold': area,
                'slug': ('y_' + profile + f'_area_{area:g}').replace('.', 'p'),
            } for profile in args.soft_y_profiles for area in args.area_thresholds)
    return candidates


def choose_point(row: np.void, area_source_px2: float,
                 profile: str, area_threshold: float | None,
                 step_at: float) -> tuple[float, float, str, float]:
    bx, by = float(row['bbox_x']), float(row['bbox_y'])
    if int(row['source_code']) != 2:
        return bx, by, 'bbox_fallback', 0.
    use_sam_x = area_threshold is None or area_source_px2 >= area_threshold
    x = float(row['sam_x']) if use_sam_x else bx
    alpha = alpha_for_height(float(row['height_px']), profile, step_at)
    y = (1. - alpha) * by + alpha * float(row['sam_y'])
    x_label = 'sam_x' if use_sam_x else 'bbox_center_x'
    return x, y, f'{x_label}_alpha_{alpha:.3f}', alpha


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
                      profile: str, area_threshold: float | None, step_at: float,
                      ) -> dict[str, int]:
    """Stream JSON to limit memory for 142k detections; keep all scores."""
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with path.open('w', encoding='utf-8') as f:
        f.write('[')
        for i, (r, area) in enumerate(zip(points, areas, strict=True)):
            px, py, src, alpha = choose_point(r, float(area), profile,
                                               area_threshold, step_at)
            counts[src] += 1
            if i: f.write(',\n')
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
                          best: dict[str, Any], step_at: float) -> None:
    """No GT needed for this CSV; can be joined to prior bbox/GT error analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id','detection_index','detection_score',
                         'bbox_height_source_px','bbox_area_source_px2',
                         'bbox_center_x_annotation_px','bbox_bottom_y_annotation_px',
                         'sam_x_annotation_px','sam_y_annotation_px',
                         'chosen_x_annotation_px','chosen_y_annotation_px',
                         'sam_y_weight_alpha','source'])
        prev_image = None
        detection_index = -1
        for r, area in zip(points, areas, strict=True):
            iid = int(r['image_id'])
            if iid != prev_image:
                prev_image = iid
                detection_index = 0
            else:
                detection_index += 1
            px,py,src,alpha = choose_point(r,float(area),best['soft_y_profile'],
                                              best['area_gate_source_px2'],step_at)
            valid = int(r['source_code']) == 2
            writer.writerow([iid,detection_index,float(r['score']),
                             float(r['height_px']),float(area),
                             float(r['bbox_x']),float(r['bbox_y']),
                             float(r['sam_x']) if valid else '',
                             float(r['sam_y']) if valid else '',
                             px,py,alpha,src])


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    yolo_root = args.yolo_cache_root.expanduser().resolve()
    cache_path = args.points_cache.expanduser().resolve()
    gt = annotation_path(dataset_root,args.split)
    dataset = read_json(gt)
    infos = {int(img['id']): img for img in dataset['images']}
    if not infos:
        raise RuntimeError('Dataset annotations have no images')
    points = load_points(cache_path,gt,yolo_root,args.split,infos)
    areas = verify_yolo_and_source_areas(points,yolo_root,args.split,infos,
                                          args.skip_yolo_cache_verification)
    if np.any(~np.isfinite(areas)):
        raise RuntimeError('Invalid source bbox areas')
    candidates = candidate_settings(args)
    if not candidates:
        raise RuntimeError('No fusion candidates')
    out = args.output_dir.expanduser().resolve() / args.split / args.strategy
    out.mkdir(parents=True, exist_ok=True)
    print(f'Strategy={args.strategy}, split={args.split}, '
          f'score threshold={args.score_threshold}, '
          f'70px reference threshold={args.reference_height_px:g} (source px)', flush=True)
    print(f'Output directory: {out}', flush=True)

    # All references are evaluated on the SAME detections, scores and LocSim setup.
    references: dict[str, dict[str, Any]] = {}
    refs = [('bbox', 'bbox_y_only', float('inf')),
            ('hybrid_a', 'bbox_y_only', None),
            ('adaptive_70px', 'step70', None)]
    for name, profile, area_gate in refs:
        path = out / f'_reference_{name}.json'
        print(f'\nEvaluating reference {name}...',flush=True)
        counts = write_predictions(path,points,areas,profile,area_gate,
                                   args.reference_height_px)
        result = evaluate_one(gt,path,args.score_threshold)
        result['source_counts_all_cached_detections'] = counts
        references[name] = result
        print_one(name,result)
        path.unlink()

    comparison: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for i,candidate in enumerate(candidates,1):
        print(f'\nCandidate {i}/{len(candidates)} '
              f'[{candidate["strategy"]}]: {candidate["slug"]}', flush=True)
        pred_file = out / f'_candidate_{candidate["slug"]}.json'
        counts = write_predictions(pred_file,points,areas,candidate['profile'],
                                   candidate['area_threshold'],args.reference_height_px)
        result = evaluate_one(gt,pred_file,args.score_threshold)
        print_one(candidate['slug'],result)
        compare = {
            'candidate': candidate['slug'],
            'strategy': candidate['strategy'],
            'soft_y_profile': candidate['profile'],
            'area_gate_source_px2': candidate['area_threshold'],
            'mAP_LocSim_50_95': result['locsim']['mAP_LocSim_50_95'],
            'delta_mAP_vs_bbox': delta(result,references['bbox'])['mAP_LocSim_50_95'],
            'delta_mAP_vs_hybrid_a': delta(result,references['hybrid_a'])['mAP_LocSim_50_95'],
            'delta_mAP_vs_adaptive_70px': delta(result,references['adaptive_70px'])['mAP_LocSim_50_95'],
            'frame_accuracy_LocSim_50': result['locsim']['frame_accuracy_LocSim_50'],
            'precision_0_25m': result['baseline_style']['precision'],
            'recall_0_25m': result['baseline_style']['recall'],
            'detected_0_25m': result['baseline_style']['detected'],
            'mean_matched_error_m_0_25m': result['baseline_style']['mean_matched_localization_error_m'],
            'n_source_bbox_fallback': counts.get('bbox_fallback',0),
            'n_source_bbox_center_x': sum(v for k,v in counts.items() if k.startswith('bbox_center_x')),
            'n_source_sam_x': sum(v for k,v in counts.items() if k.startswith('sam_x')),
        }
        comparison.append(compare)
        save_json(out / 'variants' / candidate['slug'] / 'evaluation_summary.json',{
            'strategy': candidate['strategy'],
            'split':args.split, 'score_threshold':args.score_threshold,
            'n_images':len(infos),'n_cached_detections':len(points),
            'annotation_path':str(gt),'points_cache':str(cache_path),
            'reference_height_px':args.reference_height_px,
            'soft_y_profile':candidate['profile'],
            'soft_y_formula':Y_PROFILES[candidate['profile']],
            'area_gate_source_px2':candidate['area_threshold'],
            'comparison':compare,'method':result,
            'ground_point_sources_all_cached_detections':counts,
            'delta_vs_adaptive_70px':delta(result,references['adaptive_70px']),
        })
        if best is None or compare['mAP_LocSim_50_95'] > best['mAP_LocSim_50_95']:
            best = dict(compare)
            # Keep only best prediction JSON by default (or all if requested).
            shutil.copyfile(pred_file,out/'best_predictions.json')
        if args.keep_all_predictions:
            keep_path = out / 'variants' / candidate['slug'] / 'predictions.json'
            keep_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(pred_file), str(keep_path))
        else:
            pred_file.unlink()

    csv_path = out / 'comparison.csv'
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    assert best is not None
    best_by_strategy = {}
    for strategy in ('soft_y', 'area_x', 'combined'):
        subset = [row for row in comparison if row['strategy'] == strategy]
        if not subset:
            continue
        best_by_strategy[strategy] = max(subset, key=lambda r: r['mAP_LocSim_50_95'])
        with (out / f'comparison_{strategy}.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(subset[0]))
            writer.writeheader()
            writer.writerows(subset)
    save_json(out/'overall_summary.json',{
        'selected_strategy':args.strategy, 'split':args.split,
        'score_threshold':args.score_threshold,
        'reference_height_px':args.reference_height_px,
        'points_cache':str(cache_path),
        'annotation_path':str(gt),'n_images':len(infos),
        'n_cached_detections':len(points),
        'source_image_bbox_area':True,
        'references':references,
        'best_candidate':best,
        'best_by_strategy':best_by_strategy,
        'y_profiles_tested':args.soft_y_profiles,
        'area_thresholds_tested_source_px2':args.area_thresholds,
        'selection_note':'Best result is selected on this split; confirm on held-out sequences.',
    })
    best_ground_point_csv(out/'best_ground_points.csv',points,areas,best,
                          args.reference_height_px)
    print('\nBEST:',best['candidate'], f'mAP={best["mAP_LocSim_50_95"]:.6f}',flush=True)
    print('Compare to 70px:',f'{best["delta_mAP_vs_adaptive_70px"]:+.6f}',flush=True)
    print('Share files:',csv_path,'and',out/'overall_summary.json',flush=True)
    print('Note: bbox/Hybrid A/70px references should reproduce previous scores '
          '(0.501276, 0.562704, 0.573992) for the same dataset/cache/settings.',flush=True)


if __name__ == '__main__':
    main()
