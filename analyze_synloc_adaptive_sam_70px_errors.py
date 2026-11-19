#!/usr/bin/env python3
"""Analyze localization errors of the *already evaluated* size-adaptive SAM model.

No YOLO/SAM inference, mask decompression, or mAP calculation.

Reads `extracted_sam_points.npz` produced by evaluate_synloc_sam_size_adaptive.py,
original YOLO cache (to verify per-detection identity and source-image dimensions),
and SynLoc annotations (GT ground image keypoint + GT pitch position).

For bbox-matched predictions, export signed X/Y errors in original image pixels;
full pitch error; counterfactual pitch errors with X or Y fixed to GT; and
three variants: bbox center/bottom, SAM X/bbox Y (Hybrid A), and SAM X/SAM Y.

The 0.25-m success labels here use one-to-one *bbox-IoU* correspondence to GT.
They are diagnostics, NOT the benchmark's point-distance matching or mAP-LocSim.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from sskit import image_to_ground

HEIGHT_EDGES = (0, 20, 40, 50, 60, 65, 70, 75, 80, 90, 110, 140, 200, float('inf'))
AREA_EDGES = (0, 250, 500, 1000, 2000, 4000, 8000, 16000, float('inf'))
SCORE_EDGES = (0, .43, .55, .7, .85, .95, 1.000001)
BOTTOM_FRACTION_EDGES = (0, .3, .5, .7, .85, 1.000001)

COLS = [
    'image_id', 'file_name', 'detection_index', 'detection_score', 'match_status',
    'gt_annotation_id', 'bbox_match_iou', 'source', 'threshold_px',
    'image_width_px', 'image_height_px',
    'bbox_x1_px', 'bbox_y1_px', 'bbox_x2_px', 'bbox_y2_px',
    'bbox_width_px', 'bbox_height_px', 'bbox_area_px2', 'bbox_aspect_ratio',
    'bbox_bottom_y_fraction', 'bbox_center_x_fraction',
    'bbox_height_bin', 'bbox_area_bin', 'score_bin', 'image_y_bin',
    'bbox_x_px', 'bbox_bottom_y_px', 'sam_x_px', 'sam_y_px',
    'sam_x_minus_bbox_x_px', 'sam_y_minus_bbox_bottom_px',
    'pred_x_px', 'pred_y_px', 'gt_x_px', 'gt_y_px',
    'signed_dx_px', 'signed_dy_px', 'abs_dx_px', 'abs_dy_px',
    'relative_abs_dx_bbox_width', 'relative_abs_dy_bbox_height',
    'dominant_pixel_error_axis',
    'gt_pitch_x_m', 'gt_pitch_y_m',
    'pred_pitch_x_m', 'pred_pitch_y_m', 'pitch_signed_dx_m', 'pitch_signed_dy_m',
    'adaptive_error_m', 'bbox_error_m', 'hybrid_a_error_m', 'raw_sam_error_m',
    'improvement_vs_bbox_m', 'improvement_vs_hybrid_a_m',
    'adaptive_within_025m', 'adaptive_within_050m', 'adaptive_within_100m',
    'bbox_within_025m', 'hybrid_a_within_025m', 'raw_sam_within_025m',
    # Camera-projected GT image point versus the annotated pitch position.
    'gt_projection_disagreement_m',
    # Change one coordinate at a time and compare to GT pitch position.
    'correct_x_to_gt_error_m', 'correct_y_to_gt_error_m',
    'x_correction_gain_m', 'y_correction_gain_m', 'dominant_correctable_axis',
    # The effect of each image-axis offset relative to camera-projected GT image point.
    'x_only_projection_error_m', 'y_only_projection_error_m',
    'dominant_projected_axis',
    'error_band',
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset-root', required=True, type=Path)
    p.add_argument('--yolo-cache-root', type=Path, default=Path('output/synloc_yolo_cache'))
    p.add_argument('--points-cache', type=Path, default=None,
                   help='Default: output/synloc_sam_size_adaptive_eval/{split}/band0.05_p98/extracted_sam_points.npz')
    p.add_argument('--split', choices=('val', 'test'), default='val')
    p.add_argument('--height-threshold-px', type=float, default=70.)
    p.add_argument('--score-threshold', type=float, default=.4302865565)
    p.add_argument('--match-iou', type=float, default=.5)
    p.add_argument('--output-dir', type=Path, default=Path('output/synloc_adaptive_sam_error_analysis'))
    p.add_argument('--max-images', type=int, default=0, help='0=all; 100 for a quick test')
    p.add_argument('--worst-n', type=int, default=300)
    p.add_argument('--expected-matched', type=int, default=101803,
                   help='Expected number of bbox-matched detections at the default score; 0 disables sanity check')
    a = p.parse_args()
    if not math.isfinite(a.height_threshold_px) or a.height_threshold_px < 0:
        p.error('Height threshold must be nonnegative and finite')
    if not 0 <= a.score_threshold <= 1 or not 0 <= a.match_iou <= 1:
        p.error('Confidence score and IoU thresholds must be in [0,1]')
    if a.max_images < 0 or a.worst_n < 0 or a.expected_matched < 0:
        p.error('max-images, worst-n, expected-matched must be nonnegative')
    if a.points_cache is None:
        a.points_cache = (Path('output/synloc_sam_size_adaptive_eval') / a.split /
                          'band0.05_p98' / 'extracted_sam_points.npz')
    return a


def annotation_file(root: Path, split: str) -> Path:
    candidates = [r / 'annotations' / f'{name}.json'
                  for r in (root, root / 'SpiideoSynLoc')
                  for name in (('val', 'valid', 'validation') if split == 'val' else ('test',))]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError('No annotations; looked for:\n' + '\n'.join(map(str, candidates)))


def scalar(a: Any) -> Any:
    return np.asarray(a).item()


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(0., rb - lt)
    inter = wh[:, :, 0] * wh[:, :, 1]
    aa = np.prod(np.maximum(0, a[:, 2:] - a[:, :2]), axis=1)[:, None]
    bb = np.prod(np.maximum(0, b[:, 2:] - b[:, :2]), axis=1)[None, :]
    union = aa + bb - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_bbox(det: np.ndarray, gt: np.ndarray, min_iou: float) -> dict[int, tuple[int, float]]:
    if len(det) == 0 or len(gt) == 0:
        return {}
    overlaps = iou_matrix(det, gt)
    di, gi = linear_sum_assignment(1. - overlaps)
    return {int(d): (int(g), float(overlaps[d, g])) for d, g in zip(di, gi)
            if overlaps[d, g] >= min_iou}


def ground_image_keypoint(annotation: dict[str, Any]) -> tuple[float, float] | None:
    raw = annotation.get('keypoints')
    if raw is None:
        return None
    try:
        k = np.asarray(raw, dtype=np.float64)
        if k.ndim == 1 and len(k) >= 6:
            pt = (float(k[3]), float(k[4]))
        elif k.ndim == 2 and k.shape[0] >= 2 and k.shape[1] >= 2:
            pt = (float(k[1, 0]), float(k[1, 1]))
        else:
            return None
    except (TypeError, ValueError):
        return None
    return pt if np.all(np.isfinite(pt)) else None


def bin_name(v: float, edges: tuple[float, ...]) -> str:
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= v < hi:
            return f'{lo:g}+' if math.isinf(hi) else f'{lo:g}-{hi:g}'
    return 'out_of_range'


def error_band(e: float) -> str:
    if e <= .25:
        return '<=0.25m'
    if e <= .5:
        return '0.25-0.50m'
    if e <= 1.0:
        return '0.50-1.00m'
    if e <= 2.0:
        return '1.00-2.00m'
    return '>2.00m'


def greater_axis(dx: float, dy: float) -> str:
    if not np.all(np.isfinite([dx, dy])):
        return ''
    if abs(dx - dy) <= 1e-9:
        return 'equal'
    return 'x' if dx > dy else 'y'


def project(points: np.ndarray, image: dict[str, Any]) -> np.ndarray:
    """Exact annotation-coordinate -> pitch mapping used by the adaptive evaluator."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not len(p):
        return np.empty((0, 2), dtype=np.float64)
    w, h = float(image['width']), float(image['height'])
    normalized = np.column_stack(((p[:, 0] - (w - 1) / 2) / w,
                                  (p[:, 1] - (h - 1) / 2) / w)).astype(np.float32)
    out = image_to_ground(image['camera_matrix'], image['undist_poly'], normalized)
    if hasattr(out, 'detach'):
        out = out.detach().cpu().numpy()
    arr = np.asarray(out, dtype=np.float64)
    return arr.reshape(len(p), -1)[:, :2]


def finite_or_blank(x: float | None) -> float | str:
    return '' if x is None or not np.isfinite(x) else float(x)


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def record_for_detection(p: np.void, bbox: np.ndarray, score: float, original_index: int,
                         image_id: int, file_name: str, aw: int, ah: int,
                         gw: int, gh: int, threshold: float,
                         match: tuple[int, float] | None,
                         gt_list: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    row = {name: '' for name in COLS}
    x1, y1, x2, y2 = map(float, bbox)
    bw, bh = x2 - x1, y2 - y1
    area = bw * bh
    sx, sy = aw / gw, ah / gh
    valid = int(p['source_code']) == 2
    bbox_ann = np.array([float(p['bbox_x']), float(p['bbox_y'])])
    sam_ann = np.array([float(p['sam_x']), float(p['sam_y'])]) if valid else None
    hybrid_ann = np.array([sam_ann[0], bbox_ann[1]]) if valid else bbox_ann.copy()
    raw_ann = sam_ann if valid else None
    above = valid and bh >= threshold
    adaptive_ann = raw_ann.copy() if above else hybrid_ann.copy()
    source = ('sam_x_sam_y' if above else 'sam_x_bbox_y' if valid else
              'bbox_fallback_empty_mask' if int(p['source_code']) == 1 else 'bbox_fallback_no_mask')
    row.update(
        image_id=image_id, file_name=file_name, detection_index=original_index,
        detection_score=score, match_status='unmatched_bbox_iou', source=source,
        threshold_px=threshold, image_width_px=aw, image_height_px=ah,
        bbox_x1_px=x1, bbox_y1_px=y1, bbox_x2_px=x2, bbox_y2_px=y2,
        bbox_width_px=bw, bbox_height_px=bh, bbox_area_px2=area,
        bbox_aspect_ratio=bw / bh if bh > 0 else '',
        bbox_bottom_y_fraction=y2 / ah, bbox_center_x_fraction=(x1 + x2) / (2*aw),
        bbox_height_bin=bin_name(bh, HEIGHT_EDGES), bbox_area_bin=bin_name(area, AREA_EDGES),
        score_bin=bin_name(score, SCORE_EDGES), image_y_bin=bin_name(y2 / ah, BOTTOM_FRACTION_EDGES),
        bbox_x_px=bbox_ann[0] * sx, bbox_bottom_y_px=bbox_ann[1] * sy,
        sam_x_px=sam_ann[0] * sx if valid else '',
        sam_y_px=sam_ann[1] * sy if valid else '',
        sam_x_minus_bbox_x_px=(sam_ann[0] - bbox_ann[0]) * sx if valid else '',
        sam_y_minus_bbox_bottom_px=(sam_ann[1] - bbox_ann[1]) * sy if valid else '',
        pred_x_px=adaptive_ann[0] * sx, pred_y_px=adaptive_ann[1] * sy,
    )
    points = {'bbox': bbox_ann, 'hybrid': hybrid_ann, 'adaptive': adaptive_ann}
    if raw_ann is not None:
        points['raw'] = raw_ann
    if match is None:
        return row, points
    idx, iou = match
    gt = gt_list[idx]
    row['gt_annotation_id'] = gt.get('id', '')
    row['bbox_match_iou'] = iou
    gp = np.asarray(gt.get('position_on_pitch', [])[:2], dtype=np.float64)
    if gp.shape != (2,) or not np.isfinite(gp).all():
        row['match_status'] = 'matched_without_gt_pitch'
        return row, points
    points['gt_pitch'] = gp
    xy = ground_image_keypoint(gt)
    if xy is None:
        row['match_status'] = 'matched_without_gt_ground_keypoint'
        return row, points
    gt_ann = np.array(xy, np.float64)
    points['gt'] = gt_ann
    points['x_only'] = np.array([adaptive_ann[0], gt_ann[1]])
    points['y_only'] = np.array([gt_ann[0], adaptive_ann[1]])
    # If an oracle could correct just one image axis, what residual pitch error remains?
    # x corrected -> (gt_x, predicted_y); y corrected -> (predicted_x, gt_y).
    row.update(match_status='matched', gt_x_px=gt_ann[0]*sx, gt_y_px=gt_ann[1]*sy,
               gt_pitch_x_m=gp[0], gt_pitch_y_m=gp[1])
    dx, dy = (adaptive_ann-gt_ann) * np.array([sx, sy])
    row.update(signed_dx_px=dx, signed_dy_px=dy, abs_dx_px=abs(dx), abs_dy_px=abs(dy),
               relative_abs_dx_bbox_width=abs(dx) / bw if bw > 0 else '',
               relative_abs_dy_bbox_height=abs(dy) / bh if bh > 0 else '',
               dominant_pixel_error_axis=greater_axis(abs(dx), abs(dy)))
    return row, points


def populate_projected(rows: list[dict[str, Any]], point_sets: list[dict[str, np.ndarray]],
                       image_info: dict[str, Any]) -> None:
    all_points: list[np.ndarray] = []
    jobs: list[tuple[dict[str, Any], dict[str, int], np.ndarray]] = []
    for row, pts in zip(rows, point_sets):
        if 'gt_pitch' not in pts:
            continue
        labels: dict[str, int] = {}
        for name in ('bbox', 'hybrid', 'raw', 'adaptive', 'gt', 'x_only', 'y_only'):
            if name in pts:
                labels[name] = len(all_points)
                all_points.append(pts[name])
        jobs.append((row, labels, pts['gt_pitch']))
    if not all_points:
        return
    xy = project(np.array(all_points), image_info)
    for row, idx, gt_pitch in jobs:
        def pt(key: str) -> np.ndarray | None:
            if key not in idx:
                return None
            val = xy[idx[key]]
            return val if np.isfinite(val).all() else None
        bbox, hybrid, raw, pred, gtproj, x_only, y_only = (
            pt(k) for k in ('bbox', 'hybrid', 'raw', 'adaptive', 'gt', 'x_only', 'y_only'))
        if bbox is None or pred is None:
            row['match_status'] = 'matched_projection_failed'
            continue
        eb, eh, ep = dist(bbox, gt_pitch), (dist(hybrid, gt_pitch) if hybrid is not None else math.nan), dist(pred, gt_pitch)
        row.update(
            bbox_error_m=eb, hybrid_a_error_m=finite_or_blank(eh), adaptive_error_m=ep,
            improvement_vs_bbox_m=eb-ep,
            improvement_vs_hybrid_a_m=eh-ep if np.isfinite(eh) else '',
            raw_sam_error_m=dist(raw, gt_pitch) if raw is not None else '',
            adaptive_within_025m=(ep <= .25), adaptive_within_050m=(ep <= .5),
            adaptive_within_100m=(ep <= 1.), bbox_within_025m=(eb <= .25),
            hybrid_a_within_025m=(eh <= .25) if np.isfinite(eh) else '',
            raw_sam_within_025m=(dist(raw,gt_pitch) <= .25) if raw is not None else '',
            pred_pitch_x_m=pred[0], pred_pitch_y_m=pred[1],
            pitch_signed_dx_m=pred[0]-gt_pitch[0], pitch_signed_dy_m=pred[1]-gt_pitch[1],
            error_band=error_band(ep),
        )
        if gtproj is not None and x_only is not None and y_only is not None:
            row.update(
                gt_projection_disagreement_m=dist(gtproj, gt_pitch),
                x_only_projection_error_m=dist(x_only, gtproj),
                y_only_projection_error_m=dist(y_only, gtproj),
                dominant_projected_axis=greater_axis(dist(x_only,gtproj), dist(y_only,gtproj)),
                correct_x_to_gt_error_m=dist(y_only, gt_pitch),
                correct_y_to_gt_error_m=dist(x_only, gt_pitch),
                x_correction_gain_m=ep-dist(y_only, gt_pitch),
                y_correction_gain_m=ep-dist(x_only, gt_pitch),
                dominant_correctable_axis=(
                    'neither' if max(ep-dist(y_only,gt_pitch), ep-dist(x_only,gt_pitch)) <= 0
                    else greater_axis(ep-dist(y_only,gt_pitch), ep-dist(x_only,gt_pitch))),
            )


def summarize(rows: list[dict[str, Any]], name: str, group: str) -> dict[str, Any]:
    matched = [r for r in rows if r['match_status']=='matched' and r['adaptive_error_m']!='']
    n = len(matched)
    out: dict[str, Any] = {'group':name, 'grouping':group,
                           'all_detections':len(rows), 'bbox_iou_matched':sum(r['match_status'].startswith('matched') for r in rows),
                           'evaluated':n, 'matched_percentage':n/len(rows) if rows else 0.,
                           'source_sam_y':sum(r['source']=='sam_x_sam_y' for r in rows),
                           'source_bbox_y':sum(r['source']=='sam_x_bbox_y' for r in rows)}
    if not n:
        return out
    def vec(col: str) -> np.ndarray:
        a = np.asarray([r[col] for r in matched if r[col] != '' and r[col] is not None], dtype=float)
        return a[np.isfinite(a)]
    def mean(col: str) -> float | str:
        a = vec(col)
        return float(a.mean()) if len(a) else ''
    def q(col: str, perc: float) -> float | str:
        a = vec(col)
        return float(np.percentile(a, perc)) if len(a) else ''
    for k in ('adaptive_error_m','bbox_error_m','hybrid_a_error_m','raw_sam_error_m',
              'improvement_vs_bbox_m','improvement_vs_hybrid_a_m',
              'signed_dx_px','signed_dy_px','abs_dx_px','abs_dy_px',
              'gt_projection_disagreement_m','x_only_projection_error_m','y_only_projection_error_m',
              'x_correction_gain_m','y_correction_gain_m',
              'sam_x_minus_bbox_x_px','sam_y_minus_bbox_bottom_px'):
        out['mean_'+k] = mean(k)
    out['median_adaptive_error_m'] = q('adaptive_error_m',50)
    out['p90_adaptive_error_m'] = q('adaptive_error_m',90)
    out['p95_adaptive_error_m'] = q('adaptive_error_m',95)
    out['p90_abs_dx_px'] = q('abs_dx_px',90)
    out['p90_abs_dy_px'] = q('abs_dy_px',90)
    out['within_025m_rate'] = sum(float(r['adaptive_error_m'])<=.25 for r in matched)/n
    out['within_050m_rate'] = sum(float(r['adaptive_error_m'])<=.5 for r in matched)/n
    out['within_100m_rate'] = sum(float(r['adaptive_error_m'])<=1. for r in matched)/n
    out['bbox_within_025m_rate'] = sum(float(r['bbox_error_m'])<=.25 for r in matched)/n
    out['hybrid_a_within_025m_rate'] = sum(float(r['hybrid_a_error_m'])<=.25 for r in matched if r['hybrid_a_error_m']!='')/n
    out['adaptive_beats_hybrid_a_rate'] = sum(float(r['adaptive_error_m']) < float(r['hybrid_a_error_m'])-1e-9 for r in matched if r['hybrid_a_error_m']!='')/n
    out['hybrid_a_beats_adaptive_rate'] = sum(float(r['hybrid_a_error_m']) < float(r['adaptive_error_m'])-1e-9 for r in matched if r['hybrid_a_error_m']!='')/n
    out['dominant_pixel_x_share'] = sum(r['dominant_pixel_error_axis']=='x' for r in matched)/n
    out['dominant_projected_x_share'] = sum(r['dominant_projected_axis']=='x' for r in matched)/n
    out['dominant_correctable_x_share'] = sum(r['dominant_correctable_axis']=='x' for r in matched)/n
    out['mean_bbox_height_px'] = mean('bbox_height_px')
    out['mean_bbox_area_px2'] = mean('bbox_area_px2')
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], cols: list[str] | None = None) -> None:
    if cols is None:
        cols = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({k:r.get(k,'') for k in cols} for r in rows)


def groups_write(root: Path, rows: list[dict[str, Any]], col: str, filename: str,
                 prelabels: list[str] | None = None) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        label = str(r.get(col,''))
        if label:
            groups[label].append(r)
    labels = prelabels if prelabels else sorted(groups)
    out = [summarize(groups.get(l,[]), l, col) for l in labels]
    fieldnames = list(dict.fromkeys(k for r in out for k in r))
    write_csv(root/filename, out, fieldnames)


def main() -> None:
    a = parse_args()
    annotation_path = annotation_file(a.dataset_root.expanduser().resolve(), a.split)
    yolo_dir = a.yolo_cache_root.expanduser().resolve()/a.split/'yolo_raw'
    points_path = a.points_cache.expanduser().resolve()
    out = (a.output_dir.expanduser().resolve()/a.split/f'height_{a.height_threshold_px:g}px')
    out.mkdir(parents=True,exist_ok=True)
    if not points_path.is_file():
        raise FileNotFoundError(f'Missing extracted SAM points: {points_path}\n'
             'Run evaluate_synloc_sam_size_adaptive.py first (without --restart).')
    if not yolo_dir.is_dir():
        raise FileNotFoundError(f'Missing YOLO cache directory: {yolo_dir}')
    with np.load(points_path, allow_pickle=False) as z:
        points = np.asarray(z['points'])
        settings = json.loads(str(scalar(z['settings_json'])))
        processed_ids = set(map(int,z['processed_image_ids']))
    if not set(('image_id','score','box_x','box_y','box_w','box_h','bbox_x','bbox_y',
                'sam_x','sam_y','height_px','source_code')).issubset(points.dtype.names or ()):
        raise RuntimeError('Wrong extracted_sam_points checkpoint schema')
    if (settings.get('split') != a.split or settings.get('bottom_band_frac') != .05 or
            settings.get('bottom_y_percentile') != 98.):
        raise RuntimeError('This script expects same SAM extraction parameters as your 70 px experiment (band=0.05, p98).')
    expected_annotation = settings.get('annotation_path')
    if expected_annotation and Path(expected_annotation).resolve() != annotation_path.resolve():
        raise RuntimeError('Checkpoint references a different annotation file than --dataset-root')
    expected_yolo_root = settings.get('yolo_cache_root')
    if expected_yolo_root and Path(expected_yolo_root).resolve() != a.yolo_cache_root.expanduser().resolve():
        raise RuntimeError('Checkpoint references a different YOLO cache root.')
    with annotation_path.open('r',encoding='utf-8') as f:
        dataset=json.load(f)
    images = {int(im['id']):im for im in dataset['images']}
    gt_by_image: dict[int,list[dict[str,Any]]] = defaultdict(list)
    for g in dataset.get('annotations',[]):
        if int(g.get('category_id',1)) == 1 and 'bbox' in g:
            gt_by_image[int(g['image_id'])].append(g)
    if not processed_ids.issubset(images):
        raise RuntimeError('Checkpoint contains image IDs absent from annotations')
    yolo_files = sorted(yolo_dir.glob('*.npz'))
    if a.max_images:
        yolo_files=yolo_files[:a.max_images]
    if not yolo_files:
        raise RuntimeError('No YOLO cache .npz files found')
    point_indices: dict[int, list[int]] = defaultdict(list)
    for index, p in enumerate(points):
        point_indices[int(p['image_id'])].append(index)
    if not a.max_images and (processed_ids != set(images) or len(yolo_files) != len(images)):
        raise RuntimeError('Incomplete checkpoint or YOLO image set; rerun adaptive evaluator first')
    print(f'GT annotations: {annotation_path}', flush=True)
    print(f'Extracted SAM points: {points_path}', flush=True)
    print(f'YOLO cache: {yolo_dir}', flush=True)
    print(f'Threshold: {a.height_threshold_px:g} original image pixels', flush=True)
    print(f'Output: {out}', flush=True)

    all_rows: list[dict[str,Any]]=[]
    file_counts:Counter[str]=Counter()
    for yp in tqdm(yolo_files,desc='Analyzing adaptive SAM errors',unit='image'):
        with np.load(yp, allow_pickle=False) as z:
            image_id=int(scalar(z['image_id']))
            filename=str(scalar(z['file_name']))
            aw,ah=int(scalar(z['actual_width'])),int(scalar(z['actual_height']))
            gw,gh=int(scalar(z['annotation_width'])),int(scalar(z['annotation_height']))
            boxes=np.asarray(z['bboxes'],dtype=np.float32)
            scores=np.asarray(z['detection_scores'],dtype=np.float32)
        if image_id not in images:
            raise RuntimeError(f'Image {image_id} missing from GT')
        if (gw,gh)!=(int(images[image_id]['width']),int(images[image_id]['height'])):
            raise RuntimeError(f'Annotation image dimensions mismatch in {yp.name}')
        indices=point_indices.get(image_id,[])
        cached=points[indices]
        if len(cached)!=len(boxes):
            raise RuntimeError(f'YOLO/checkpoint count mismatch: {yp.name}: {len(boxes)} vs {len(cached)}')
        if len(boxes):
            exp_box = np.column_stack((boxes[:,0]*gw/aw,boxes[:,1]*gh/ah,
                                       (boxes[:,2]-boxes[:,0])*gw/aw,
                                       (boxes[:,3]-boxes[:,1])*gh/ah))
            got_box = np.column_stack([cached[k] for k in ('box_x','box_y','box_w','box_h')])
            if not np.allclose(exp_box,got_box,rtol=1e-6,atol=1e-3):
                raise RuntimeError(f'YOLO/checkpoint bbox mismatch: {yp.name}; do NOT compare mismatched runs')
            if not np.array_equal(scores,cached['score']):
                raise RuntimeError(f'YOLO/checkpoint confidence mismatch: {yp.name}')
            if not np.allclose(boxes[:,3]-boxes[:,1],cached['height_px'],rtol=1e-6,atol=1e-3):
                raise RuntimeError(f'YOLO/checkpoint height mismatch: {yp.name}')
        selected=np.flatnonzero(scores>=a.score_threshold)
        gt_list=gt_by_image.get(image_id,[])
        dt_boxes=boxes[selected].astype(float)
        dt_ann=dt_boxes*np.array([gw/aw,gh/ah,gw/aw,gh/ah])
        gt_boxes=np.array([[g['bbox'][0],g['bbox'][1],g['bbox'][0]+g['bbox'][2],g['bbox'][1]+g['bbox'][3]]
                           for g in gt_list],dtype=float).reshape(-1,4)
        matches=match_bbox(dt_ann,gt_boxes,a.match_iou)
        image_rows=[]; point_sets=[]
        for j,i in enumerate(selected):
            r,ps=record_for_detection(cached[i],boxes[i],float(scores[i]),int(i),image_id,filename,
                                      aw,ah,gw,gh,a.height_threshold_px,matches.get(j),gt_list)
            image_rows.append(r)
            point_sets.append(ps)
        populate_projected(image_rows,point_sets,images[image_id])
        all_rows.extend(image_rows)
        file_counts['images']+=1
    # The full-size val split should reproduce previous bbox-IoU matching stats.
    match_count=sum(r['match_status'].startswith('matched') for r in all_rows)
    if not a.max_images and a.expected_matched and a.split=='val' and a.score_threshold==.4302865565 and a.match_iou==.5:
        if match_count!=a.expected_matched:
            raise RuntimeError(f'Matched {match_count} detections but expected {a.expected_matched}. '
                               'Check original YOLO cache and split; no CSV written.')
    write_csv(out/f'adaptive_{a.height_threshold_px:g}px_per_detection_errors.csv',all_rows,COLS)
    matched=[r for r in all_rows if r['match_status']=='matched' and r['adaptive_error_m']!='']
    groups_write(out,all_rows,'bbox_height_bin','summary_by_bbox_height.csv',
                 [bin_name(v,HEIGHT_EDGES) for v in HEIGHT_EDGES[:-1]])
    groups_write(out,all_rows,'bbox_area_bin','summary_by_bbox_area.csv',
                 [bin_name(v,AREA_EDGES) for v in AREA_EDGES[:-1]])
    groups_write(out,all_rows,'source','summary_by_coordinate_source.csv')
    groups_write(out,all_rows,'error_band','summary_by_error_band.csv',
                 ['<=0.25m','0.25-0.50m','0.50-1.00m','1.00-2.00m','>2.00m'])
    groups_write(out,all_rows,'image_y_bin','summary_by_image_vertical_position.csv',
                 [bin_name(v,BOTTOM_FRACTION_EDGES) for v in BOTTOM_FRACTION_EDGES[:-1]])
    groups_write(out,all_rows,'score_bin','summary_by_yolo_confidence.csv',
                 [bin_name(v,SCORE_EDGES) for v in SCORE_EDGES[:-1]])
    groups_write(out,all_rows,'dominant_correctable_axis','summary_by_correctable_axis.csv')
    worst=sorted(matched,key=lambda r:float(r['adaptive_error_m']),reverse=True)[:a.worst_n]
    write_csv(out/'worst_localizations.csv',worst,COLS)
    overall = summarize(all_rows,'all','all')
    overall.update(
        method='size_adaptive_sam_y',threshold_px=a.height_threshold_px,
        source_image_coordinates=True,score_threshold=a.score_threshold,
        iou_match_threshold=a.match_iou,images=file_counts['images'],
        matched_image_keypoints=sum(r['match_status']=='matched' for r in all_rows),
        total_gt_players=sum(len(gt_by_image.get(int(im['id']),[])) for im in images.values()),
        source_counts=dict(Counter(r['source'] for r in all_rows)),
        match_status_counts=dict(Counter(r['match_status'] for r in all_rows)),
        analysis_notes=[
            'Ground point predictions are the EXACT adaptive evaluator cached points; no SAM extraction or inference.',
            'Bounding-box width/height/area and signed X/Y pixel offsets use ORIGINAL source-image pixels.',
            'Every row corresponds to one YOLO detection above the given confidence score.',
            'One-to-one Hungarian bbox IoU matching is used for diagnostic GT correspondence, independently of ground localization.',
            'Per-detection within-25cm rate is NOT benchmark mAP-LocSim and NOT baseline-style distance-matching precision/recall.',
            'X/Y projected-only errors use the camera-projected GT image point as reference; they are NOT additive.',
            'Correct-X/Y oracle errors project (GT_x, predicted_y)/(predicted_x, GT_y) then compare to annotated pitch position.',
            'The difference between GT camera projection and annotated position_on_pitch is reported separately.',
            'Unmatched detections have no X/Y ground-error labels: no unambiguous GT player correspondence.',
        ])
    (out/'overall_summary.json').write_text(json.dumps(overall,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\n'+'='*58)
    print('Images:',overall['images'],'  detections:',overall['all_detections'],'  bbox matched:',overall['bbox_iou_matched'])
    print('Evaluated:',overall['evaluated'],'  within 25 cm:',f"{overall.get('within_025m_rate',0):.2%}")
    if matched:
        print('Mean adaptive pitch error (bbox-matched):', f"{overall['mean_adaptive_error_m']:.4f} m")
        print('Mean abs X and Y errors (source image pixels):',
              f"{overall['mean_abs_dx_px']:.3f} px", f"{overall['mean_abs_dy_px']:.3f} px")
    print('Files saved in:',out)
    for name in ('overall_summary.json','summary_by_bbox_height.csv','summary_by_bbox_area.csv',
                 'summary_by_coordinate_source.csv','summary_by_error_band.csv',
                 'summary_by_image_vertical_position.csv','summary_by_correctable_axis.csv',
                 'summary_by_yolo_confidence.csv','worst_localizations.csv',
                 f'adaptive_{a.height_threshold_px:g}px_per_detection_errors.csv'):
        print(' ',name)


if __name__=='__main__':
    main()
