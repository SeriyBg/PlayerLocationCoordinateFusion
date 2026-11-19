#!/usr/bin/env python3
"""Paired, cluster-aware bootstrap for official SynLoc mAP-LocSim.

Python 3.11. Requires numpy, matplotlib, tqdm, xtcocotools and sskit.
No GPU or neural-network inference is performed.

For each method, official LocSimCOCOeval.evaluate() is run ONCE. The program
caches the per-image, per-category, area='all', maxDet=100 evaluation records.
Each bootstrap replicate then calls the UNMODIFIED LocSimCOCOeval.accumulate()
on a paired multiset of image records, where repeated images occupy distinct
sample slots. Only the ('all', maxDets[-1]) cells are accumulated: those are
exactly the cells used for the official mAP-LocSim stats[0]. The optimization
is checked against the full original evaluation AND against independently
materialized duplicated-image mini-datasets before bootstrap begins.

An explicit sequence/match grouping field is used only if every image has
that field. No grouping is inferred from image filenames, numeric IDs, or
image ordering. Otherwise this is an exploratory image-level bootstrap.

Example:
  python bootstrap_synloc.py \\
    --dataset-root data/spiideo \\
    --yolo-cache-root output/synloc_yolo_cache \\
    --points-cache output/synloc_sam_size_adaptive_eval/val/band0.05_p98/extracted_sam_points.npz \\
    --iterations 1000 --seed 42
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import json
import math
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from xtcocotools.coco import COCO
from sskit.coco import LocSimCOCOeval

POINT_DTYPE = np.dtype([
    ('image_id', np.int64), ('score', np.float32),
    ('box_x', np.float64), ('box_y', np.float64),
    ('box_w', np.float64), ('box_h', np.float64),
    ('bbox_x', np.float64), ('bbox_y', np.float64),
    ('sam_x', np.float64), ('sam_y', np.float64),
    ('height_px', np.float64), ('source_code', np.uint8),
])
METHODS = ('bbox', 'hybrid_a', 'hard_fusion', 'continuous_fusion')
PAIRS = (('bbox', 'continuous_fusion'),
         ('hybrid_a', 'continuous_fusion'),
         ('hard_fusion', 'continuous_fusion'))
EXPECTED = {'bbox': 0.5012757915459799,
            'hybrid_a': 0.5627036263330816,
            'hard_fusion': 0.6009988626237528,
            'continuous_fusion': 0.604249210980587}
GROUP_FIELDS = ('match_id', 'match_uid', 'match_uuid', 'game_id', 'game_uid',
                'sequence_id', 'sequence_uid', 'video_id', 'video_uid',
                'recording_id', 'recording_uid', 'clip_id', 'clip_uid')


def args_parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset-root', type=Path,
                   default=Path('data/spiideo'))
    p.add_argument('--annotation', type=Path, default=None,
                   help='Explicit path to annotations/val.json')
    p.add_argument('--yolo-cache-root', type=Path,
                   default=Path('output/synloc_yolo_cache'))
    p.add_argument('--points-cache', type=Path,
                   default=Path('output/synloc_sam_size_adaptive_eval/val/band0.05_p98/extracted_sam_points.npz'))
    p.add_argument('--split', default='val', choices=['val', 'test'])
    p.add_argument('--output-dir', type=Path,
                   default=Path('output/synloc_paired_bootstrap'))
    p.add_argument('--iterations', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--score-threshold', type=float, default=0.4302865565)
    p.add_argument('--map-tolerance', type=float, default=5e-7,
                   help='Maximum full-set absolute deviation from expected mAP')
    p.add_argument('--group-by', default='auto',
                   help="auto, image, or explicit image metadata path, e.g. 'match_id' / 'metadata.match_id'")
    p.add_argument('--allow-other-image-count', action='store_true',
                   help='Allow other image counts (recorded score validation still mandatory)')
    p.add_argument('--skip-yolo-verification', action='store_true',
                   help='NOT recommended: disables raw YOLO/point identity checks')
    p.add_argument('--parity-test-images', type=int, default=4,
                   help='Number of original distinct images in duplicated-image test (min 2)')
    a = p.parse_args()
    if a.iterations < 1 or a.parity_test_images < 2:
        p.error('--iterations must be >=1 and --parity-test-images >=2')
    if not (0 <= a.score_threshold <= 1 and math.isfinite(a.score_threshold)):
        p.error('--score-threshold must be in [0,1]')
    if not (a.map_tolerance > 0 and math.isfinite(a.map_tolerance)):
        p.error('--map-tolerance must be finite and >0')
    return a


def read_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def save_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix('.json.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
    tmp.replace(path)


def find_annotation(a: argparse.Namespace) -> Path:
    if a.annotation is not None:
        result = a.annotation.expanduser().resolve()
        if not result.is_file():
            raise FileNotFoundError(f'Annotation file not found: {result}')
        return result
    roots = (a.dataset_root, a.dataset_root / 'SpiideoSynLoc')
    attempts = [root / 'annotations' / 'val.json' for root in roots]
    for path in attempts:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f'Cannot find SynLoc val annotations; looked in: {attempts}')


def validate_dataset(dataset: dict) -> tuple[list[dict], dict[int, dict], dict[int, list[dict]]]:
    if not all(k in dataset for k in ('images', 'annotations', 'categories')):
        raise ValueError('Annotation JSON must contain images, annotations, categories')
    images = dataset['images']
    infos = {int(im['id']): im for im in images}
    if len(infos) != len(images) or not images:
        raise ValueError('Duplicate or missing image IDs in annotation JSON')
    if 1 not in [int(cat['id']) for cat in dataset['categories']]:
        raise ValueError('SynLoc person category_id=1 absent')
    ann_by_image: dict[int, list[dict]] = defaultdict(list)
    ann_ids = set()
    for ann in dataset['annotations']:
        iid, aid = int(ann['image_id']), int(ann['id'])
        if iid not in infos or aid in ann_ids:
            raise ValueError(f'Invalid/duplicate GT annotation id={aid}, image={iid}')
        ann_ids.add(aid)
        ann_by_image[iid].append(ann)
    return images, infos, ann_by_image


def load_points(cache_path: Path, annotation: Path, yolo_root: Path,
                infos: dict[int, dict]) -> np.ndarray:
    if not cache_path.is_file():
        raise FileNotFoundError(f'SAM point cache not found: {cache_path}')
    with np.load(cache_path, allow_pickle=False) as z:
        for k in ('points', 'processed_image_ids', 'settings_json'):
            if k not in z.files:
                raise ValueError(f'Cache missing {k}: {cache_path}')
        points = np.asarray(z['points'])
        done = np.asarray(z['processed_image_ids'], dtype=np.int64)
        settings = json.loads(str(np.asarray(z['settings_json']).item()))
    if points.dtype != POINT_DTYPE:
        raise ValueError(f'Unexpected point dtype; expected {POINT_DTYPE}, got {points.dtype}')
    if settings.get('split') != 'val' and settings.get('split') != 'test':
        raise ValueError(f'Point cache is for split {settings.get("split")!r}, expected val')
    if Path(settings['annotation_path']).expanduser().resolve() != annotation:
        raise ValueError('Points were extracted from another annotation path')
    if Path(settings['yolo_cache_root']).expanduser().resolve() != yolo_root:
        raise ValueError('Points were extracted from a different YOLO cache root')
    if (settings.get('bottom_band_frac') != 0.05 or
            settings.get('bottom_y_percentile') != 98.0):
        raise ValueError('Unexpected SAM extraction settings: expected bottom_band_frac=0.05, bottom_y_percentile=98')
    if len(done) != len(infos) or len(set(map(int, done))) != len(infos) or set(map(int, done)) != set(infos):
        raise ValueError('Point cache incomplete; processed image IDs do not equal annotation images')
    if len(points) != 141924:
        raise ValueError(f'Expected 141,924 cached detections, found {len(points):,}')
    if not set(np.unique(points['image_id']).tolist()).issubset(infos):
        raise ValueError('Point cache references unknown image IDs')
    if np.any(points['source_code'] > 2) or not np.all(np.isfinite(points['height_px'])):
        raise ValueError('Invalid cache source codes or bbox heights')
    for k in ('box_x', 'box_y', 'box_w', 'box_h', 'bbox_x', 'bbox_y', 'score'):
        if not np.all(np.isfinite(points[k])):
            raise ValueError(f'Nonfinite cached point field: {k}')
    valid = points['source_code'] == 2
    if np.any(~np.isfinite(points['sam_x'][valid])) or np.any(~np.isfinite(points['sam_y'][valid])):
        raise ValueError('Nonfinite valid SAM points')
    if np.any((points['score'] < 0) | (points['score'] > 1)):
        raise ValueError('Cached YOLO scores outside [0,1]')
    return points


def source_areas(points: np.ndarray, yolo_root: Path,
                 infos: dict[int, dict], verify: bool) -> np.ndarray:
    """Exact float64 YOLO/source-area conversion from continuous-X evaluator."""
    by_img: dict[int, list[int]] = defaultdict(list)
    for j, iid in enumerate(points['image_id']):
        by_img[int(iid)].append(j)
    root = yolo_root / 'val' / 'yolo_raw'
    files = sorted(root.glob('*.npz'))
    if not files:
        raise FileNotFoundError(f'No YOLO cache files under {root}')
    seen = set()
    areas = np.full(len(points), np.nan, dtype=np.float64)
    for path in tqdm(files, desc='Check YOLO cache / source area', unit='image'):
        with np.load(path, allow_pickle=False) as z:
            iid = int(np.asarray(z['image_id']).item())
            boxes = np.asarray(z['bboxes'], dtype=np.float32)
            scores = np.asarray(z['detection_scores'], dtype=np.float32)
            aw, ah = int(z['actual_width']), int(z['actual_height'])
            gw, gh = int(z['annotation_width']), int(z['annotation_height'])
        if iid in seen or iid not in infos:
            raise ValueError(f'Duplicate or unknown YOLO cache image ID {iid}: {path}')
        seen.add(iid)
        idx = by_img.get(iid, [])
        if boxes.shape != (len(scores), 4) or len(idx) != len(boxes):
            raise ValueError(f'YOLO/points length mismatch at {path}')
        if (gw, gh) != (int(infos[iid]['width']), int(infos[iid]['height'])) or min(aw, ah, gw, gh) <= 0:
            raise ValueError(f'YOLO/GT size mismatch: {path}')
        if not np.all(np.isfinite(boxes)):
            raise ValueError(f'Nonfinite YOLO box at {path}')
        b = boxes.astype(np.float64)
        x1, y1, x2, y2 = (b[:, j] for j in range(4))
        w, h = x2 - x1, y2 - y1
        areas[idx] = np.maximum(0.0, w) * np.maximum(0.0, h)
        if verify and idx:
            r = points[idx]
            sx, sy = gw / aw, gh / ah
            checks = (
                ('score', scores), ('box_x', x1 * sx), ('box_y', y1 * sy),
                ('box_w', w * sx), ('box_h', h * sy),
                ('bbox_x', (x1 + x2) / 2 * sx), ('bbox_y', y2 * sy),
                ('height_px', h),
            )
            for name, calc in checks:
                if not np.allclose(r[name], calc, rtol=0, atol=2e-4):
                    error = np.abs(r[name] - calc)
                    k = int(np.argmax(error))
                    raise ValueError(f'Point/YOLO {name} mismatch in {path.name}, detection {k}: '
                                     f'saved={r[name][k]}, computed={calc[k]}, delta={error[k]}. '
                                     'Do not skip verification without examining this.')
    if seen != set(infos):
        raise ValueError(f'YOLO cache image set mismatch: missing {len(set(infos)-seen)}, extra {len(seen-set(infos))}')
    if not np.all(np.isfinite(areas)):
        raise ValueError('Missing source areas for detections')
    return areas


def localization(row: np.void, area: float, method: str) -> tuple[float, float]:
    """Bit-for-bit arithmetic order of evaluate_synloc_sam_continuous_x.py."""
    bx, by = float(row['bbox_x']), float(row['bbox_y'])
    if int(row['source_code']) != 2:
        return bx, by
    sx, sy = float(row['sam_x']), float(row['sam_y'])
    if method == 'bbox':
        beta, alpha = 0.0, 0.0
    elif method == 'hybrid_a':
        beta, alpha = 1.0, 0.0
    elif method == 'hard_fusion':
        beta = float(area >= 400.0)
        h = float(row['height_px'])
        if h < 40: alpha = 0.0
        elif h < 50: alpha = 0.2
        elif h < 60: alpha = 0.4
        elif h < 70: alpha = 0.6
        elif h < 140: alpha = 0.8
        else: alpha = 1.0
    elif method == 'continuous_fusion':
        beta = float(np.clip((area - 200.0) / 500.0, 0.0, 1.0))
        alpha = float(np.clip((float(row['height_px']) - 40.0) / 50.0, 0.0, 1.0))
    else:
        raise ValueError(f'Unknown localization method {method}')
    if beta == 0.0: x = bx
    elif beta == 1.0: x = sx
    else: x = (1.0-beta) * bx + beta * sx
    # Intentionally preserve Y arithmetic, even at alpha endpoints.
    y = (1.0-alpha) * by + alpha * sy
    return x, y


def make_predictions(points: np.ndarray, areas: np.ndarray, method: str) -> list[dict]:
    result = []
    for n, (r, area) in enumerate(zip(points, areas, strict=True), start=1):
        x, y, w, h = (float(r[k]) for k in ('box_x','box_y','box_w','box_h'))
        px, py = localization(r, float(area), method)
        result.append({'id': n, 'image_id': int(r['image_id']), 'category_id': 1,
                       'score': float(r['score']), 'bbox': [x,y,w,h],
                       'area': max(w,0.0)*max(h,0.0),
                       'keypoints': [x+w/2., y+h/2., 1., px, py, 1.]})
    return result


def build_evaluator(coco_gt: COCO, detections: list[dict], score_th: float) -> LocSimCOCOeval:
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(detections)
        evaluator = LocSimCOCOeval(coco_gt, coco_dt, 'bbox', [0.089, 0.089], True)
        evaluator.params.useSegm = None
        evaluator.params.position_from_keypoint_index = 1
        evaluator.params.score_threshold = float(score_th)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return evaluator


def ap_from_eval(evaluator: LocSimCOCOeval) -> float:
    """Exactly the COCOeval._summarize(ap=1, iouThr=None, areaRng='all', maxDets=100) mean."""
    precision = evaluator.eval['precision']
    all_idx = evaluator.params.areaRngLbl.index('all')
    max_idx = evaluator.params.maxDets.index(max(evaluator.params.maxDets))
    values = precision[:, :, :, all_idx, max_idx]
    valid = values[values > -1]
    return float(np.mean(valid)) if valid.size else -1.0


@dataclass
class EvalCache:
    name: str
    img_ids: list[int]
    categories: list[int]
    area_all: tuple[float, float]
    max_det: int
    # [category][image index] frozen official per-image evaluation dict or None
    records: list[list[dict | None]]
    template: LocSimCOCOeval
    official_map: float


def cache_eval(name: str, ev: LocSimCOCOeval) -> EvalCache:
    pe = ev._paramsEval
    images = list(map(int, pe.imgIds))
    cats = list(map(int, pe.catIds)) if pe.useCats else [-1]
    if len(set(images)) != len(images):
        raise ValueError('Original evaluator has duplicate image IDs')
    a = pe.areaRngLbl.index('all')
    I, A = len(images), len(pe.areaRng)
    table = []
    for k in range(len(cats)):
        base = k * A * I + a * I
        table.append([ev.evalImgs[base + j] for j in range(I)])
    if any(len(x) != I for x in table):
        raise ValueError('Invalid evaluator evalImgs dimensions')
    ap = float(ev.stats[0])
    if not np.isclose(ap, ap_from_eval(ev), rtol=0, atol=1e-12):
        raise RuntimeError(f'{name}: official stats[0] differs from direct COCO AP extraction')
    return EvalCache(name, images, cats, tuple(map(float, pe.areaRng[a])),
                     int(pe.maxDets[-1]), table, ev, ap)


def sampled_map(cache: EvalCache, indices: np.ndarray) -> float:
    """Official accumulate of an explicit multiset of per-image evalImgs.

    Each sampled occurrence gets a unique VIRTUAL image slot; duplicate source
    images are retained and counted repeatedly as distinct observations.
    The original matching and all-area/maxDet100 AP definition are unchanged.
    """
    indices = np.asarray(indices, dtype=np.int64)
    n = len(indices)
    if not n or np.min(indices) < 0 or np.max(indices) >= len(cache.img_ids):
        raise ValueError('Invalid bootstrap sample image indices')
    virtual_ids = list(range(1, n + 1))
    template = cache.template
    ev = copy.copy(template)
    ev.params = copy.deepcopy(template.params)
    ev._paramsEval = copy.deepcopy(template._paramsEval)
    for p in (ev.params, ev._paramsEval):
        p.imgIds = virtual_ids
        p.areaRng = [list(cache.area_all)]
        p.areaRngLbl = ['all']
        p.maxDets = [cache.max_det]
        p.catIds = cache.categories if p.useCats else [-1]
    order = indices.tolist()
    ev.evalImgs = [records[i] for records in cache.records for i in order]
    ev.eval = {}
    ev.stats = []
    with contextlib.redirect_stdout(io.StringIO()):
        ev.accumulate()  # real LocSimCOCOeval.accumulate(), not an AP approximation
    return ap_from_eval(ev)


def check_original_parity(cache: EvalCache, tol: float = 2e-12) -> None:
    got = sampled_map(cache, np.arange(len(cache.img_ids), dtype=np.int64))
    if not math.isclose(got, cache.official_map, rel_tol=0.0, abs_tol=tol):
        raise RuntimeError(f'{cache.name}: cached accumulation original-set AP {got:.15g} '
                           f'!= official AP {cache.official_map:.15g}; bootstrap ABORTED')
    print(f'  {cache.name:<20}: full official={cache.official_map:.12f}  cached={got:.12f}  OK', flush=True)


def duplicated_image_parity(cache: EvalCache, original_gt: dict,
                            gt_by_img: dict[int, list[dict]],
                            predictions: list[dict], num_imgs: int,
                            score: float, tol: float = 2e-10) -> None:
    """Compare repeated-image cached AP with a REAL materialized COCO subset.

    This validates duplicate image handling and the virtual-slot optimization,
    not merely cache identity on the full, non-resampled dataset.
    """
    image_ids = cache.img_ids
    # deterministic, includes nonempty frames (if available)
    by_detection: dict[int, list[dict]] = defaultdict(list)
    for d in predictions:
        by_detection[int(d['image_id'])].append(d)
    eligible = [j for j, iid in enumerate(image_ids)
                if gt_by_img.get(iid) and by_detection.get(iid)]
    if len(eligible) < 2:
        raise RuntimeError('Not enough populated images for duplicate-image parity test')
    chosen = eligible[:max(2, num_imgs)]
    chosen = chosen[:num_imgs]
    # Distinct images, plus repetitions of the first two; resampling is not deduplicated.
    sampled_indices = np.asarray([chosen[0], chosen[1], chosen[0], *chosen[2:], chosen[1]], dtype=np.int64)
    cached = sampled_map(cache, sampled_indices)
    info_map = {int(i['id']): i for i in original_gt['images']}
    imgs, anns, dts = [], [], []
    next_gt_id = 1
    next_dt_id = 1
    for new_img_id, old_index in enumerate(sampled_indices, start=1):
        old_id = image_ids[int(old_index)]
        image = copy.deepcopy(info_map[old_id])
        image['id'] = new_img_id
        imgs.append(image)
        for ann in gt_by_img.get(old_id, []):
            new_ann = copy.deepcopy(ann)
            new_ann['id'] = next_gt_id
            new_ann['image_id'] = new_img_id
            # xtcocotools.COCO(annotation_file) adds this default in __init__
            # AFTER createIndex(). Our in-memory COCO() + createIndex() skips
            # that normalization, so reproduce it explicitly for exact parity.
            # Preserve existing crowd annotations without modifying them.
            new_ann.setdefault('iscrowd', False)
            anns.append(new_ann)
            next_gt_id += 1
        for det in by_detection.get(old_id, []):
            new_det = copy.deepcopy(det)
            new_det['id'] = next_dt_id
            new_det['image_id'] = new_img_id
            dts.append(new_det)
            next_dt_id += 1
    gt_data = {'images': imgs, 'annotations': anns,
               'categories': copy.deepcopy(original_gt['categories']),
               'info': copy.deepcopy(original_gt.get('info', {}))}
    with contextlib.redirect_stdout(io.StringIO()):
        mini_gt = COCO()
        mini_gt.dataset = gt_data
        # xtcocotools.COCO.anno_file is a list (not a string). Its loadRes()
        # implementation appends the result source when the list has length 1.
        mini_gt.anno_file = ['<in-memory-bootstrap-parity-test>']
        mini_gt.createIndex()
    official = build_evaluator(mini_gt, dts, score)
    if not math.isclose(cached, float(official.stats[0]), rel_tol=0., abs_tol=tol):
        raise RuntimeError(f'{cache.name}: duplicate-image parity FAILED: '
                           f'cached={cached:.15g}, official={official.stats[0]:.15g}, '
                           f'abs delta={abs(cached - official.stats[0]):.6g}. Bootstrap ABORTED.')
    print(f'  {cache.name:<20}: duplicated mini COCO cached={cached:.12f} '
          f'official={official.stats[0]:.12f} OK', flush=True)


def lookup_path(row: dict, path: str) -> Any:
    item: Any = row
    for key in path.split('.'):
        if not isinstance(item, dict) or key not in item:
            return None
        item = item[key]
    return item


def sample_groups(images: list[dict], ordered_ids: list[int], rule: str
                  ) -> tuple[list[list[int]], dict[str, Any]]:
    lookup = {int(im['id']): im for im in images}
    if set(lookup) != set(ordered_ids):
        raise ValueError('Evaluator and annotation image ID sets differ')
    ordered = [lookup[i] for i in ordered_ids]
    tested = []
    if rule == 'image':
        chosen = None
    elif rule == 'auto':
        # Prefer explicit match identifiers over explicit sequence identifiers.
        for prefix in ('', 'metadata.', 'meta.', 'attributes.'):
            for field in GROUP_FIELDS:
                tested.append(prefix + field)
        chosen = None
        for field in tested:
            vals = [lookup_path(im, field) for im in ordered]
            if all(isinstance(v, (str, int)) and str(v).strip() for v in vals):
                if 1 < len(set(map(str, vals))) < len(vals):
                    chosen = field
                    break
    else:
        chosen = rule
    if chosen is not None:
        vals = [lookup_path(im, chosen) for im in ordered]
        if not all(isinstance(v, (str, int)) and str(v).strip() for v in vals):
            raise ValueError(f'Grouping metadata {chosen!r} missing/invalid for one or more images')
        members: dict[str, list[int]] = defaultdict(list)
        for i, value in enumerate(vals):
            members[str(value)].append(i)
        if len(members) < 2:
            if rule != 'auto':
                raise ValueError(f'Grouping {chosen!r} contains only one cluster; cannot bootstrap')
            chosen = None
        else:
            sizes = [len(x) for x in members.values()]
            info = {'strategy': 'cluster', 'group_field': chosen,
                    'n_sampling_groups': len(members), 'group_size_min': min(sizes),
                    'group_size_median': float(np.median(sizes)), 'group_size_max': max(sizes),
                    'interpretation': ('Paired cluster bootstrap by the explicit annotation image field '
                                       + chosen + '; cluster labels have not been independently audited '
                                       'for true match/sequence independence.')}
            if len(members) < 20:
                warnings.warn('Fewer than 20 clusters; percentile CIs may be unstable', stacklevel=2)
            return list(members.values()), info
    if rule not in ('auto', 'image'):
        raise ValueError(f'Invalid grouping field {rule}')
    warnings.warn('No verified match/sequence grouping found in annotation image metadata. '
                  'Using IMAGE-LEVEL paired bootstrap (exploratory); images from the same match/sequence '
                  'may be correlated, making these CIs overly narrow.', stacklevel=2)
    return [[i] for i in range(len(ordered))], {
        'strategy': 'image_exploratory', 'group_field': None,
        'n_sampling_groups': len(ordered),
        'group_size_min': 1, 'group_size_median': 1, 'group_size_max': 1,
        'interpretation': 'Image-level paired resampling because trustworthy grouping identifiers '
                          'were unavailable. Within-sequence correlation is not modeled.'}


def bootstrap_indices(rng: np.random.Generator, groups: list[list[int]]) -> np.ndarray:
    selected = rng.integers(0, len(groups), size=len(groups))
    return np.fromiter((i for grp in selected for i in groups[int(grp)]), dtype=np.int64)


def conclusion(delta: float, lo: float, hi: float) -> str:
    if lo > 0:
        return 'percentile_CI_above_zero'
    if hi < 0:
        return 'percentile_CI_below_zero'
    return 'percentile_CI_includes_zero'


def make_plot(rows: list[dict], path: Path, n_iter: int) -> None:
    labels = [r['comparison_label'] for r in rows]
    effects = np.array([r['delta_percentage_points'] for r in rows])
    lower = np.array([r['ci_lower_percentage_points'] for r in rows])
    upper = np.array([r['ci_upper_percentage_points'] for r in rows])
    fig, ax = plt.subplots(figsize=(9.4, 4.5), dpi=180)
    y = np.arange(len(rows))[::-1]
    ax.errorbar(effects, y, xerr=[effects - lower, upper - effects],
                fmt='o', color='#1a5276', ecolor='#2874a6',
                markersize=6, capsize=5, elinewidth=2)
    ax.axvline(0, color='#7f8c8d', linestyle='--', linewidth=1.2)
    ax.set_yticks(y, labels)
    ax.set_xlabel('Improvement in dataset-level mAP-LocSim (percentage points)')
    ax.set_title(f'Paired bootstrap 95% percentile confidence intervals (B={n_iter:,})')
    ax.grid(axis='x', alpha=0.22)
    ax.set_axisbelow(True)
    xspan = max(float(np.max(upper)-np.min(lower)), 0.3)
    ax.set_xlim(float(min(0.,np.min(lower))-0.07*xspan),
                float(max(0.,np.max(upper))+0.1*xspan))
    for yi, val in zip(y, effects):
        ax.annotate(f'{val:+.3f} pp', (val, yi), xytext=(8,7),
                    textcoords='offset points', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def build_interpretation(rows: list[dict], group_info: dict, n: int,
                         images: int, seed: int) -> str:
    lines = [
        'Paired bootstrap statistical interpretation — SynLoc validation',
        '',
        f'We evaluated {images:,} images using {n:,} paired bootstrap resamples '
        f'(random seed {seed}). Resampling was performed at the '
        f'{"explicit " + group_info["group_field"] + " cluster" if group_info["strategy"] == "cluster" else "image"} '
        f'level ({group_info["n_sampling_groups"]:,} sampling units). '
        'For each resample, dataset-level mAP-LocSim was recomputed with the '
        'official LocSimCOCOeval matching and accumulation rules. '
        'Both methods in each comparison used the identical resampled units.',
        '',
    ]
    for row in rows:
        if row['ci_includes_zero']:
            inference = ('The 95% percentile interval includes zero; this resampling analysis '
                         'does not establish a reliably positive incremental gain.')
        else:
            inference = ('The 95% percentile interval excludes zero in the positive direction '
                         'under the stated resampling assumptions.' if row['ci_lower_absolute'] > 0 else
                         'The 95% percentile interval excludes zero in the negative direction '
                         'under the stated resampling assumptions.')
        lines.append(
            f"{row['comparison_label']}: original mAP-LocSim was "
            f"{row['comparator_map']:.6f} versus {row['proposed_map']:.6f}, "
            f"a difference of {row['delta_absolute']:+.6f} "
            f"({row['delta_percentage_points']:+.3f} percentage points; "
            f"{row['relative_improvement_percent']:+.2f}% relative). "
            f"Bootstrap mean difference was {row['bootstrap_mean_absolute']:+.6f} "
            f"(SD {row['bootstrap_sd_absolute']:.6f}); the 95% percentile CI was "
            f"[{row['ci_lower_absolute']:+.6f}, {row['ci_upper_absolute']:+.6f}], "
            f"with positive differences in {row['proportion_positive']:.1%} of resamples. "
            + inference)
    lines += [
        '', 'Limitations:',
        'The hyperparameters for the fusion rules were selected using this same validation set. '
        'The paired bootstrap quantifies uncertainty conditional on that selection and does not '
        'correct for model-selection optimism or establish out-of-sample generalization.',
        ('Image-level resampling is exploratory because images from the same sequence or camera '
         'may be statistically dependent; reported confidence intervals could be too narrow.'
         if group_info['strategy'] != 'cluster' else
         'Cluster inference assumes the selected annotation grouping genuinely identifies independent '
         'sequences/matches; cross-cluster camera or event dependencies might remain.'),
        'Neither a narrow confidence interval nor an above-zero bootstrap proportion proves transfer '
        'to unseen matches, camera configurations, or data domains. '
        'The proportion of positive resamples is not interpreted as a formal p-value.',
        'The fixed detection score threshold is retained from the experiments; '
        'mAP-LocSim is computed over the precision-recall curve rather than simply averaging '
        'detections above that threshold.'
    ]
    return '\n\n'.join(item for item in lines if item.strip()) + '\n'


def main() -> None:
    a = args_parse()
    ann_file = find_annotation(a)
    yolo_root = a.yolo_cache_root.expanduser().resolve()
    points_cache = a.points_cache.expanduser().resolve()
    out = a.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    original_gt = read_json(ann_file)
    images, infos, gt_by_img = validate_dataset(original_gt)
    if len(images) != 6777 and not a.allow_other_image_count:
        raise ValueError(f'Expected 6,777 validation images, got {len(images)}. '
                         'Pass --allow-other-image-count only if intentionally evaluating a different set.')
    print(f'Annotation: {ann_file}\nImages: {len(images):,}; cache: {points_cache}', flush=True)
    print('Reconstructing all four methods from frozen points; no model inference.', flush=True)
    points = load_points(points_cache, ann_file, yolo_root, infos)
    areas = source_areas(points, yolo_root, infos, not a.skip_yolo_verification)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(ann_file))
    caches: dict[str, EvalCache] = {}
    # Predictions are not assumed to have been saved by the fusion experiments.
    # Keep lightweight GT/DT references only while their original evaluator is needed.
    print('\nReproducing recorded official results (mandatory checks):', flush=True)
    for method in METHODS:
        preds = make_predictions(points, areas, method)
        try:
            evaluator = build_evaluator(coco_gt, preds, a.score_threshold)
        except Exception as exc:
            raise RuntimeError(f'Official LocSim evaluation failed for {method}: {exc}') from exc
        result = float(evaluator.stats[0])
        expected = EXPECTED[method]
        deviation = abs(result - expected)
        print(f'  {method:<20}: measured={result:.12f}, expected={expected:.12f}, '
              f'|difference|={deviation:.3g}', flush=True)
        if deviation > a.map_tolerance:
            raise RuntimeError(f'STOP: mandatory {method} mAP parity failure; '
                               f'computed {result:.12f}, expected {expected:.12f}, '
                               f'tolerance={a.map_tolerance}. '
                               'Check cache, category, evaluator version, coordinate scaling, '
                               'and sskit installation. No bootstrap output will be generated.')
        cache = cache_eval(method, evaluator)
        caches[method] = cache
        # Detach COCO data from the cached evaluator to reduce persistent memory.
        cache.template.cocoDt = None
        cache.template.cocoGt = None
        cache.template._gts = {}
        cache.template._dts = {}
        cache.template.ious = {}
        cache.template.evalImgs = []  # Per-image all-area entries preserved in cache.records.
        cache.template.eval = {}
        del preds, evaluator

    id_order = caches['bbox'].img_ids
    for method in METHODS[1:]:
        c = caches[method]
        if c.img_ids != id_order or c.categories != caches['bbox'].categories or c.max_det != caches['bbox'].max_det:
            raise RuntimeError(f'Evaluator image/category/maxDet settings differ for {method}')
    print('\nChecking original evaluation vs optimized cached accumulation:', flush=True)
    for cache in caches.values():
        check_original_parity(cache)
    print('\nChecking repeated sampled images vs independently materialized official evaluation:', flush=True)
    # Reconstruct ONLY detections of the 4 images chosen below, saving memory.
    point_ids = set(map(int, np.unique(points['image_id'])))
    candidates = [i for i in id_order if gt_by_img.get(i) and i in point_ids]
    if len(candidates) < a.parity_test_images:
        raise RuntimeError('Not enough images with GT and detections for duplicate parity check')
    mini_ids = set(candidates[:a.parity_test_images])
    is_mini = np.isin(points['image_id'], list(mini_ids))
    mini_points, mini_areas = points[is_mini], areas[is_mini]
    for method in METHODS:
        duplicated_image_parity(caches[method], original_gt, gt_by_img,
                                make_predictions(mini_points, mini_areas, method),
                                a.parity_test_images, a.score_threshold)
    del points, areas, mini_points, mini_areas

    groups, group_info = sample_groups(images, id_order, a.group_by)
    print(f'\nSampling: {group_info["strategy"]}; {group_info["n_sampling_groups"]:,} groups; '
          f'field={group_info["group_field"]}; iterations={a.iterations:,}; seed={a.seed}', flush=True)
    print(group_info['interpretation'], flush=True)
    rng = np.random.default_rng(a.seed)
    raw: dict[tuple[str, str], list[float]] = {pair: [] for pair in PAIRS}
    per_method: dict[str, list[float]] = {key: [] for key in METHODS}
    dist_path = out / 'bootstrap_distributions.csv'
    with dist_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['iteration', 'sampled_groups', 'sampled_images',
                         'bbox_map', 'hybrid_a_map', 'hard_fusion_map', 'continuous_fusion_map',
                         'delta_vs_bbox', 'delta_vs_hybrid_a', 'delta_vs_hard_fusion'])
        for b in tqdm(range(1, a.iterations + 1), desc='Paired bootstrap (official COCO accumulate)', unit='sample'):
            idx = bootstrap_indices(rng, groups)
            vals = {key: sampled_map(caches[key], idx) for key in METHODS}
            for key in METHODS:
                per_method[key].append(vals[key])
            deltas = {}
            for left, right in PAIRS:
                diff = vals[right] - vals[left]
                raw[(left, right)].append(diff)
                deltas[left] = diff
            writer.writerow([b, len(groups), len(idx),
                             *[f'{vals[m]:.16g}' for m in METHODS],
                             *[f'{deltas[m]:.16g}' for m in ('bbox', 'hybrid_a', 'hard_fusion')]])
            if b % 25 == 0:
                handle.flush()

    comparison_rows = []
    for comparator, proposed in PAIRS:
        d = np.asarray(raw[(comparator, proposed)], dtype=np.float64)
        obs0, obs1 = caches[comparator].official_map, caches[proposed].official_map
        low, high = np.percentile(d, [2.5, 97.5])
        delta = obs1 - obs0
        rec = {
            'comparison_label': f'{proposed} vs {comparator}',
            'comparator': comparator, 'proposed': proposed,
            'comparator_map': obs0, 'proposed_map': obs1,
            'delta_absolute': delta, 'delta_percentage_points': 100.0 * delta,
            'relative_improvement_percent': 100.0 * delta / obs0,
            'ci_lower_absolute': float(low), 'ci_upper_absolute': float(high),
            'ci_lower_percentage_points': 100.0 * float(low),
            'ci_upper_percentage_points': 100.0 * float(high),
            'bootstrap_mean_absolute': float(np.mean(d)),
            'bootstrap_sd_absolute': float(np.std(d, ddof=1)) if len(d) > 1 else None,
            'proportion_positive': float(np.mean(d > 0)),
            'ci_includes_zero': bool(low <= 0 <= high),
            'interval_conclusion': conclusion(delta, float(low), float(high)),
            'n_iterations': a.iterations,
            'sampling_strategy': group_info['strategy'],
            'n_sampling_groups': group_info['n_sampling_groups'],
        }
        comparison_rows.append(rec)
    csv_file = out / 'bootstrap_comparisons.csv'
    with csv_file.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    make_plot(comparison_rows, out / 'bootstrap_ci.png', a.iterations)
    interpretation = build_interpretation(comparison_rows, group_info,
                                          a.iterations, len(images), a.seed)
    (out / 'academic_interpretation.txt').write_text(interpretation, encoding='utf-8')
    summary = {
        'methodology': {
            'dataset': 'Spiideo SoccerNet SynLoc validation',
            'annotation': str(ann_file), 'points_cache': str(points_cache),
            'n_images': len(images), 'cached_detections': 141924,
            'sampling': group_info, 'n_iterations': a.iterations,
            'seed': a.seed, 'ci_method': 'percentile_2.5_97.5',
            'same_sample_used_for_all_methods': True,
            'resampled_repeated_images_counted_separately': True,
            'score_threshold': a.score_threshold,
            'official_eval': {'evaluator': 'sskit.coco.LocSimCOCOeval',
                              'iou_type': 'bbox', 'sigmas': [0.089, 0.089],
                              'use_area': True, 'useSegm': None,
                              'position_from_keypoint_index': 1,
                              'cached_ap_optimization': 'official COCOeval accumulate on area=all and maxDet=100; verified against full official score and independent repeated-image official subset'},
            'validated_original_scores': {m: caches[m].official_map for m in METHODS},
            'warning': ('Fusion hyperparameters were selected on this validation set; '
                        'bootstrap does not remove selection optimism or prove transfer to unseen cameras.'),
        },
        'comparisons': comparison_rows,
        'bootstrap_method_distributions': {
            m: {'mean_map': float(np.mean(v)),
                'sd_map': float(np.std(v, ddof=1)) if len(v) > 1 else None}
            for m,v in per_method.items()},
        'academic_interpretation': interpretation,
    }
    save_json(out / 'bootstrap_summary.json', summary)
    print('\n' + '=' * 100)
    print('PAIRED BOOTSTRAP: DATASET-LEVEL mAP-LocSim (95% PERCENTILE CI)')
    print(f'{len(images):,} images; {len(groups):,} groups; {group_info["strategy"]}; '
          f'B={a.iterations:,}; seed={a.seed}')
    for m in METHODS:
        print(f'  {m:<20}: original mAP-LocSim = {caches[m].official_map:.10f}')
    print('-' * 100)
    for row in comparison_rows:
        print(f"{row['comparison_label']}: +{row['delta_percentage_points']:.4f} pp "
              f"({row['relative_improvement_percent']:+.2f}% relative); "
              f"95% CI [{row['ci_lower_percentage_points']:+.4f}, "
              f"{row['ci_upper_percentage_points']:+.4f}] pp; "
              f"bootstrap mean={row['bootstrap_mean_absolute']:+.6f}, "
              f"SD={row['bootstrap_sd_absolute'] if row['bootstrap_sd_absolute'] is not None else 'N/A'}; "
              f"positive={row['proportion_positive']:.1%}; "
              f"includes zero={row['ci_includes_zero']}")
    print('-' * 100)
    print('The proportion of positive bootstrap differences is NOT a formal p-value.')
    print('Fusion hyperparameters selected on val: confidence intervals do not correct selection optimism.')
    if group_info['strategy'] != 'cluster':
        print('EXPLORATORY IMAGE BOOTSTRAP: within-match dependence may make intervals too narrow.')
    print('\n' + interpretation)
    print('Saved results:', out, flush=True)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, FileNotFoundError, KeyError) as exc:
        print(f'\nERROR: {exc}', file=sys.stderr, flush=True)
        sys.exit(2)
