"""
ggcnn_region_filter_capture.py

Copy of ggcnn_detection_only_capture.py (left unchanged) with the single-box
grasp acceptance rule from grasp_region_filter.py added.

Still DETECTION ONLY: RobotGrasp is not imported and there is no robot
connection object, so no code path in this file can move the arm.

Requires grasp/ggcnn_torch.py with Edit F (it stores the angle and width maps).

Keys (click the preview window first):
    r        define the grasp region: in the pop-up window, click the two ENDS
             of the box along its long (161 mm) side, then press Enter.
             Redo this whenever the box or the camera moves. It is cleared
             automatically when you enter new labels.
    n        new object/position labels
    s        save the current scene + log one grasp row (as before), and print
             the filter decision for that frame
    b        consecutive-frame sequence (as before) plus the filter outputs
    q / ESC  quit

Everything the original script wrote is still written, unchanged. Extra files
in each sequence folder:
    frame_00_angle.npy          model angle map (radians, model map coordinates)
    frame_00_width.npy          model width map (pixels, model map coordinates)
    frame_00_filtered.png       yellow  = grasp region (box ends excluded)
                                green   = model's own prediction (unchanged)
                                cyan    = best candidate in region, passed checks
                                orange  = best candidate in region, failed a check
                                magenta = ACCEPTED candidate
    filtered_grasps.csv         per frame: model prediction, region candidate,
                                every check, and the final decision
    filtered_contact_sheet.png  all filtered overlays in one image
    sequence_meta.json          now also records the region and all thresholds

The consistency check starts fresh at the first frame of each sequence, so
with STABLE_FRAMES = 5 the first 4 frames of a sequence cannot be accepted.
"""

import os
import sys
import csv
import json
import time
import math
import cv2
import numpy as np
from queue import Queue

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from camera.rs_camera import RealSenseCamera
from camera.utils import get_combined_img
from grasp.ggcnn_torch import TorchGGCNN
import grasp_region_filter as grf
# NOTE: RobotGrasp is deliberately NOT imported here.

WIN_NAME = 'RealSense-D435 -- DETECTION ONLY, NO ROBOT'
REGION_WIN = 'Click both box ENDS  |  Enter = accept, c = clear, ESC = cancel'
CAM_WIDTH = 640
CAM_HEIGHT = 480

MODEL_FILE = os.path.join(os.path.dirname(__file__), '../../models/ggcnn_epoch_23_cornell')    # GGCNN
# MODEL_FILE = os.path.join(os.path.dirname(__file__), '../../models/epoch_50_cornell')        # GGCNN2
OPEN_LOOP_HEIGHT = 340  # mm
GGCNN_IN_THREAD = False
SHOW_GRASP_IMG = True

# Fixed placeholder in METERS (get_grasp_img expects robot z in meters).
# Only affects prediction tracking; NOT a camera-height calibration.
ASSUMED_ROBOT_Z_M = 0.4

OBJECT_NAME = 'obj1'
POSITION_NAME = 'posA'
SCENE_NAME = f'{OBJECT_NAME}_{POSITION_NAME}'

# Grasp region (set with 'r') and the cross-frame consistency tracker.
REGION = None
TRACKER = grf.StabilityTracker()


def _clean_label(text):
    """Make a label safe for file names: spaces -> _, drop odd characters."""
    text = text.strip().replace(' ', '_')
    return ''.join(c for c in text if c.isalnum() or c in '_-')


def set_labels():
    """Ask in the terminal for object and position. Enter keeps the current value.
    If either label changes, the grasp region is cleared (the box has moved)."""
    global OBJECT_NAME, POSITION_NAME, SCENE_NAME, REGION
    print("\n--- Set labels (camera preview pauses while you type) ---")
    print("Type a new value, or just press Enter to keep the current one.")
    obj = _clean_label(input(f"  1) OBJECT (what it is, e.g. iphone_box)    currently '{OBJECT_NAME}': "))
    pos = _clean_label(input(f"  2) POSITION (where it is, e.g. posA/posB/posC)  currently '{POSITION_NAME}': "))
    changed = (obj and obj != OBJECT_NAME) or (pos and pos != POSITION_NAME)
    if obj:
        OBJECT_NAME = obj
    if pos:
        POSITION_NAME = pos
    SCENE_NAME = f'{OBJECT_NAME}_{POSITION_NAME}'
    if changed and REGION is not None:
        REGION = None
        TRACKER.reset()
        print("Labels changed, so the grasp region was cleared. Press 'r' to define it again.")
    print(f"Now capturing: object={OBJECT_NAME}, position={POSITION_NAME}")
    print("Click the preview window again, then press 'r' / 's' / 'b'.\n")

BURST_FRAMES = 10        # consecutive frames captured per 'b' press
AUTO_PROMPT_AFTER_BURST = True

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'grasp_scenes')
LOG_CSV = os.path.join(SAVE_DIR, 'grasp_log.csv')

CAM_FX = None   # set in main() from camera intrinsics
CAM_K = None    # 3x3 intrinsics matrix, set in main(); used by the grasp overlay

SELECTION_MODE = 'unknown'

# ---- grasp overlay settings ----
# If the drawn line runs ALONG the long side of a narrow object instead of
# across it, the angle sign convention is flipped: change this to +1.0.
# Do not change this to make a picture look better -- only after checking
# the angle convention in grasp/ggcnn_torch.py. The filter's angle check
# uses this same value.
ANGLE_SIGN = -1.0        # -1 = standard GGCNN convention (image rows point down)
JAW_TICK_PX = 10
CONTACT_SHEET_COLS = 5
CONTACT_SHEET_TILE = (320, 240)

REGION_COLOR = (0, 255, 255)     # yellow
MODEL_COLOR = (0, 255, 0)        # green
CAND_OK_COLOR = (255, 255, 0)    # cyan
CAND_FAIL_COLOR = (0, 165, 255)  # orange
ACCEPT_COLOR = (255, 0, 255)     # magenta

# NOTE: depth_patch_mm in the logs is NOT the depth at the grasp. ggcnn_torch.py
# takes it from a fixed patch (rows 100-140, cols 130-170) of the 480x480 crop.

DEPTH_UNIT_TO_MM = 1000.0

LOG_FIELDS = [
    'timestamp', 'scene', 'object', 'position', 'capture_type', 'burst_id',
    'frame_in_burst', 'result_valid',
    'x_cam_m', 'y_cam_m', 'z_cam_m',
    'angle_rad', 'angle_deg',
    'width_px', 'width_mm_est',
    'depth_patch_mm',
    'assumed_robot_z_m', 'image_prefix', 'raw_result',
]

SEQ_FIELDS = [
    'frame', 'ms_since_first_frame', 'result_valid', 'invalid_reason',
    'x_cam_m', 'y_cam_m', 'z_cam_m',
    'center_u_px', 'center_v_px',
    'angle_rad', 'angle_deg',
    'width_px', 'width_mm_est',
    'jaw1_u_px', 'jaw1_v_px', 'jaw2_u_px', 'jaw2_v_px',
    'depth_patch_mm', 'depth_at_center_m',
    'selected_score',
    'crop_invalid_depth_fraction',
    'rgb_file', 'overlay_file', 'heatmap_file', 'quality_file', 'depth_file',
    'raw_result',
]

# Columns of filtered_grasps.csv: the model's own prediction, then the
# region candidate and each check, then the decision.
FILTER_FIELDS = [
    'frame',
    'model_result_valid', 'model_center_u_px', 'model_center_v_px',
    'model_angle_deg', 'model_width_mm_est', 'model_selected_score',
    'region_defined', 'region_found', 'region_u_px', 'region_v_px', 'region_score',
    'x_cam_m', 'y_cam_m', 'z_cam_m', 'angle_deg', 'width_px', 'width_mm',
    'along_mm', 'across_mm', 'angle_from_perpendicular_deg', 'depth_valid_fraction',
    'pass_depth', 'pass_opening', 'pass_angle', 'frame_ok', 'frame_reason',
    'model_to_region_mm',
    'window_count', 'pos_spread_mm', 'angle_spread_deg',
    'decision', 'decision_reason',
    'filtered_overlay_file', 'angle_file', 'width_file',
]


def depth_to_png16_mm(depth):
    """Float depth -> 16-bit PNG array in mm (invalid -> 0). For viewing only."""
    mm = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0) * DEPTH_UNIT_TO_MM
    return np.clip(mm, 0, 65535).astype(np.uint16)


def valid_depth_mask(depth):
    """True where depth is a usable reading (finite and > 0)."""
    return np.isfinite(depth) & (depth > 0)


def parse_result(result):
    """Split the GGCNN result into named fields. Returns (valid, dict)."""
    fields = {k: '' for k in ['x_cam_m', 'y_cam_m', 'z_cam_m', 'angle_rad',
                              'angle_deg', 'width_px', 'width_mm_est',
                              'depth_patch_mm']}
    try:
        vals = [float(v) for v in list(result)]
    except Exception:
        return False, fields
    if len(vals) < 6 or not all(math.isfinite(v) for v in vals[:5]):
        return False, fields
    fields.update({
        'x_cam_m': vals[0],
        'y_cam_m': vals[1],
        'z_cam_m': vals[2],
        'angle_rad': vals[3],
        'angle_deg': math.degrees(vals[3]),
        'width_px': vals[4],
        'width_mm_est': vals[4] * vals[2] / CAM_FX * 1000.0 if CAM_FX else '',
        'depth_patch_mm': vals[5],
    })
    return True, fields


def invalid_reason(result, valid, fields):
    """Why a frame has no drawable grasp ('' if it has one)."""
    if valid:
        if grasp_geometry(fields) is None:
            return 'no_depth_at_selected_pixel_z_0'
        return ''
    if result is None:
        if SELECTION_MODE.startswith('global max'):
            return 'no_depth_at_selected_pixel_nan'
        return 'model_returned_none'
    try:
        vals = [float(v) for v in list(result)]
    except Exception:
        return 'result_not_numeric'
    if len(vals) < 6:
        return f'result_has_only_{len(vals)}_values'
    return 'non_finite_value_in_result'


def grasp_to_pixel(fields):
    """Camera-frame x,y,z (m) -> pixel (u, v) in the FULL RGB image."""
    if CAM_K is None:
        return None
    z = fields.get('z_cam_m')
    if not isinstance(z, float) or not math.isfinite(z) or z <= 0:
        return None
    u = CAM_K[0, 0] * fields['x_cam_m'] / z + CAM_K[0, 2]
    v = CAM_K[1, 1] * fields['y_cam_m'] / z + CAM_K[1, 2]
    return u, v


def grasp_geometry(fields):
    """Pixel geometry of the grasp exactly as it is drawn (full-image pixels)."""
    uv = grasp_to_pixel(fields)
    if uv is None:
        return None
    u, v = uv
    a = fields['angle_rad']
    half = 0.5 * fields['width_px']
    dx = half * math.cos(a)
    dy = ANGLE_SIGN * half * math.sin(a)
    return {'center': (u, v), 'jaw1': (u - dx, v - dy), 'jaw2': (u + dx, v + dy)}


def draw_grasp(img, fields, color=(0, 255, 0), label=True):
    """Draw the grasp on img (BGR, full frame): center, opening line, jaw ticks."""
    geom = grasp_geometry(fields)
    if geom is None:
        return False
    u, v = geom['center']
    a = fields['angle_rad']

    tx = -ANGLE_SIGN * JAW_TICK_PX * math.sin(a)
    ty = JAW_TICK_PX * math.cos(a)

    for px, py in (geom['jaw1'], geom['jaw2']):
        cv2.line(img, (int(round(px - tx)), int(round(py - ty))),
                 (int(round(px + tx)), int(round(py + ty))), color, 2)
    (x1, y1), (x2, y2) = geom['jaw1'], geom['jaw2']
    cv2.line(img, (int(round(x1)), int(round(y1))),
             (int(round(x2)), int(round(y2))), color, 2)
    cv2.circle(img, (int(round(u)), int(round(v))), 4, (0, 0, 255), -1)

    if label:
        text = f"{fields['width_mm_est']:.0f} mm, {fields['angle_deg']:.0f} deg"
        cv2.putText(img, text, (int(u) + 10, int(v) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return True


def draw_region(img, region, label=True):
    """Draw the grasp region outline, the clicked box ends, and a label."""
    corners = np.round(region.corners()).astype(np.int32)
    cv2.polylines(img, [corners.reshape(-1, 1, 2)], True, REGION_COLOR, 2)
    e1 = (int(round(region.end1[0])), int(round(region.end1[1])))
    e2 = (int(round(region.end2[0])), int(round(region.end2[1])))
    cv2.line(img, e1, e2, REGION_COLOR, 1)
    for p in (e1, e2):
        cv2.circle(img, p, 4, REGION_COLOR, -1)
    if label:
        x, y = corners[np.argmin(corners[:, 1])]
        cv2.putText(img, f'GRASP REGION (ends -{grf.END_MARGIN_MM:.0f} mm)',
                    (int(x) + 5, max(12, int(y) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, REGION_COLOR, 1, cv2.LINE_AA)


def cand_to_fields(cand):
    """Region candidate -> the same field names draw_grasp() uses (or None)."""
    if cand.get('z_cam_m') is None:
        return None
    return {
        'x_cam_m': float(cand['x_cam_m']), 'y_cam_m': float(cand['y_cam_m']),
        'z_cam_m': float(cand['z_cam_m']), 'angle_rad': float(cand['angle_rad']),
        'angle_deg': float(cand['angle_deg']), 'width_px': float(cand['width_px']),
        'width_mm_est': float(cand['width_mm']),
    }


def draw_candidate(img, cand, color):
    """Draw the region candidate; if it has no usable depth, mark its pixel with an X."""
    f = cand_to_fields(cand)
    if f is not None and draw_grasp(img, f, color=color, label=True):
        return True
    if cand.get('region_u_px') is not None:
        p = (int(round(cand['region_u_px'])), int(round(cand['region_v_px'])))
        cv2.drawMarker(img, p, color, cv2.MARKER_TILTED_CROSS, 14, 2)
        return True
    return False


def decision_text(decision):
    return 'ACCEPTED' if decision['accepted'] else f"NO GRASP: {decision['reason']}"


def save_burst_overlay(color_full, burst_grasps, burst_id, out_path=None):
    """Draw every valid grasp from one burst on a single RGB image and save it."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    overlay = color_full.copy()
    drawn = 0
    for frame_idx, fields in burst_grasps:
        shade = np.uint8([[int(255 * frame_idx / max(1, BURST_FRAMES - 1))]])
        color = cv2.applyColorMap(shade, cv2.COLORMAP_JET)[0, 0].tolist()
        if draw_grasp(overlay, fields, color=color, label=False):
            drawn += 1
    cv2.putText(overlay, f"{SCENE_NAME}  burst {burst_id}  {drawn}/{BURST_FRAMES} drawn",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    path = out_path or os.path.join(SAVE_DIR, f'{SCENE_NAME}_burst_{burst_id}_overlay.png')
    cv2.imwrite(path, overlay)
    print(f"Saved all-frames overlay ({drawn}/{BURST_FRAMES} grasps drawn): {path}")
    return path


def _banner(img, text, color=(255, 255, 255), y=22):
    """Write a one-line label, readable on any background."""
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _csv(v):
    if v is None:
        return ''
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return round(v, 5)
    return v


def _contact_sheet(tiles, path):
    if not tiles:
        return
    cols = CONTACT_SHEET_COLS
    blank = np.zeros_like(tiles[0])
    while len(tiles) % cols:
        tiles.append(blank)
    sheet = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])
    cv2.imwrite(path, sheet)


def save_filter_outputs(seq_dir, frames):
    """Per-frame filtered overlays, filtered_grasps.csv, filtered contact sheet."""
    n = len(frames)
    rows, tiles = [], []
    n_accepted = 0
    for fr in frames:
        i = fr['frame']
        stem = f'frame_{i:02d}'
        cand, dec = fr['cand'], fr['decision']
        angle_file = width_file = ''
        if fr['angle_map'] is not None:
            angle_file = stem + '_angle.npy'
            np.save(os.path.join(seq_dir, angle_file), fr['angle_map'])
        if fr['width_map'] is not None:
            width_file = stem + '_width.npy'
            np.save(os.path.join(seq_dir, width_file), fr['width_map'])

        img = fr['color'].copy()
        if fr['region'] is not None:
            draw_region(img, fr['region'])
        if fr['valid']:
            draw_grasp(img, fr['fields'], color=MODEL_COLOR, label=False)
        if cand['region_found']:
            if dec['accepted']:
                color = ACCEPT_COLOR
            else:
                color = CAND_OK_COLOR if cand['frame_ok'] else CAND_FAIL_COLOR
            draw_candidate(img, cand, color)
        if dec['accepted']:
            n_accepted += 1
        _banner(img, f"{SCENE_NAME}  frame {i + 1}/{n}  {decision_text(dec)}",
                color=ACCEPT_COLOR if dec['accepted'] else (0, 0, 255))
        _banner(img, "yellow=region  green=model  cyan/orange=region cand  magenta=accepted",
                y=img.shape[0] - 10)
        filtered_file = stem + '_filtered.png'
        cv2.imwrite(os.path.join(seq_dir, filtered_file), img)
        tiles.append(cv2.resize(img, CONTACT_SHEET_TILE, interpolation=cv2.INTER_AREA))

        geom = grasp_geometry(fr['fields']) if fr['valid'] else None
        model_to_region_mm = None
        if geom is not None and cand.get('z_cam_m') is not None:
            model_to_region_mm = 1000.0 * math.hypot(fr['fields']['x_cam_m'] - cand['x_cam_m'],
                                                     fr['fields']['y_cam_m'] - cand['y_cam_m'])
        row = {k: _csv(cand.get(k)) for k in grf.CAND_KEYS if k in FILTER_FIELDS}
        row.update({
            'frame': i,
            'model_result_valid': int(geom is not None),
            'model_center_u_px': _csv(geom['center'][0]) if geom else '',
            'model_center_v_px': _csv(geom['center'][1]) if geom else '',
            'model_angle_deg': _csv(fr['fields']['angle_deg']) if geom else '',
            'model_width_mm_est': _csv(fr['fields']['width_mm_est']) if geom else '',
            'model_selected_score': _csv(fr['score']),
            'model_to_region_mm': _csv(model_to_region_mm),
            'window_count': dec['window_count'],
            'pos_spread_mm': _csv(dec['pos_spread_mm']),
            'angle_spread_deg': _csv(dec['angle_spread_deg']),
            'decision': dec['decision'],
            'decision_reason': dec['reason'],
            'filtered_overlay_file': filtered_file,
            'angle_file': angle_file,
            'width_file': width_file,
        })
        rows.append(row)

    with open(os.path.join(seq_dir, 'filtered_grasps.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FILTER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _contact_sheet(tiles, os.path.join(seq_dir, 'filtered_contact_sheet.png'))
    print(f"  Filter: {n_accepted}/{n} frames accepted (see filtered_grasps.csv).")
    return n_accepted


def save_sequence(frames, burst_id, crop):
    """Write one folder for a consecutive-frame sequence (original outputs
    unchanged), then the filter outputs."""
    seq_name = f'{SCENE_NAME}_seq_{burst_id}'
    seq_dir = os.path.join(SAVE_DIR, seq_name)
    os.makedirs(seq_dir, exist_ok=True)
    y0, x0, size = crop
    n = len(frames)
    t0 = frames[0]['t'] if frames else 0.0

    rows = []
    tiles = []
    n_valid = 0
    for fr in frames:
        i = fr['frame']
        stem = f'frame_{i:02d}'
        fields = fr['fields']
        geom = grasp_geometry(fields) if fr['valid'] else None
        reason = invalid_reason(fr['result'], fr['valid'], fields)

        rgb_file = stem + '_rgb.png'
        overlay_file = stem + '_overlay.png'
        depth_file = stem + '_depth.npy'
        heatmap_file = ''
        quality_file = ''

        cv2.imwrite(os.path.join(seq_dir, rgb_file), fr['color'])
        np.save(os.path.join(seq_dir, depth_file), fr['depth'])
        if fr['heatmap'] is not None:
            if fr['heatmap'].dtype == np.uint8:
                heatmap_file = stem + '_heatmap.png'
                cv2.imwrite(os.path.join(seq_dir, heatmap_file), fr['heatmap'])
            else:
                heatmap_file = stem + '_heatmap.npy'
                np.save(os.path.join(seq_dir, heatmap_file), fr['heatmap'])
        if fr['quality'] is not None:
            quality_file = stem + '_quality.npy'
            np.save(os.path.join(seq_dir, quality_file), fr['quality'])

        overlay = fr['color'].copy()
        if geom is not None:
            draw_grasp(overlay, fields)
            _banner(overlay, f"{SCENE_NAME}  frame {i + 1}/{n}")
            n_valid += 1
        else:
            _banner(overlay, f"{SCENE_NAME}  frame {i + 1}/{n}  NO VALID PREDICTION ({reason})",
                    color=(0, 0, 255))
        cv2.imwrite(os.path.join(seq_dir, overlay_file), overlay)
        tiles.append(cv2.resize(overlay, CONTACT_SHEET_TILE, interpolation=cv2.INTER_AREA))

        depth_at_center = ''
        if geom is not None:
            cu, cv_ = int(round(geom['center'][0])), int(round(geom['center'][1]))
            h, w = fr['depth'].shape[:2]
            if 0 <= cv_ < h and 0 <= cu < w:
                depth_at_center = float(fr['depth'][cv_, cu])

        crop_depth = fr['depth'][y0:y0 + size, x0:x0 + size]
        row = {
            'frame': i,
            'ms_since_first_frame': round((fr['t'] - t0) * 1000.0, 1),
            'result_valid': int(geom is not None),
            'invalid_reason': reason,
            'depth_at_center_m': depth_at_center,
            'selected_score': '' if fr['score'] is None else fr['score'],
            'crop_invalid_depth_fraction': round(float((~valid_depth_mask(crop_depth)).mean()), 4),
            'rgb_file': rgb_file,
            'overlay_file': overlay_file,
            'heatmap_file': heatmap_file,
            'quality_file': quality_file,
            'depth_file': depth_file,
            'raw_result': fr['raw_result'],
            'center_u_px': '', 'center_v_px': '',
            'jaw1_u_px': '', 'jaw1_v_px': '', 'jaw2_u_px': '', 'jaw2_v_px': '',
            **fields,
        }
        if geom is not None:
            row.update({
                'center_u_px': round(geom['center'][0], 2),
                'center_v_px': round(geom['center'][1], 2),
                'jaw1_u_px': round(geom['jaw1'][0], 2),
                'jaw1_v_px': round(geom['jaw1'][1], 2),
                'jaw2_u_px': round(geom['jaw2'][0], 2),
                'jaw2_v_px': round(geom['jaw2'][1], 2),
            })
        rows.append(row)

    with open(os.path.join(seq_dir, 'predictions.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SEQ_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    _contact_sheet(tiles, os.path.join(seq_dir, 'contact_sheet.png'))

    if frames:
        save_burst_overlay(frames[-1]['color'],
                           [(fr['frame'], fr['fields']) for fr in frames if fr['valid']],
                           burst_id,
                           out_path=os.path.join(seq_dir, 'all_frames_overlay.png'))

    n_accepted = save_filter_outputs(seq_dir, frames)
    region = frames[0]['region'] if frames else None

    with open(os.path.join(seq_dir, 'sequence_meta.json'), 'w') as f:
        json.dump({
            'scene': SCENE_NAME,
            'object': OBJECT_NAME,
            'position': POSITION_NAME,
            'sequence_id': burst_id,
            'frames_captured': n,
            'frames_with_valid_prediction': n_valid,
            'frames_accepted_by_filter': n_accepted,
            'model_file': os.path.basename(MODEL_FILE),
            'selection_mode': SELECTION_MODE,
            'open_loop_height_config': OPEN_LOOP_HEIGHT,
            'assumed_robot_z_m': ASSUMED_ROBOT_Z_M,
            'camera_matrix': CAM_K.tolist() if CAM_K is not None else None,
            'crop_y0_x0_size': [int(y0), int(x0), int(size)],
            'angle_sign': ANGLE_SIGN,
            'drawing_convention': (
                'center = model x,y,z projected to the full RGB image with camera_matrix; '
                'jaw1/jaw2 = center -/+ (width_px/2) * (cos(angle), angle_sign*sin(angle)); '
                'width_px is the model width value used as-is, with no rescaling'),
            'quality_map': (
                'frame_XX_quality.npy = float map after Gaussian filter, clip to [0, 1), and '
                'exclusion of invalid-depth pixels; selected_score = its value at the chosen pixel'),
            'cross_frame_processing': (
                'predictions.csv: none. filtered_grasps.csv: consistency check over consecutive '
                'frames (accept/reject only; the accepted candidate is not averaged)'),
            'grasp_region': region.to_dict() if region is not None else None,
            'grasp_filter_settings': grf.settings_dict(),
        }, f, indent=2)

    print(f"Saved sequence: {seq_dir}")
    print(f"  {n_valid}/{n} frames have a valid model prediction; all {n} frames were saved.")
    return seq_dir


def log_grasp(result, capture_type, burst_id='', frame_in_burst='', image_prefix=''):
    """Append one row to grasp_log.csv."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    new_file = not os.path.exists(LOG_CSV)
    valid, fields = parse_result(result)
    row = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'scene': SCENE_NAME,
        'object': OBJECT_NAME,
        'position': POSITION_NAME,
        'capture_type': capture_type,
        'burst_id': burst_id,
        'frame_in_burst': frame_in_burst,
        'result_valid': int(valid),
        'assumed_robot_z_m': ASSUMED_ROBOT_Z_M,
        'image_prefix': os.path.basename(image_prefix) if image_prefix else '',
        'raw_result': repr(result),
        **fields,
    }
    with open(LOG_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return valid, fields


def save_scene(camera, color_full, depth_aligned, grasp_img, result, crop):
    """Save RGB, raw depth, aligned depth, invalid mask, heatmap, and metadata."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    prefix = os.path.join(SAVE_DIR, f"{SCENE_NAME}_{time.strftime('%Y%m%d_%H%M%S')}")
    y0, x0, size = crop

    try:
        _, depth_raw = camera.get_images(align=False)
    except Exception as e:
        depth_raw = None
        print(f"Could not get raw depth: {e}")

    valid = valid_depth_mask(depth_aligned)
    invalid_mask = (~valid).astype(np.uint8) * 255

    cv2.imwrite(prefix + '_rgb_full.png', color_full)
    cv2.imwrite(prefix + '_rgb_crop.png', color_full[y0:y0 + size, x0:x0 + size])

    np.save(prefix + '_depth_aligned.npy', depth_aligned)
    cv2.imwrite(prefix + '_depth_aligned_mm.png', depth_to_png16_mm(depth_aligned))

    cv2.imwrite(prefix + '_invalid_mask_full.png', invalid_mask)
    cv2.imwrite(prefix + '_invalid_mask_crop.png', invalid_mask[y0:y0 + size, x0:x0 + size])

    if depth_raw is not None:
        np.save(prefix + '_depth_raw.npy', depth_raw)
        cv2.imwrite(prefix + '_depth_raw_mm.png', depth_to_png16_mm(depth_raw))

    if grasp_img is not None:
        np.save(prefix + '_heatmap.npy', grasp_img)
        if grasp_img.dtype == np.uint8:
            cv2.imwrite(prefix + '_heatmap.png', grasp_img)

    with open(prefix + '_meta.json', 'w') as f:
        json.dump({
            'scene': SCENE_NAME,
            'object': OBJECT_NAME,
            'position': POSITION_NAME,
            'assumed_robot_z_m': ASSUMED_ROBOT_Z_M,
            'depth_dtype': str(depth_aligned.dtype),
            'depth_unit_to_mm_assumed': DEPTH_UNIT_TO_MM,
            'depth_min_valid': float(depth_aligned[valid].min()) if valid.any() else None,
            'depth_max_valid': float(depth_aligned[valid].max()) if valid.any() else None,
            'invalid_pixel_fraction_aligned': float((~valid).mean()),
            'crop_y0_x0_size': [int(y0), int(x0), int(size)],
            'grasp_result_repr': repr(result),
            'grasp_region': REGION.to_dict() if REGION is not None else None,
        }, f, indent=2)

    print(f"Saved scene files with prefix: {prefix}")
    if valid.any():
        print(f"  depth range={depth_aligned[valid].min():.4f} to "
              f"{depth_aligned[valid].max():.4f} m, invalid fraction={(~valid).mean():.3f}")
    else:
        print("  WARNING: no valid depth pixels in this frame.")
    return prefix


def print_grasp(valid, fields, label):
    if valid:
        print(f"  [{label}] x={fields['x_cam_m']*1000:.1f} mm, y={fields['y_cam_m']*1000:.1f} mm, "
              f"z={fields['z_cam_m']*1000:.1f} mm, angle={fields['angle_deg']:.1f} deg, "
              f"width={fields['width_px']:.1f} px (~{fields['width_mm_est']:.0f} mm)")
    else:
        print(f"  [{label}] no valid grasp result")


def print_filter(cand, decision, label):
    if cand['region_found'] and cand.get('width_mm') is not None:
        detail = (f"region cand width {cand['width_mm']:.1f} mm, "
                  f"{cand['angle_from_perpendicular_deg']:.1f} deg from across-box, "
                  f"along {cand['along_mm']:+.1f} mm")
    else:
        detail = 'no usable region candidate'
    print(f"  [{label}] filter: {decision_text(decision)}  ({detail})")


def define_region(color_full, depth_full):
    """Pop-up window: click the two ends of the box. Returns a GraspRegion or None."""
    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
            clicks.append((x, y))

    cv2.namedWindow(REGION_WIN)
    cv2.setMouseCallback(REGION_WIN, on_mouse)
    print("\nDefine grasp region: click the CENTER of each END of the box "
          f"(the {grf.BOX_LENGTH_MM:.0f} mm side), then press Enter.")
    region = None
    error = ''
    while True:
        img = color_full.copy()
        for p in clicks:
            cv2.circle(img, p, 5, REGION_COLOR, -1)
        region = None
        if len(clicks) == 2:
            try:
                region = grf.GraspRegion(clicks[0], clicks[1])
                draw_region(img, region)
                error = ''
            except ValueError as e:
                error = str(e)
        _banner(img, error or f"{len(clicks)}/2 ends clicked",
                color=(0, 0, 255) if error else (255, 255, 255))
        cv2.imshow(REGION_WIN, img)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and region is not None:
            break
        if k == ord('c'):
            clicks.clear()
        if k == 27:
            region = None
            break
    cv2.destroyWindow(REGION_WIN)

    if region is None:
        print("Grasp region not changed.")
        return None
    info = region.check_scale(depth_full, CAM_FX)
    print(f"Grasp region set: box length {region.length_px:.0f} px, "
          f"{region.px_per_mm:.2f} px/mm from your clicks.")
    if info.get('ratio') is not None:
        print(f"  Scale check: intrinsics give {info['px_per_mm_from_intrinsics']:.2f} px/mm "
              f"at {info['box_depth_m']:.3f} m (ratio {info['ratio']:.2f}).")
        if abs(info['ratio'] - 1.0) > 0.10:
            print("  WARNING: more than 10% off. Re-click the box ends (press 'r') "
                  "or check that the camera is looking roughly straight down.")
    else:
        print(f"  Scale check skipped: {info.get('note')}")
    return region


def main():
    global CAM_FX, CAM_K, SELECTION_MODE, REGION
    camera = RealSenseCamera(width=CAM_WIDTH, height=CAM_HEIGHT)
    color_intrin, depth_intrin = camera.get_intrinsics()
    CAM_FX = float(color_intrin.fx)
    DEPTH_CAM_K = np.array([
        [color_intrin.fx, 0, color_intrin.ppx],
        [0, color_intrin.fy, color_intrin.ppy],
        [0, 0, 1]
    ])
    CAM_K = DEPTH_CAM_K

    depth_img_que = Queue(1)
    ggcnn_cmd_que = Queue(1)   # unused -- no RobotGrasp thread consumes it
    ggcnn_config = {
        'MODEL_FILE': MODEL_FILE,
        'OPEN_LOOP_HEIGHT': OPEN_LOOP_HEIGHT,
        'GGCNN_IN_THREAD': GGCNN_IN_THREAD,
        'DEPTH_CAM_K': DEPTH_CAM_K,
    }
    ggcnn = TorchGGCNN(ggcnn_config, depth_img_que, ggcnn_cmd_que)
    if not hasattr(ggcnn, 'last_angle_map'):
        print("ERROR: grasp/ggcnn_torch.py does not have Edit F (angle/width maps). "
              "The filter will report 'model_returned_no_candidate' on every frame.")

    crop_size = 300
    crop_y_offset = 0
    crop_y_inx = -1
    crop_x_inx = -1

    burst_remaining = 0
    burst_id = ''
    burst_frame = 0
    burst_frames = []

    print("Detection-only mode: no robot IP needed, no motion commands possible.")
    olh = getattr(ggcnn, 'open_loop_height', None)
    print(f"fx = {CAM_FX:.1f} px")
    print(f"ggcnn.open_loop_height = {olh!r}, ASSUMED_ROBOT_Z_M = {ASSUMED_ROBOT_Z_M}")
    if not isinstance(olh, (int, float)):
        SELECTION_MODE = f'unknown (ggcnn.open_loop_height = {olh!r})'
    elif ASSUMED_ROBOT_Z_M > olh:
        SELECTION_MODE = f'global max each frame (assumed z {ASSUMED_ROBOT_Z_M} > open_loop_height {olh!r})'
        print("  Model mode: GLOBAL MAX each frame (independent per frame).")
    else:
        SELECTION_MODE = (f'local-peak tracking with smoothing inside the model '
                          f'(assumed z {ASSUMED_ROBOT_Z_M} <= open_loop_height {olh!r})')
        print("  Model mode: LOCAL-PEAK TRACKING with smoothing.")
    print(f"Filter: commanded opening {grf.REQUIRED_OPENING_MM:.0f} mm "
          f"(box {grf.BOX_WIDTH_MM:.0f} + 2 x {grf.CLEARANCE_PER_SIDE_MM:.0f} mm clearance, "
          f"gripper max {grf.GRIPPER_MAX_OPENING_MM} mm); model width is logged only. "
          f"{grf.STABLE_FRAMES} consistent frames needed.")
    print(f"Keys: 'r' grasp region, 'n' labels, 's' save, 'b' {BURST_FRAMES}-frame sequence, 'q'/ESC quit.")
    set_labels()

    try:
        while True:
            color_image, depth_image = camera.get_images(align=True)
            frame_time = time.time()
            color_full = color_image.copy()

            if crop_y_inx < 0:
                imh, imw = depth_image.shape
                crop_size = min(imh, imw)
                crop_y_inx = max(0, imh - crop_size) // 2 - crop_y_offset
                crop_x_inx = max(0, imw - crop_size) // 2

            grasp_img, result = ggcnn.get_grasp_img(depth_image, DEPTH_CAM_K, ASSUMED_ROBOT_Z_M)
            heatmap_for_saving = grasp_img
            # Copy now: the next get_grasp_img call replaces these.
            quality_for_saving = None if ggcnn.last_quality_map is None else ggcnn.last_quality_map.copy()
            score_for_saving = ggcnn.last_score
            angle_map = getattr(ggcnn, 'last_angle_map', None)
            width_map = getattr(ggcnn, 'last_width_map', None)
            angle_map = None if angle_map is None else angle_map.copy()
            width_map = None if width_map is None else width_map.copy()
            crop_origin = getattr(ggcnn, 'last_crop_origin', None)
            crop_scale = getattr(ggcnn, 'last_crop_scale', None)

            # Grasp acceptance rule (does not change the model's own prediction).
            cand = grf.evaluate_frame(REGION, quality_for_saving, angle_map, width_map,
                                      crop_origin, crop_scale, depth_image, CAM_K, ANGLE_SIGN)
            decision = TRACKER.update(cand)

            # Live preview: region, model prediction (green), region candidate.
            live_valid, live_fields = parse_result(result)
            preview_full = color_full.copy()
            if REGION is not None:
                draw_region(preview_full, REGION)
            if live_valid:
                draw_grasp(preview_full, live_fields, color=MODEL_COLOR)
            if cand['region_found']:
                color = ACCEPT_COLOR if decision['accepted'] else (
                    CAND_OK_COLOR if cand['frame_ok'] else CAND_FAIL_COLOR)
                draw_candidate(preview_full, cand, color)
            color_image = preview_full[crop_y_inx:crop_y_inx + crop_size,
                                       crop_x_inx:crop_x_inx + crop_size, :].copy()
            if REGION is None:
                _banner(color_image, "press 'r' to define the grasp region", color=(0, 0, 255))
            else:
                _banner(color_image, decision_text(decision),
                        color=ACCEPT_COLOR if decision['accepted'] else (0, 0, 255))

            if burst_remaining > 0:
                valid, fields = log_grasp(result, 'burst', burst_id, burst_frame,
                                          image_prefix=f'{SCENE_NAME}_seq_{burst_id}')
                print_grasp(valid, fields, f"frame {burst_frame + 1}/{BURST_FRAMES}")
                print_filter(cand, decision, f"frame {burst_frame + 1}/{BURST_FRAMES}")
                burst_frames.append({
                    'frame': burst_frame,
                    't': frame_time,
                    'color': color_full,
                    'depth': depth_image.copy(),
                    'heatmap': None if heatmap_for_saving is None else heatmap_for_saving.copy(),
                    'quality': quality_for_saving,
                    'score': score_for_saving,
                    'angle_map': angle_map,
                    'width_map': width_map,
                    'region': REGION,
                    'cand': cand,
                    'decision': decision,
                    'result': result,
                    'raw_result': repr(result),
                    'valid': valid,
                    'fields': fields,
                })
                burst_frame += 1
                burst_remaining -= 1
                if burst_remaining == 0:
                    print(f"Sequence {burst_id} captured, writing files...")
                    save_sequence(burst_frames, burst_id,
                                  (crop_y_inx, crop_x_inx, crop_size))
                    burst_frames = []
                    if AUTO_PROMPT_AFTER_BURST:
                        print("Move the object to the next placement, then enter its labels.")
                        set_labels()

            if not SHOW_GRASP_IMG or grasp_img is None:
                grasp_img = depth_image[crop_y_inx:crop_y_inx + crop_size,
                                        crop_x_inx:crop_x_inx + crop_size]

            combined_img = get_combined_img(color_image, grasp_img)
            cv2.imshow(WIN_NAME, combined_img)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('r'):
                if burst_remaining > 0:
                    print("Wait for the sequence to finish before changing the region.")
                else:
                    new_region = define_region(color_full, depth_image)
                    if new_region is not None:
                        REGION = new_region
                        TRACKER.reset()
            elif key & 0xFF == ord('n'):
                if burst_remaining > 0:
                    print("Wait for the sequence to finish before changing labels.")
                else:
                    set_labels()
            elif key & 0xFF == ord('s'):
                prefix = save_scene(camera, color_full, depth_image, heatmap_for_saving, result,
                                    (crop_y_inx, crop_x_inx, crop_size))
                valid, fields = log_grasp(result, 'snapshot', image_prefix=prefix)
                print_grasp(valid, fields, 'snapshot')
                print_filter(cand, decision, 'snapshot')
                if quality_for_saving is not None:
                    np.save(prefix + '_quality.npy', quality_for_saving)
                    print(f"  Saved quality map: {prefix}_quality.npy "
                          f"(selected score {score_for_saving:.4f})")
                if valid:
                    overlay = color_full.copy()
                    if draw_grasp(overlay, fields):
                        cv2.imwrite(prefix + '_rgb_grasp_overlay.png', overlay)
                        print(f"  Saved grasp overlay: {prefix}_rgb_grasp_overlay.png")
                    else:
                        print("  No overlay saved: grasp point has no usable depth.")
            elif key & 0xFF == ord('b') and burst_remaining == 0:
                if REGION is None:
                    print("NOTE: no grasp region defined; every frame will be 'no grasp'. "
                          "Press 'r' first if you want the filter results.")
                burst_id = time.strftime('%Y%m%d_%H%M%S')
                burst_remaining = BURST_FRAMES
                burst_frame = 0
                burst_frames = []
                TRACKER.reset()
                print(f"Starting sequence {burst_id} ({BURST_FRAMES} consecutive frames) "
                      f"-- keep the object and camera still.")
            elif key & 0xFF == ord('q') or key == 27:
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
