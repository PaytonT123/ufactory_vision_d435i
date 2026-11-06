"""
test_detection_only_RGB_Depth_Heatmap.py

Object-detection-only with scene saving
and grasp-result logging.

This intentionally does NOT import RobotGrasp. There is no robot connection
object anywhere in this script, so no code path in this file can send a
motion command to the arm -- it only streams the D435i, runs the GGCNN
grasp-detection model on the depth image, and displays/logs the result.
No robot IP is needed to run it.

At startup you are asked (in the terminal) for the object and position.

Keys (click the preview window first):
    n        new labels any time: type a new object/position in the terminal
             (you are also asked automatically after every 'b' burst)
    s        save the current scene images + log one grasp row
             (also saves *_rgb_grasp_overlay.png: the grasp drawn on the RGB)
    b        burst: log grasp rows for the next BURST_FRAMES frames, then save
             ONE overlay image with every frame's grasp drawn on the RGB
             (<scene>_burst_<id>_overlay.png; blue = first frame, red = last)
    q / ESC  quit

Every logged row goes to grasp_scenes/grasp_log.csv.

GRASP OVERLAY (added for the "check the proposed grasp on the actual object"
task): the selected point is projected from camera-frame x,y,z back to a
pixel in the FULL RGB image using the same intrinsics the model used, so no
crop offsets are involved. The line is the jaw-opening axis (length =
predicted width, rotated to the predicted angle); the short ticks at each
end are the jaws; the red dot is the selected center.
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
from ufactory_vision.ggcnn_grasping_demo.grasp.ggcnn_torch_orig_2 import TorchGGCNN
# NOTE: RobotGrasp is deliberately NOT imported here.

WIN_NAME = 'RealSense-D435 -- DETECTION ONLY, NO ROBOT'
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

# ---- labels for the 3 objects x 3 positions test ----
# These are entered in the terminal at startup and whenever you press 'n'.
OBJECT_NAME = 'obj1'
POSITION_NAME = 'posA'
SCENE_NAME = f'{OBJECT_NAME}_{POSITION_NAME}'


def _clean_label(text):
    """Make a label safe for file names: spaces -> _, drop odd characters."""
    text = text.strip().replace(' ', '_')
    return ''.join(c for c in text if c.isalnum() or c in '_-')


def set_labels():
    """Ask in the terminal for object and position. Enter keeps the current value."""
    global OBJECT_NAME, POSITION_NAME, SCENE_NAME
    print("\n--- Set labels (camera preview pauses while you type) ---")
    print("Type a new value, or just press Enter to keep the current one.")
    obj = _clean_label(input(f"  1) OBJECT (what it is, e.g. iphone_box)    currently '{OBJECT_NAME}': "))
    pos = _clean_label(input(f"  2) POSITION (where it is, e.g. posA/posB/posC)  currently '{POSITION_NAME}': "))
    if obj:
        OBJECT_NAME = obj
    if pos:
        POSITION_NAME = pos
    SCENE_NAME = f'{OBJECT_NAME}_{POSITION_NAME}'
    print(f"Now capturing: object={OBJECT_NAME}, position={POSITION_NAME}")
    print("Click the preview window again, then press 's' / 'b'.\n")

BURST_FRAMES = 10        # rows logged per 'b' press

# After each burst finishes, automatically ask for the next object/position.
# Press Enter at the prompt to keep the current value.
AUTO_PROMPT_AFTER_BURST = True

# ---- output locations ----
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'grasp_scenes')
LOG_CSV = os.path.join(SAVE_DIR, 'grasp_log.csv')

CAM_FX = None   # set in main() from camera intrinsics
CAM_K = None    # 3x3 intrinsics matrix, set in main(); used by the grasp overlay

# ---- grasp overlay settings ----
# If the drawn line runs ALONG the long side of a narrow object instead of
# across it, the angle sign convention is flipped: change this to +1.0.
ANGLE_SIGN = -1.0        # -1 = standard GGCNN convention (image rows point down)
JAW_TICK_PX = 10         # half-length of the jaw tick marks, pixels

# Depth from camera.get_images() is float32 meters (confirmed 2026-09-29:
# valid range 0.455-1.299). PNGs are written in mm; .npy keeps meters.
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


def grasp_to_pixel(fields):
    """Camera-frame x,y,z (m) -> pixel (u, v) in the FULL RGB image.

    Returns None if the grasp has no usable depth or intrinsics are not set.
    """
    if CAM_K is None:
        return None
    z = fields.get('z_cam_m')
    if not isinstance(z, float) or z <= 0:
        return None
    u = CAM_K[0, 0] * fields['x_cam_m'] / z + CAM_K[0, 2]
    v = CAM_K[1, 1] * fields['y_cam_m'] / z + CAM_K[1, 2]
    return u, v


def draw_grasp(img, fields, color=(0, 255, 0), label=True):
    """Draw the grasp on img (BGR, full frame): center, opening line, jaw ticks.

    Returns True if something was drawn, False if the grasp could not be
    projected (no depth at the grasp point).
    """
    uv = grasp_to_pixel(fields)
    if uv is None:
        return False
    u, v = uv
    a = fields['angle_rad']
    half = 0.5 * fields['width_px']

    # Jaw-opening axis
    dx = half * math.cos(a)
    dy = ANGLE_SIGN * half * math.sin(a)
    # Perpendicular direction, for the jaw tick marks
    tx = -ANGLE_SIGN * JAW_TICK_PX * math.sin(a)
    ty = JAW_TICK_PX * math.cos(a)

    for side in (-1, 1):
        px, py = u + side * dx, v + side * dy
        cv2.line(img, (int(round(px - tx)), int(round(py - ty))),
                 (int(round(px + tx)), int(round(py + ty))), color, 2)
    cv2.line(img, (int(round(u - dx)), int(round(v - dy))),
             (int(round(u + dx)), int(round(v + dy))), color, 2)
    cv2.circle(img, (int(round(u)), int(round(v))), 4, (0, 0, 255), -1)

    if label:
        text = f"{fields['width_mm_est']:.0f} mm, {fields['angle_deg']:.0f} deg"
        cv2.putText(img, text, (int(u) + 10, int(v) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return True


def save_burst_overlay(color_full, burst_grasps, burst_id):
    """Draw every valid grasp from one burst on a single RGB image and save it.

    burst_grasps is a list of (frame_index, fields). Blue = first frame,
    red = last frame, so you can see whether the selection moves over time.
    """
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
    path = os.path.join(SAVE_DIR, f'{SCENE_NAME}_burst_{burst_id}_overlay.png')
    cv2.imwrite(path, overlay)
    print(f"Saved burst overlay ({drawn}/{BURST_FRAMES} grasps drawn): {path}")
    return path


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
    invalid_mask = (~valid).astype(np.uint8) * 255   # 255 = no depth

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


def main():
    camera = RealSenseCamera(width=CAM_WIDTH, height=CAM_HEIGHT)
    color_intrin, depth_intrin = camera.get_intrinsics()
    global CAM_FX, CAM_K
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

    crop_size = 300
    crop_y_offset = 0
    crop_y_inx = -1
    crop_x_inx = -1

    burst_remaining = 0
    burst_id = ''
    burst_frame = 0
    burst_grasps = []   # (frame_index, fields) for each valid grasp in the current burst

    print("Detection-only mode: no robot IP needed, no motion commands possible.")
    olh = getattr(ggcnn, 'open_loop_height', None)
    print(f"fx = {CAM_FX:.1f} px")
    print(f"ggcnn.open_loop_height = {olh!r}, ASSUMED_ROBOT_Z_M = {ASSUMED_ROBOT_Z_M}")
    if olh is None:
        print("  Could not read open_loop_height -- check TorchGGCNN.__init__.")
    elif ASSUMED_ROBOT_Z_M > olh:
        print("  Mode: GLOBAL MAX each frame (independent per frame) -- good for consistency test.")
    else:
        print("  Mode: LOCAL-PEAK TRACKING with smoothing -- bursts will look steadier than they are.")
    print(f"Keys: 'n' new object/position, 's' save images + log, "
          f"'b' log {BURST_FRAMES}-frame burst, 'q'/ESC quit.")
    set_labels()

    try:
        while True:
            color_image, depth_image = camera.get_images(align=True)
            color_full = color_image.copy()

            if crop_y_inx < 0:
                imh, imw = depth_image.shape
                crop_size = min(imh, imw)
                crop_y_inx = max(0, imh - crop_size) // 2 - crop_y_offset
                crop_x_inx = max(0, imw - crop_size) // 2

            grasp_img, result = ggcnn.get_grasp_img(depth_image, DEPTH_CAM_K, ASSUMED_ROBOT_Z_M)
            heatmap_for_saving = grasp_img

            # Live preview: draw the current grasp on a copy of the full RGB,
            # then crop it for display. color_full itself stays clean for saving.
            live_valid, live_fields = parse_result(result)
            preview_full = color_full.copy()
            if live_valid:
                draw_grasp(preview_full, live_fields)
            color_image = preview_full[crop_y_inx:crop_y_inx + crop_size,
                                       crop_x_inx:crop_x_inx + crop_size, :]

            # Burst logging: one row per frame, plus one overlay image at the end
            if burst_remaining > 0:
                valid, fields = log_grasp(result, 'burst', burst_id, burst_frame)
                print_grasp(valid, fields, f"burst {burst_frame + 1}/{BURST_FRAMES}")
                if valid:
                    burst_grasps.append((burst_frame, fields))
                burst_frame += 1
                burst_remaining -= 1
                if burst_remaining == 0:
                    print(f"Burst {burst_id} done.")
                    save_burst_overlay(color_full, burst_grasps, burst_id)
                    burst_grasps = []
                    if AUTO_PROMPT_AFTER_BURST:
                        print("Move the object to the next placement, then enter its labels.")
                        set_labels()

            if not SHOW_GRASP_IMG or grasp_img is None:
                grasp_img = depth_image[crop_y_inx:crop_y_inx + crop_size,
                                        crop_x_inx:crop_x_inx + crop_size]

            combined_img = get_combined_img(color_image, grasp_img)
            cv2.imshow(WIN_NAME, combined_img)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('n'):
                if burst_remaining > 0:
                    print("Wait for the burst to finish before changing labels.")
                else:
                    set_labels()
            elif key & 0xFF == ord('s'):
                prefix = save_scene(camera, color_full, depth_image, heatmap_for_saving, result,
                                    (crop_y_inx, crop_x_inx, crop_size))
                valid, fields = log_grasp(result, 'snapshot', image_prefix=prefix)
                print_grasp(valid, fields, 'snapshot')
                if valid:
                    overlay = color_full.copy()
                    if draw_grasp(overlay, fields):
                        cv2.imwrite(prefix + '_rgb_grasp_overlay.png', overlay)
                        print(f"  Saved grasp overlay: {prefix}_rgb_grasp_overlay.png")
                    else:
                        print("  No overlay saved: grasp point has no usable depth.")
            elif key & 0xFF == ord('b') and burst_remaining == 0:
                burst_id = time.strftime('%Y%m%d_%H%M%S')
                burst_remaining = BURST_FRAMES
                burst_frame = 0
                burst_grasps = []
                print(f"Starting burst {burst_id} ({BURST_FRAMES} frames) -- keep the scene still.")
            elif key & 0xFF == ord('q') or key == 27:
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
