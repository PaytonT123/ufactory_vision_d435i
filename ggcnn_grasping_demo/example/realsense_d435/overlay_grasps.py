"""
overlay_grasps.py -- draw logged grasps on the full-resolution RGB image.

Usage (from any folder):
    python overlay_grasps.py                      # lists scenes found in the log
    python overlay_grasps.py wifi_box_c_posC      # latest burst of that scene
    python overlay_grasps.py wifi_box_c_posC 20261006_135546   # a specific burst
    python overlay_grasps.py wifi_box_c_posC --dir "C:\\path\\to\\grasp_scenes"

Reads grasp_log.csv and intrinsics.json from the grasp_scenes folder (default:
a grasp_scenes folder next to this script). Writes overlay PNGs there.

Drawn per grasp: a red dot at the selected pixel, a line along the jaw opening
(length = predicted width in pixels), and two short finger bars at its ends.
Burst grasps are colored by frame index (blue -> red). The snapshot's top-5
quality peaks are drawn as small white circles with their scores.
"""

import argparse
import csv
import json
import math
import os

import cv2
import numpy as np

ANGLE_SIGN = 1                # flip to -1 if the angle looks mirrored
OPENING_ALONG_ANGLE = True    # set False if the opening looks rotated 90 degrees


def wrap_half_pi(d):
    """Angle differences wrap at 180 deg (grasp angle is only defined mod pi)."""
    return (d + math.pi / 2) % math.pi - math.pi / 2


def draw_grasp(img, u, v, theta, w_px, color, label=None, thick=2):
    th = ANGLE_SIGN * theta
    a = np.array([math.cos(th), math.sin(th)])     # opening axis (between fingers)
    b = np.array([-a[1], a[0]])                    # finger axis
    if not OPENING_ALONG_ANGLE:
        a, b = b, a
    c = np.array([u, v], dtype=float)
    p1, p2 = c - a * w_px / 2, c + a * w_px / 2
    jaw = max(w_px * 0.3, 16)
    for p in (p1, p2):
        cv2.line(img, tuple((p - b * jaw / 2).astype(int)),
                 tuple((p + b * jaw / 2).astype(int)), color, thick, cv2.LINE_AA)
    cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), color, 1, cv2.LINE_AA)
    cv2.circle(img, (int(round(u)), int(round(v))), 4, (0, 0, 255), -1, cv2.LINE_AA)
    if label is not None:
        cv2.putText(img, str(label), (int(u) + 6, int(v) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scene', nargs='?')
    ap.add_argument('burst', nargs='?')
    ap.add_argument('--dir', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'grasp_scenes'))
    ap.add_argument('--log', default='grasp_log.csv')
    args = ap.parse_args()

    d = args.dir
    rows = list(csv.DictReader(open(os.path.join(d, args.log), newline='')))
    scenes = sorted({r['scene'] for r in rows})
    if not args.scene:
        print('Scenes in the log:')
        for s in scenes:
            print('  ', s)
        return
    if args.scene not in scenes:
        print(f"Scene '{args.scene}' not found. Available: {scenes}")
        return

    K = json.load(open(os.path.join(d, 'intrinsics.json')))
    fx, fy, ppx, ppy = K['fx'], K['fy'], K['ppx'], K['ppy']

    rows = [r for r in rows if r['scene'] == args.scene]
    snaps = [r for r in rows if r['capture_type'] == 'snapshot' and r['image_prefix']]
    bursts = [r for r in rows if r['capture_type'] == 'burst']
    if not args.burst and bursts:
        args.burst = bursts[-1]['burst_id']
    bursts = [r for r in bursts if r['burst_id'] == args.burst]

    base = None
    if snaps:
        p = os.path.join(d, snaps[-1]['image_prefix'] + '_rgb_full.png')
        if os.path.exists(p):
            base = cv2.imread(p)
    if base is None:
        print('No snapshot RGB found for this scene (press s in the capture script).')
        return

    img = base.copy()
    n = max(len(bursts), 1)
    cmap = cv2.applyColorMap(np.linspace(0, 255, n).astype(np.uint8).reshape(-1, 1),
                             cv2.COLORMAP_JET)

    # snapshot's top-5 quality peaks, for context
    if snaps and snaps[-1].get('peaks_full_json'):
        for r_, c_, s_ in json.loads(snaps[-1]['peaks_full_json']):
            cv2.circle(img, (int(c_), int(r_)), 9, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"{s_:.2f}", (int(c_) + 10, int(r_) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    print(f"Scene {args.scene}, burst {args.burst}")
    print('frame status  sel(row,col)  score   jump_mm  d_ang_deg  w_mm   z_mm  reproj_err_px')
    prev = None
    for i, r in enumerate(bursts):
        if r['result_valid'] != '1':
            print(f"{i:>5} {r['status']}  (no valid result)")
            prev = None
            continue
        x, y, z = (float(r[k]) for k in ('x_cam_m', 'y_cam_m', 'z_cam_m'))
        th, w = float(r['angle_rad']), float(r['width_px'])
        row_s, col_s = float(r['sel_row_full']), float(r['sel_col_full'])
        # reprojection check: camera-frame x,y,z should land back on the selected pixel
        u, v = fx * x / z + ppx, fy * y / z + ppy
        err = math.hypot(u - col_s, v - row_s)
        color = tuple(int(c) for c in cmap[i, 0])
        draw_grasp(img, col_s, row_s, th, w, color, label=i)
        jump = dang = ''
        if prev is not None:
            jump = f"{math.dist((x, y, z), prev[0]) * 1000:7.1f}"
            dang = f"{math.degrees(wrap_half_pi(th - prev[1])):9.1f}"
        print(f"{i:>5} {r['status']:<7} ({int(row_s):>3},{int(col_s):>3})  "
              f"{float(r['quality_score']):.3f}  {jump:>7}  {dang:>9}  "
              f"{float(r['width_mm_est']):5.0f}  {z * 1000:5.0f}  {err:6.2f}")
        prev = ((x, y, z), th)

    out = os.path.join(d, f"overlay_{args.scene}_{args.burst}.png")
    cv2.imwrite(out, img)
    print('Wrote', out)


if __name__ == '__main__':
    main()
