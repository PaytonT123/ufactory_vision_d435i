"""
grasp_region_filter.py

DETECTION ONLY: this module contains no robot code and sends no commands.

Per frame:
  1. Restrict candidates to a labelled GRASP REGION: the centre section of the
     box, excluding END_MARGIN_MM at each end, and within ACROSS_TOL_MM of the
     box centreline. The region is defined by clicking the two ends of the box
     in the RGB image.
  2. Take the highest-quality pixel INSIDE that region from the model's own
     quality map. (The model's own global-max prediction is not changed; the
     capture script logs it next to this result.)
  3. Check that candidate: valid depth around the centre, a predicted opening
     of at least box width + clearance on both sides, and a jaw axis that
     crosses the box rather than running along it.

Across frames:
  4. Accept only when STABLE_FRAMES consecutive frames pass steps 1-3 and their
     positions and angles agree within the limits below. The accepted candidate
     is the most recent frame's candidate exactly as predicted -- nothing is
     averaged or smoothed.

If any step fails, the answer is "no grasp", with the reason.

The model's width is used as-is (no rescaling). The angle check uses the same
jaw-axis convention as the drawing code (ANGLE_SIGN in the capture script).
If that convention is wrong, the angle check will reject every frame: fix the
convention after checking it, do not loosen the tolerance.

"""

import math
from collections import deque

import numpy as np

# ---- measured object: iphone_box (mm) ----
BOX_LENGTH_MM = 161.0   # long side
BOX_WIDTH_MM = 29.0     # side the jaws close across
BOX_HEIGHT_MM = 8.8

# ---- grasp region ----
END_MARGIN_MM = 40.0    # excluded at EACH end of the box -> central 81 mm allowed
ACROSS_TOL_MM = 6.0     # max offset of the grasp centre from the box centreline

# ---- opening ----
CLEARANCE_PER_SIDE_MM = 5.0
REQUIRED_OPENING_MM = BOX_WIDTH_MM + 2.0 * CLEARANCE_PER_SIDE_MM   # 39 mm
GRIPPER_MAX_OPENING_MM = None   # set from the gripper spec; None = not checked

# ---- jaw angle ----
ANGLE_TOL_DEG = 15.0    # max deviation of the jaw axis from perpendicular to the box

# ---- depth ----
DEPTH_WINDOW_HALF_PX = 2          # 5x5 window around the centre
MIN_VALID_DEPTH_FRACTION = 0.8

# ---- consistency over frames ----
STABLE_FRAMES = 5
MAX_POS_SPREAD_MM = 5.0           # max distance of any window frame from the window median
MAX_ANGLE_SPREAD_DEG = 10.0       # max - min angle across the window


def settings_dict():
    """All thresholds, for writing into sequence_meta.json."""
    return {
        'box_length_mm': BOX_LENGTH_MM,
        'box_width_mm': BOX_WIDTH_MM,
        'box_height_mm': BOX_HEIGHT_MM,
        'end_margin_mm': END_MARGIN_MM,
        'across_tol_mm': ACROSS_TOL_MM,
        'clearance_per_side_mm': CLEARANCE_PER_SIDE_MM,
        'required_opening_mm': REQUIRED_OPENING_MM,
        'gripper_max_opening_mm': GRIPPER_MAX_OPENING_MM,
        'angle_tol_deg': ANGLE_TOL_DEG,
        'depth_window_half_px': DEPTH_WINDOW_HALF_PX,
        'min_valid_depth_fraction': MIN_VALID_DEPTH_FRACTION,
        'stable_frames': STABLE_FRAMES,
        'max_pos_spread_mm': MAX_POS_SPREAD_MM,
        'max_angle_spread_deg': MAX_ANGLE_SPREAD_DEG,
        'accepted_candidate': 'most recent frame of the stable window, as predicted (no averaging)',
        'opening_source': 'model width map value at the region candidate, used as-is (no rescaling)',
    }


class GraspRegion:
    """Grasp region on the box, defined in full-image pixels by the two box ends."""

    def __init__(self, end1_px, end2_px, box_length_mm=BOX_LENGTH_MM):
        p1 = np.asarray(end1_px, dtype=float)
        p2 = np.asarray(end2_px, dtype=float)
        length_px = float(np.linalg.norm(p2 - p1))
        if length_px < 20:
            raise ValueError('box ends are too close together; click both ends of the long side')
        self.end1, self.end2 = p1, p2
        self.mid = (p1 + p2) / 2.0
        self.axis = (p2 - p1) / length_px                    # unit vector along the box (u, v)
        self.normal = np.array([-self.axis[1], self.axis[0]])
        self.length_px = length_px
        self.px_per_mm = length_px / box_length_mm
        self.half_along_px = (box_length_mm / 2.0 - END_MARGIN_MM) * self.px_per_mm
        self.half_across_px = ACROSS_TOL_MM * self.px_per_mm
        if self.half_along_px <= 0:
            raise ValueError('END_MARGIN_MM leaves no region on the box')
        self.scale_info = None
        self._mask_cache = {}

    def local(self, u, v):
        """(along, across) offsets from the box centre, in pixels."""
        d = np.stack([np.asarray(u, float) - self.mid[0],
                      np.asarray(v, float) - self.mid[1]], axis=-1)
        return d @ self.axis, d @ self.normal

    def contains(self, u, v):
        along, across = self.local(u, v)
        return (np.abs(along) <= self.half_along_px) & (np.abs(across) <= self.half_across_px)

    def corners(self):
        """4 region corners in full-image pixels (u, v), in drawing order."""
        a = self.axis * self.half_along_px
        n = self.normal * self.half_across_px
        return np.array([self.mid - a - n, self.mid + a - n, self.mid + a + n, self.mid - a + n])

    def mask(self, shape, crop_origin, crop_scale):
        """Boolean region mask in the model's map coordinates."""
        key = (tuple(shape), tuple(crop_origin), float(crop_scale))
        if key not in self._mask_cache:
            rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
            v = rows * crop_scale + crop_origin[0]
            u = cols * crop_scale + crop_origin[1]
            self._mask_cache[key] = self.contains(u, v)
        return self._mask_cache[key]

    def check_scale(self, depth_full, fx):
        """Sanity check on the clicks: compare px/mm from the clicked length with
        px/mm implied by the intrinsics at the box depth. Assumes the camera looks
        roughly straight down; this is NOT a calibration."""
        ts = np.linspace(-0.3, 0.3, 25)
        pts = self.mid[None, :] + ts[:, None] * self.length_px * self.axis[None, :]
        h, w = depth_full.shape[:2]
        zs = []
        for u, v in pts:
            ui, vi = int(round(u)), int(round(v))
            if 0 <= vi < h and 0 <= ui < w:
                z = float(depth_full[vi, ui])
                if math.isfinite(z) and z > 0:
                    zs.append(z)
        if len(zs) < 5:
            self.scale_info = {'ratio': None, 'note': 'not enough valid depth along the box'}
            return self.scale_info
        z = float(np.median(zs))
        expected = fx / (z * 1000.0)
        self.scale_info = {
            'box_depth_m': z,
            'px_per_mm_from_clicks': self.px_per_mm,
            'px_per_mm_from_intrinsics': expected,
            'ratio': self.px_per_mm / expected,
        }
        return self.scale_info

    def to_dict(self):
        return {
            'end1_px': self.end1.tolist(),
            'end2_px': self.end2.tolist(),
            'length_px': self.length_px,
            'px_per_mm': self.px_per_mm,
            'half_along_px': self.half_along_px,
            'half_across_px': self.half_across_px,
            'corners_px': self.corners().tolist(),
            'scale_check': self.scale_info,
        }


def angle_from_perpendicular_deg(region, angle_rad, angle_sign):
    """0 deg = jaw axis exactly across the box; 90 deg = jaw axis along the box."""
    jaw = np.array([math.cos(angle_rad), angle_sign * math.sin(angle_rad)])
    along = min(1.0, abs(float(jaw @ region.axis)))
    return math.degrees(math.asin(along))


CAND_KEYS = [
    'region_defined', 'region_found', 'region_u_px', 'region_v_px', 'region_score',
    'x_cam_m', 'y_cam_m', 'z_cam_m', 'angle_rad', 'angle_deg', 'width_px', 'width_mm',
    'along_mm', 'across_mm', 'angle_from_perpendicular_deg', 'depth_valid_fraction',
    'pass_depth', 'pass_opening', 'pass_angle', 'frame_ok', 'frame_reason',
]


def evaluate_frame(region, quality, angle_map, width_map, crop_origin, crop_scale,
                   depth_full, cam_k, angle_sign):
    """Best candidate inside the region for one frame, and its per-frame checks."""
    c = {k: None for k in CAND_KEYS}
    c.update(region_defined=region is not None, region_found=False, pass_depth=False,
             pass_opening=False, pass_angle=False, frame_ok=False, frame_reason='')

    if region is None:
        c['frame_reason'] = 'no_region_defined'
        return c
    if any(x is None for x in (quality, angle_map, width_map, crop_origin, crop_scale)):
        c['frame_reason'] = 'model_returned_no_candidate'
        return c

    masked = np.where(region.mask(quality.shape, crop_origin, crop_scale), quality, 0.0)
    if not np.any(masked > 0):
        c['frame_reason'] = 'no_quality_in_region'
        return c

    r, col = np.unravel_index(np.argmax(masked), masked.shape)
    u = col * crop_scale + crop_origin[1]
    v = r * crop_scale + crop_origin[0]
    ui, vi = int(round(u)), int(round(v))
    a = float(angle_map[r, col])
    w_px = float(width_map[r, col]) * crop_scale
    along, across = region.local(u, v)
    c.update(
        region_found=True, region_u_px=float(u), region_v_px=float(v),
        region_score=float(masked[r, col]),
        angle_rad=a, angle_deg=math.degrees(a), width_px=w_px,
        along_mm=float(along) / region.px_per_mm, across_mm=float(across) / region.px_per_mm,
        angle_from_perpendicular_deg=angle_from_perpendicular_deg(region, a, angle_sign),
    )

    # Depth: the centre pixel and most of a small window around it must be valid.
    h, w = depth_full.shape[:2]
    k = DEPTH_WINDOW_HALF_PX
    win = depth_full[max(0, vi - k):vi + k + 1, max(0, ui - k):ui + k + 1]
    frac = float(np.mean(np.isfinite(win) & (win > 0))) if win.size else 0.0
    z = float(depth_full[vi, ui]) if (0 <= vi < h and 0 <= ui < w) else float('nan')
    c['depth_valid_fraction'] = frac
    c['pass_depth'] = bool(math.isfinite(z) and z > 0 and frac >= MIN_VALID_DEPTH_FRACTION)

    ok_min = ok_max = False
    if c['pass_depth']:
        fx, fy = float(cam_k[0][0]), float(cam_k[1][1])
        cx, cy = float(cam_k[0][2]), float(cam_k[1][2])
        c['x_cam_m'] = (ui - cx) / fx * z
        c['y_cam_m'] = (vi - cy) / fy * z
        c['z_cam_m'] = z
        c['width_mm'] = w_px * z / fx * 1000.0
        ok_min = c['width_mm'] >= REQUIRED_OPENING_MM
        ok_max = GRIPPER_MAX_OPENING_MM is None or c['width_mm'] <= GRIPPER_MAX_OPENING_MM
        c['pass_opening'] = bool(ok_min and ok_max)
    c['pass_angle'] = bool(c['angle_from_perpendicular_deg'] <= ANGLE_TOL_DEG)

    if not c['pass_depth']:
        c['frame_reason'] = 'insufficient_valid_depth'
    elif not ok_min:
        c['frame_reason'] = 'opening_too_small'
    elif not ok_max:
        c['frame_reason'] = 'opening_exceeds_gripper'
    elif not c['pass_angle']:
        c['frame_reason'] = 'jaw_axis_not_across_box'
    c['frame_ok'] = c['frame_reason'] == ''
    return c


def _spreads(window):
    pts = np.array([[c['x_cam_m'] * 1000.0, c['y_cam_m'] * 1000.0] for c in window])
    pos = float(np.max(np.linalg.norm(pts - np.median(pts, axis=0), axis=1)))
    a0 = window[0]['angle_rad']
    # grasp angles are equivalent modulo 180 deg
    d = [((c['angle_rad'] - a0 + math.pi / 2) % math.pi) - math.pi / 2 for c in window]
    ang = math.degrees(max(d) - min(d))
    return pos, ang


class StabilityTracker:
    """Requires STABLE_FRAMES consecutive passing frames that agree with each other."""

    def __init__(self):
        self.window = deque(maxlen=STABLE_FRAMES)

    def reset(self):
        self.window.clear()

    def update(self, cand):
        if not cand['frame_ok']:
            self.reset()
            return {'accepted': False, 'decision': 'no_grasp', 'reason': cand['frame_reason'],
                    'window_count': 0, 'pos_spread_mm': None, 'angle_spread_deg': None}
        self.window.append(cand)
        n = len(self.window)
        pos, ang = _spreads(self.window)
        if n < STABLE_FRAMES:
            reason = f'need_{STABLE_FRAMES}_consecutive_ok_frames_have_{n}'
        elif pos > MAX_POS_SPREAD_MM:
            reason = 'position_not_consistent'
        elif ang > MAX_ANGLE_SPREAD_DEG:
            reason = 'angle_not_consistent'
        else:
            reason = ''
        return {'accepted': reason == '', 'decision': 'accepted' if reason == '' else 'no_grasp',
                'reason': reason, 'window_count': n, 'pos_spread_mm': pos, 'angle_spread_deg': ang}
