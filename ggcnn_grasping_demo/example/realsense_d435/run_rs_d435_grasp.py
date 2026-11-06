import os
import sys
import cv2
import time
import numpy as np
from queue import Queue

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from camera.rs_camera import RealSenseCamera
from camera.utils import get_combined_img
from ufactory_vision.ggcnn_grasping_demo.grasp.ggcnn_torch_orig_2 import TorchGGCNN
from grasp.robot_grasp import RobotGrasp

WIN_NAME = 'RealSense-D435'
CAM_WIDTH = 640
CAM_HEIGHT = 480

MODEL_FILE = os.path.join(os.path.dirname(__file__), '../../models/ggcnn_epoch_23_cornell')    # GGCNN
# MODEL_FILE = os.path.join(os.path.dirname(__file__), '../../models/epoch_50_cornell')        # GGCNN2
# use open-loop solution when robot height is over OPEN_LOOP_HEIGHT
OPEN_LOOP_HEIGHT = 340 # mm
GGCNN_IN_THREAD = False

# show the grasp image of ggcnn or not, otherwise show native depth images.
SHOW_GRASP_IMG = False

# rgb camera calibration result
EULER_EEF_TO_COLOR_OPT = [0.067052239, -0.0311387575, 0.021611456, -0.004202176, -0.00848499, 1.5898775] # xyzrpy meters_rad
# EULER_COLOR_TO_DEPTH_OPT = [0.015, 0, 0, 0, 0, 0]
EULER_COLOR_TO_DEPTH_OPT = [0, 0, 0, 0, 0, 0]

# The range of motion of the robot grasping
# If it exceeds the range, it will return to the initial detection position.
GRASPING_RANGE = [180, 600, -300, 300] # [x_min, x_max, y_min, y_max]

# initial detection position
DETECT_XYZ = [300, 0, 400] # [x, y, z]

# release grasping pos
RELEASE_XYZ = [400, 400, 270]

# lift offset based on DETECT_XYZ[2] after grasping or release
LIFT_OFFSET_Z = 100 # lift_height = DETECT_XYZ[2] + LIFT_OFFSET_Z

# The distance between the gripping point of the robot grasping and the end of the robot arm flange
# The value needs to be fine-tuned according to the actual situation.
GRIPPER_Z_MM = 150 # mm

# minimum z for grasping
GRASPING_MIN_Z = 175 # mm

# DEPTH_CAM_K = np.array([
#     [fx, 0, cx],
#     [0, fy, cy],
#     [0, 0, 1]
# ])


def main():
    if len(sys.argv) < 2:
        print('Usage: {} {{robot_ip}}'.format(sys.argv[0]))
        exit(1)
    
    robot_ip = sys.argv[1]

    depth_img_que = Queue(1)
    ggcnn_cmd_que = Queue(1)
    
    camera = RealSenseCamera(width=CAM_WIDTH, height=CAM_HEIGHT)
    color_intrin, depth_intrin = camera.get_intrinsics()
    # depth is aligned to color, use color intrinsics
    DEPTH_CAM_K = np.array([
        [color_intrin.fx, 0, color_intrin.ppx],
        [0, color_intrin.fy, color_intrin.ppy],
        [0, 0, 1]
    ])
    ggcnn_config = {
        'MODEL_FILE': MODEL_FILE,
        'OPEN_LOOP_HEIGHT': OPEN_LOOP_HEIGHT,
        'GGCNN_IN_THREAD': GGCNN_IN_THREAD,
        'DEPTH_CAM_K': DEPTH_CAM_K,
    }
    ggcnn = TorchGGCNN(ggcnn_config, depth_img_que, ggcnn_cmd_que)
    time.sleep(2)
    euler_opt = {
        'EULER_EEF_TO_COLOR_OPT': EULER_EEF_TO_COLOR_OPT,
        'EULER_COLOR_TO_DEPTH_OPT': EULER_COLOR_TO_DEPTH_OPT,
    }
    grasp_config = {
        'GRASPING_RANGE': GRASPING_RANGE,
        'DETECT_XYZ': DETECT_XYZ,
        'RELEASE_XYZ': RELEASE_XYZ,
        'LIFT_OFFSET_Z': LIFT_OFFSET_Z,
        'GRIPPER_Z_MM': GRIPPER_Z_MM,
        'GRASPING_MIN_Z': GRASPING_MIN_Z,
        'MIN_RESULT_Z_MM': 250,
    }
    grasp = RobotGrasp(robot_ip, ggcnn_cmd_que, euler_opt, grasp_config)

    crop_size = 300
    crop_y_offset = 0

    crop_y_inx = -1
    crop_x_inx = -1

    while grasp.is_alive():
        color_image, depth_image = camera.get_images(align=True)

        if crop_y_inx < 0:
            imh, imw = depth_image.shape
            crop_size = min(imh, imw)
            crop_y_inx = max(0, imh - crop_size) // 2 - crop_y_offset   # crop height(y) start index
            crop_x_inx = max(0, imw - crop_size) // 2                   # crop width(x) start index

        color_image = color_image[crop_y_inx:crop_y_inx + crop_size, crop_x_inx:crop_x_inx + crop_size, :]

        robot_pos = grasp.get_eef_pose_m()
        if GGCNN_IN_THREAD:
            if not depth_img_que.empty():
                depth_img_que.get()
            depth_img_que.put([robot_pos, depth_image])
            grasp_img = ggcnn.grasp_img
            if not SHOW_GRASP_IMG or grasp_img is None:
                grasp_img = depth_image[crop_y_inx:crop_y_inx + crop_size, crop_x_inx:crop_x_inx + crop_size]

            combined_img = get_combined_img(color_image, ggcnn.grasp_img)
            cv2.imshow(WIN_NAME, combined_img)
        else:
            grasp_img, result = ggcnn.get_grasp_img(depth_image, DEPTH_CAM_K, robot_pos[2])
            if result:
                if not ggcnn_cmd_que.empty():
                    ggcnn_cmd_que.get()
                ggcnn_cmd_que.put([robot_pos, result])

            if not SHOW_GRASP_IMG or grasp_img is None:
                grasp_img = depth_image[crop_y_inx:crop_y_inx + crop_size, crop_x_inx:crop_x_inx + crop_size]

            combined_img = get_combined_img(color_image, grasp_img)
            cv2.imshow(WIN_NAME, combined_img)
        
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            camera.stop()
            break

if __name__ == '__main__':
    main()
