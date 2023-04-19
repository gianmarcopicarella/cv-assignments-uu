import pickle

import glm

import os

from modules.io import load_cameras_xml, load_checkerboard_xml
from modules.subtraction import *
from modules.utils import *
from engine.config import config

block_size = 1.0


def get_video_source(data_path, camera_number):
    return cv2.VideoCapture(os.path.join(data_path, "cam" + str(camera_number) + "/video.avi"))


def get_subtractor(data_path, camera_number):
    return train_hsv_MOG_subtractor(os.path.join(data_path, "cam" + str(camera_number) + "/background.avi"), 80, 0.95)


def load_lookup_table(filepath):
    with open(filepath, 'rb') as handle:
        lookup = pickle.load(handle)
        return lookup


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -block_size, z * block_size - depth / 2])
            colors.append([1.0, 1.0, 1.0] if (x + z) % 2 == 0 else [0, 0, 0])
    return data, colors


# Global variables
base_project_data = "../data"
camera_data = load_cameras_xml(base_project_data)
frame_numbers_count = count_video_frames(os.path.join(base_project_data, "cam1/background.avi"))
current_frame_index = -1
last_frame_masks = None
voxels_on = set()
video_frame_shape = (486, 644)

video_sources = [
    get_video_source(base_project_data, 1),
    get_video_source(base_project_data, 2),
    get_video_source(base_project_data, 3),
    get_video_source(base_project_data, 4)
]

subtractors = [
    get_subtractor(base_project_data, 1),
    get_subtractor(base_project_data, 2),
    get_subtractor(base_project_data, 3),
    get_subtractor(base_project_data, 4)
]

checkerboard_data = load_checkerboard_xml(os.path.join(base_project_data, "checkerboard.xml"))
lookup_table = load_lookup_table("../lookup")
voxel_visibility = np.zeros((config["world_width"], config["world_height"], config["world_depth"], len(camera_data)),
                            dtype=np.uint8)


def custom_cam3_mask_process(mask):
    """
    This function applies a CLOSE operation to the input mask (additional pre-processing only used for camera 3)
    :param mask: The foreground binary mask
    :return: The foreground binary mask with a CLOSE operation applied
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def set_voxel_positions(width, height, depth):
    """
    This function computes the position and color vectors for each voxel to be drawn
    :param width: The voxel volume width
    :param height: The voxel volume height
    :param depth: The voxel volume depth
    :return: The lists of voxel positions and colors
    """
    global frame_numbers_count, current_frame_index, voxel_visibility, lookup_table, last_frame_masks, voxels_on

    # Fetch the next video frame from each camera
    current_frame_index = (current_frame_index + 1) % frame_numbers_count
    frames = np.array([v.read()[1] for v in video_sources])

    # Exit from the app if any of the cameras used failed to fetch the next frame
    if None in frames:
        return [], [], True

    # Computes the foreground masks for each camera
    masks = compute_foreground_masks(frames,
                                     subtractors,
                                     [80, 80, 80, 40], False,
                                     [lambda x: x, lambda x: x, custom_cam3_mask_process, lambda x: x])

    if last_frame_masks is not None:
        # Compute the masks XOR and detect the on/off voxel pixels
        masks_xor = cv2.bitwise_xor(masks, last_frame_masks)
        masks_on = cv2.bitwise_and(masks, masks_xor)
        masks_off = cv2.bitwise_and(cv2.bitwise_not(masks), masks_xor)

        # Set voxels to off
        M = np.transpose((masks_off == 255).nonzero())
        R = [(v[0], v[1], v[2], i) for i, y, x in M if (x, y) in lookup_table[i] for v in lookup_table[i][(x, y)]]
        voxel_visibility[tuple(np.array(R).T)] = False

        # Set voxels to on
        M = np.transpose((masks_on == 255).nonzero())
        A = [(v[0], v[1], v[2], i) for i, y, x in M if (x, y) in lookup_table[i] for v in lookup_table[i][(x, y)]]
        voxel_visibility[tuple(np.array(A).T)] = True
        A = [v for v in A if np.bitwise_and.reduce(voxel_visibility[v[:3]])]

        # Update the current voxel set
        voxels_on = voxels_on.difference(set(R))
        voxels_on = voxels_on.union(set(A))
    else:
        # Compute the voxels to draw without using the XOR optimization (only for the first video frame)
        M = np.transpose((masks == 255).nonzero())
        A = [(v[0], v[1], v[2], i) for i, y, x in M if (x, y) in lookup_table[i] for v in lookup_table[i][(x, y)]]
        voxel_visibility[tuple(np.array(A).T)] = True
        A = [v for v in A if np.bitwise_and.reduce(voxel_visibility[v[:3]])]
        # Update the current voxel set
        voxels_on = voxels_on.union(set(A))

    # Set the last frame mask equal to the current
    last_frame_masks = masks

    # Ad hoc volume scaling and offsettings
    scaling_factor = 0.2
    manual_z_offset = 7
    should_close_app = False

    voxels = [(scaling_factor * (vx * block_size - width / 2), scaling_factor * (vz * block_size - manual_z_offset),
               scaling_factor * (vy * block_size - height / 2)) for vx, vy, vz, _ in
              voxels_on]

    colors = [(vx / width, vz / depth, vy / height)
              for vx, vy, vz, _ in voxels_on]

    return voxels, colors, should_close_app


def get_cam_positions():
    """
    This function computes the position and color vectors for each camera
    :return: The list of position and color vectors for all the cameras
    """
    global camera_data, checkerboard_data

    camera_positions = []
    scaling_factor = 1.0 / checkerboard_data["CheckerBoardSquareSize"]

    for data in camera_data:
        rotation = cv2.Rodrigues(data["RotationVector"])[0]
        position = -np.matrix(rotation).T * np.matrix(data["TranslationVector"]) * scaling_factor
        position = [position[0][0], position[2][0], -position[1][0]]
        camera_positions.append(position)

    return camera_positions, [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    """
    This function computes the rotation matrix for each camera
    :return: The list of rotation matrices for all the cameras
    """
    global camera_data

    rotations = []

    # Flip Y axis sign + Rotate 90 degrees around the Y-axis
    adjustment = glm.rotate(np.pi / 2.0, glm.vec3(0, 1, 0)) * \
                 glm.mat4(1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)

    for data in camera_data:
        cv_rotation = cv2.Rodrigues(data["RotationVector"])[0]

        # Apply adjustment matrix to the transposed matrix with inverted Y and Z axis
        gl_rotation = adjustment * glm.mat4(cv_rotation[0][0], cv_rotation[1][0], cv_rotation[2][0], 0,
                                            cv_rotation[0][2], cv_rotation[1][2], cv_rotation[2][2], 0,
                                            cv_rotation[0][1], cv_rotation[1][1], cv_rotation[2][1], 0,
                                            0, 0, 0, 1)
        rotations.append(gl_rotation)

    return rotations
