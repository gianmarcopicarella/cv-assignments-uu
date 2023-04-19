import os
import numpy as np
import cv2
import pickle


def load_xml(filepath, tags, custom_process=lambda x: int(x.real())):
    """
    This function acts as a basic xml loading routine
    :param filepath: The xml file path
    :param tags: The list of tags to load
    :param custom_process: A custom function applied to the xml node
    :return: The dictionary containing all the tags and their values
    """
    file = cv2.FileStorage(filepath, cv2.FileStorage_READ)
    return {tag: custom_process(file.getNode(tag)) for tag in tags}


def save_xml(filepath, tags, values):
    """
    This function saves the list of key-value pairs in a file located at filepath
    :param filepath: The path of the file
    :param tags: The list of tags of type string
    :param values: The list of values of type OpenCV compliant
    """
    file = cv2.FileStorage(filepath, cv2.FileStorage_WRITE)
    for tag, value in zip(tags, values):
        file.write(tag, value)
    file.release()


def load_checkerboard_xml(filepath, tags=None):
    """
    This function loads the checkerboard xml file
    :param filepath: The checkerboard directory path
    :param tags: The list of tags to load
    :return: The dictionary containing all the tags and their values
    """
    if tags is None:
        tags = ['CheckerBoardWidth', 'CheckerBoardHeight', 'CheckerBoardSquareSize']
    return load_xml(filepath, tags)


def load_camera_xml(path):
    """
    This function loads all the camera data contained in intrinsics.xml/extrinsics.xml
    :param path: The path pointing to the camera directory
    :return: The intrinsics and extrinsics data stored in a dictionary
    """
    intrinsics_path = os.path.join(path, "intrinsics.xml")
    extrinsics_path = os.path.join(path, "extrinsics.xml")

    camera = load_xml(intrinsics_path, ['CameraMatrix', 'DistortionCoeffs'], lambda x: x.mat())
    camera.update(load_xml(extrinsics_path, ['RotationVector', 'TranslationVector'], lambda x: x.mat()))

    return camera


def load_cameras_xml(path, store_camera_position=False, scaling_factor=1.0):
    """
    This function loads all the cameras with a subfolder in the provided path sorted by alphanumerical order
    :param scaling_factor: The scaling factor used during camera world position rescaling. It should be always set to 1/calibration_square_size
    :param store_camera_position: Specify if the scaled world position of each camera should be computed
    :param path: The cameras path
    :return: A list of cameras infos sorted by alphanumerical order
    """
    data = []

    for filename in sorted(os.listdir(path)):
        potential_cam_path = os.path.join(path, filename)
        if os.path.isdir(potential_cam_path):
            data.append(load_camera_xml(potential_cam_path))

            if store_camera_position:
                cam_rot = cv2.Rodrigues(data[-1]["RotationVector"])[0]
                cam_pos = -np.matrix(cam_rot).T * np.matrix(data[-1]["TranslationVector"]) * scaling_factor
                data[-1]["RescaledWorldPosition"] = np.array([cam_pos[0][0], cam_pos[2][0], cam_pos[1][0]])

    return data


def fetch_video_frame(filepath, frame_index):
    """
    This function returns the frame at the specified index for the video located at filepath
    :param filepath: The video path
    :param frame_index: The frame index
    :return: The requested frame
    """
    video = cv2.VideoCapture(filepath)
    while True:
        success, current_frame = video.read()
        if not success:
            break
        elif frame_index == 0:
            return current_frame
        frame_index -= 1
    return None


def count_video_frames(filepath):
    """
    This function counts the number of frames in the video located at filepath
    :param filepath: The video path
    :return: The number of frames
    """
    frame_count = 0
    video = cv2.VideoCapture(filepath)
    while True:
        success, _ = video.read()
        if not success:
            break
        frame_count += 1
    return frame_count


def save_voxel_shape(filepath, voxel_flags):
    """
    This function saves the current voxel model in binary format
    :param filepath: The location where the voxel model should be saved
    :param voxel_flags: The 3D array of flags stating if a voxel is on or off
    """
    voxel_shape = np.zeros((voxel_flags.shape[0], voxel_flags.shape[1], voxel_flags.shape[2]), dtype=bool)

    for x in range(voxel_flags.shape[0]):
        for y in range(voxel_flags.shape[1]):
            for z in range(voxel_flags.shape[2]):
                if np.bitwise_and.reduce(voxel_flags[x, y, z]):
                    voxel_shape[x, y, z] = True

    pickle.dump(voxel_shape, open(filepath, "wb"))
