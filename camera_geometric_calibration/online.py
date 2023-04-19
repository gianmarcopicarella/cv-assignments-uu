import copy
import os
import cv2
import numpy as np
import argparse
from warnings import warn


def cli_arguments_parse():
    """
    This function defines a CLI, parses and checks the CLI arguments and returns them.
    There is only 1 required argument: square_size.
    If the parsing fails or one of the parameters is not well-formed then the entire script is aborted and a CLI message displayed.
    """
    parser = argparse.ArgumentParser(description="Camera Calibration")
    parser._action_groups.pop()

    required = parser.add_argument_group('Required arguments')
    optional = parser.add_argument_group('Optional arguments')

    required.add_argument("-p", "--pattern_shape", type=int, nargs=2, help='Pattern shape (Px, Py)', required=True)
    required.add_argument("-s", "--square_size", type=int, help='Square size (mm)', required=True)

    required_group = required.add_mutually_exclusive_group(required=True)
    required_group.add_argument("-t", "--test_image_path", type=str, help='Test image path')
    required_group.add_argument("-w", "--use_webcam", help='Use real-time webcam video', action='store_true')

    optional.add_argument("-c", "--cam_params_dir_path", type=str, help="Training images folder path",
                          default=os.getcwd())

    args = parser.parse_args()

    if not os.path.isdir(args.cam_params_dir_path):
        parser.error("The camera parameters directory path is not well-formed!")

    if not args.use_webcam and not os.path.isfile(args.test_image_path):
        parser.error("The test image path is not well-formed!")

    return args


def load_camera_parameters(dir_path):
    """
    This function loads all the estimated camera parameters and sort them by calibration run
    :param dir_path: loading directory
    :return: A list containing the estimated camera parameters for each calibration run
    """
    return list(sorted([
        np.load(os.path.join(dir_path, fn))
        for fn in os.listdir(dir_path)
        if fn.endswith(".npz")
    ], key=lambda x: x["run"]))


def draw_cube(current_image, cam_matrix, r_vec, t_vec, color, side_length):
    """
    This function draws a cube at the (0, 0, 0) world origin
    :param current_image: the current image
    :param cam_matrix: the camera matrix
    :param r_vec: the rotation vector
    :param t_vec: the translation vector
    :param color: the cube color
    :param side_length: the length of each cube side
    """
    cube = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                       [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]]) * side_length

    cube_pixels, _ = cv2.projectPoints(cube.astype(np.float32), r_vec, t_vec, cam_matrix, None)
    cube_pixels = cube_pixels.astype(np.int32).reshape(-1, 2)

    cv2.drawContours(current_image, [cube_pixels[0:4]], -1, color, 2, lineType=cv2.LINE_AA)

    for i, j in zip(range(4), range(4, 8)):
        cv2.line(current_image, tuple(cube_pixels[i]), tuple(cube_pixels[j]), color, 2, lineType=cv2.LINE_AA)

    cv2.drawContours(current_image, [cube_pixels[4:]], -1, color, 2, lineType=cv2.LINE_AA)


def draw_frame_axis(current_image, cam_matrix, r_vec, t_vec, origin, axis_length):
    """
    This function draws the 3D axis frame centered at the world origin
    :param current_image: the current image
    :param cam_matrix: the camera matrix
    :param r_vec: the rotation vector
    :param t_vec: the translation vector
    :param origin: the world origin
    :param axis_length: the length of each axis
    """
    axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]) * axis_length

    axis_pixels, _ = cv2.projectPoints(axis.astype(np.float32), r_vec, t_vec, cam_matrix, None)
    axis_pixels = axis_pixels.astype(np.int32)

    origin = origin.astype(np.int32)
    cv2.arrowedLine(current_image, origin, axis_pixels[0][0], (255, 0, 0), 3, line_type=cv2.LINE_AA)
    cv2.arrowedLine(current_image, origin, axis_pixels[1][0], (0, 255, 0), 3, line_type=cv2.LINE_AA)
    cv2.arrowedLine(current_image, origin, axis_pixels[2][0], (0, 0, 255), 3, line_type=cv2.LINE_AA)


def process_frame(frame, corners, calibration, pattern_shape):
    global default_object_points

    ret, r_vec, t_vec = cv2.solvePnP(default_object_points, corners, calibration["mtx"], None)

    cv2.drawChessboardCorners(frame, pattern_shape, corners, found)
    draw_frame_axis(frame, calibration["mtx"], r_vec, t_vec, corners[0][0], 60)
    draw_cube(frame, calibration["mtx"], r_vec, t_vec, (255, 255, 0), 40)


if __name__ == "__main__":
    # CLI arguments parsing and error detection stage
    args = cli_arguments_parse()

    # Loading calibrations parameters
    calibrations = load_camera_parameters(args.cam_params_dir_path)

    # Defining object points
    default_object_points = np.zeros((args.pattern_shape[0] * args.pattern_shape[1], 3), dtype=np.float32)
    default_object_points[:, :2] = np.mgrid[0:args.pattern_shape[0], 0:args.pattern_shape[1]].T.reshape(-1,
                                                                                                        2) * args.square_size

    # If a test image path is specified then use it
    if args.test_image_path is not None:
        test_image = cv2.imread(args.test_image_path)
        test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        found, coords = cv2.findChessboardCorners(test_gray, args.pattern_shape,
                                                  flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                        cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                        cv2.CALIB_CB_FAST_CHECK +
                                                        cv2.CALIB_CB_FILTER_QUADS)

        # If no corner is found then print a warning and abort the execution
        if not found:
            warn("[CORNERS_DET_FAIL]: Couldn't detect chessboard corners automatically, aborting the script now!")
            exit(0)

        # Otherwise open a window
        cv2.namedWindow("Test Image")
        cv2.setWindowTitle("Test Image", "Calibration N.1")

        calibration_index = 0

        while True:
            current_image = copy.deepcopy(test_image)
            process_frame(current_image, coords, calibrations[calibration_index], args.pattern_shape)

            cv2.imshow("Test Image", current_image)

            key = cv2.waitKey(0)

            # If the user presses d then it shows the result with the calibration parameters estimated in the next run
            if key == ord('d'):
                calibration_index = (calibration_index + 1) % len(calibrations)
                cv2.setWindowTitle("Test Image", "Calibration N." + str(calibration_index + 1))
            # If the user presses a then it shows the result with the calibration parameters estimated in the
            # previous run
            elif key == ord('a'):
                calibration_index = (calibration_index - 1) % len(calibrations)
                cv2.setWindowTitle("Test Image", "Calibration N." + str(calibration_index + 1))
            # If the user presses q then the script ends
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()

    elif args.use_webcam:
        src_video_flow = cv2.VideoCapture(0)

        # If it's not open then trying to open it manually may work
        if not src_video_flow.isOpened():
            src_video_flow.open()

        # If the video flow is still closed then raise an exception
        if not src_video_flow.isOpened():
           raise Exception("[CAM_STREAM]: Cannot open the camera stream!")

        calibration_index = 0

        cv2.namedWindow("RealTime Webcam Video")
        cv2.setWindowTitle("RealTime Webcam Video", "Calibration N.1")

        while True:
            ret, current_frame = src_video_flow.read()

            if not ret:
                raise Exception("[CAM_RECV_NEXT_FRAME]: Cannot read the next video frame!")

            key = cv2.waitKey(1)

            # If the user presses d then it shows the result with the calibration parameters estimated in the next run
            if key == ord('d'):
                calibration_index = (calibration_index + 1) % len(calibrations)
                cv2.setWindowTitle("RealTime Webcam Video", "Calibration N." + str(calibration_index + 1))
            # If the user presses a then it shows the result with the calibration parameters estimated in the
            # previous run
            elif key == ord('a'):
                calibration_index = (calibration_index - 1) % len(calibrations)
                cv2.setWindowTitle("RealTime Webcam Video", "Calibration N." + str(calibration_index + 1))
            # If the user presses q then the script ends
            elif key == ord('q'):
                break

            # Compute a grayscale version of the current frame and try to detect the chessboard corners automatically
            frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

            found, coords = cv2.findChessboardCorners(frame_gray, args.pattern_shape,
                                                      flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                            cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                            cv2.CALIB_CB_FAST_CHECK +
                                                            cv2.CALIB_CB_FILTER_QUADS)

            if found:
                process_frame(current_frame, coords, calibrations[calibration_index], args.pattern_shape)

            cv2.imshow("RealTime Webcam Video", current_frame)

        cv2.destroyAllWindows()
        src_video_flow.release()

    exit(0)
