import os
import argparse
from modules.io import load_checkerboard_xml, fetch_video_frame, save_xml
from modules.subtraction import train_KNN_background_subtractor
from modules.utils import *
import copy

# Global variables
checkerboard_corners = set()
checkerboard_corners_sort = []
current_image = None


def sample_valid_video_frames(filepath, pattern_shape, sampling_step):
    """
    This function samples a list of frames from the video located at the specified filepath. A frame is considered
    valid only if cv2.findChessboardCorners is able to detect the chessboard corners.
    :param filepath: The video filepath
    :param pattern_shape: The number of squares along each side of the checkerboard (Ysize, Xsize)
    :param sampling_step: The number of samples to skip before a new frame is processed
    :return: The list of selected samples and a tuple containing the height and width of the video frame
    """
    video = cv2.VideoCapture(filepath)
    samples = []
    local_sample_distance = 1

    while True:
        success, frame = video.read()
        local_sample_distance = (local_sample_distance - 1) % sampling_step

        if not success:
            break
        elif local_sample_distance == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, coords = cv2.findChessboardCorners(gray, pattern_shape, flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                                                 cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                                                 cv2.CALIB_CB_FAST_CHECK +
                                                                                 cv2.CALIB_CB_FILTER_QUADS)
            if found:
                samples.append(coords)

    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return samples, (video_height, video_width)


def estimate_camera_intrinsics(image_points, image_shape, checkerboard_pattern, square_size):
    """
    This function returns the re-projection error, camera matrix and standard deviations using calibrateCameraExtended
    :param image_points: a list containing the 2D corner points for each training image
    :param image_shape: the shape of the image
    :param checkerboard_pattern: the shape of the chessboard pattern
    :param square_size: the size of each square
    """

    default_object_points = np.zeros((checkerboard_pattern[0] * checkerboard_pattern[1], 3), dtype=np.float32)
    default_object_points[:, :2] = np.mgrid[0:checkerboard_pattern[0], 0:checkerboard_pattern[1]].T.reshape(-1, 2) \
                                   * square_size

    object_points = [default_object_points] * len(image_points)

    re_err, matrix, dist, _, _, std_in, _, _ = \
        cv2.calibrateCameraExtended(object_points, image_points, image_shape, None, None)

    return re_err, matrix, dist, std_in[0:4]


def corners_selection_callback(event, x, y, flags, params):
    """
    This function is used as a callback during the corners selection phase. When the user presses the left mouse
    button, a new corner point is added. When the user presses the right mouse button, the nearest corner point is
    removed. When the user is done it's sufficient to press a keyboard button to go to the next step.
    :param event: The window event
    :param x: The mouse x location
    :param y: The mouse y location
    :param flags: Not used
    :param params: Not used
    """
    global checkerboard_corners, current_image

    # Local UI update flag
    update = False

    if event == cv2.EVENT_LBUTTONDOWN and len(checkerboard_corners) < 4:
        checkerboard_corners.add((x, y))
        update = True
    elif event == cv2.EVENT_RBUTTONDOWN and len(checkerboard_corners) > 0:
        nearest_point = min(checkerboard_corners,
                            key=lambda p: np.linalg.norm(np.array([x, y]) - np.array([p[0], p[1]])))
        checkerboard_corners.remove(nearest_point)
        update = True

    if update:
        show_image = copy.deepcopy(current_image)
        for point in checkerboard_corners:
            cv2.circle(show_image, point, 2, (255, 0, 0), -1)
        cv2.imshow("Corner selection phase", show_image)


def corners_sorting_callback(event, x, y, flags, params):
    """
    This function is used as a callback during the corners sorting phase. When the user presses the left mouse
    button, the nearest non-selected point is marked as the next point in the ordering. When the user presses the
    right mouse button, the nearest selected point is removed from the ordering.
    :param event: The window event
    :param x: The mouse x location
    :param y: The mouse y location
    :param flags: Not used
    :param params: Not used
    """
    global checkerboard_corners, checkerboard_corners_sort, current_image

    # Local UI update flag
    update = False

    if event == cv2.EVENT_RBUTTONDOWN and len(checkerboard_corners_sort) > 0:
        nearest_point = min(checkerboard_corners_sort,
                            key=lambda p: np.linalg.norm(np.array([x, y]) - np.array([p[0], p[1]])))
        checkerboard_corners_sort.pop(checkerboard_corners_sort.index(nearest_point))
        update = True
    elif event == cv2.EVENT_LBUTTONDOWN and len(checkerboard_corners_sort) < 4:
        nearest_point = min(list(filter(lambda p: p not in checkerboard_corners_sort, checkerboard_corners)),
                            key=lambda p: np.linalg.norm(np.array([x, y]) - np.array([p[0], p[1]])))
        checkerboard_corners_sort.append(nearest_point)
        update = True

    if update:
        show_image = copy.deepcopy(current_image)
        for point in checkerboard_corners:
            cv2.circle(show_image, point, 2, (255, 0, 0), -1)
            if point in checkerboard_corners_sort:
                index = checkerboard_corners_sort.index(point)
                cv2.putText(show_image, "P" + str(index + 1), point, cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2,
                            cv2.LINE_AA)
        cv2.imshow("Corner selection phase", show_image)


def find_checkerboard_corners(checkerboard_path, background_path, pattern_shape):
    """
    This function uses an automated logic to approximate the location of the 4 outer checkerboard corners and computes
    all the intermediate corners. In case no detection occurred or the quality of the detection is not sufficiently
    accurate, the user can remove (with a right click to the nearest point) or add (with a left click) a point.
    :param checkerboard_path: The checkerboard video path
    :param background_path: The background video path
    :param pattern_shape: The number of squares along each side of the checkerboard (Ysize, Xsize)
    :return: The estimated corner points
    """
    global checkerboard_corners, current_image

    # fetch first video frame
    first_frame = fetch_video_frame(checkerboard_path, 0)
    knn_subtractor = train_KNN_background_subtractor(background_path)

    # find_checkerboard_polygon with frame
    polygon_points = find_checkerboard_polygon(first_frame, knn_subtractor)
    checkerboard_corners = {(point[0], point[1]) for point in polygon_points}

    # draw polygon points
    current_image = copy.deepcopy(first_frame)
    show_image = copy.deepcopy(first_frame)

    for point in checkerboard_corners:
        cv2.circle(show_image, point, 2, (255, 0, 0), -1)

    cv2.imshow("Corner selection phase", show_image)
    cv2.setWindowTitle("Corner selection phase", "Checkerboard Corners Selection")
    cv2.setMouseCallback("Corner selection phase", corners_selection_callback)

    while True:
        cv2.waitKey(0)
        if len(checkerboard_corners) == 4:
            break

    cv2.setWindowTitle("Corner selection phase", "Checkerboard Corners Sorting")
    cv2.setMouseCallback("Corner selection phase", corners_sorting_callback)

    while True:
        cv2.waitKey(0)
        if len(checkerboard_corners_sort) == 4:
            cv2.setMouseCallback("Corner selection phase", lambda *args: None)
            break

    # compute internal corners
    corners = np.array([np.array([x, y]) for x, y in checkerboard_corners_sort])
    corners = interpolate_and_project_corners(corners, pattern_shape, True)
    corners = np.array([[p[0] for p in corners], [p[1] for p in corners]]) \
        .transpose().astype(np.float32)

    # Show corners and return
    cv2.drawChessboardCorners(current_image, pattern_shape, corners, True)

    cv2.setWindowTitle("Corner selection phase", "Checkerboard Corners")
    cv2.imshow("Corner selection phase", current_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return np.array(corners)


def estimate_camera_extrinsics(corners, camera_matrix, square_size, pattern_shape):
    """
    This function estimates the camera extrinsics using the openCV's solvePnP function
    :param corners: The list of detected corners
    :param camera_matrix: The camera matrix
    :param square_size: The size in mm of each square
    :param pattern_shape: The number of squares along each side of the checkerboard (Ysize, Xsize)
    :return: The estimated translation, rotation vectors and the re-projection error
    """
    default_object_points = np.zeros((pattern_shape[0] * pattern_shape[1], 3), dtype=np.float32)
    default_object_points[:, :2] = np.mgrid[0:pattern_shape[0], 0:pattern_shape[1]].T.reshape(-1, 2) \
                                   * square_size
    return cv2.solvePnP(default_object_points, corners, camera_matrix, None)


def log_camera_intrinsics_confidence(mtx, ret, std, calibration_name="", rounding=3):
    """
    This function logs the estimated value and standard deviation for each intrinsic camera parameter
    :param mtx: camera matrix
    :param ret: re-projection error
    :param std: standard deviation for each estimated intrinsic parameter
    :param calibration_name: calibration name
    :param rounding: number of decimal figures considered while rounding
    """
    title = "Confidence Of Estimated Camera Parameters" if calibration_name == "" \
        else "[" + calibration_name + "] Confidence Of Estimated Camera Parameters"

    print(title)
    print("Overall RMS Re-Projection Error", round(ret, 3))
    print("Focal Length (Fx)", round(mtx[0][0], rounding), "\tSTD +/-", round(std[0][0], rounding))
    print("Focal Length (Fy)", round(mtx[1][1], rounding), "\tSTD +/-", round(std[1][0], rounding))
    print("Camera Center (Cx)", round(mtx[0][2], rounding), "\tSTD +/-", round(std[2][0], rounding))
    print("Camera Center (Cy)", round(mtx[1][2], rounding), "\tSTD +/-", round(std[3][0], rounding), "\n")


def cli_arguments_parse():
    """
    This function defines a CLI, parses and checks the CLI arguments and returns them. There are 2 required argument:
    checkerboard_settings_path and camera_path. If the parsing fails or one of the parameters is not well-formed then
    the entire script is aborted and a CLI message displayed.
    """
    parser = argparse.ArgumentParser(description="Camera Calibration")
    parser._action_groups.pop()

    required = parser.add_argument_group('Required arguments')
    optional = parser.add_argument_group('Optional arguments')

    required.add_argument("-l", "--checkerboard_settings_path", type=str, help='Board settings XML path', required=True)
    required.add_argument("-c", "--camera_path", type=str, help='Camera folder path', required=True)
    optional.add_argument("-s", "--sampling_step", type=int, help='Training frames sampling step', default=50)

    args = parser.parse_args()

    if not os.path.isfile(args.checkerboard_settings_path):
        parser.error("The checkerboard XML settings filepath is not well-formed or doesn't exist!")

    if not args.checkerboard_settings_path.lower().endswith('.xml'):
        parser.error("The checkerboard XML settings filepath doesn't lead to an XML file!")

    if not os.path.isdir(args.camera_path):
        parser.error("The camera folder path is not well-formed or doesn't exist!")

    return args


if __name__ == '__main__':
    # CLI arguments parsing and error detection stage
    args = cli_arguments_parse()

    # Retrieve checkerboard data
    checkerboard_data = load_checkerboard_xml(args.checkerboard_settings_path)

    # Sample the training video for the specified camera
    intrinsic_video_path = os.path.join(args.camera_path, "intrinsics.avi")
    checkerboard_pattern_shape = (checkerboard_data["CheckerBoardHeight"], checkerboard_data["CheckerBoardWidth"])

    calibration_images_points, images_shape = sample_valid_video_frames(intrinsic_video_path,
                                                                        checkerboard_pattern_shape, args.sampling_step)

    # Log number of frames used for calibration
    print("Using", len(calibration_images_points), "frames for intrinsics calibration!")

    # Calibrate the specified camera
    re_err_i, matrix, dist, std_in, = estimate_camera_intrinsics(calibration_images_points,
                                                                 images_shape,
                                                                 checkerboard_pattern_shape,
                                                                 checkerboard_data["CheckerBoardSquareSize"])

    # Log the estimated intrinsics parameters
    log_camera_intrinsics_confidence(matrix, re_err_i, std_in, "Camera Calibration")

    # Save the estimated intrinsics to args.camera_path/intrinsics.xml
    intrinsics_xml_path = os.path.join(args.camera_path, "intrinsics.xml")
    save_xml(intrinsics_xml_path,
             ["CameraMatrix", "DistortionCoeffs"],
             [matrix, dist])

    # Calibrate the specified camera extrinsics
    extrinsic_video_path = os.path.join(args.camera_path, "checkerboard.avi")
    background_video_path = os.path.join(args.camera_path, "background.avi")

    checkerboard_corners = find_checkerboard_corners(extrinsic_video_path,
                                                     background_video_path,
                                                     checkerboard_pattern_shape)

    re_err_e, r_vecs, t_vecs = estimate_camera_extrinsics(checkerboard_corners,
                                                          matrix,
                                                          checkerboard_data["CheckerBoardSquareSize"],
                                                          checkerboard_pattern_shape)

    # Saving the estimated extrinsics to args.camera_path/extrinsics.xml
    extrinsics_xml_path = os.path.join(args.camera_path, "extrinsics.xml")
    save_xml(extrinsics_xml_path,
             ["RotationVector", "TranslationVector"],
             [r_vecs, t_vecs])

    # Final calibration test
    calibration_test_frame = fetch_video_frame(extrinsic_video_path, 0)
    cv2.drawFrameAxes(calibration_test_frame, matrix, None, r_vecs, t_vecs,
                      checkerboard_data["CheckerBoardSquareSize"] * 4, thickness=2)

    # Plotting the calibration frame
    cv2.imshow("Calibration Frame Test", calibration_test_frame)

    # Saving the calibration frame to args.camera_path/calibration_test_frame.jpg
    calibration_test_frame_path = os.path.join(args.camera_path, "calibration_test_frame.jpg")
    cv2.imwrite(calibration_test_frame_path, calibration_test_frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit(0)
