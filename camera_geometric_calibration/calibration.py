import cv2
import os
import numpy as np
import argparse
from random import sample
from warnings import warn
from copy import deepcopy
import matplotlib.pyplot as plt

# Defined some global variables
checkerboard_edges = np.empty(shape=(4, 2), dtype=np.int32)
checkerboard_edges_count = 0
current_image = None
current_image_cache = []


def manual_polygon_selection_callback(event, x, y, flags, params):
    """
    This function is a callback invoked by the opencv window every time an event is detected.
    It draws a 2D polygon based on the user specified 2D points.
    The user can left-click to select a point and right click to cancel the last added point.
    Exactly 4 points in clockwise order (TL, TR, BR, BL) have to be specified by the user.
    :param event: type of event
    :param x: click x coordinate
    :param y: click y coordinate
    :param flags: used by opencv, not needed
    :param params: used by opencv, not needed
    """
    global checkerboard_edges, checkerboard_edges_count, current_image, current_image_cache

    if event == cv2.EVENT_LBUTTONDOWN and checkerboard_edges_count < 4:
        checkerboard_edges[checkerboard_edges_count] = [x, y]
        checkerboard_edges_count += 1

        current_image_cache.append(deepcopy(current_image))

        point_text_label = "P" + str(checkerboard_edges_count)

        cv2.putText(current_image, point_text_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)

        cv2.circle(current_image, checkerboard_edges[checkerboard_edges_count - 1], 3, (255, 0, 0), -1)

        if checkerboard_edges_count > 1:
            cv2.line(current_image, checkerboard_edges[checkerboard_edges_count - 2],
                     checkerboard_edges[checkerboard_edges_count - 1], (255, 0, 0), 2, cv2.LINE_AA)

        if checkerboard_edges_count == 4:
            cv2.line(current_image, checkerboard_edges[0],
                     checkerboard_edges[checkerboard_edges_count - 1], (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Manual Corners Selection", current_image)

    elif event == cv2.EVENT_RBUTTONDOWN and checkerboard_edges_count > 0:
        checkerboard_edges_count -= 1
        current_image = current_image_cache[checkerboard_edges_count]

        cv2.imshow("Manual Corners Selection", current_image)


def cli_arguments_parse():
    """
    This function defines a CLI, parses and checks the CLI arguments and returns them.
    There are only 2 required arguments: pattern_shape and square_size.
    If the parsing fails or one of the parameters is not well-formed then the entire script is aborted and a CLI message displayed.
    """
    parser = argparse.ArgumentParser(description="Camera Calibration")
    parser._action_groups.pop()

    required = parser.add_argument_group('Required arguments')
    optional = parser.add_argument_group('Optional arguments')

    required.add_argument("-p", "--pattern_shape", type=int, help='Pattern shape (YDim, XDim)', nargs=2, required=True)
    required.add_argument("-s", "--square_size", type=int, help='Square size (mm)', required=True)

    optional.add_argument("-t", "--train_dir_path", type=str, help="Training images folder path", default=os.getcwd())
    optional.add_argument("-o", '--ser_dir_path', type=str, help='Estimated camera parameters serialization '
                                                                 'path', default=os.getcwd())
    optional.add_argument("-c", "--plot_dir_path", type=str, help='Estimated camera parameters comparison plot path',
                          default=os.getcwd())

    args = parser.parse_args()

    if not os.path.isdir(args.train_dir_path):
        parser.error("The training directory path is not well-formed!")

    if not os.path.isdir(args.ser_dir_path):
        parser.error("The serialization directory path is not well-formed!")

    if not os.path.isdir(args.plot_dir_path):
        parser.error("The plot directory path is not well-formed!")

    return args


def serialize_estimated_camera_params(file, ret, mtx, dist, rvs, tvs):
    """
    This function serializes the estimated camera intrinsics and extrinsic parameters to disk
    :param file: file used for serialization
    :param ret: re-projection error
    :param mtx: camera matrix
    :param dist: distortion coefficients
    :param rvs: rotation vectors
    :param tvs: translation vectors
    """
    np.save(file, ret)
    np.save(file, mtx)
    np.save(file, dist)
    np.save(file, rvs)
    np.save(file, tvs)


def plot_intrinsics_comparisons(mtxs, stds, out_dir_path):
    """
    This function plots a series of graphs containing the intrinsic parameter's estimated value and standard
    deviation for each calibration run
    :param mtxs: camera matrices list of N elements
    :param stds: standard deviations list of N elements
    :param out_dir_path: plot out directory path
    :return: True if the plot was successful false otherwise
    """
    if len(mtxs) != len(stds):
        warn("[PLOT_INTRINSICS]: No plot will be generated because the number of camera intrinsics and standard "
             "deviations must be the same!")
        return False

    # Create some lists containing the intrinsic parameters
    x_focal_lengths = np.array([cm[0][0] for cm in mtxs])
    y_focal_lengths = np.array([cm[1][1] for cm in mtxs])
    x_center = np.array([cm[0][2] for cm in mtxs])
    y_center = np.array([cm[1][2] for cm in mtxs])

    # The x-axis value will correspond to the calibration index
    x_axis = np.array(["Calibration " + str(i) for i in range(len(mtxs))])

    # Create some lists containing the standard deviations for each intrinsic parameter
    x_focal_lengths_stds = np.array([std[0][0] for std in stds])
    y_focal_lengths_stds = np.array([std[1][0] for std in stds])
    x_centers_stds = np.array([std[2][0] for std in stds])
    y_centers_stds = np.array([std[3][0] for std in stds])

    # Create a plot with 4 rows and one column
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(15, 15))

    # Plot the results
    ax[0].set_title("Focal Length (Fx)")
    ax[0].errorbar(x_axis, x_focal_lengths, x_focal_lengths_stds, linestyle='None', marker='x')

    ax[1].set_title("Focal Length (Fy)")
    ax[1].errorbar(x_axis, y_focal_lengths, y_focal_lengths_stds, linestyle='None', marker='x')

    ax[2].set_title("Center Point (Cx)")
    ax[2].errorbar(x_axis, x_center, x_centers_stds, linestyle='None', marker='x')

    ax[3].set_title("Center Point (Cy)")
    ax[3].errorbar(x_axis, y_center, y_centers_stds, linestyle='None', marker='x')

    # Save the plot in a .pdf file
    plt.savefig(os.path.join(out_dir_path, "intrinsic_params_runs_comparison.pdf"))

    return True


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


def interpolate_and_project_corners(edges, pattern_shape):
    """
    This function computes and returns the interpolated 2D corner points using 4 manually selected board corners.
    :param edges: a list containing 4 manually selected board corners sorted in clockwise order (TP, TR, BR, BL)
    :param pattern_shape: the shape of the checkerboard
    :return: the list of interpolated 2D checkerboard corners
    """
    if len(edges) != 4:
        raise ValueError("[PROJ_CORNERS]: Edges should always contain 4 different points!")

    if len(pattern_shape) != 2:
        raise ValueError("[PROJ_CORNERS]: Pattern shape should always contain 2 dimensions!")

    # Find the max width and height of the manually selected polygon
    max_width = max(np.linalg.norm(edges[0] - edges[1]),
                    np.linalg.norm(edges[3] - edges[2]))
    max_height = max(np.linalg.norm(edges[1] - edges[2]),
                     np.linalg.norm(edges[3] - edges[0]))

    # Define the mapping coordinates for perspective transform
    output_points = np.float32([[0, 0],
                                [max_width - 1, 0],
                                [max_width - 1, max_height - 1],
                                [0, max_height - 1]])

    # Compute the inverse perspective transform
    p_matrix = cv2.getPerspectiveTransform(edges.astype(np.float32), output_points)
    inv_p_matrix = np.linalg.inv(p_matrix)

    # Compute the horizontal and vertical step
    w_step = max_width / (pattern_shape[1] - 1)
    h_step = max_height / (pattern_shape[0] - 1)

    projected_corners = []

    # Compute each projected point
    for y in range(0, pattern_shape[0]):
        for x in range(0, pattern_shape[1]):
            point = np.array([x * w_step, y * h_step, 1])
            point = np.matmul(inv_p_matrix, point)
            # Divide each point by its Z component
            point *= (1.0 / point[2])
            # Append only the first 2 elements of each point
            projected_corners.append(point[0:2])

    return projected_corners


def estimate_camera_params(images_info, image_shape, pattern_shape, square_size):
    """
    This function estimates the camera parameters using calibrateCameraExtended
    :param images_info: a list containing (found, 2D Points) for each training image
    :param image_shape: the shape of the image
    :param pattern_shape: the shape of the chessboard pattern
    :param square_size: the size of each square
    """

    default_object_points = np.zeros((pattern_shape[0] * pattern_shape[1], 3), dtype=np.float32)
    default_object_points[:, :2] = np.mgrid[0:pattern_shape[0], 0:pattern_shape[1]].T.reshape(-1, 2) \
                                   * square_size

    image_points = list(map(lambda info: info[1], images_info))
    object_points = [default_object_points for _ in range(len(image_points))]

    return cv2.calibrateCameraExtended(object_points, image_points, image_shape, None, None)


def check_images_have_same_shape(train_dir_path):
    """
    This function checks if every readable image in the training directory has the same shape
    :param train_dir_path: directory path containing all the training images
    :return: the common shape if all the readable images share the same shape, None otherwise
    """
    shapes = []

    for filename in os.listdir(train_dir_path):
        file_path = os.path.join(train_dir_path, filename)
        image = cv2.imread(file_path)
        if image is None:
            continue
        shapes.append(image.shape)

    return shapes[0] if len(set(shapes)) == 1 else None


def calibration(args):
    """
    This function computes the chessboard corners for each training picture and calibrate the camera in 3 different runs
    Additionally it logs and plots the calibration intrinsic parameters fidelity stats for each calibration run
    :param args: CLI arguments
    """
    global current_image, checkerboard_edges, checkerboard_edges_count

    # Check if all the images in the calibration directory share the same shape
    image_shape = check_images_have_same_shape(args.train_dir_path)[:-1]

    # If not then raise an exception
    if image_shape is None:
        raise ValueError("[IMAGE_SHAPES_MISMATCH]: All the training images used for calibration should share at least "
                         "the same shape!")

    images_info = []

    for filename in os.listdir(args.train_dir_path):
        file_path = os.path.join(args.train_dir_path, filename)
        current_image = cv2.imread(file_path)

        if current_image is None:
            warn("[IMREAD_FAIL]: Cannot load " + filename + ". It won't be considered during camera calibration!")
            continue

        # Convert the picture to grayscale and try to detect the chessboard corners automatically
        gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        found, coords = cv2.findChessboardCorners(gray, args.pattern_shape, flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                                                  cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                                                  cv2.CALIB_CB_FAST_CHECK +
                                                                                  cv2.CALIB_CB_FILTER_QUADS)

        # If cv2.findChessboardCorners fails then it fallbacks to the manual corners selection flow
        if not found:
            warn("[CORNERS_FAIL]: Cannot detect corners for image " + filename + ". It requires manual corners "
                                                                                 "selection!")

            # Keep a cache of the same pictures so that if the user undo a point selection then the updated image can
            # be shown
            current_image_cache.append(deepcopy(current_image))

            # Show the corner selection window
            cv2.imshow("Manual Corners Selection", current_image)
            cv2.setMouseCallback("Manual Corners Selection", manual_polygon_selection_callback)

            # Once the user is done then it's sufficient to press any key to proceed to the interpolation phase
            while True:
                cv2.waitKey(0)
                if checkerboard_edges_count == 4:
                    cv2.destroyAllWindows()
                    break

            # Corner interpolation phase
            coords = interpolate_and_project_corners(checkerboard_edges, args.pattern_shape)

            # Set the current image equal to the first entry in the cache list
            current_image = current_image_cache[0]

            # Reset the cache, edges list and count
            current_image_cache.clear()
            checkerboard_edges = np.empty(shape=(4, 2), dtype=np.int32)
            checkerboard_edges_count = 0

        # Perform a refinement algorithm to the projected corner points
        sub_pixels_t_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        points = cv2.cornerSubPix(gray, np.array(coords).astype(np.float32), (11, 11), (-1, -1),
                                  sub_pixels_t_criteria)

        images_info.append((found, points))

        print(points.shape, points)

        # Draw the final chessboard corners and wait for a user input to proceed
        cv2.drawChessboardCorners(current_image, args.pattern_shape, points, True)
        cv2.imshow('Checkerboard Corners', current_image)
        cv2.waitKey(200)

    # Calibration run 1
    ret_1, mtx_1, dist_1, r_vecs_1, t_vecs_1, std_in_1, _, _ = estimate_camera_params(images_info, image_shape,
                                                                                      args.pattern_shape,
                                                                                      args.square_size)

    # Calibration run 2
    images_info = list(filter(lambda info: info[0], images_info))
    if len(images_info) < 10:
        raise Exception("Cannot perform calibration run 2 with less than 10 pictures")

    ret_2, mtx_2, dist_2, r_vecs_2, t_vecs_2, std_in_2, _, _ = estimate_camera_params(sample(images_info, 10),
                                                                                      image_shape,
                                                                                      args.pattern_shape,
                                                                                      args.square_size)

    # Calibration run 3
    ret_3, mtx_3, dist_3, r_vecs_3, t_vecs_3, std_in_3, _, _ = estimate_camera_params(sample(images_info, 5),
                                                                                      image_shape,
                                                                                      args.pattern_shape,
                                                                                      args.square_size)

    # Log Calibration Parameters Fidelity And Plot results of different calibrations
    log_camera_intrinsics_confidence(mtx_1, ret_1, std_in_1, "CALIBRATION_RUN_1")
    log_camera_intrinsics_confidence(mtx_2, ret_2, std_in_2, "CALIBRATION_RUN_2")
    log_camera_intrinsics_confidence(mtx_3, ret_3, std_in_3, "CALIBRATION_RUN_3")

    plot_intrinsics_comparisons([mtx_1, mtx_2, mtx_3], [std_in_1, std_in_2, std_in_3], args.plot_dir_path)

    # serialize camera parameters
    cal_run_1_path = os.path.join(args.ser_dir_path, "calibration_run_1.npz")
    np.savez(cal_run_1_path, ret=np.array(ret_1), mtx=mtx_1, dist=dist_1, r_vecs=r_vecs_1,
             t_vecs=t_vecs_1, s_size=args.square_size, run=1)

    cal_run_2_path = os.path.join(args.ser_dir_path, "calibration_run_2.npz")
    np.savez(cal_run_2_path, ret=np.array(ret_2), mtx=mtx_2, dist=dist_2, r_vecs=r_vecs_2,
             t_vecs=t_vecs_2, s_size=args.square_size, run=2)

    cal_run_3_path = os.path.join(args.ser_dir_path, "calibration_run_3.npz")
    np.savez(cal_run_3_path, ret=np.array(ret_3), mtx=mtx_3, dist=dist_3, r_vecs=r_vecs_3,
             t_vecs=t_vecs_3, s_size=args.square_size, run=3)


if __name__ == "__main__":
    # CLI arguments parsing and error detection stage
    args = cli_arguments_parse()

    # Calibration stage
    calibration(args)
    exit(0)
