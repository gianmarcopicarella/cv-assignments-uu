import numpy as np
import cv2


def interpolate_and_project_corners(edges, pattern_shape, adjust_outer_edges=False):
    """
    This function computes and returns the interpolated 2D corner points using 4 manually selected board corners.
    :param edges: a list containing 4 manually selected board corners sorted in clockwise order (TP, TR, BR, BL)
    :param pattern_shape: the shape of the checkerboard
    :param adjust_outer_edges: adjusts the checkerboard width and height if set to True. If edges contains outer edges then adjust_outer_edges should be always True
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

    # Adjust width and height lengths if specified
    w_adjust, h_adjust = 0, 0

    if adjust_outer_edges:
        w_adjust = max_width / (pattern_shape[1] + 1)
        h_adjust = max_height / (pattern_shape[0] + 1)
        max_width -= 2.0 * w_adjust
        max_height -= 2.0 * h_adjust

    # Compute the inverse perspective transform
    p_matrix = cv2.getPerspectiveTransform(edges.astype(np.float32), output_points)
    inv_p_matrix = np.linalg.inv(p_matrix)

    # Compute the horizontal and vertical step
    w_step = max_width / (pattern_shape[1] - 1)
    h_step = max_height / (pattern_shape[0] - 1)

    projected_corners = []

    # Compute each projected point
    for x in range(0, pattern_shape[1]):
        for y in range(0, pattern_shape[0]):
            point = np.array([w_adjust + x * w_step, h_adjust + y * h_step, 1])
            point = np.matmul(inv_p_matrix, point)
            # Divide each point by its Z component
            point *= (1.0 / point[2])
            # Append only the first 2 elements of each point
            projected_corners.append(point[0:2])

    return projected_corners


def find_checkerboard_polygon(frame, background_subtractor):
    """
    This function attempts to locate and approximate the checkerboard polygon containing all the black squares
    :param frame: The frame where the checkerboard polygon should be detected
    :param background_subtractor: The background subtractor used
    :return: The polygon's list of points sorted in clock-wise order
    """
    # Background subtraction
    kernel = np.ones((3, 3), np.uint8)

    foreground_seg = background_subtractor.apply(frame, None, 0.0001)
    f_mask = cv2.erode(foreground_seg, kernel, iterations=1)
    f_mask = cv2.dilate(f_mask, kernel, iterations=1)

    # Foreground noise reduction and histogram equalization
    foreground = cv2.bitwise_and(frame, frame, mask=f_mask)
    foreground_gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    foreground = cv2.equalizeHist(foreground_gray)

    # Thresholding phase
    _, f_mask = cv2.threshold(foreground, 127, 255, cv2.THRESH_BINARY)
    foreground = cv2.bitwise_and(foreground, foreground, mask=f_mask)

    # Find final square mask
    f_mask = cv2.erode(f_mask, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(f_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    flat_contours = []
    for ctr in contours:
        flat_contours += [pt[0] for pt in ctr]

    hull = cv2.convexHull(np.array(flat_contours))

    final_mask = np.zeros_like(f_mask)
    cv2.drawContours(final_mask, [hull], -1, 255, -1)

    _, fs_mask = cv2.threshold(foreground, 127, 255, cv2.THRESH_BINARY_INV)
    squares_mask = cv2.bitwise_and(fs_mask, fs_mask, mask=final_mask)

    squares_mask = cv2.erode(squares_mask, kernel, iterations=1)
    squares_mask = cv2.dilate(squares_mask, kernel, iterations=1)

    # Final rectangle corners search
    lst = []

    for y in range(squares_mask.shape[0]):
        for x in range(squares_mask.shape[1]):
            if squares_mask[y, x]:
                lst.append(np.array([x, y]))

    hull = cv2.convexHull(np.array(lst))

    perimeter = cv2.arcLength(np.array(hull), True)
    approx = cv2.approxPolyDP(np.array(hull), 0.02 * perimeter, True)

    return [p[0] for p in approx]
