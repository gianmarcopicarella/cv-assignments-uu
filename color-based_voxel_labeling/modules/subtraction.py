import cv2
import numpy as np

from modules.io import count_video_frames


def train_KNN_background_subtractor(filepath):
    """
    This function trains a KNN background subtractor using the video located at filepath
    :param filepath: The training video path
    :return: The trained KNN background subtractor
    """
    video = cv2.VideoCapture(filepath)

    subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=False)

    while True:
        success, current_frame = video.read()
        if not success:
            break
        subtractor.apply(current_frame, None)

    return subtractor


def train_hsv_KNN_subtractor(filepath, threshold, shadows=False):
    """
        This function trains a KNN background subtractor using the hsv version of the video located at filepath
        :param filepath: The training video path
        :param threshold: The foreground threshold
        :return: The trained KNN background subtractor
    """
    subtractor = cv2.createBackgroundSubtractorKNN(history=count_video_frames(filepath),
                                                   detectShadows=shadows,
                                                   dist2Threshold=threshold)
    video = cv2.VideoCapture(filepath)

    while True:
        success, frame = video.read()
        if not success:
            break
        #frame = cv2.bilateralFilter(frame, 9, 75, 75)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        subtractor.apply(hsv)

    return subtractor


def train_hsv_MOG_subtractor(filepath, n_mixtures=5, background_ratio=0.7, noise_sigma=0):
    """
        This function trains a MOG background subtractor using the hsv version of the video located at filepath
        :param filepath: The training video path
        :param n_mixtures: The number of gaussian used
        :param background_ratio: The background ratio
        :param noise_sigma: The noise sigma
        :return: The trained MOG background subtractor
    """
    subtractor = cv2.bgsegm.createBackgroundSubtractorMOG(history=count_video_frames(filepath),
                                                          nmixtures=n_mixtures,
                                                          backgroundRatio=background_ratio,
                                                          noiseSigma=noise_sigma)
    video = cv2.VideoCapture(filepath)

    while True:
        success, frame = video.read()
        if not success:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        subtractor.apply(hsv, None, -1)

    return subtractor


def compute_foreground_masks(bgr_frames, subtractors, outer_areas_thresholds, inner_areas_thresholds,
                             custom_per_frame=None):
    """
    This function computes the foreground mask for each frame provided with the corresponding subtractor
    :param bgr_frames: The list of frames
    :param subtractors: The list of subtractors
    :param contours_thresholds: The list of contours perimeters thresholds
    :param use_denoising: Apply cv2.fastNlMeansDenoisingColored to each frame before applying the background subtractor
    :param custom_per_frame: The list of custom logic for each frame
    :return: The list of foreground masks
    """
    masks = np.zeros((len(bgr_frames), bgr_frames[0].shape[0], bgr_frames[0].shape[1]), dtype=np.uint8)

    for index, frame in enumerate(bgr_frames):

        current_frame = frame.copy()

        # Transform color space from BGR to HSV
        hsv_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

        # Apply background subtraction
        mask = subtractors[index].apply(hsv_frame, None, 0.00001)

        # Apply custom function if present
        if custom_per_frame is not None:
            mask = custom_per_frame[index](mask)

        # Find contours
        mask = cv2.dilate(mask, np.ones((3, 3)), iterations=1)
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Sort contours by size
        contours_sorted_indices = sorted(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]), reverse=True)

        silhouette = np.zeros(mask.shape, dtype=np.uint8)
        childs = [[]] * len(contours_sorted_indices)

        for c_index in contours_sorted_indices:
            if cv2.contourArea(contours[c_index]) > outer_areas_thresholds[index]:
                silhouette = cv2.drawContours(silhouette, contours, c_index, 255)
                silhouette = cv2.fillPoly(silhouette, pts=[contours[c_index]], color=255)

                # find all black holes to fill
                next_child_index = hierarchy[0][c_index][2]

                while next_child_index > -1:
                    if cv2.contourArea(contours[next_child_index], True) >= inner_areas_thresholds[index]:
                        childs[c_index].append(next_child_index)
                    next_child_index = hierarchy[0][next_child_index][0]

            else:
                break

        silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, np.ones((3,3)))

        for c_index in range(len(childs)):
            for cc_index in childs[c_index]:
                silhouette = cv2.fillPoly(silhouette, pts=[contours[cc_index]], color=0, lineType=cv2.LINE_AA)
                silhouette = cv2.drawContours(silhouette, contours, cc_index, 255)

        silhouette[silhouette > 0] = 255
        masks[index] = silhouette

    return masks
