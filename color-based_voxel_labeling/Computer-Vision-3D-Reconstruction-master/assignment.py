import pickle

import glm

import os

from modules.io import load_cameras_xml, load_checkerboard_xml, fetch_video_frame
from modules.subtraction import *
from modules.utils import *
from engine.config import config

from scipy import interpolate

block_size = 1.0


def get_video_path(data_path, camera_number, video_name="video.avi"):
    return os.path.join(data_path, "cam" + str(camera_number) + "/" + video_name)


def get_video_source(data_path, camera_number):
    return cv2.VideoCapture(get_video_path(data_path, camera_number))


def get_subtractor(data_path, camera_number):
    return train_hsv_KNN_subtractor(get_video_path(data_path, camera_number, "background.avi"), 2600)


def load_lookup_table(filepath):
    """
    Loads the look-up table
    :param filepath: look-up table filepath
    :return: the look-up table dictionary
    """
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
frame_numbers_count = count_video_frames(os.path.join(base_project_data, "cam1/video.avi"))
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
trajectories = [[] for _ in range(4)]
trajectories_map = np.zeros((400, int(400 * (config["world_height"] / config["world_width"])), 3), dtype=np.uint8)
offline_color_models = None


def build_voxel_to_pixel_lookup():
    """
    Compute the inverse look-up table for each camera based on the look-up table
    :return: the inverse look-up table for each camera
    """
    global lookup_table

    D = {i: dict() for i in range(len(lookup_table))}

    for i in range(len(D)):
        for key in lookup_table[i].keys():
            for v in lookup_table[i][key]:
                D[i][v] = key

    return D


voxel_lookup = build_voxel_to_pixel_lookup()


def compute_clustered_groups(items, labels, k):
    """
    Given some items, labels and the number of clusters this function computes each group of clustered elements as a separate list
    :param items: The clustered items
    :param labels: The items' labels
    :param k: The number of clusters
    :return: The list of lists containing all the clusters
    """
    clustered_items = [[] for _ in range(k)]

    for item_idx, item_label_idx in enumerate(labels):
        clustered_items[item_label_idx[0]].append(items[item_idx])

    return clustered_items


def compute_kmeans_clusters(items, k, items_preprocess=None, with_outliers=True, std_factor=1.0):
    """
    Performs a clustering with k clusters over all the items specified. If specified, it filters out all the outliers based on the distance mean and standard deviation
    :param items: The items to be clustered
    :param k: The number of clusters
    :param items_preprocess: A pre-processing function
    :param with_outliers: If set to true then the function removes the outliers and performs an additional clustering operation
    :param std_factor: How many standard deviations a point can be far from the mean to be considered valid
    :return: The list of lists containing all the clusters
    """
    if items_preprocess is None:
        items_preprocess = lambda x: x

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(items_preprocess(items), k, None, criteria, 100, cv2.KMEANS_PP_CENTERS)

    clustered_items = compute_clustered_groups(items, labels, k)

    if with_outliers:
        return clustered_items, centers

    # Detect and remove outliers from each cluster
    for cluster_idx in range(len(clustered_items)):
        cluster_count = len(clustered_items[cluster_idx])
        D = np.zeros(shape=cluster_count, dtype=np.float32)

        for item_idx, item in enumerate(items_preprocess(clustered_items[cluster_idx])):
            D[item_idx] = np.linalg.norm(item - centers[cluster_idx])

        mean = np.mean(D)
        std = np.std(D)
        outliers_indices = []

        for item_idx in range(cluster_count):
            if abs(D[item_idx] - mean) >= std * std_factor:
                outliers_indices.append(item_idx)

        for outlier_idx in reversed(outliers_indices):
            clustered_items[cluster_idx].pop(outlier_idx)

    # Flatten clustered_items list
    flattened_items = [item for cluster in clustered_items for item in cluster]

    # Apply K-Means without outliers
    _, labels, centers = cv2.kmeans(items_preprocess(flattened_items), k, None, criteria, 100, cv2.KMEANS_PP_CENTERS)

    clustered_items = compute_clustered_groups(flattened_items, labels, k)

    return clustered_items, centers


def compute_color_list(clustered_voxels, palette, matching_found=True):
    """
    Computes the list of colors for each cluster based on the color palette
    :param clustered_voxels: the list of clustered voxels
    :param palette: the list of colors to be used
    :param matching_found: If set to false then all the voxels are colored black
    :return: The list of colors for each cluster
    """
    colours = []

    if matching_found:
        for cluster_idx in range(len(clustered_voxels)):
            colours += [palette[cluster_idx]] * len(clustered_voxels[cluster_idx])
    else:
        colours = [palette[-1]] * sum([len(c) for c in clustered_voxels])

    return colours


def kmeans_preprocess_voxels(voxels):
    """
    Removes the y axis from each voxel position before the clustering is applied
    :param voxels: list of 3D voxels
    :return: list of 2D voxels without the Y axis
    """
    X = np.array(voxels, dtype=np.float32)
    X = np.delete(X, 2, axis=1)
    return X


def project_and_sample_pixels(voxels, camera_index, samples_count=200, height_weigths=[0.1, 0.4],
                              width_weigths=[0.2, 0.2]):
    """
    Projects and samples the pixel colors for a specific cluster.
    :param voxels: The list of voxels
    :param camera_index: The camera index
    :param samples_count: The number of samples
    :param height_weigths: The weights used to focus the sampling operation within a certain height range
    :param width_weigths: The weights used to focus the sampling operation within a certain width range
    :return: The list of samples 2D pixels
    """
    global voxel_lookup

    # Project voxels to 2D pixels
    unique_pixels = set()
    min_height, max_height = np.inf, -np.inf
    min_width, max_width = np.inf, -np.inf

    # Remove pixel not in x-range
    for voxel in voxels:
        pixel = voxel_lookup[camera_index][voxel]
        unique_pixels.add(pixel)
        min_width = min(min_width, pixel[0])
        max_width = max(max_width, pixel[0])
        min_height = min(min_height, pixel[1])
        max_height = max(max_height, pixel[1])

    width_range = max_width - min_width + 1
    height_range = max_height - min_height + 1

    min_height += height_range * height_weigths[0]
    max_height -= height_range * height_weigths[1]

    min_width += width_range * width_weigths[0]
    max_width -= width_range * width_weigths[1]

    unique_pixels = list(
        filter(lambda p: min_width <= p[0] <= max_width and min_height <= p[1] <= max_height, unique_pixels))
    samples_indices = np.random.choice(np.arange(0, len(unique_pixels)), samples_count, replace=True)
    samples = [unique_pixels[pixel_index] for pixel_index in samples_indices]

    return samples


def find_matching(online_color_models, camera_index, camera_bonus=1):
    """
    Computes the matching with the highest score among all the possible ones.
    :param online_color_models: The list of online color models
    :param camera_index: The camera index
    :param camera_bonus: The bonus factor assigned to the current camera
    :return: The matching with the highest score and its score
    """
    global offline_color_models

    matching_matrix = np.zeros((len(online_color_models), len(offline_color_models[camera_index])), dtype=np.float32)

    # Computing the matching matrix
    for online_index, online_model in enumerate(online_color_models):
        for offline_index, offline_model in enumerate(offline_color_models[camera_index]):
            aggregate_score = 0.0

            for channel_index in range(len(offline_model)):
                aggregate_score += cv2.compareHist(online_model[channel_index], offline_model[channel_index], 0)

            matching_matrix[online_index, offline_index] = aggregate_score

    # Select best matching
    total_similarity = 0.0
    match = [0] * len(online_color_models)

    for _ in range(len(online_color_models)):
        max_match_index = np.unravel_index(matching_matrix.argmax(), matching_matrix.shape)

        total_similarity += matching_matrix[max_match_index]
        match[max_match_index[0]] = max_match_index[1]

        matching_matrix[max_match_index[0], :] = [-np.inf] * matching_matrix.shape[1]
        matching_matrix[:, max_match_index[1]] = [-np.inf] * matching_matrix.shape[0]

    return [tuple(match), total_similarity * camera_bonus]


def compute_hsv_histogram(cluster, hsv_frame, camera_index):
    """
    Computes a histogram for the specified cluster, hsv_frame and camera index
    :param cluster: The list of 3D voxels
    :param hsv_frame: The fetched frame from camera with index = camera_index in hsv color space
    :param camera_index: The camera index
    :return: A list containing one normalized histogram per color channel
    """
    pixels = project_and_sample_pixels(cluster, camera_index)

    h = np.zeros(len(pixels), dtype=np.uint8)
    s = np.zeros(len(pixels), dtype=np.uint8)
    v = np.zeros(len(pixels), dtype=np.uint8)

    for i in range(len(pixels)):
        pixel = hsv_frame[pixels[i][1], pixels[i][0]]
        h[i] = pixel[0]
        s[i] = pixel[1]
        v[i] = pixel[2]

    hh = cv2.calcHist([h], [0], None, [180], [0, 180])
    hs = cv2.calcHist([s], [0], None, [256], [0, 256])
    hv = cv2.calcHist([v], [0], None, [256], [0, 256])

    cv2.normalize(hh, hh, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hs, hs, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hv, hv, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return [hh, hs, hv]


def compute_color_models(clustered_voxels, cameras_indices, cameras_frames):
    """
    Computes the color models for a specified frame
    :param clustered_voxels: The list of all clusters found in the current frame
    :param cameras_indices: The list of camera indices to be used
    :param cameras_frames: The list of camera frames to be used
    :return: The list of color models for the current frame
    """
    global offline_color_models, base_project_data

    offline_color_models = {camera_index: [] for camera_index in cameras_indices}

    for camera_index, real_camera_index in enumerate(cameras_indices):

        # Fetch frame
        frame = fetch_video_frame(get_video_path(base_project_data, real_camera_index + 1),
                                  cameras_frames[camera_index])

        # Convert frame to target color space (HSV for now)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for cluster in clustered_voxels:
            offline_color_models[real_camera_index].append(compute_hsv_histogram(cluster, hsv_frame, real_camera_index))


def count_binary_blobs(mask):
    """
    Computes the number of white blobs in a binary mask
    :param mask: The mask to be used
    :return: The number of white blobs in the mask
    """
    r, l = cv2.connectedComponents(mask)
    return len([np.argwhere(l == i) for i in range(1, r)])


def cluster_matching(clustered_voxels, cluster_centers, frames, masks):
    """
    Computes a matching between the offline and online color models
    :param clustered_voxels: The list of clusters
    :param cluster_centers: The list of clusters' centers
    :param frames: The list of frames to be used
    :param masks: The list of foreground binary masks to be used
    :return: The list of sorted clusters and centers based on the computed matching
    """
    global offline_color_models

    matchings = []

    for camera_index in offline_color_models.keys():
        # Convert frame to target color space (HSV for now)
        hsv_frame = cv2.cvtColor(frames[camera_index], cv2.COLOR_BGR2HSV)

        # Create a list of color models for the current camera
        online_color_models = []

        for cluster in clustered_voxels:
            online_color_models.append(compute_hsv_histogram(cluster, hsv_frame, camera_index))

        # Compute matching and score
        camera_bonus_factor = 3 if count_binary_blobs(masks[camera_index]) >= 4 else 1
        matchings.append(find_matching(online_color_models, camera_index, camera_bonus_factor))

    matchings = list(sorted(matchings, key=lambda x: x[0]))
    aggregate_matches = [matchings[0]]

    for match in matchings[1:]:
        if match[0] == aggregate_matches[-1][0]:
            aggregate_matches[-1][1] += match[1]
        else:
            aggregate_matches.append(match)

    matchings = list(sorted(aggregate_matches, key=lambda x: x[1], reverse=True))

    reliable_matching = True if (len(matchings) == 1 or abs(matchings[0][1] - matchings[1][1]) >= 0) else False

    sorted_clustered_voxels = [None] * len(clustered_voxels)
    sorted_cluster_centers = [None] * len(clustered_voxels)

    for cluster_idx, sorted_cluster_idx in enumerate(matchings[0][0]):
        sorted_clustered_voxels[sorted_cluster_idx] = clustered_voxels[cluster_idx]
        sorted_cluster_centers[sorted_cluster_idx] = cluster_centers[cluster_idx]

    return reliable_matching, sorted_clustered_voxels, sorted_cluster_centers


color_palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 0, 0)]


def save_trajectories(dirpath, smooth=False, sampling=100000):
    """
    Computes the raw and smoothed trajectories map and saves them
    :param dirpath: Directory path where the trajectories maps should be saved
    :param smooth: True if the smoothed map has to be computed
    :param sampling: The number of samples used during the smoothed map computation
    """
    global trajectories_map, trajectories

    # Save trajectories list
    pickle.dump(trajectories, open(os.path.join(dirpath, "trajectories"), "wb"))

    # Save raw trajectories map
    cv2.imwrite(os.path.join(dirpath, "raw_trajectories_v.jpg"), trajectories_map)

    if smooth:
        trajectories_map.fill(0)
        for label, trajectory in enumerate(trajectories):

            x = [trajectory[0][0]]
            y = [trajectory[0][1]]

            for i in range(1, len(trajectory)):
                if x[-1] == trajectory[i][0] and y[-1] == trajectory[i][1]:
                    continue
                else:
                    x.append(trajectory[i][0])
                    y.append(trajectory[i][1])

            f, u = interpolate.splprep([x, y], s=0)
            x_int, y_int = interpolate.splev(np.linspace(0, 1, sampling), f)
            samples = np.array([np.array([x_int[i], y_int[i]], dtype=np.int32) for i in range(sampling)])

            rgb_cluster_color = color_palette[label]
            bgr_cluster_color = (rgb_cluster_color[2], rgb_cluster_color[1], rgb_cluster_color[0])
            trajectories_map = cv2.polylines(trajectories_map, [samples], False, bgr_cluster_color, 1,
                                             cv2.LINE_AA)

        # Save smooth trajectories map
        cv2.imwrite(os.path.join(dirpath, "smooth_trajectories_v.jpg"), trajectories_map)


def draw_trajectories(voxel_centers):
    """
        Draws the cluster centers with the correct matching color on the 2D map of raw trajectories
        :param voxel_centers: The list of clusters' center for the current frame
        """
    global trajectories_map, trajectories
    for cluster_idx, center in enumerate(voxel_centers):
        t_shape = trajectories_map.shape
        y_scale = t_shape[0] / config["world_width"]
        x_scale = t_shape[1] / config["world_height"]

        # Compute local coordinates
        local_center = np.array([trajectories_map.shape[1] - center[1] * x_scale,
                                 trajectories_map.shape[0] - center[0] * y_scale], dtype=np.int32)

        trajectories[cluster_idx].append(local_center)

        rgb_cluster_color = color_palette[cluster_idx]
        bgr_cluster_color = (rgb_cluster_color[2], rgb_cluster_color[1], rgb_cluster_color[0])

        cv2.drawMarker(trajectories_map, local_center, bgr_cluster_color, cv2.MARKER_CROSS, 3, 1)


def set_voxel_positions(width, height, depth):
    """
    This function computes the position and color vectors for each voxel to be drawn
    :param width: The voxel volume width
    :param height: The voxel volume height
    :param depth: The voxel volume depth
    :return: The lists of voxel positions and colors
    """
    global frame_numbers_count, current_frame_index, voxel_visibility, lookup_table, last_frame_masks, voxels_on, color_palette

    # Fetch the next video frame from each camera
    current_frame_index = (current_frame_index + 1) % frame_numbers_count
    frames = np.array([v.read()[1] for v in video_sources])

    if 0 < current_frame_index < 365:
        return [], [], False

    # Exit from the app if any of the cameras used failed to fetch the next frame
    if None in frames or current_frame_index == 465:
        save_trajectories("../", True)
        return [], [], True

    #print("Current FRAME index", current_frame_index)

    # Computes the foreground masks for each camera
    masks = compute_foreground_masks(frames,
                                     subtractors,
                                     [350, 350, 350, 350],
                                     [100, 100, 100, 100])

    cv2.imshow("Silhouettes", np.concatenate([masks[0], masks[1], masks[2], masks[3]], axis=1))

    if last_frame_masks is not None:
        # Compute the masks XOR and detect the on/off voxel pixels
        masks_xor = cv2.bitwise_xor(masks, last_frame_masks)
        masks_on = cv2.bitwise_and(masks, masks_xor)
        masks_off = cv2.bitwise_and(cv2.bitwise_not(masks), masks_xor)

        # Set voxels to off
        M = np.transpose((masks_off == 255).nonzero())
        R = [(v[0], v[1], v[2], i) for i, y, x in M if (x, y) in lookup_table[i] for v in lookup_table[i][(x, y)]]
        voxel_visibility[tuple(np.array(R).T)] = False
        R = set(map(lambda k: k[:3], R))

        # Set voxels to on
        M = np.transpose((masks_on == 255).nonzero())
        A = [(v[0], v[1], v[2], i) for i, y, x in M if (x, y) in lookup_table[i] for v in lookup_table[i][(x, y)]]
        voxel_visibility[tuple(np.array(A).T)] = True
        A = set(map(lambda k: k[:3], A))
        A = [v for v in A if np.bitwise_and.reduce(voxel_visibility[v])]

        voxels_on = voxels_on.difference(set(R))
        voxels_on = voxels_on.union(set(A))
    else:
        # Compute the voxels to draw without using the XOR optimization (only for the first video frame)
        voxel_visibility.fill(False)
        M = np.transpose((masks == 255).nonzero())
        A = [(v[0], v[1], v[2], i) for i, y, x in M if (x, y) in lookup_table[i] for v in lookup_table[i][(x, y)]]
        voxel_visibility[tuple(np.array(A).T)] = True
        A = [v for v in A if np.bitwise_and.reduce(voxel_visibility[v[:3]])]

        # Update the current voxel set
        voxels_on = set(map(lambda x: x[:3], A))

    # Ad hoc volume scaling and offsettings
    scaling_factor = 0.5
    manual_z_offset = 0
    manual_y_offset = 0
    should_close_app = False
    matching_found = True

    voxel_clusters, voxel_centers = compute_kmeans_clusters(list(voxels_on), 4, kmeans_preprocess_voxels, False, 1.2)

    if last_frame_masks is None:
        compute_color_models(voxel_clusters, [0, 1, 3], [0, 0, 0])
    else:
        # Match clusters with offline results
        res, voxel_clusters, voxel_centers = cluster_matching(voxel_clusters, voxel_centers, frames, masks)
        matching_found = res

    # update trajectories
    if matching_found and current_frame_index >= 365:
        draw_trajectories(voxel_centers)

    # Show trajectories map
    cv2.imshow("Trajectories map", trajectories_map)

    # Set the last frame mask equal to the current
    last_frame_masks = masks

    colors = compute_color_list(voxel_clusters, color_palette, matching_found)

    voxels = [(scaling_factor * (vx * block_size - width / 2), scaling_factor * (vz * block_size - manual_z_offset),
               scaling_factor * (-vy * block_size + height / 2 + manual_y_offset)) for cluster_idx in
              range(len(voxel_clusters)) for
              vx, vy, vz in voxel_clusters[cluster_idx]]

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
