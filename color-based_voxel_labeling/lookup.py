import cv2
import numpy as np
import ctypes
import os
from multiprocessing import Pool, shared_memory
from contextlib import closing
import pickle
import functools
import argparse

from modules.io import load_cameras_xml


def cli_arguments_parse():
    """
    This function defines a CLI, parses and checks the CLI arguments and returns them. There is only 1 required argument:
    cameras_path. If the parsing fails or one of the parameters is not well-formed then
    the entire script is aborted and a CLI message displayed.
    """
    parser = argparse.ArgumentParser(description="Look-up table creation")
    parser._action_groups.pop()

    required = parser.add_argument_group('Required arguments')
    optional = parser.add_argument_group('Optional arguments')

    required.add_argument("-c", "--cameras_path", type=str, help='Cameras base path', required=True)

    optional.add_argument("-l", "--lookup_save_path", type=str, help='Look-up table save path', default=os.getcwd())
    optional.add_argument("-k", "--calibration_scale", type=int, help='Calibration scale in mm', default=115)
    optional.add_argument("-s", "--step_size", type=int, help='Sampling step in mm', default=15)
    optional.add_argument("-f", "--frame_shape", type=int, help='Frame shape (DimY, DimX)', nargs=2, default=[486, 644])
    optional.add_argument("-o", "--volume_center", type=float, help='Volume Center', nargs=3, default=[5, 3.8, 0])
    optional.add_argument("-v", "--volume_size", type=float, help='Volume Center', nargs=3, default=[10, 12, 15])
    optional.add_argument("-d", "--sort_by_camera_distance", action='store_true', help='Sort each voxel by distance '
                                                                                       'from each camera')

    args = parser.parse_args()

    if not os.path.isdir(args.cameras_path):
        parser.error("The cameras folder path is not well-formed or doesn't exist!")

    if not os.path.isdir(args.lookup_save_path):
        parser.error("The look-up folder save path is not well-formed or doesn't exist!")

    return args


def compute_world_points(args, chunk_size):
    """
    This function computes the real world coordinates in mm for each voxel point
    :param args: The command line arguments provided by the user
    :param chunk_size: The size of a volume chunk
    :return: The real world points and the number of chunks per dimension
    """

    # Compute volume location and ranges
    center = np.array(args.volume_center, dtype=np.float32) * args.calibration_scale
    size = np.array(args.volume_size, dtype=np.float32) * args.calibration_scale

    sx, sy, _ = center - size / 2.0
    ex, ey, _ = center + size / 2.0
    sz, ez = center[2], size[2]

    samples_size = (size / args.step_size).astype(ctypes.c_int32)

    print("The requested look-up table will have size", samples_size)

    # Sample points
    X = np.linspace(sx, ex, samples_size[0], dtype=np.float32)
    Y = np.linspace(sy, ey, samples_size[1], dtype=np.float32)
    Z = np.linspace(sz, ez, samples_size[2], dtype=np.float32)

    chunks = np.ceil(samples_size / chunk_size).astype(np.int32)

    return X, Y, Z, chunks


def create_shared_memory_chunk(shape, type):
    """
    This function creates a shared memory buffer and binds it to a numpy array
    :param shape: The shape of the numpy array
    :param type: The type of data stored in the numpy array
    :return: The numpy array backed with a shared memory region
    """
    temp = np.zeros(shape=shape, dtype=type)
    shared = shared_memory.SharedMemory(create=True, size=temp.nbytes)
    array = np.ndarray(temp.shape, dtype=type, buffer=shared.buf)
    array[:] = temp[:]
    return shared, array


def parallel_projection(chunk, camera):
    """
    This function is used as a parallel worker during the computation
    :param chunk: The current shared memory chunk
    :param camera_index: The camera index used by this worker process
    :return: The list of 2D projected points
    """
    flattened_pixels, _ = cv2.projectPoints(chunk,
                                            camera["RotationVector"],
                                            camera["TranslationVector"],
                                            camera["CameraMatrix"],
                                            camera["DistortionCoeffs"])
    return flattened_pixels


def is_valid_pixel(pixel, shape):
    return not np.any(np.isinf(pixel)) and \
        -1 < pixel[0] < shape[1] and \
        -1 < pixel[1] < shape[0]


# CLI arguments
args = None
# Hardcoded chunk size
chunk_size = 128
# Cameras
cameras = None
X, Y, Z, chunks_count = None, None, None, None
shared_mem, shared_voxels_chunk = None, None


def main_process_exclusive_work():
    global args, cameras, X, Y, Z, chunks_count, shared_mem, shared_voxels_chunk
    # CLI arguments parsing and error detection stage
    args = cli_arguments_parse()

    # Load global data
    cameras = load_cameras_xml(args.cameras_path, args.sort_by_camera_distance, 1.0 / args.calibration_scale)
    X, Y, Z, chunks_count = compute_world_points(args, chunk_size)
    shared_mem, shared_voxels_chunk = create_shared_memory_chunk((chunk_size ** 3, 3), ctypes.c_float)


def compute_shared_voxels_chunk(lx, ly, lz, cx, cy, cz, size):
    """
    This function fills the shared memory region with the real world coordinates that will be used by cv2.projectPoints
    :param lx: The list of real world points along the x-axis
    :param ly: The list of real world points along the y-axis
    :param lz: The list of real world points along the z-axis
    :param cx: The chunk index along the x-axis
    :param cy: The chunk index along the y-axis
    :param cz: The chunk index along the z-axis
    :param size: The size of a chunk
    :return: The length of the chunk along each axis
    """
    global shared_voxels_chunk

    llx = lx[(cx * size):((cx + 1) * size)]
    lly = ly[(cy * size):((cy + 1) * size)]
    llz = lz[(cz * size):((cz + 1) * size)]

    for ix, x in enumerate(llx):
        for iy, y in enumerate(lly):
            for iz, z in enumerate(llz):
                i = ix + iy * len(llx) + iz * len(llx) * len(lly)
                shared_voxels_chunk[i] = [x, y, z]

    return len(llx), len(lly), len(llz)


if __name__ == '__main__':

    main_process_exclusive_work()

    parallel_processes_count = len(cameras)
    lookup = {i: dict() for i in range(len(cameras))}

    print("Starting look-up table computation!")
    print("Total number of chunks: ", chunks_count[0] * chunks_count[1] * chunks_count[2])

    # Starting a pool with one worker per camera
    with closing(Pool(parallel_processes_count)) as p:
        for cx in range(chunks_count[0]):
            for cy in range(chunks_count[1]):
                for cz in range(chunks_count[2]):
                    # Fill the current memory chunk
                    chunk_filled_size = compute_shared_voxels_chunk(X, Y, Z, cx, cy, cz, chunk_size)

                    # Compute starting x, y, z indices for current volume chunk
                    chunk_start_x = cx * chunk_size
                    chunk_start_y = cy * chunk_size
                    chunk_start_z = cz * chunk_size

                    print("Working on chunk ", str((cx, cy, cz)), "of", chunks_count)

                    for camera_index, pixels in enumerate(
                            p.imap(functools.partial(parallel_projection, shared_voxels_chunk),
                                   [camera for camera in cameras])):

                        print("Working on camera", camera_index, "of chunk", str((cx, cy, cz)))

                        pixels = pixels.astype(np.int32)

                        # Link 3D voxel to a pixel for a given camera if the key is withing the image shape
                        pixels_index = 0
                        for ix in range(chunk_filled_size[0]):
                            for iy in range(chunk_filled_size[1]):
                                for iz in range(chunk_filled_size[2]):
                                    i = ix + iy * chunk_filled_size[0] + iz * chunk_filled_size[0] * chunk_filled_size[
                                        1]
                                    pixel_t = tuple(pixels[i][0])
                                    if is_valid_pixel(pixel_t, args.frame_shape):
                                        world_voxel = (chunk_start_x + ix, chunk_start_y + iy, chunk_start_z + iz)
                                        lookup[camera_index].setdefault(pixel_t, []).append(world_voxel)

                        # Sort by camera distance each voxel
                        if args.sort_by_camera_distance:
                            for pixel in lookup[camera_index].keys():
                                lookup[camera_index][pixel] = list(sorted(lookup[camera_index][pixel],
                                                                          key=lambda v: np.sum(np.square(
                                                                              np.array(v) - cameras[camera_index][
                                                                                  "RescaledWorldPosition"]))))

    # Save the look-up table in a binary file
    pickle.dump(lookup, open(os.path.join(args.lookup_save_path, "lookup"), "wb"))

    # Free the shared memory region
    del shared_voxels_chunk
    shared_mem.close()
    shared_mem.unlink()

    exit(0)
