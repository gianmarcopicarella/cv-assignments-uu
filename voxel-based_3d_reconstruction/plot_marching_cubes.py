import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure
import pickle
import argparse


def cli_arguments_parse():
    """
    This function defines a CLI, parses and checks the CLI arguments and returns them. There is only 1 required argument:
    voxel_model_path. If the parsing fails or one of the parameters is not well-formed then
    the entire script is aborted and a CLI message displayed.
    """
    parser = argparse.ArgumentParser(description="Voxel surface estimation")
    parser._action_groups.pop()

    required = parser.add_argument_group('Required arguments')
    optional = parser.add_argument_group('Optional arguments')

    required.add_argument("-v", "--voxel_model_path", type=str, help='Voxel model path', required=True)

    optional.add_argument("-s", "--surface_plot_path", type=str, help='Surface plot save path', default=os.getcwd())
    optional.add_argument("-r", "--rotate180_voxel", help='Rotate the voxel 180 degrees around the y-axis',
                          action='store_true')

    args = parser.parse_args()

    if not os.path.isfile(args.voxel_model_path):
        parser.error("The voxel model path is not well-formed or doesn't exist!")

    if not os.path.isdir(args.surface_plot_path):
        parser.error("The surface plot directory save path is not well-formed or doesn't exist!")

    return args


def get_volumetric_data(filepath):
    with open(filepath, 'rb') as handle:
        voxel = pickle.load(handle)
        return voxel


if __name__ == '__main__':
    # CLI arguments parsing and error detection stage
    args = cli_arguments_parse()

    # Load voxel model
    volumetric_data = get_volumetric_data(args.voxel_model_path)

    # Rotate voxel
    if args.rotate180_voxel:
        volumetric_data = np.rot90(volumetric_data, 2)

    # Apply marching cubes algorithm
    verts, faces, normals, values = measure.marching_cubes(volumetric_data, 0)

    # Plot result
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_zlabel("z-axis")
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")

    ax.set_zlim(0, volumetric_data.shape[0])
    ax.set_xlim(0, volumetric_data.shape[2])
    ax.set_ylim(0, volumetric_data.shape[1])

    plt.tight_layout()

    voxel_surface_save_path = os.path.join(args.surface_plot_path, 'voxel_surface.png')
    plt.savefig(voxel_surface_save_path)

    plt.show()
    exit(0)
