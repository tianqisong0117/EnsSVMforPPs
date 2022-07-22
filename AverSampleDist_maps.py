"""
Module to generate the texture maps(AverSampleDist) for cortical surface.

# Authors: Tianqi SONG <tianqisong0117@gmail.com>
# License: GNU GENERAL PUBLIC LICENSE

"""


import surface_profiling as sf
import numpy as np
import utils


def compute_aver_sample_dist(mesh, rot_angle=5, samp_step=0.1, samp_max=45, save_bool=False):
    """
    This function is used to generate the feature map of AverSampleDis for a given mesh.
    1. Compute the surface profiling of the triangle mesh (cortical surface)
    2. Compute the AverSampleDist of cortical surface.
    *Default parameters are used in our paper.*

    :param mesh: trimesh object
        The cortical surface mesh.
    :param rot_angle: float
        Degree of rotation angle.
    :param samp_step: float
        Length of sampling steps.
    :param samp_max: int
        Maximum of samples in one profiles.
    :param save_bool: BOOL
        If save_bool=True, must set the PATH to save the generated feature maps.
    :return:
    """

    # sampling the cortical surface and save the result
    print('Begin sampling.')
    _, samples_y = sf.cortical_surface_profiling(mesh, rot_angle, samp_step, samp_max)
    print('Finish sampling.')

    # Compute the AverSampleDist feature maps.
    aver_sample_y = np.average(samples_y, axis=2)
    for i in range(len(samples_y)):
        for j in range(len(samples_y[i])):
            aver_sample_y[i][j] = np.average(samples_y[i][j])
    aver_sample_dis = np.average(aver_sample_y, axis=1)

    if save_bool:
        # define your own save path
        print('Save the ASD maps.')
        utils.write_texture([aver_sample_dis], 'data/example_asd_tex.gii')

    return aver_sample_dis
