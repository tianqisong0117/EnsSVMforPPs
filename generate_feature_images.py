"""
Module to generate the square feature images and feature disks.

# Authors: Tianqi SONG <tianqisong0117@gmail.com>
# License: GNU GENERAL PUBLIC LICENSE

Abbreviations:
N_vertex = Number of center vertices for surface profiling.
N_p = Number of profiles for each center.
N_s = Number of sampling points on each profiles.

# For the storage of results, users can custom own paths by set the 'save_path'

"""


import numpy as np
from PIL import Image
import trimesh
import utils


def sts_profile_feature_image(
        sub_id,
        mesh_path,
        texture_path,
        ordered_fundus_path,
        sample_points_path,
        sam_points_cof_id_path,
        visual_img=True,
        save_bool=False):
    """
    For a given subject,
        1. Generate the feature (in feature list) values for all profiling points
        2. Generate the feature disks of STS fundus
        3. Generate the square feature images of STS fundus if visual_img=True.

    :param sub_id: string
        Index of given subject.
    :param mesh_path: string
        Path of cortical surface triangle mesh.
    :param texture_path: string
        Path of texture maps (AverSampleDis).
    :param ordered_fundus_path: string
        Path of specific region to generate the feature images.
        e.g. The fundus of Superior Temporal Sulcus in our paper.
    :param sample_points_path: string
        Path of profiling sample points
    :param sam_points_cof_id_path: string
        Path of indices of polygons(faces) which the profiling sample points belongs to.
    :param visual_img: Bool
        If True, generate the feature images for visualization.
    :param save_bool: Bool
        If True, must set the PATH to save the generated feature maps.

    :return: None
    """

    # load mesh
    mesh = utils.load_mesh(mesh_path)

    # load texture map
    feature_tex = utils.read_texture(texture_path)

    # load the profiling results
    profiling_sample_points = np.load(sample_points_path)
    prof_sam_points_cof_id = np.load(sam_points_cof_id_path)

    # load ordered fundus ids for fundus points
    ordered_fundus = np.load(ordered_fundus_path)

    # compute the texture values of each profiling sample points
    feature_profile = compute_features_of_profiles(mesh, feature_tex,
                                                   profiling_sample_points, prof_sam_points_cof_id)

    if save_bool:
        # define your own path
        save_path = 'data/'
        sts_profile_asd_save_path = save_path + '{}white_sts_profile_asd.npy'.format(sub_id)
        utils.write_np_array(sts_profile_asd_save_path, feature_profile)

    generate_feature_disk(sub_id, feature_profile, feature_tex, ordered_fundus, save_path='data/')

    # Visualization
    if visual_img:
        feature_to_image_vis(feature_profile, sub_id, ordered_fundus, save_path='data/')

    print('ASD is done.')

    return


def compute_features_of_profiles(
        mesh,
        feature_tex,
        profiling_sample_points,
        prof_sam_points_cof_id):
    """
    Compute feature values of profiling sample points using the barycentric interpolation.

    :param mesh: trimesh object
        Triangle mesh of cortical surface.
    :param feature_tex: array
        Texture maps (AverSampleDis) for cortical surface.
    :param profiling_sample_points: (N_vertex, N_p, N_s, 3, 3)
        Profiling sample results.
        For each profile points contain [p1, p2, sample_points],
        where p1, p2 are the points used to calculate the sampling points.
    :param prof_sam_points_cof_id: (N_vertex, N_p, N_s)
        Indices of polygons(faces) which the profiling sample points belongs to.

    :return: array of float, (N_vertex, N_p, N_s)
        The feature values of profiling sample points.
    """

    # compute the barycentric parameters of each profile point to its co-faces
    # of mesh
    barycentric_para = compute_barycentric_para(profiling_sample_points, mesh, prof_sam_points_cof_id)
    print('barycentric_para is done')

    # compute the features of each profile
    feature_profile = get_profile_texture_value(feature_tex, mesh, prof_sam_points_cof_id, barycentric_para)
    print('feature_profile is done')

    return feature_profile


def compute_barycentric_para(profile_sample_points, mesh, triangle_id):
    """
    Compute the barycentric parameters of each points in profiles

    :param profile_sample_points: array of float, (N_vertex, N_p, N_s, 3, 3)
        Profiling sample points.
    :param mesh: trimesh object
        Triangle mesh of cortical surface.
    :param triangle_id: array of int, (N_vertex, N_p, N_s)
        The indices of polygons(triangles).

    :return: array of float, (N_vertex, N_p, N_s, 3)
        The barycentric interpolation parameters for each polygon.
    """

    vert = mesh.vertices
    poly = mesh.faces

    sample_points_profile = profile_sample_points[:, :, :, 2]

    sample_points = sample_points_profile.reshape(
        sample_points_profile.size // 3, 3)

    triangle_id = triangle_id.reshape(triangle_id.size)

    triangles_v = vert[poly[triangle_id]]

    barycentric = trimesh.triangles.points_to_barycentric(
        triangles_v, sample_points)
    barycentric = barycentric.reshape(
        len(sample_points_profile), len(sample_points_profile[0]), len(sample_points_profile[0][0]), 3)

    return barycentric


def get_profile_texture_value(feature_tex, mesh, triangle_id, barycentric_para):
    """
    For each given points of profiles, calculate the texture values.

    :param feature_tex: array of float, (N_vertex,)
        Texture maps.
    :param mesh: trimesh object
        Triangle mesh of cortical surface.
    :param triangle_id: array of int, (N_vertex, N_p, N_s)
        The indices of polygons(triangles).
    :param barycentric_para: array of float, (N_vertex, N_p, N_s, 3)
        The barycentric interpolation parameters for each polygon.

    :return: array of float, (N_vertex, N_p, N_s)
        The feature values of profiling sample points.
    """

    num_points_sts = len(barycentric_para)
    num_areas = len(barycentric_para[0])
    num_sides = len(barycentric_para[0][0])
    poly = mesh.faces

    triangle_id = triangle_id.reshape(triangle_id.size)
    barycentric = barycentric_para.reshape(barycentric_para.size // 3, 3)

    feature_tri_points = feature_tex[poly[triangle_id]]

    feature_profile = np.dot(feature_tri_points * barycentric, [1, 1, 1])

    return feature_profile.reshape(num_points_sts, num_areas, num_sides)


def feature_to_image_vis(feature_profile, sub_id, ordered_fundus, save_path):
    """
    Generate the squared feature images for visualization.

    :param feature_profile: array of float, (N_vertex, N_p, N_s)
        The feature values of profiling sample points.
    :param sub_id: int
        Index of subjects.
    :param ordered_fundus: array of int,
        Ordered fundus points ids from anterior to posterior.
    :param save_path: string
        Path to save the images.

    :return: None
    """

    feature_y_max = np.max(feature_profile)
    feature_y_min = np.min(feature_profile)

    scale_a = 0
    scale_b = 255

    sts_samples_image = np.round((feature_profile - feature_y_min) /
                                 (feature_y_max - feature_y_min) *
                                 (scale_b - scale_a) + scale_a)

    for i in range(len(feature_profile)):
        newim = Image.fromarray(sts_samples_image[i])
        newim = newim.convert("L")

        # define your own paths
        save_files = save_path + '{}white/sts_profiles_image/vertex_{}_{}_asd_img.jpeg'\
                                 .format(sub_id, ordered_fundus[i], i)
        utils.makedir(save_files)
        newim.save(save_files)
    return


def generate_feature_disk(sub_id, feature_profile, vertex_feature, ordered_fundus, save_path):
    """
    Generate the feature disk (Vector)
    NOTE: In order to visualize the feature disk in BrainVisa/Anatomist software as a 3D disk model,
          here we save the origin point value twice.
          For the same reason, the format of result is GIFIT file.

    :param sub_id: string
        Index of subject.
    :param feature_profile: array of float, (N_vertex, N_p, N_s)
        The feature values of profiling sample points.
    :param vertex_feature: array of float, (N_vertex,)
        Texture maps.
    :param ordered_fundus: array of int,
        Ordered fundus points ids from anterior to posterior.
    :param save_path: string,
        Path to save the feature disk of all fundus points.
        The format of files is GIFIT (.gii).

    :return: None
    """
    # profile_feature_file = sts_profiles_img_file + 'data/{}white_sts_profile_asd.npy'.format(sub, sub, feature_name)
    # feature_profile = np.load(profile_feature_file)

    num_profiles = len(feature_profile[0])
    num_samples = len(feature_profile[0][0])

    for i in range(len(feature_profile)):
        origin_feature = vertex_feature[ordered_fundus[i]]
        profile_features_i = feature_profile[i].reshape(num_profiles * num_samples)

        # save the origin point value twice to visualize the results in Anatomist software
        disk_mesh_feature = np.hstack((origin_feature, profile_features_i, origin_feature))

        # Here the info of save_path are
        # sub_id, mesh vertices ids, ordered fundus ids (anterior-posterior)
        save_file = save_path + '{}white/sts_profiles_disk/vertex_{}_{}_asd_disk.gii'\
                                .format(sub_id, ordered_fundus[i], i)
        utils.write_texture([disk_mesh_feature], save_file)

    return
