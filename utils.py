import numpy as np
import os
import nibabel as nib
import trimesh


def get_rotate_matrix(rot_axis, angle):
    """
    for a pair of rotation axis and angle, calculate the rotate matrix
    :param rot_axis: rotation axis
    :param angle: float, the rotation angle is a real number rather than degree
    :return: rotate matrix of [3, 3]
    """

    if np.linalg.norm(rot_axis) == 0:
        raise Exception('The axis of rotation cannot be 0.')

    # normalize the rotate axis
    r_n = rot_axis / np.linalg.norm(rot_axis)
    rot_matrix = np.zeros((3, 3), dtype='float32')

    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    x = r_n[0]
    y = r_n[1]
    z = r_n[2]

    rot_matrix[0, 0] = cos_theta + (1 - cos_theta) * np.power(x, 2)
    rot_matrix[0, 1] = (1 - cos_theta) * x * y - sin_theta * z
    rot_matrix[0, 2] = (1 - cos_theta) * x * z + sin_theta * y

    rot_matrix[1, 0] = (1 - cos_theta) * y * x + sin_theta * z
    rot_matrix[1, 1] = cos_theta + (1 - cos_theta) * np.power(y, 2)
    rot_matrix[1, 2] = (1 - cos_theta) * y * z - sin_theta * x

    rot_matrix[2, 0] = (1 - cos_theta) * z * x - sin_theta * y
    rot_matrix[2, 1] = (1 - cos_theta) * z * y + sin_theta * x
    rot_matrix[2, 2] = cos_theta + (1 - cos_theta) * np.power(z, 2)

    return rot_matrix


def project_vector2tangent_plane(v_n, v_p):
    """
    calculate the projection vector of v_p onto tangent plane of v_n
    :param v_n: array of (3,)  float
        normal vector of v
    :param v_p: array of (n, 3) float
        vector projected
    :return: v_t (n, 3) float
        projection result
    """

    if np.linalg.norm(v_n) == 0:
        unitev_n = v_n
    else:
        unitev_n = v_n / np.linalg.norm(v_n)

    coeff_v_pn = np.dot(v_p, unitev_n)

    coeff = coeff_v_pn.reshape([coeff_v_pn.size, 1])

    v_pn = coeff * unitev_n

    v_t = v_p - np.array(v_pn)

    return v_t


def project_vector2vector(v_n, v_p):
    """
    calculate the projection vector of v_p onto v_n,
    v_pn = (v_p dot v_n) / |v_n| * unite vector of (v_n)
    :param v_n: array of (3,)  float
        direction vector
    :param v_p: array of (n, 3) float
        vectors projected
    :return: (n, 3) float
        projection result
    """

    if np.linalg.norm(v_n) == 0:
        unitev_n = v_n
    else:
        unitev_n = v_n / np.linalg.norm(v_n)

    coeff_v_pn = np.dot(v_p, unitev_n)

    coeff = coeff_v_pn.reshape([coeff_v_pn.size, 1])

    v_pn = coeff * unitev_n

    return v_pn


"""
Functions for IO
"""


def load_mesh(mesh_file):
    """
    Load the .gii file as trimesh object.
    :param mesh_file:
    :return:
    """
    nib_mesh = nib.load(mesh_file)
    vert = nib_mesh.darrays[0].data.astype(np.float64)
    poly = nib_mesh.darrays[1].data

    mesh = trimesh.Trimesh(vert, poly, process=False)

    return mesh


def read_texture(texture_path):
    """
    Load the texture data
    :param texture_path: string
        Path of given texture GIFTI files
    :return: array
        array of texture maps
    """
    nib_tex = nib.load(texture_path)
    tex = nib_tex.darrays[0].data

    return tex


def write_texture(darray, gifti_file):
    """
    write a TextureND object to disk as a gifti file
    :param darray: list of array
    :param gifti_file: string
        Path of the GIFTI file.
    :return: the corresponding TextureND object
    """

    file_name = os.path.split(gifti_file)[-1]
    dirs = gifti_file[: - len(file_name) - 1]

    if not os.path.exists(dirs):
        os.makedirs(dirs)

    darrays_list = []
    for d in darray:
        gdarray = nib.gifti.GiftiDataArray().from_array(
            d.astype(np.float32), 0)
        darrays_list.append(gdarray)
    out_texture_gii = nib.gifti.GiftiImage(darrays=darrays_list)

    nib.gifti.write(out_texture_gii, gifti_file)

    return


def write_np_array(path, np_array):
    """
    Write a numpy array to disk
    :param path:
    :param np_array:
    :return:
    """
    file_name = os.path.split(path)[-1]
    dirs = path[: - len(file_name) - 1]

    if not os.path.exists(dirs):
        os.makedirs(dirs)

    np.save(path, np_array)

    return


def makedir(path):

    file_name = os.path.split(path)[-1]
    dirs = path[: -len(file_name) - 1]
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    return

