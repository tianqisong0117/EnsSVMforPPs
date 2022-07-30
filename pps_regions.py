import numpy as np
import linecache
import utils
import ensemble_learning


class PPsRegion(object):
    """
    PPs regions.
    """
    def __init__(self, pps_reg, pps_reg_id):
        self.pps_reg = pps_reg
        self.del_vert_id = []
        self.pps_reg_id = pps_reg_id

    def set_pps_reg(self, vlist):
        self.pps_reg = vlist
        return

    def set_pps_reg_id(self, id_list):
        self.pps_reg_id = id_list
        return

    def get_pps_reg(self):
        return self.pps_reg

    def get_pps_reg_id(self):
        return self.pps_reg_id

    def add_pps_reg(self, vlist):
        return self.pps_reg.append(vlist)

    def add_pps_reg_id(self, id_list):
        return self.pps_reg_id.append(id_list)

    def record_del_pps_vert(self, vert_id):
        self.del_vert_id.append(vert_id)
        return


def generate_pps_region_from_pred_res(sub_id, predict_res_path, save_bool=False):
    """
    Generate the PPs regions from the predict result.

    :param sub_id: string
        Index of given subject.
    :param predict_res_path: string
        Path of predict result of STS fundus points.
    :param save_bool: Bool
        If True, must set the PATH to save the generated feature maps.
    :return:
        pps_reg_pos: start vertex ids for each PPs regions
        pps_reg_len: length of each PPs regions
        sub_pps_info: [subject index, number of PPs regions]
    """

    pps_reg_pos,  pps_reg_len, sub_pps_info = get_info_pps_regions(sub_id, predict_res_path)

    if save_bool:

        pps_reg_pos_file = 'data/{}/pps region position.npy'.format(sub_id)
        utils.write_np_array(pps_reg_pos_file, pps_reg_pos)

        pps_reg_len_file = 'data/{}/pps region length.npy'.format(sub_id)
        utils.write_np_array(pps_reg_len_file, pps_reg_len)

        sub_pps_info_file = 'data/{}/number of pps region.npy'.format(sub_id)
        utils.write_np_array(sub_pps_info_file, sub_pps_info)

    return pps_reg_pos,  pps_reg_len, sub_pps_info


def get_info_pps_regions(sub_id, predict_res_path):
    """
    For a given subject, get the pps region result from the predict labels.
    We record the following info:
    1. The start position for each PPs regions (from anterior to posterior)
    2. The length of each PPs region
    3. The number of PPs regions.

    :param sub_id: string
        Index of subject.
    :param predict_res_path: string
        Path of predict results.

    :return:
        pps_reg_pos: start vertex ids for each PPs regions
        pps_reg_len: length of each PPs regions
        sub_pps_info: [subject index, number of PPs regions]
    """

    # sub_id_len = 7
    pps_reg_pos = []
    pps_reg_len = []
    sub_pps_info = []

    pre_res_file = predict_res_path.format(sub_id)
    label_str = get_pre_label_str(pre_res_file)

    # get the num of zero between pp region and non-pp region
    label_1 = np.array(label_str.split('0'))
    num_str_zero = np.where(label_1 != '')[0]

    pp_region_arr = label_1[num_str_zero]

    num_pp_region = len(pp_region_arr)
    pp_reg_start_id = np.zeros(num_pp_region).astype(int)
    pp_reg_len_arr_i = np.zeros(num_pp_region).astype(int)

    sum_pp_reg_len = 0
    for j in range(num_pp_region):
        pp_region_j = pp_region_arr[j]
        len_pp_reg_j = len(pp_region_j)
        num_zero_str_j = num_str_zero[j]

        if j == 0:
            pp_reg_start_id[j] = num_str_zero[j]
        else:
            len_str = sum_pp_reg_len + num_zero_str_j
            pp_reg_start_id[j] = len_str

        pp_reg_len_arr_i[j] = len_pp_reg_j
        sum_pp_reg_len += len_pp_reg_j

    pps_reg_pos.append(pp_reg_start_id)
    pps_reg_len.append(pp_reg_len_arr_i)
    sub_num_list = [sub_id, num_pp_region]
    sub_pps_info.append(sub_num_list)
    print(sub_id, 'PPs info is done.')

    return np.array(pps_reg_pos, dtype=object), np.array(pps_reg_len, dtype=object), np.array(sub_pps_info)


def get_pre_label_str(predict_path):
    """
    Generate the prediction results as a sting consisting of '0' or '1'.

    :param predict_path: string
        Path of prediction result.
    :return: string
        Predict label string.
    """

    f1 = open(predict_path, 'r')
    number = len(f1.readlines())

    predict_str = ''
    for i in range(1, number + 1):
        pl = linecache.getline(predict_path, i).strip()[-1:]
        predict_str += pl

    return predict_str


def generate_pps_label_tex(predict_pps_arr, ordered_fundus_path, fundus_tex_path, save_label_tex_path):
    """
    Generate texture maps labelled the PPs points

    :param predict_pps_arr:
        Prediction results of PPs
    :param ordered_fundus_path:
        Ordered fundus points indices
    :param fundus_tex_path:
        Path of STS fundus texture
    :param save_label_tex_path:
        Storage path of texture maps
    :return:
    """

    subs_id_arr = predict_pps_arr[:, 0]
    pps_loc_vert_id = predict_pps_arr[:, 1]

    for i in range(len(subs_id_arr)):
        sub_i = subs_id_arr[i]
        pps_loc_vert_i = pps_loc_vert_id[i]

        ordered_fundus_file = ordered_fundus_path.format(sub_i, sub_i)
        ordered_fundus = np.load(ordered_fundus_file)

        tex_arr = utils.read_texture(fundus_tex_path.format(sub_i, sub_i))

        vert_id = ordered_fundus[pps_loc_vert_i]

        post_process_tex_arr = np.zeros(len(tex_arr))
        post_process_tex_arr[vert_id] = 1

        label_save_file = save_label_tex_path.format(sub_i, sub_i)
        utils.write_texture([post_process_tex_arr], label_save_file)

        print(sub_i, 'Label texture is done.')

    return


def generate_predict_probability_tex(sub_id,
                                     ordered_fundus_path,
                                     fundus_tex_path,
                                     pre_proba_path,
                                     save_bool=False):
    """
    Generate texture of prediction probability of PPs on cortical surface

    :param sub_id: string
        Index of subject
    :param ordered_fundus_path: string
        Path of ordered fundus index array
    :param fundus_tex_path:
        Path of STS fundus texture
    :param pre_proba_path:
        Path of prediction probability array
    :param save_bool:
        If True, must set the PATH to save the texture maps of prediction probability.

    :return: GIFTI file
        Texture maps
    """

    # get vertices ids
    ordered_fundus_file = ordered_fundus_path.format(sub_id, sub_id)
    ordered_fundus = np.load(ordered_fundus_file)

    sts_id_arr = np.linspace(0, len(ordered_fundus) - 1, len(ordered_fundus)).astype(int)

    # get proba
    predict_proba = np.load(pre_proba_path.format(sub_id, sub_id))
    pps_reg_svm_proba = ensemble_learning.get_specific_svm_predict_proba(predict_proba, sts_id_arr, 21)

    new_fundus_tex = utils.read_texture(fundus_tex_path.format(sub_id, sub_id))  # fundus value = 100

    len_sts = len(ordered_fundus)
    for j in range(len_sts):
        new_fundus_tex[ordered_fundus[j]] = pps_reg_svm_proba[j]

    if save_bool:
        tex_save_path = 'data/{}/{}_STS_proba_fundus.gii'
        save_file = tex_save_path.format(sub_id, sub_id)
        utils.write_texture([new_fundus_tex], save_file)

    return new_fundus_tex
