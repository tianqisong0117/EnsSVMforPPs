import numpy as np
import pps_regions
import ensemble_learning
import utils


def post_processing(pps_reg_pos,
                    pps_reg_len,
                    sub_pps_info,
                    ordered_fundus_path,
                    fundus_tex_path,
                    predict_proba_path,
                    post_pps_path,
                    post_pps_tex_path):
    """
    Post-processing

    1. For each vertices on STS fundus, get its prediction label and the associated probability from the SVMs.
    2. Compute PP regions as connected components of the PP label set.
    3. Discard all regions with less than 3 vertices.
    4. For regions with 5 vertices or less, select the vertex with maximum probability to represent the PP.
    5. For regions with more than 5 vertices,
        a. Split the region at the vertex with minimum PP probability.
        b. If sub-region contains 5 vertices or less, go to step 4.
           Otherwise, iterate step 5.

    :param pps_reg_pos:
        Start vertex ids for each PPs regions
    :param pps_reg_len:
        Length of each PPs regions
    :param sub_pps_info:
        [subject index, number of PPs regions]
    :param ordered_fundus_path: string
        Path of ordered fundus (array)
    :param fundus_tex_path: string
        Path of STS fundus texture (GIFTI file)
    :param predict_proba_path: string
        Path of prediction results (array) with probability from machine learning models.
    :param post_pps_path: string
        Storage path of PPs results after post-processing
    :param post_pps_tex_path: string
        Storage path of PPs texture after post-processing

    :return:
        The selected PPs points after post-processing step

    """

    # get the number of subjects
    num_sub = len(pps_reg_len)
    # set minimum PPs regions
    min_pps_reg_len = 5

    for i in range(num_sub):
        sub_id = sub_pps_info[i][0]

        # Load prediction probability file
        sub_pre_proba_file = predict_proba_path.format(sub_id, sub_id)
        predict_probabilities = np.load(sub_pre_proba_file)

        # Load ordered fundus arr
        ordered_fundus_file = ordered_fundus_path.format(sub_id, sub_id)
        sub_ordered_fundus = np.load(ordered_fundus_file)

        # 1. discard the single pp point region
        sig_pps_reg_id = np.where(pps_reg_len[i] == 1)[0]

        if len(sig_pps_reg_id) != 0:
            pps_reg_pos[i] = np.delete(pps_reg_pos[i], sig_pps_reg_id)
            pps_reg_len[i] = np.delete(pps_reg_len[i], sig_pps_reg_id)

        # discard the pps reg with 2 point
        p2_pps_reg_id = np.where(pps_reg_len[i] == 2)[0]

        if len(p2_pps_reg_id) != 0:
            pps_reg_pos[i] = np.delete(pps_reg_pos[i], p2_pps_reg_id)
            pps_reg_len[i] = np.delete(pps_reg_len[i], p2_pps_reg_id)

        # 2. discard the anterior reg
        # check the 1st pps region
        len_ant = pps_reg_pos[i][-1] + pps_reg_len[i][-1]
        if len_ant == len(sub_ordered_fundus):
            pps_reg_pos[i] = np.delete(pps_reg_pos[i], -1)
            pps_reg_len[i] = np.delete(pps_reg_len[i], -1)

        # 3. check and split the outlier region
        outlier_pps_reg_id = np.where(pps_reg_len[i] >= 6)[0]

        if len(outlier_pps_reg_id) != 0:
            new_reg_pos = []
            new_reg_len = []
            for j in outlier_pps_reg_id:
                split_res = split_pps_reg_proba(predict_probabilities, min_pps_reg_len,
                                                pps_reg_len[i][j], pps_reg_pos[i][j],
                                                num_base_clf=21)
                for k in split_res:
                    new_reg_pos.append(k[0])
                    new_reg_len.append(len(k))

            # delete outlier region info
            pps_reg_len[i] = np.delete(pps_reg_len[i], outlier_pps_reg_id)
            pps_reg_pos[i] = np.delete(pps_reg_pos[i], outlier_pps_reg_id)

            # add new region info
            pps_reg_pos[i] = np.hstack([pps_reg_pos[i], np.array(new_reg_pos)])
            pps_reg_len[i] = np.hstack([pps_reg_len[i], np.array(new_reg_len)])

        sub_pps_info[i][1] = len(pps_reg_pos[i])
        # print(i)

        ###############################################################################
        # Selection of PPs points in new regions and save the res
        proba_tex = pps_regions.generate_predict_probability_tex(sub_id,
                                                                 ordered_fundus_path,
                                                                 fundus_tex_path,
                                                                 predict_proba_path, save_bool=False)

        post_pps_arr = select_pps_from_pps_reg(np.array(pps_reg_pos),
                                               np.array(pps_reg_len),
                                               np.array(sub_pps_info),
                                               sub_ordered_fundus,
                                               proba_tex, post_pps_path)

        pps_regions.generate_pps_label_tex(post_pps_arr, ordered_fundus_path, fundus_tex_path, post_pps_tex_path)

    return


def split_pps_reg_proba(predict_probabilities,
                        min_len_pps_reg,
                        len_outlier_reg,
                        reg_start_pos,
                        num_base_clf=21):
    """
    Split the PPs regions according to the prediction probability of each points in PPs regions.
    :param predict_probabilities: array,
        Prediction probabilities for all vertices on the STS fundus.
    :param min_len_pps_reg: int
        Minimum of points in PPs region
    :param len_outlier_reg: int
        Length of PPs region
    :param reg_start_pos: int
        Start points of given PPs region
    :param num_base_clf: int
        Number of base classifiers in EnsSVM,
        the parameter is needed to compute the probability.

    :return:
        Generated PPs regions after split.
    """

    pps_reg_end_pos = reg_start_pos + len_outlier_reg
    reg_vert_id_arr = np.linspace(reg_start_pos, pps_reg_end_pos - 1, len_outlier_reg).astype(int)

    # get probability
    pps_reg_svm_proba = ensemble_learning.get_specific_svm_predict_proba(predict_probabilities,
                                                                         reg_vert_id_arr,
                                                                         num_base_clf)

    # split the region
    pps_region = pps_regions.PPsRegion([], [])  # used to save the split result
    new_pps_reg = binary_split(pps_reg_svm_proba, reg_vert_id_arr, min_len_pps_reg, pps_region)

    split_reg_id = new_pps_reg.get_pps_reg_id()

    return split_reg_id


def binary_split(split_reg_arr, split_arr_id, num_min_points, pps_reg):
    """
    Split the regions that exceed the maximum number of points.
    Iterate util the generated PPs region fit the constraints.

    :param split_reg_arr:
        The PPs region to be split
    :param split_arr_id:
        The index of vertices in PPs region.
    :param num_min_points: int,
        Minimum number of points in a PPs region.
    :param pps_reg:
        Array to record all PPs regions.

    :return: array
        New PPs regions after split.
    """

    if len(split_reg_arr) < num_min_points:
        pps_reg.add_pps_reg(split_reg_arr)
        pps_reg.add_pps_reg_id(split_arr_id)

    else:
        min_idx = np.argmin(split_reg_arr)

        if min_idx == 0:
            # the left point
            reg_r = split_reg_arr[min_idx + 1:]
            reg_r_id = split_arr_id[min_idx + 1:]

            binary_split(reg_r, reg_r_id, num_min_points, pps_reg)

        elif min_idx + 1 == len(split_reg_arr):
            # the right point
            reg_l = split_reg_arr[:min_idx]
            reg_l_id = split_arr_id[:min_idx]

            binary_split(reg_l, reg_l_id, num_min_points, pps_reg)

        else:
            reg_l = split_reg_arr[:min_idx]
            reg_l_id = split_arr_id[:min_idx]
            binary_split(reg_l, reg_l_id, num_min_points, pps_reg)

            reg_r = split_reg_arr[min_idx + 1:]
            reg_r_id = split_arr_id[min_idx + 1:]
            binary_split(reg_r, reg_r_id, num_min_points, pps_reg)

    return pps_reg


def select_pps_from_pps_reg(pps_reg_pos,
                            pps_reg_len,
                            sub_pps_info,
                            ordered_fundus,
                            proba_tex,
                            predict_pps_path):
    """
    For each PPs regions, select the single point with maximum probability to represent the PPs regions.

    :param pps_reg_pos:
        Start vertex ids for each PPs regions
    :param pps_reg_len:
        Length of each PPs regions
    :param sub_pps_info:
        [subject index, number of PPs regions]
    :param ordered_fundus:
        Ordered STS fundus points indices from anterior to posterior
    :param proba_tex:
        Prediction probability of PPs points
    :param predict_pps_path:
        Storage Path of selected PPs points

    :return: array
        Selected PPs points indices
    """

    sub_id_arr = sub_pps_info[:, 0]
    pre_pps_list = []

    for i in sub_id_arr:

        sub_i = np.where(sub_id_arr == i)[0]

        # generate predict labeled points
        sub_pps_pos = pps_reg_pos[sub_i][0]
        sub_pps_len = pps_reg_len[sub_i][0]

        labeled_pps = []
        for j in range(len(sub_pps_pos)):
            reg_len_j = sub_pps_len[j]
            reg_sta_j = sub_pps_pos[j]
            local_pps_reg_arr = np.linspace(reg_sta_j, reg_sta_j + reg_len_j - 1, reg_len_j).astype(int)
            labeled_pps.append(local_pps_reg_arr)

        # select PPs points
        pps_local_idx = []
        pps_local_pre = []
        for j in range(len(labeled_pps)):
            pps_reg_j = labeled_pps[j]

            sts_proba_value = proba_tex[ordered_fundus]
            pps_reg_j_proba = sts_proba_value[pps_reg_j]
            max_pps_points = np.argmax(pps_reg_j_proba)
            pps_local_idx.append(pps_reg_j[max_pps_points])
            pps_local_pre.append(pps_reg_j_proba[max_pps_points])
        pre_pps_list.append([i, pps_local_idx, pps_local_pre])

        predict_pps_file = predict_pps_path.format(i, i)
        utils.write_np_array(predict_pps_file, np.array(pre_pps_list, dtype=object))

    return np.array(pre_pps_list, dtype=object)
