import numpy as np
from sklearn.utils import resample
from sklearn import preprocessing
from sklearn import svm

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

import joblib
import pandas as pd

import utils


def train_enssvm(train_data_csv_file, feature_img_path, model_path, num_estimators, C_range, gamma_range):
    """
    Train the EnsSVM models. This function contains:
    1. Load the training data csv file
    2. Generate the sub-dataset for each base SVMs.
    3. Train the base SVMs and store them.

    :param train_data_csv_file:
        File of training data indices
    :param feature_img_path:
        Path of feature images.
    :param model_path:
        Path of storage models.
    :param num_estimators:
        Number of base classifiers.
    :param C_range:
        Range of hyper-parameters for the optimization of SVM
    :param gamma_range:
        Range of hyper-parameters for the optimization of SVM

    :return:

    """

    # load training data index
    train_data_ids = pd.read_csv(train_data_csv_file)
    pps_ids = np.array(train_data_ids[train_data_ids['label'] == 1]['id'])
    non_pps_ids = np.array(train_data_ids[train_data_ids['label'] == 0]['id'])

    # get pps and non-pps data index respectively
    for i in range(len(pps_ids)):
        pps_ids[i] = pps_ids[i].split('.')[0]
    for i in range(len(non_pps_ids)):
        non_pps_ids[i] = non_pps_ids[i].split('.')[0]

    # generate based dataset for each classifiers
    enssvm_data, enssvm_data_labels = generate_based_svm_dataset(num_estimators, pps_ids, non_pps_ids)

    # train base classifiers
    for i in range(num_estimators):
        print('Begin training model: ', i)
        estimator = select_parameters_grid(enssvm_data[i], enssvm_data_labels[i],
                                           feature_img_path, C_range, gamma_range)
        save_file = model_path.format(i)
        joblib.dump(estimator, save_file)

    return


def predict_enssvm(train_data, predict_data_csv_file, model_path, num_estimators, fea_img_path, predict_path):
    """
    This function is used to identify the PPs using EnsSVM and contains the following steps:

        1. Prediction process using EnsSVM
        2. Generate the texture maps of cortical surface.

    :param train_data: 
        Training data of the EnsSVM models.
        The data is used for the scale standardization.
    :param predict_data_csv_file:
        Path of predict data
    :param model_path:
        Path of machine learning models
    :param num_estimators:
        Number of base classifiers in EnsSVM
    :param fea_img_path:
        Path of feature images
    :param predict_path:
        Path of storage prediction results
    :return:
    """

    # load predict data
    predict_data = pd.read_csv(predict_data_csv_file)
    predict_data_ids_arr = np.array(predict_data['id'])

    predict_data_imgs, _ = load_data(predict_data_ids_arr, _, fea_img_path)

    # predict using the based classifiers
    predict_list = []
    for i in range(num_estimators):
        based_train_data = train_data[i]
        based_model_path = model_path.format(i)
        # res_arr is array of 0 or 1
        res_arr = predict_based_svm(predict_data_imgs, based_train_data, based_model_path)
        predict_list.append(res_arr)

    # major vote
    weight_vec = np.zeros(num_estimators) + 1
    enssvm_predict_res = major_vote(weight_vec, np.array(predict_list))

    # generate texture
    fundus_file = 'data/{}/{}_STS_fundus.gii'
    pps_region_fundus_file = 'data/{}/{}_predict_Fundus.gii'
    generate_label_tex(enssvm_predict_res, predict_data_ids_arr, fundus_file, pps_region_fundus_file)

    return


def predict_based_svm(predict_data_list, train_data_set, model_file):
    """
    Prediction with base classifiers

    :param predict_data_list: (N_predict_samples,)
        Predicted data
    :param train_data_set:
        Training dataset
    :param model_file:
        Storage of base classifiers

    :return: (N_predict_samples,)
        Prediction results
    """

    # get the training data standardization and apply on the predict dataset
    scaler = preprocessing.StandardScaler().fit(train_data_set)
    X_val = scaler.transform(predict_data_list)

    based_svm = joblib.load(model_file)
    print('Finish loading models.')

    y_pre_val = based_svm.predict(X_val)
    print('Finish prediction.')

    return y_pre_val


def major_vote(weight_vec, predict_data_arr):
    """
    This function is a weighted majority vote.

    :param weight_vec:
        Weights assigned to each base classifiers
    :param predict_data_arr: (N_cls, N_predict_samples)
        Prediction results of EnsSVM

    :return: (N_predict_samples,)
        Prediction results of majority vote

    """
    ensemble_pre = []
    num_samples = len(predict_data_arr[0])

    select_svm_id = np.where(weight_vec != 0)[0]
    for i in range(num_samples):
        pre_list = predict_data_arr[select_svm_id, i]
        # print(pre_list)
        n_pos = np.nonzero(pre_list)[0].size
        n_neg = pre_list.size - n_pos

        if n_pos > n_neg:
            ensemble_pre.append(1)
        else:
            # n_pos < n_neg
            ensemble_pre.append(0)

    return ensemble_pre


def generate_based_svm_dataset(n_based_svm, pps_arr, non_pps_arr, save_bool=True):
    """
    Generate the dataset used to train base classifier

    :param n_based_svm: int
        Number of base classifiers
    :param pps_arr:
        Array of PPs indices
    :param non_pps_arr:
        Array of non-PPs indices
    :param save_bool:
         If True, must set the PATH to save the trained base models.
    :return:
    """

    num_pps = len(pps_arr)
    num_non_pps = len(non_pps_arr)
    non_pps_index = np.linspace(0, num_non_pps - 1, num_non_pps - 1).astype(int)

    resample_non_pps_ids = []
    for i in range(n_based_svm):
        sub_arg = resample(non_pps_index, replace=True, n_samples=num_pps, random_state=i)
        resample_non_pps_ids.append(sub_arg)

    print(resample_non_pps_ids)

    ensvm_data = []
    ensvm_data_label = []
    for i in resample_non_pps_ids:
        sub_svm_non_pps = non_pps_arr[i]
        sub_dataset = np.hstack((pps_arr, sub_svm_non_pps))
        ensvm_data.append(sub_dataset)

        # labels
        sub_data_labels = np.hstack(([1]*len(pps_arr), [0]*len(sub_svm_non_pps)))
        ensvm_data_label.append(sub_data_labels)

    if save_bool:
        # Define your own storage path
        save_path = 'data/{} estimators_dataset.npy'.format(n_based_svm)
        utils.write_np_array(save_path, np.array(ensvm_data))
    print('Finish generating dataset.')
    return np.array(ensvm_data), np.array(ensvm_data_label)


def select_parameters_grid(train_dataset, train_data_labels, fea_img_path, C_range, gamma_range):
    """
    Select the optimization hyper-parameters for each base classifiers using the Grid-search method.

    :param train_dataset:
        Training data index
    :param train_data_labels:
        Manual labels
    :param fea_img_path:
        Path of feature images
    :param C_range: 1D array
        The value range of parameter C for SVM
    :param gamma_range: 1D array
        The value range of parameter gamma for SVM

    :return:
        Optimal classifiers
    """

    X_train, y_train = load_data(train_dataset, train_data_labels, fea_img_path)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    param_grid = dict(gamma=gamma_range, C=C_range)
    # 10-cross validation
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=1)
    grid = GridSearchCV(svm.SVC(class_weight='balanced'), param_grid=param_grid, cv=cv, refit=True)
    grid.fit(X_train_scaled, y_train)

    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))

    return grid.best_estimator_


def load_data(dataset, labels, fea_img_path):
    """
    Load the data

    :param dataset:
        Index of training data
    :param labels:
        Manual labels of training data
    :param fea_img_path:
        Path of feature images
        GIFTI file

    :return:
        data_list: (N_samples, )
            List of feature vectors
        data_label: (N_samples,)
            List of manual labels
    """
    data_list = []

    for i in dataset:
        #  load the feature image
        sub_tex_path = fea_img_path + '{}_asd.gii'.format(i)
        sub_tex = utils.read_texture(sub_tex_path)
        data_list.append(sub_tex[: -1])

    data_label = labels

    return data_list, data_label


def get_specific_svm_predict_proba(predict_probabilities, local_vert_arr, num_base_clf):
    """
    Compute the average predict probability from based classifiers of EnsSVM.

    :param predict_probabilities: array of float
        Predict probabilities
    :param local_vert_arr: array of int
        Array of vertex index.
    :param num_base_clf: int
        Number of base classifiers in EnsSVM

    :return:
        The prediction probabilities for all points in PPs regions.
    """

    sum_proba = np.copy(predict_probabilities[0])
    for i in range(0, len(predict_probabilities)):
        sum_proba += predict_probabilities[i]

    avg_proba = sum_proba / num_base_clf
    # [not_pps_porba, pps_proba]
    pre_proba_pps = avg_proba[:, 1]
    # print(avg_proba[:, 1])
    reg_proba = pre_proba_pps[local_vert_arr]

    return reg_proba


def generate_label_tex(predict_label_arr, predict_data_ids_arr, fundus_texture_path, storage_path):
    """
    Generate the texture maps with predicted labels of PPs.

    :param predict_label_arr: (N_pdt_samples,)
        Prediction result of EnsSVM
    :param predict_data_ids_arr: (N_pdt_samples,)
        Array of prediction data index
    :param fundus_texture_path: string
        Path of fundus texture (GIFTI file)
    :param storage_path: string
        Storage path of generated label texture (GIFTI file)

    :return:
    """

    # identify the subject ids
    sub_ids = []
    for i in predict_data_ids_arr:
        sub_i = '{}_{}'.format(i.split('_')[0], i.split('_')[1])
        if not sub_i in sub_ids:
            sub_ids.append(sub_i)

    for j in sub_ids:
        # fundus value = 100
        new_fundus_tex = utils.read_texture(fundus_texture_path.format(j, j))
        for k in range(len(predict_data_ids_arr)):
            if j in k:
                mesh_vert_ids = int(predict_data_ids_arr[k].split('_')[2])
                # pps vertex value = 50
                new_fundus_tex[mesh_vert_ids] = 100 - predict_label_arr[k] * 50

        # store the texture files
        save_file = storage_path.format(j)
        utils.write_texture([new_fundus_tex], save_file)
        print('Subject {} texture is done.'.format(j, j))

    return
