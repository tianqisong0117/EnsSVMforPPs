"""
.. _example_post_processing:

===================================
example of post-processing
===================================

# Authors: Tianqi SONG <tianqisong0117@gmail.com>
# License: GNU GENERAL PUBLIC LICENSE

"""

###############################################################################
# importation of modules
import pps_regions
import post_processing as pp_post

###############################################################################
# example subjects index
sub_id = 'AWY13_R'

###############################################################################
# Load prediction results
# prediction labels
example_predict_res_file = 'data/{}white/{}_ensvm_predict_label.txt'.format(sub_id, sub_id)
# prediction probability
example_predict_proba_path = 'data/{}white/{}_ensvm_predict_label_proba.npy'
# ordered fundus
ordered_fundus_path = 'data/{}white/{}_STS_ordered_fundus.npy'
# fundus texture
fundus_tex_path = 'data/{}white/{}_STS_fundus.gii'

# post-processing results
post_pps_path = 'data/{}white/{}_pps_post_process.npy'
post_pps_tex_path = 'data/{}white/{}_pps_tex_post_process.gii'

# get the PPs regions
ori_pps_reg_pos,  ori_pps_reg_len, ori_sub_pps_info = \
    pps_regions.generate_pps_region_from_pred_res(sub_id, example_predict_res_file)

# Post-processing
pp_post.post_processing(list(ori_pps_reg_pos),
                        list(ori_pps_reg_len),
                        list(ori_sub_pps_info),
                        ordered_fundus_path,
                        fundus_tex_path,
                        example_predict_proba_path,
                        post_pps_path,
                        post_pps_tex_path)
