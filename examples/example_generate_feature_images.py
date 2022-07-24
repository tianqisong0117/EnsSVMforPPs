"""
.. _example_generate_feature_images:

===================================
example of feature images generation
===================================

# Authors: Tianqi SONG <tianqisong0117@gmail.com>
# License: GNU GENERAL PUBLIC LICENSE

"""


###############################################################################
# importation of modules
import generate_feature_images as gfi

###############################################################################
# Load the files used to generate feature images
# Example brain hemi-sphere id
sub_id = '118124_L'

# mesh file
mesh_file = 'data/{}white/{}white_to_dwi.gii'.format(sub_id, sub_id)

# texture map of AverSampleDist
asd_maps_file = 'data/{}white/{}_AverSampleDist.gii'.format(sub_id, sub_id)

# ordered fundus file
ordered_fundus_file = 'data/{}white/{}_STS_ordered_fundus.npy'.format(sub_id, sub_id)

# Profiling samples results on STS fundus
sample_points_file = 'data/{}white/{}_sts_samples_points_fnum.npy'.format(sub_id, sub_id)
sam_points_cof_id_file = 'data/{}white/{}_sts_samples_points_fnum_face_id.npy'.format(sub_id, sub_id)

###############################################################################
# Generate the feature disks and images for given subject.
gfi.sts_profile_feature_image(sub_id,
                              asd_maps_file,
                              mesh_file,
                              ordered_fundus_file,
                              sample_points_file,
                              sam_points_cof_id_file,
                              visual_img=True, save_bool=False)

print(sub_id, 'is done.')

