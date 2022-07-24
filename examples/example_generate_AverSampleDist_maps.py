"""
.. _example_generate_AverSampleDist_maps:

===================================
example of AverSampleDist maps
===================================

# Authors: Tianqi SONG <tianqisong0117@gmail.com>
# License: GNU GENERAL PUBLIC LICENSE

"""

###############################################################################
# importation of modules
import AverSampleDist_maps as asd_maps
import trimesh.visual.color
import utils
from matplotlib import pyplot as plt

###############################################################################
# Load example mesh
mesh = utils.load_mesh('data/example_mesh.gii')

###############################################################################
# Generate the AverSampleDist map for the example mesh
texture = asd_maps.compute_aver_sample_dist(mesh, rot_angle=5, samp_step=0.1, samp_max=45)

# In this example, we have already generated the ASD map.
# For a quick look, you can directly load the texture map use the following code.

# texture = utils.read_texture('data/example_asd_tex.gii')

###############################################################################
# Visualize result
#
# set color maps
color_map = plt.get_cmap('jet', 12)

# set texture value to mesh
mesh.visual.vertex_colors = trimesh.visual.color.interpolate(texture, color_map=color_map)
scene = trimesh.Scene(mesh)
scene.show(smooth=False)
