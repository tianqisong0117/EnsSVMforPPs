# EnsSVMforPPs
This is a source code for "Ensemble learning for the detection of pli-de-passages in the superior temporal sulcus"

## Highlights

We propose the first machine learning based method to automatically detect plis-de-passage (PP) 
in the superior temporal sulcus (STS). The method contains the following steps:
+ Generate the cortical texture maps AverSampleDist (ASD) .
+ Generate the local feature images for each given vertex on the STS. 
+ Ensemble Support Vector Machine (EnsSVM) to classify the feature images as PPs or not.
+ Post-processing algorithm to further select the PPs from PPs regions. 

## Dependencies
- Python 3.6+
- [gdist](https://github.com/the-virtual-brain/external_geodesic_library)
- [Numpy](https://numpy.org)
- [NiBabel](https://nipy.org/nibabel/)
- [Pillow](https://python-pillow.org)
- [Scipy](https://scipy.org)
- [Scikit-learn](https://scikit-learn.org/)
- [Trimesh](https://github.com/mikedh/trimesh)



## Usage
There are three main steps to identify the PPs on a specific cortical surface areas, 
and each steps contains following python modules. 
+ Data pre-processing: ``surface_profiling.py``, ``AverSampleDist_map.py``
+ Machine learning
+ Post-processing 

The details of each python modules are shown as follow:
+ ``surface_profiling.py``: Our implementation of the Cortical Surface Profiling method in [Li et al. (2010)](https://doi.org/10.1016/j.neuroimage.2010.04.263).
+ ``AverSampleDist_map.py``: Generate the texture map ASD for cortical surface.
+
+
+
# Citations
EnsSVMforPPs is an open-source library and is licensed under the GNU General Public License (v3)

If you are using this library, please cite our paper.
