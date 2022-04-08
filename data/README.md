This folder contains small files that are used in the code.

## Fetal brain atlases

* the folder ```Neurotypical_Gholipour2017``` contains the fetal brain atlas for normal brain development proposed in [1].
We have manually refined the segmentation compared to the original release fo the atlas.

* the folder ```Neurotypical_Wu2021``` contains the fetal brain atlas for normal brain development proposed in [2].
We have manually refined the segmentation compared to the original release fo the atlas.

* the folder ```SpinaBifida_Fidon2021``` contains the fetal brain atlas for spina bifida aperta that we proposed in [3].
The 3D MRI and segmentations in this folder are the exactly the same as the one at
https://zenodo.org/record/5524312#.Yk7SD3XMI5k

## Fetal MRI data
The full fetal brain MRI is not open access for now.

A subset of the testing dataset (88 3D MRIs with manual segmentations) can be downloaded at
https://zenodo.org/record/6405632#.YkbWPCTMI5k

This subset corresponds to the FeTA dataset [4] with our manual refinement of the segmentations and addition of the 
segmentation of the corpus callosum.


## How to cite
If you use one or several of the atlases in this repository, please cite the corresponding papers (see above).

The manual refinement of the segmentations were performed by Michael Aertsen and Lucas Fidon.
To acknowledge the work on the manual segmentations please also cite

* L. Fidon, M. Aertsen, F. Kofler, A. Bink, A. L. David, T. Deprest, D. Emam, F. Guffens, A. Jakab, G. Kasprian,
 P. Kienast, A. Melbourne, B. Menze, N. Mufti, I. Pogledic, D. Prayer, M. Stuempflen, E. Van Elslander, S. Ourselin, 
 J. Deprest, T. Vercauteren.
 [A Dempster-Shafer approach to trustworthy AI with application to fetal brain MRI segmentation][twai]


## References
[1] Gholipour, Ali, et al. "A normative spatiotemporal MRI atlas of the fetal brain for automatic segmentation and 
analysis of early brain growth." Scientific reports 7.1 (2017): 1-13.

[2] Wu, Jiangjie, et al. "Age-specific structural fetal brain atlases construction and cortical development 
quantification for chinese population." Neuroimage 241 (2021): 118412.

[3] Fidon, Lucas, et al. "A spatio-temporal atlas of the developing fetal brain with spina bifida aperta." 
Open Research Europe (2021).

[4] Payette, Kelly, et al. "An automatic multi-tissue human fetal brain segmentation benchmark using the 
Fetal Tissue Annotation Dataset." Scientific Data 8.1 (2021): 1-14.


[twai]: https://arxiv.org/abs/2204.02779