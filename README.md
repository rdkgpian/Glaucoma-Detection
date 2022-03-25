# Glaucoma Detection

## Problem Statement Description
Glaucoma is a serious ocular disease and leads to blindness if it can’t be detected and treated in a proper way. The diagnostic criteria for glaucoma include intraocular pressure measurement, optic nerve head evaluation, retinal nerve fiber layer and visual field defect. The observation of optic nerve head, cup to disc ratio and neural rim configuration are important for early detecting glaucoma in clinical practice. Hence, it is still important to develop various detection techniques to assist clinicians to diagnose glaucoma at early stages.

## Methodology
A three-step methodology has been used for the automated diagnosis of glaucoma:
- Optic Disc Segmentation
- Optic Cup Segmentation
- Ellipse Fitting of segmented optic cup and disc.

The ratio of the diameters of the ellipses is calculated to get the CDR. The retinal fundus image having **CDR greater than 0.5 is classified as glaucomous.**

Link to the Dataset: [Dataset](https://github.com/seva100/optic-nerve-cnn/tree/master/data/DRISHTI_GS)

## Results

For the accuracy of segmenetation of Optic Disc and cup, two metrics were used: **FScore and Bounday Localization**. Here are the results:

![](https://user-images.githubusercontent.com/32013812/160123780-16d142c4-f352-46ad-a768-3dbaf144a457.png)

The calculated CDR values for train and test data can be found in the _Results/train_x_ and _Results/test_x_ files of the repository. The corresponding classification (glaucomous or non-glaucomous) is saved in _train_y_ and _test_y_ files.

## References

- Yin, Fengshou, et al. "Automated segmentation of optic disc and optic cup in fundus images for glaucoma diagnosis." 2012 25th IEEE international symposium on computer-based medical systems (CBMS). IEEE, 2012.
- Norouzifard, Mohammad, et al. "Unsupervised optic cup and optic disk segmentation for glaucoma detection by icica." 2018 15th International Symposium on Pervasive Systems, Algorithms and Networks (I-SPAN). IEEE, 2018.
- Roslin, M., and S. Sumathi. "Glaucoma screening by the detection of blood vessels and optic cup to disc ratio." 2016 International Conference on Communication and Signal Processing (ICCSP). IEEE, 2016.
