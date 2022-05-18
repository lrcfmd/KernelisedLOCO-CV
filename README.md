# KernelisedLOCO-CV

A repository for code associated with the the publication "Random projections and Kernelised Leave One Cluster Out Cross-Validation: Universal baselines and evaluation tools for supervised machine learning for materials properties"


## Requirements

This requires a working installations of:
* python (tested using 3.7.9)

* numpy (tested using version 1.21.5)
* pandas (tested using version 1.2.3)
* scipy (tested using verseion 1.6.1)
* sci-kit learn (tested using version 2.4.1)

* git (only required for set up)

## Setup

To featurise data and download [required repository](https://github.com/rynmurdock/domain_knowledge) run the provided script by running the following the top level of this repository:
``` 
chmod +x setup.sh
./setup.sh
```
This will take a while (an hour or more) to run and result in ~30GB of data being generated

## Utilities

In utilities.py we provide methods to do the following:
* Featurise data (this mostly a light wrapper on an [existing library](https://github.com/rynmurdock/domain_knowledge)
* Make a LOCO-CV clustering file given data
* Perform LOCO-CV given a clustering file and some data

## Notebooks

We provide tutorial notebooks:
* comparing_CBFV_performance.ipynb takes you through an example of comparing two different featurisation methods for one of the tasks. It uses 80/20 train/test splits, then LOCO-CV and Kernelised LOCO-CV. It then compares each featurisation method to a random projection of the same size
* prepare_loco_cv_splits.ipynb goes through an example of preparing clusterings for use with LOCO-CV both with and without application of the radial basis function
* create_and_apply_random_projection.ipynb takes you through the process of creating and applying a random projection (**note: random projections used for scores reported in the associated paper are found in the data/linear_seperability/random_projections folder**)
* analyse_linear_seperability.ipynb takes you through an example of comparing the standard deviation in cluster size (cluster size uneveness) and the mean distance between a point in a cluster its centroid (spread of cluster) before and after application of radial basis function

## Data 

We are currently facing some issues with uploading large files to github, will get a workarround sorted in the next day or so that data can be included in this repository

original data sources can also be found in the data availability statement, and the featurisation method used is provided in utilities.py

Data for section 3.1 can be found in data/case_studies it is split using either LOCO-CV (kernelised or otherwise) or in an 80/20 train/test split

Data for section 3.2 can be found in data/linear_seperability/datasets these datasets are not split into training/test categories (this is not applicable here), and some datasets do not have target values associated with them. The ICSD is also included in these datasets **note:** 3 of the tasks in case_studes (T_c, T_c>10K and T_c|T_c>10k) use the same base data and can be seen in the data/linear_seperability/datasets/supercon folder. 

Data for section 3.3 can be found in data/linear_seperability/dataset/random_projectons_applied_to_datasets


## Citation

The paper associated with this work is currently under peer review. Citation will be provided once this is finished.