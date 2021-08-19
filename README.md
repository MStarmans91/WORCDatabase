# WORC Database: Downloaders and Experiments
This repository contains scripts to download six public datasets
and reproduce radiomics experiments on these six datasets and two other
public datasets. The six public datasets are described in the following paper:

``Starmans, M. P. A. et al. (2021). The WORC* database: MRI and CT scans, segmentations, and clinical labels for 932 patients from six radiomics studies, Submitted``

The data used for six of the datasets can be found at https://xnat.bmia.nl/data/projects/worc.
The experiments are described in the following paper:

``Starmans, M. P. A. et al. (2021). Reproducible radiomics through automated machine learning validated on twelve clinical applications, Submitted``

## License
When using parts of this code or the above datasets, please cite the two
above mentioned papers. Feel free to also cite the code from this repository:

``Martijn P.A. Starmans. WORCDatabase. Zenodo (2021). Available from:  https://github.com/MStarmans91/WORCDatabase. DOI: http://doi.org/10.5281/zenodo.5119040``

For the DOI, visit [![][DOI]][DOI-lnk].

[DOI]: https://zenodo.org/badge/DOI/10.5281/zenodo.5119040.svg
[DOI-lnk]: https://doi.org/10.5281/zenodo.5119040

## Installation
The experiments only require the WORC Python package to be installed,
which can be done using pip. In the paper, version 3.4.5 was used:

    pip install "WORC==3.5.0"

The requirements for WORC itself can be found at https://github.com/MStarmans91/WORC.

## Usage: download data
The six public datasets as published in the Data in Brief paper can be
downloaded using the provided ``datadownloader.py`` script. These are
downloaded from the XNAT repository at https://xnat.bmia.nl/data/projects/worc.
Simply import one of the following functions from the script and run it:

1. ``download_Lipo``
2. ``download_Desmoid``
3. ``download_Liver``
4. ``download_GIST``
5. ``download_CRLM``
6. ``download_Melanoma``

Additionally, we provide downloaders for two of the three previously publicly
published datasets used in the WORC MEDIA paper:

7. ``download_HeadAndNeck``
8. ``download_Glioma``

Documentation for these functions can be found in the docstrings of the
functions.

## Usage: experiment reproduction
To reproduce eight experiments presented in the WORC paper,
i.e., those on the above mentioned eight datasets, simply import
and run the ``run_experiment`` function from the ``run_experiments.py`` script.
The *dataset* argument can be used to switch between the eight datasets. See
the docstring of the function  or the *==help* option when running the
function on the command line for further documentation.

For the six public datasets as published in the Data in Brief paper, the data
is directly fed from XNAT into WORC. For the two previously publicly published
datasets used in the WORC MEDIA paper, some additional steps are performed.

### Glioma dataset
For theg glioma dataset, radiomics features are supplied instead of raw
imaging data. As these have been created using Python2 and a specific version
of pandas, these files cannot directly be fed into WORC. Therefore,
we have created two scripts to convert the feature files to a format
that can be fed into WORC, which can be found in the *helpers* folder:

1. ``convert_features_hdf5_to_csv``: this script should be run in a Python2
  environment with the pandas 0.19.0 and tables package installed. The .hdf5
  files containing the features are converted to .csv files.
2. ``convert_features_csv_to_hdf5.py``: this script should be run in a Python3
  environment with pandas installed (any version, as long as you are using
  the same one when running the WORC experiment). The .csv files which were
  created by the previous function are converted back to .hdf5 files, but
  now with the same pandas version as you are using for WORC. The features
  are also renamed to correspond with the formatting used in WORC

Documentation for these functions can be found in the respective files. Note
that you first have to download the features from http://dx.doi.org/10.17632/rssf5nxxby.3.

Afterwards, we recommend to use the ``SimpleWORC`` module as explained in
the WORC Tutorial (https://github.com/MStarmans91/WORCTutorial) to run the
actual WORC experiment.

### Head and neck dataset
For the head and neck dataset, the data can be found at https://xnat.bmia.nl/data/projects/stwstrategyhn1.
This data can be directly fed into WORC. However, in the
experiment conducted in the paper, we predict the T-stage based on GTV-1 segmentations,
which are both missing for some of the patients. Hence, before running the experiment,
we first check for which patients these are available and only include those.
This is automatically done in the ``run_experiment`` function when using the
head and neck dataset.

## Known Issues
See the WORC FAQ: https://worc.readthedocs.io/en/latest/static/faq.html
