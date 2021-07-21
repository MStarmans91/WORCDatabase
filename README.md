# WORC Database: Downloaders and Experiments
This repository contains scripts to download six public datasets
and reproduce radiomics experiments on these six datasets and two other
public datasets. The six public datasets are described in the following paper:

``Starmans, M. P. A. et al. (2021). The WORC* database: MRI and CT scans, segmentations, and clinical labels for 932 patients from six radiomics studies, In Preparation``

The data used for six of the datasets can be found at https://xnat.bmia.nl/data/projects/worc.
The experiments are described in the following paper:

``Starmans, M. P. A. et al. (2021). Reproducible radiomics through automated machine learning validated on twelve clinical applications, In Preparation``

## License
When using parts of this code or the above datasets, please cite the two
above mentioned papers. Feel free to also cite the code from this repository:

``Martijn P.A. Starmans. WORCDatabase. Zenodo (2021). Available from:  https://github.com/MStarmans91/WORCDatabase. DOI: http://doi.org/10.5281/zenodo.5119040``

For the DOI, visit [![][DOI]][DOI-lnk].

[DOI]: https://zenodo.org/badge/388076660.svg
[DOI-lnk]: https://zenodo.org/badge/latestdoi/388076660

## Installation
The experiments only require the WORC Python package to be installed,
which can be done using pip. In the paper, version 3.4.5 was used:

    pip install "WORC==3.4.5"

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
the docstring of the function for further documentation.

For the six public datasets as published in the Data in Brief paper, the data
is directly fed from XNAT into WORC. For the two previously publicly published
datasets used in the WORC MEDIA paper, some additional steps are performed.

### Glioma dataset
To Do

### Head and neck dataset
To Do

## Known Issues
See the WORC FAQ: https://worc.readthedocs.io/en/latest/static/faq.html
