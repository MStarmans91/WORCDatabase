#!/usr/bin/env python

# Copyright 2016-2021 Biomedical Imaging Group Rotterdam, Departments of
# Medical Informatics and Radiology, Erasmus MC, Rotterdam, The Netherlands
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import xnat
import WORC
import pandas as pd
import os
import fastr


valid_datasets = ['HN']
CT_datasets = ['HN']


def get_source_data_HN(verbose=True, csv_label_file=None):
    """
    Get the source data for WORC experiment on the Head and Neck Cancer
    dataset from https://xnat.bmia.nl/data/projects/stwstrategyhn1
    """
    # Some settings
    xnat_url = 'https://xnat.bmia.nl'
    project_name = 'stwstrategyhn1'
    this_directory = os.path.dirname(os.path.abspath(__file__))
    if csv_label_file is None:
        csv_label_file = os.path.join(this_directory, 'pinfo_HN.csv')

    # Connect to XNAT and retreive project
    session = xnat.connect(xnat_url)
    project = session.projects[project_name]

    # Loop over patients
    images = dict()
    segmentations = dict()
    tstages = dict()
    for subject_name in project.subjects:
        subject = project.subjects[subject_name]
        has_GTV1 = False
        has_image_Nifti = False

        # Check if subject has a CT scan
        for experiment_name in subject.experiments:
            experiment = subject.experiments[experiment_name]

            if experiment.session_type is None:  # some files in project don't have _CT postfix
                if verbose:
                    print(f"\tSkipping patient {subject.label}, experiment {experiment.label}: type is not CT but {experiment.session_type}.")
                continue

            if '_CT' not in experiment.session_type:
                if verbose:
                    print(f"\tSkipping patient {subject.label}, experiment {experiment.label}: type is not CT but {experiment.session_type}.")
                continue

            # Patient has CT scan, check for scans with a GTV-1 and image NIFTI
            for scan_name in experiment.scans:
                scan = experiment.scans[scan_name]
                for res in scan.resources:
                    resource_label = scan.resources[res].label
                    if resource_label == 'NIFTI':
                        for file_name in scan.resources[res].files:
                            if 'mask_GTV-1.nii.gz' in file_name:
                                # Patient has GTV1 mask
                                has_GTV1 = True

                                # Save uri to GTV1 mask
                                GTV1_uri = scan.resources[res].files[file_name].external_uri()

                            elif 'image.nii.gz' in file_name:
                                # Patient has image NIFTI
                                has_image_Nifti = True

                                # Save uri to image NIFTI
                                image_uri = scan.resources[res].files[file_name].external_uri()

            if not has_GTV1:
                print(f"\tExcluding patient {subject.label}: no GTV1 mask.")
                continue

            if not has_image_Nifti:
                print(f"\tExcluding patient {subject.label}: no image Nifti.")
                continue

            # Get the T-stage and binarize
            tstage = subject.fields['clin_t']
            if tstage in ['0', '1', '2']:
                tstage_binary = 0
            elif tstage in ['3', '4']:
                tstage_binary = 1
            else:
                print(f"\tExcluding patient {subject.label}: unknown t-stage {tstage}.")
                continue

            # Patient has GTV1 mask, image Nifti, and known t-stage, so include
            if verbose:
                print(f"\tIncluding patient {subject.label}, T-stage {tstage}, binarized to {tstage_binary}.")

            label = subject.label
            images[label] = image_uri
            segmentations[label] = GTV1_uri
            tstages[label] = tstage_binary

    # Convert tstages to a label CSV file
    df = pd.DataFrame({'Patient': list(tstages.keys()),
                       'Tstage': list(tstages.values())})
    df.to_csv(csv_label_file, index=False)

    return images, segmentations, csv_label_file


def run_experiment(dataset='HN', coarse=True, name=None,
                   add_evaluation=False):
    """Run a radiomics experiment using WORC on one of eight public datasets.

    Parameters
    ----------
    dataset: string, default HN
        Determine which dataset to use. Valid values: HN

    coarse: boolean, default False
        Determine whether to run a coarse or full experiment.

    name: string, default None
        Name of the experiment to be used in the output. If None, the dataset
        name will be used.

    add_evaluation: boolean, default False
        Decide whether additional evaluation tools will be run, see
        https://worc.readthedocs.io/en/latest/static/user_manual.html#evaluation-of-your-network

    """
    # Check if dataset is valid
    if dataset not in valid_datasets:
        raise KeyError(f"{dataset} is not a valid dataset, should be one of {valid_datasets}.")

    print(f"Running experiment for dataset {dataset}.")

    # Create an experiment
    if name is None:
        name = dataset

    network = WORC.WORC(name)

    # Create the configuration for WORC
    config = network.defaultconfig()

    # Look for the source data
    if dataset == 'HN':
        images, segmentations, label_file = get_source_data_HN()
        config['Labels']['label_names'] = 'Tstage'

    # Check modality
    if dataset in CT_datasets:
        print(f"Dataset {dataset} is CT, disabling image normalization.")
        config['Preprocessing']['Normalize'] = 'False'
        config['ImageFeatures']['image_type'] = 'CT'

    # Set all sources
    network.images_train.append(images)
    network.segmentations_train.append(segmentations)
    network.labels_train.append(label_file)
    network.configs.append(config)

    # Run experiment
    network.build()
    if add_evaluation:
        network.add_evaluation(label_type=config['Labels']['label_names'])
    network.set()
    network.execute()

    # Check the output performance, if the experiment finished succesfully
    outputfolder = os.path.join(fastr.config.mounts['output'], name)


if __name__ == '__main__':
    run_experiment('HN')
