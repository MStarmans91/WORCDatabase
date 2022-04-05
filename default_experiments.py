#!/usr/bin/env python

# Copyright 2016-2022 Biomedical Imaging Group Rotterdam, Departments of
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
from WORC import BasicWORC
import pandas as pd
import os
import fastr
import argparse
import json

# General settings
valid_datasets = ['HN', 'Lipo', 'Desmoid', 'GIST', 'Liver', 'CRLM', 'Melanoma']
CT_datasets = ['HN', 'GIST', 'CRLM', 'Melanoma']
MRI_datasets = ['Lipo', 'Desmoid', 'Liver']


def main():
    parser = argparse.ArgumentParser(description='WORC Database experiments')
    parser.add_argument('-dataset', '--dataset', metavar='dataset',
                        dest='dataset', type=str, required=True,
                        help='Name of dataset to be used for experiment')
    parser.add_argument('-coarse', '--coarse', metavar='coarse', dest='coarse',
                        type=str, required=False,
                        help='Determine whether to run a coarse experiment or not')
    parser.add_argument('-name', '--name', metavar='name',
                        dest='name', type=str, required=False,
                        help='Determine name of the experiment')
    parser.add_argument('-eval', '--eval', metavar='eval', dest='eval',
                        type=str, required=False,
                        help='Determine wheter to add evaluation or not')
    parser.add_argument('-verbose', '--verbose', metavar='verbose', dest='verbose',
                        type=str, required=False,
                        help='Verbose')
    args = parser.parse_args()

    # Convert strings to Booleans
    if args.coarse == 'True':
        args.coarse = True
    else:
        args.coarse = False

    if args.verbose == 'True':
        args.verbose = True
    else:
        args.verbose = False

    if args.eval == 'True':
        args.eval = True
    else:
        args.eval = False

    # Run the experiment
    run_experiment(
        dataset=args.dataset,
        coarse=args.coarse,
        name=args.name,
        add_evaluation=args.eval,
        verbose=args.verbose)


def get_source_data_HN(verbose=True, csv_label_file=None):
    """Get the source data for WORC experiment on the Head and Neck Cancer.

    Dataset from https://xnat.bmia.nl/data/projects/stwstrategyhn1
    """
    # Some settings
    xnat_url = 'https://xnat.bmia.nl'
    project_name = 'stwstrategyhn1'
    this_directory = os.path.dirname(os.path.abspath(__file__))
    if csv_label_file is None:
        csv_label_file = os.path.join(this_directory, 'pinfo_HN.csv')

    # Connect to XNAT and retreive project
    with xnat.connect(xnat_url) as session:
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


def get_source_data_WORC(dataset="Lipo", verbose=True, csv_label_file=None):
    """Get the source data for a WORC experiment using the WORC database.

    Dataset from https://xnat.bmia.nl/data/projects/worc
    """
    # Some settings
    xnat_url = 'https://xnat.bmia.nl'
    project_name = 'worc'
    this_directory = os.path.dirname(os.path.abspath(__file__))
    if csv_label_file is None:
        csv_label_file = os.path.join(this_directory, f'pinfo_{dataset}.csv')

    # Connect to XNAT and retreive project
    with xnat.connect(xnat_url) as session:
        project = session.projects[project_name]

        # Loop over patients
        images = dict()
        segmentations = dict()
        ground_truths = dict()
        for subject_name in project.subjects:
            subject = project.subjects[subject_name]
            label = subject.label
            subject_dataset = subject.fields['dataset']
            if subject_dataset != dataset:
                # This subject belongs to a different dataset than the one requested
                continue

            # Obtain the clinical ground_truth
            ground_truth = int(subject.fields['diagnosis_binary'])

            # Get the scan and mask, assuming subject has one experiment and one scan
            experiment = subject.experiments[0]
            scan = experiment.scans[0]

            if dataset in ['Melanoma', 'CRLM']:
                # Multiple lesions per patient, thus loop over segmentations
                image_uri = scan.resources['NIFTI'].files['image.nii.gz'].external_uri()
                for file_name in scan.resources['NIFTI'].files:
                    # Check if file is an image or segmentation
                    if 'segmentation' in file_name:
                        # CRLM: only include CNN segmentations for default experiment
                        if dataset == 'CRLM' and '_CNN' not in file_name:
                            continue

                        segmentation_uri = scan.resources['NIFTI'].files[file_name].external_uri()

                        # Add found sources to data dicitonaries
                        images[label] = image_uri
                        segmentations[label] = segmentation_uri
                        ground_truths[label] = ground_truth
                        if verbose:
                            print(f"\tIncluding patient {label}, diagnosis {ground_truth}, segmentation {file_name}.")

            else:
                # There are patients where the data is organized differently
                if subject.label == 'Lipo-073':
                    print("Patient Lipo-073 has two lesions, including both.")
                    image_uri = scan.resources['NIFTI'].files['image.nii.gz'].external_uri()

                    # First lesion: Lipoma
                    segmentation_uri = scan.resources['NIFTI'].files['segmentation_Lipoma.nii.gz'].external_uri()
                    images[label] = image_uri
                    segmentations[label] = segmentation_uri
                    ground_truths[label] = 0
                    if verbose:
                        print(f"\tIncluding patient {label}, diagnosis 0.")

                    # Second lesion: WDLPS
                    segmentation_uri = scan.resources['NIFTI'].files['segmentation_WDLPS.nii.gz'].external_uri()
                    ground_truth = 1

                elif subject.label == 'GIST-018':
                    print("Patient GIST-018 has two lesions, including both.")

                    # First lesion
                    image_uri = scan.resources['NIFTI'].files['image_lesion_0.nii.gz'].external_uri()
                    segmentation_uri = scan.resources['NIFTI'].files['segmentation_lesion_0.nii.gz'].external_uri()
                    images[label] = image_uri
                    segmentations[label] = segmentation_uri
                    ground_truths[label] = ground_truth
                    if verbose:
                        print(f"\tIncluding patient {label}, diagnosis 0.")

                    # Second lesion
                    image_uri = scan.resources['NIFTI'].files['image_lesion_1.nii.gz'].external_uri()
                    segmentation_uri = scan.resources['NIFTI'].files['segmentation_lesion_1.nii.gz'].external_uri()

                else:
                    image_uri = scan.resources['NIFTI'].files['image.nii.gz'].external_uri()
                    segmentation_uri = scan.resources['NIFTI'].files['segmentation.nii.gz'].external_uri()
                    if verbose:
                        print(f"\tIncluding patient {label}, diagnosis {ground_truth}.")

                # Add found sources to data dicitonaries
                images[label] = image_uri
                segmentations[label] = segmentation_uri
                ground_truths[label] = ground_truth

    # Convert diagnosis labels to a label CSV file
    df = pd.DataFrame({'Patient': list(ground_truths.keys()),
                       'Diagnosis': list(ground_truths.values())})
    df.to_csv(csv_label_file, index=False)

    return images, segmentations, csv_label_file


def run_experiment(dataset='Lipo', coarse=False, name=None,
                   add_evaluation=True, verbose=True):
    """Run a radiomics experiment using WORC on one of eight public datasets.

    Parameters
    ----------
    dataset: string, default Lipo
        Determine which dataset to use.
         Valid values: Lipo, Desmoid, GIST, Liver, CRLM, Melanoma, HN

    coarse: boolean, default False
        Determine whether to run a coarse or full experiment.

    name: string, default None
        Name of the experiment to be used in the output. If None, the dataset
        name will be used.

    add_evaluation: boolean, default True
        Decide whether additional evaluation tools will be run, see
        https://worc.readthedocs.io/en/latest/static/user_manual.html#evaluation-of-your-network

    """
    # Check if dataset is valid
    if dataset not in valid_datasets:
        raise KeyError(f"{dataset} is not a valid dataset, should be one of {valid_datasets}.")

    # To Do: check if the given ensembling method is valid

    print(f"Running experiment for dataset {dataset}.")

    # Create an experiment for binary classification
    if name is None:
        name = dataset

    experiment = BasicWORC(name)
    experiment.binary_classification(coarse=coarse)

    # Look for the source data
    print(f"Scanning for source data on XNAT.")
    if dataset == 'HN':
        images, segmentations, label_file = get_source_data_HN(verbose=verbose)
        label_to_predict = 'Tstage'
    else:
        images, segmentations, label_file =\
            get_source_data_WORC(dataset=dataset, verbose=verbose)
        label_to_predict = 'Diagnosis'

    # Check modality
    if dataset in CT_datasets:
        print(f"Dataset {dataset} is CT, add for fingerprinting.")
        experiment.set_image_types(['CT'])

    elif dataset in MRI_datasets:
        print(f"Dataset {dataset} is MRI, enabling image normalization.")
        experiment.set_image_types(['MRI'])

    # NOTE: PyRadiomics might throw an error due to slight differences
    # in metadata of the image and mask. Thus, we assume they
    # have the same
    overrides = {
        'General': {
            'AssumeSameImageAndMaskMetadata': 'True'
        }}
    experiment.add_config_overrides(overrides)

    # We fix the random seed of the train-test cross-validation splits to
    # facilitate reproducbility
    overrides = {
        'CrossValidation': {
            'fixed_seed': 'True'
        }}
    experiment.add_config_overrides(overrides)

    # Set all sources
    experiment.images_train.append(images)
    experiment.segmentations_train.append(segmentations)
    experiment.labels_file_train = label_file
    experiment.predict_labels([label_to_predict])

    # Run experiment
    if add_evaluation:
        experiment.add_evaluation()

    experiment.set_multicore_execution()
    experiment.execute()

    # WORC outputs multiple evaluation tools. Here, we only
    # check the performance metrics to see if the experiment finished succesfully
    outputfolder = os.path.join(fastr.config.mounts['output'], name)
    performance_file = os.path.join(outputfolder, 'performance_all_0.json')
    if not os.path.exists(performance_file):
        raise ValueError(f'No performance file {performance_file} found: your network has failed.')

    with open(performance_file, 'r') as fp:
        performance = json.load(fp)

    # Print the output performance
    print("\n Performance:")
    stats = performance['Statistics']
    for k, v in stats.items():
        print(f"\t {k} {v}.")


if __name__ == '__main__':
    main()
