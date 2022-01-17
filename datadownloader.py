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
import os
import sys
import shutil
from glob import glob
import argparse

from xnat.exceptions import XNATResponseError

# General settings
valid_datasets = ['Lipo', 'Desmoid', 'GIST', 'Liver', 'CRLM', 'Melanoma']


def main():
    parser = argparse.ArgumentParser(description='WORC Database experiments')
    parser.add_argument('-dataset', '--dataset', metavar='dataset',
                        dest='dataset', type=str, required=False,
                        default='Lipo',
                        help='Name of dataset to be downloaded.')
    parser.add_argument('-datafolder', '--datafolder', metavar='datafolder',
                        dest='datafolder', type=str, required=False,
                        help='Folder to download the dataset to.')
    parser.add_argument('-nsubjects', '--nsubjects', metavar='nsubjects',
                        dest='nsubjects', type=str, required=False,
                        default='all',
                        help='Number of subjects to be downloaded.')
    args = parser.parse_args()

    # Run the experiment
    download_WORCDatabase(dataset=args.dataset,
                          datafolder=args.datafolder,
                          nsubjects=args.nsubjects)


def download_subject(project, subject, datafolder, session, verbose=False):
    """Download data of a single XNAT subject."""
    # Download all data and keep track of resources
    download_counter = 0
    resource_labels = list()
    for e in subject.experiments:
        resmap = {}
        experiment = subject.experiments[e]

        for s in experiment.scans:
            scan = experiment.scans[s]
            print(("\tDownloading patient {}, experiment {}, scan {}.").format(subject.label, experiment.label,
                                                                               scan.id))
            for res in scan.resources:
                resource_label = scan.resources[res].label
                if resource_label == 'NIFTI':
                    # Create output directory
                    outdir = datafolder + '/{}'.format(subject.label)
                    if not os.path.exists(outdir):
                        os.makedirs(outdir)

                    resmap[resource_label] = scan
                    print(f'resource is {resource_label}')
                    scan.resources[res].download_dir(outdir)
                    resource_labels.append(resource_label)
                    download_counter += 1

    # Parse resources and throw warnings if they not meet the requirements
    subject_name = subject.label
    if download_counter == 0:
        print(f'\t[WARNING] Skipping subject {subject_name}: no (suitable) resources found.')
        return False

    if 'NIFTI' not in resource_labels:
        print(f'\t[WARNING] Skipping subject {subject_name}: no NIFTI resources found.')
        return False

    # Reorder files to a easier to read structure
    NIFTI_files = glob(os.path.join(outdir, '*', 'scans', '*', 'resources', 'NIFTI', 'files', '*.nii.gz'))
    for NIFTI_file in NIFTI_files:
        basename = os.path.basename(NIFTI_file)
        shutil.move(NIFTI_file, os.path.join(outdir, basename))

    for folder in glob(os.path.join(outdir, '*')):
        if os.path.isdir(folder):
            shutil.rmtree(folder)

    return True


def download_project(project_name, xnat_url, datafolder, nsubjects='all',
                     verbose=True, dataset='all'):
    """Download data of full XNAT project."""
    # Connect to XNAT and retreive project
    with xnat.connect(xnat_url) as session:
        project = session.projects[project_name]

        # Create the data folder if it does not exist yet
        datafolder = os.path.join(datafolder, project_name)
        if not os.path.exists(datafolder):
            os.makedirs(datafolder)

        subjects_len = len(project.subjects)
        if nsubjects != 'all':
            nsubjects = min(nsubjects, subjects_len)

        subjects_counter = 1
        downloaded_subjects_counter = 0
        for s in range(0, subjects_len):
            s = project.subjects[s]
            if dataset is not 'all':
                # Check if patient belongs to required dataset
                subject_dataset = s.fields['dataset']
                if subject_dataset != dataset:
                    print(f'\t Skipping subject {s.label}: belongs to a different dataset than {dataset}.')
                    continue

            print(f'Processing on subject {subjects_counter}/{subjects_len}')
            subjects_counter += 1

            success = download_subject(project_name, s, datafolder, session,
                                       verbose)
            if success:
                downloaded_subjects_counter += 1

            # Stop downloading if we have reached the required number of subjects
            if downloaded_subjects_counter == nsubjects:
                break

        # Disconnect the session
        session.disconnect()
        if nsubjects != 'all':
            if downloaded_subjects_counter < nsubjects:
                raise ValueError(f'Number of subjects downloaded {downloaded_subjects_counter} is smaller than the number required {nsubjects}.')

        print('Done downloading!')


def download_WORCDatabase(dataset=None, datafolder=None, nsubjects='all'):
    """Download a dataset from the WORC Database.

    Download all Nifti images and segmentations from a dataset from the WORC
    database from https://xnat.bmia.nl/data/projects/worc

    dataset: string, default None
        If None, download the full XNAT project. If string, download one
        of the six datasets. Valid values: Lipo, Desmoid, GIST, Liver, CRLM,
        Melanoma
    """
    # Check if dataset is valid
    if dataset not in valid_datasets:
        raise KeyError(f"{dataset} is not a valid dataset, should be one of {valid_datasets}.")

    if datafolder is None:
        # Download data to path in which this script is located + Data
        cwd = os.getcwd()
        datafolder = os.path.join(cwd, 'Data')
        if not os.path.exists(datafolder):
            os.makedirs(datafolder)

    xnat_url = 'https://xnat.bmia.nl'
    project_name = 'worc'
    download_project(project_name, xnat_url, datafolder, nsubjects=nsubjects,
                     verbose=True, dataset=dataset)


if __name__ == '__main__':
    main()
