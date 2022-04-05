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

import json
from WORC import BasicWORC
import os
import fastr
import argparse
from default_experiments import (
    get_source_data_HN, get_source_data_WORC,
    valid_datasets, CT_datasets, MRI_datasets
    )


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
    parser.add_argument('-use_smac', '--use_smac', metavar='use_smac',
                        dest='use_smac', type=str, required=False,
                        help='If True, use Bayesian optimization through SMAC.')
    parser.add_argument('-smac_budget', '--smac_budget', metavar='smac_budget',
                        dest='smac_budget', type=str, required=False,
                        help='If using SMAC, use either low, medium, or high budget.')
    parser.add_argument('-ensemble_method', '--ensemble_method', metavar='ensemble_method',
                        dest='ensemble_method', type=str, required=False,
                        help='Determine ensembling method.')
    parser.add_argument('-ensemble_size', '--ensemble_size', metavar='ensemble_size',
                        dest='ensemble_size', type=str, required=False,
                        help='Determine ensembling size, if required by ensembling method to set.')
    parser.add_argument('-RS_iterations', '--RS_iterations', metavar='RS_iterations',
                        dest='RS_iterations', type=str, required=False,
                        help='Number of random search iterations, if random search is used.')
    parser.add_argument('-radiomics_sota', '--radiomics_sota', metavar='radiomics_sota',
                        dest='radiomics_sota', type=str, required=False,
                        help='Use a radiomics SOTA: only LASSO + logistic regression.')
    parser.add_argument('-verbose', '--verbose', metavar='verbose', dest='verbose',
                        type=str, required=False,
                        help='Verbose')
    args = parser.parse_args()

    # Convert strings to Booleans
    if args.coarse == 'True':
        args.coarse = True

    if args.verbose == 'True':
        args.verbose = True

    if args.eval == 'True':
        args.eval = True

    # Run the experiment
    run_automl_experiment(
        dataset=args.dataset,
        coarse=args.coarse,
        name=args.name,
        add_evaluation=args.eval,
        verbose=args.verbose,
        use_smac=args.use_smac,
        smac_budget=args.smac_budget,
        ensemble_method=args.ensemble_method,
        ensemble_size=args.ensemble_size,
        RS_iterations=args.RS_iterations,
        radiomics_sota=args.radiomics_sota
        )


def run_automl_experiment(dataset='Lipo', coarse=False, name=None,
                          add_evaluation=True, verbose=True, use_smac=False,
                          smac_budget='low', ensemble_method='top_N',
                          ensemble_size=100, RS_iterations=1000,
                          radiomics_sota=False):
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
            'AssumeSameImageAndMaskMetadata': 'True',
        }}
    experiment.add_config_overrides(overrides)

    # We fix the random seed of the train-test cross-validation splits to
    # facilitate reproducbility
    overrides = {
        'CrossValidation': {
            'fixed_seed': 'True'
        }}
    experiment.add_config_overrides(overrides)

    # Reduce the number of train-test cross-validation iterations to
    # reduce the computational burden for these comparison experiments
    overrides = {
        'CrossValidation':{
            'N_iterations': '20',
        }}
    experiment.add_config_overrides(overrides)

    # AutoML Parameters
    if use_smac:
        # Use Baysian optimization through use_smac
        if smac_budget == 'low':
            overrides = {
                'SMAC':
                    {
                        'use':  'True',
                        # NOTE: if you have more cores available, change this to speed up the optimization
                        'n_smac_cores': '60',
                        'budget_type': 'time',
                        'budget': '36',
                        'init_method': 'random',
                        'init_budget': '1',
                    },
                }
        elif smac_budget == 'medium':
            overrides = {
                'SMAC':
                    {
                        'use':  'True',
                        # NOTE: if you have more cores available, change this to speed up the optimization
                        'n_smac_cores': '90',
                        'budget_type': 'time',
                        'budget': '355',
                        'init_method': 'random',
                        'init_budget': '5',
                    },
                }
        elif smac_budget == 'high':
            overrides = {
                'SMAC':
                    {
                        'use':  'True',
                        # NOTE: if you have more cores available, change this to speed up the optimization
                        'n_smac_cores': '120',
                        'budget_type': 'time',
                        'budget': '1842',
                        'init_method': 'random',
                        'init_budget': '40',
                    },
                }
        else:
            raise ValueError(f'Budget for SMAC should be low, medium, or high: received {smac_budget}.')
        experiment.add_config_overrides(overrides)

    # Use radiomics SOTA: PyRadiomics features, LASSO, LR, no ensembling
    if radiomics_sota:
        overrides = {
            # One feature extraction toolbox: PyRadiomics
            'General':
                {
                    'FeatureCalculators': '[pyradiomics/Pyradiomics:1.0]'
                },
            # # One imputation strategy: median
            'Imputation':
                {
                    'strategy': 'median'
                },
            # One feature selection method: LASSO
            'Featsel':
                {
                    'Variance': '0.0',
                    'GroupwiseSearch': 'False',
                    'SelectFromModel': '1.0',
                    'SelectFromModel_estimator': 'Lasso',
                    'UsePCA': '0.0',
                    'StatisticalTestUse': '0.0',
                    'ReliefUse': '0.0'
                },
            # No resampling
            'Resampling':
                {'Use': '0.00'},
            # One classification method: logistic regression
            'Classification':
                {'classifiers': 'LR'},
            # No ensembling
            'Ensemble':
                {
                    'Method': 'top_N',
                    'Size': '1'
                }
            }
        experiment.add_config_overrides(overrides)

    # Randomized search overrides
    if RS_iterations is not None:
        overrides = {
            'HyperOptimization':
                {
                    'N_iterations': f'{RS_iterations}'
                }
            }
        experiment.add_config_overrides(overrides)

    # Ensembling overrides
    if ensemble_method is not None:
        overrides = {
            'Ensemble':
                {
                    'Method': f'{ensemble_method}',
                    'Size': f'{ensemble_size}'
                }
            }
        experiment.add_config_overrides(overrides)

    # Set all sources
    experiment.images_train.append(images)
    experiment.segmentations_train.append(segmentations)
    experiment.labels_file_train = label_file
    experiment.predict_labels([label_to_predict])

    # Run experiment
    if add_evaluation:
        experiment.add_evaluation()

    experiment.execute()

    # WORC outputs multiple evaluation tools. Here, we only
    # check the performance metrics to see if the experiment finished succesfully
    outputfolder = os.path.join(fastr.config.mounts['output'], f'WORC_{name}')
    performance_file = os.path.join(outputfolder, 'performance_all_0.json')
    if not os.path.exists(performance_file):
        raise ValueError('No performance file found: your network has failed.')

    with open(performance_file, 'r') as fp:
        performance = json.load(fp)

    # Print the output performance
    print("\n Performance:")
    stats = performance['Statistics']
    for k, v in stats.items():
        print(f"\t {k} {v}.")


if __name__ == '__main__':
    main()
