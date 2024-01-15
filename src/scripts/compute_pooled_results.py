#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#######################################
# Script to evaluate model selection performance
#######################################

import os
import pickle as pkl
from sklearn.model_selection import ParameterGrid

from tsadams.utils.utils import get_args_from_cmdline
from tsadams.utils.set_all_seeds import set_all_seeds
from tsadams.evaluation.evaluation import get_pooled_aggregate_stats


def set_eval_params(args, datasets):
    base_config = {
        'save_dir': [args['results_path']],
        'evaluation_metric': [args['evaluation_metric']],
        'n_validation_splits': [args['n_validation_splits']],
        'n_neighbors': [args['n_neighbors']],
        'random_state': [args['random_seed']],
        'n_splits': [args['n_splits']],
        'sliding_window': [args['sliding_window']],
        'metric': [args['metric']],
        'top_k': [args['top_k']],
        'use_all_ranks': [args['use_all_ranks']],
        'top_kr': [args['top_kr']],
        'n_jobs': [args['n_jobs']]
    }
    evaluation_params = []

    for d in datasets:
        if d == 'autotsad':
            evaluation_params.append({
                'dataset': ['autotsad'],
                'data_family': [
                    'GutenTAG',
                    'IOPS',
                    'KDD-TSAD',
                    'MGAB',
                    'NAB',
                    'NASA-MSL',
                    'NASA-SMAP',
                    'NormA',
                    'SAND',
                    'TSB-UAD-artificial',
                    'TSB-UAD-synthetic',
                    'WebscopeS5',

                ],
                **base_config
            })
        elif d == 'anomaly_archive':
            evaluation_params.append({
                'dataset': ['anomaly_archive'],
                'data_family': [
                    'Atrial Blood Pressure (ABP)',
                    'Electrocardiogram (ECG) Arrhythmia',
                    'Insect Electrical Penetration Graph (EPG)',
                    'Power Demand',
                    'NASA Data',
                    'Gait',
                    'Respiration Rate (RESP)',
                    'Acceleration Sensor Data',
                    'Air Temperature',
                ],
                **base_config
            })
        else:
            evaluation_params.append({
                'dataset': [d],
                'data_family': [d.upper()],
                **base_config
            })
    return evaluation_params


def main(datasets=['smd', 'anomaly_archive']):
    args = get_args_from_cmdline()
   
    set_all_seeds(args['random_seed']) # Reduce randomness

    evaluation_params = set_eval_params(args, datasets)

    aggregate_stats = {}
    for params in list(ParameterGrid(evaluation_params)):
        print(42 * "=")
        print(f"Pooling results for dataset: {params['dataset']}")
        print(42 * "=")

        if not args['overwrite']:
            if os.path.exists(os.path.join(args['results_path'], f"aggregate_stats_{params['dataset']}.pkl")):
                print(f"Results for dataset {args['dataset']} are already pooled!")
                continue

        stats = get_pooled_aggregate_stats(**params)

        if 'data_family' in params.keys():
            aggregate_stats[params['data_family']] = stats
        else:
            aggregate_stats[params['dataset'].upper()] = stats

        with open(os.path.join(args['results_path'], f"aggregate_stats_{params['dataset']}.pkl"), 'wb') as f:
            pkl.dump(aggregate_stats, f)


if __name__ == '__main__':
    main()    
