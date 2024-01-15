#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#######################################
# Script to train models on all the datasets/entities
#######################################

import traceback
from joblib import Parallel, delayed
from tsadams.model_trainer.train import TrainModels
from tsadams.model_trainer.entities import ANOMALY_ARCHIVE_ENTITIES, MACHINES
from tsadams.utils.utils import get_args_from_cmdline
from tsadams.utils.set_all_seeds import set_all_seeds


def train_model_wrapper(dataset, entity, args):
    print()
    print(42 * "=")
    print(f"Training models on entity: {entity}")
    print(42 * "=")
    model_trainer = TrainModels(dataset=dataset,
                                entity=entity,
                                downsampling=args['downsampling'],
                                min_length=args['min_length'],
                                root_dir=args['dataset_path'],
                                training_size=args['training_size'],
                                overwrite=args['overwrite'],
                                verbose=args['verbose'],
                                save_dir=args['trained_model_path'])
    try:
        model_trainer.train_models(model_architectures=args['model_architectures'])
    except:  # Handle exceptions to allow continue training
        print(f'Traceback for Entity: {entity} Dataset: {dataset}')
        print(traceback.format_exc())
    print(42 * "=")


def main(datasets=['anomaly_archive', 'smd'], entities=[ANOMALY_ARCHIVE_ENTITIES, MACHINES]):
    args = get_args_from_cmdline()
   
    print('Set all seeds!')
    set_all_seeds(args['random_seed']) # Reduce randomness

    for d_i, dataset in enumerate(datasets):
        _ = Parallel(n_jobs=args['n_jobs'])(delayed(train_model_wrapper)(dataset, entity, args) for entity in entities[d_i])


if __name__ == '__main__':
    main()    