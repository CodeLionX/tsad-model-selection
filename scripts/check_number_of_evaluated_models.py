#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

######################################################
# Function to check the number of evaluated entities
######################################################

import os
from model_trainer.entities import ANOMALY_ARCHIVE_ENTITIES, MACHINES, MSL_CHANNELS, SMAP_CHANNELS

DATASETS = ['anomaly_archive', 'smd', 'msl', 'smap'] 
ENTITIES = [ANOMALY_ARCHIVE_ENTITIES, MACHINES, MSL_CHANNELS, SMAP_CHANNELS]
EVALUATED_MODEL_BASE_PATH = r'/home/ubuntu/efs/results/'

total_models = 0
for d, dataset in enumerate(DATASETS): 
    n_evaluated_models = 0
    if not os.path.exists(os.path.join(EVALUATED_MODEL_BASE_PATH, dataset)):
        print(f'No models evaluated for dataset {dataset}')
    else: 
        n_evaluated_models = int(len(os.listdir(os.path.join(EVALUATED_MODEL_BASE_PATH, dataset))))
        print(f'Total entities evaluated in {dataset} = {n_evaluated_models}')
    
    total_models = total_models + n_evaluated_models
print(f'Total number of entities evaluated = {total_models}')