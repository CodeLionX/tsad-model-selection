#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from evaluation.evaluation import get_pooled_aggregate_stats
from sklearn.model_selection import ParameterGrid

import sys
sys.path.append('/home/ubuntu/PyMAD/')
from src.pymad.evaluation.numpy import adjusted_precision_recall_f1_auc

evaluation_params = [
    {
        'save_dir': [r'/home/ubuntu/efs/results/'], 
        'dataset':['anomaly_archive'],  
        'data_family':[
            'Acceleration Sensor Data', 
            'Air Temperature', 
            'Atrial Blood Pressure (ABP)', 
            'Electrocardiogram (ECG) Arrhythmia', 
            'Gait', 
            'Insect Electrical Penetration Graph (EPG)', 
            'NASA Data', 
            'Power Demand', 
            'Respiration Rate (RESP)'],  
        'evaluation_metric': ['Best F-1'], 
        'n_validation_splits': [5], 
        'n_neighbors': [[2, 4, 6]],
        'random_state': [13], 
        'n_splits': [100], 
        'metric': ['influence'],
        'top_k': [3],
        'n_jobs': [3]
    },  
    {
        'save_dir': [r'/home/ubuntu/efs/results/'], 
        'dataset':['smd'],  
        'data_family': [''],
        'evaluation_metric': ['Best F-1'], 
        'n_validation_splits': [5], 
        'n_neighbors': [[2, 4, 6]],
        'random_state': [13], 
        'n_splits': [100], 
        'metric': ['influence'],
        'top_k': [3],
        'n_jobs': [3]
    },
]

aggregate_stats = {}
for params in list(ParameterGrid(evaluation_params)):
    try: 
        stats = get_pooled_aggregate_stats(**params)
    except:
        continue

    if 'data_family' in params.keys(): 
        aggregate_stats[params['data_family']] = stats
    else:
        aggregate_stats['smd'] = stats
 
    with open(f"aggregate_stats_{params['dataset']}.pkl", 'wb') as f: 
        pkl.dump(aggregate_stats, f)    
