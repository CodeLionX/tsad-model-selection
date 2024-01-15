#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

######################################################
# Function to check the number of evaluated entities
######################################################

import os
from tsadams.utils.utils import get_args_from_cmdline


# DATASETS = ['anomaly_archive', 'smd']
DATASETS = ['autotsad']


def main(datasets=DATASETS):
    args = get_args_from_cmdline()

    total_models = 0
    for d, dataset in enumerate(datasets):
        n_evaluated_models = 0
        if not os.path.exists(os.path.join(args['results_path'], dataset)):
            print(f'No models evaluated for dataset {dataset}')
        else:
            n_evaluated_models = int(
                len(os.listdir(os.path.join(args['results_path'], dataset))))
            print(f'Total entities evaluated in {dataset} = {n_evaluated_models}')

        total_models = total_models + n_evaluated_models
    print(f'Total number of entities evaluated = {total_models}')


if __name__ == '__main__':
    main()
