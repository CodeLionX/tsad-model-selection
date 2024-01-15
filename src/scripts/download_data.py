#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#######################################
# Script to download the Server Machine and Anomaly Archive datasets
#######################################

from tsadams.datasets.load import load_data
from tsadams.utils.utils import get_args_from_cmdline

def main(datasets=['smd', 'anomaly_archive']):
    args = get_args_from_cmdline()

    for d in datasets:
        _ = load_data(dataset=d,
                      group='train',
                      entities=None,
                      downsampling=None,
                      min_length=None,
                      root_dir=args['dataset_path'],
                      verbose=args['verbose'])


if __name__ == '__main__':
    main()    