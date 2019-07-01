# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 09:29:52 2018

@author: michaelek
"""
import os
import yaml
import pandas as pd

#####################################
### Parameters

base_dir = os.path.realpath(os.path.dirname(__file__))

with open(os.path.join(base_dir, 'parameters.yml')) as param:
    param = yaml.safe_load(param)

param['run_time'] = pd.Timestamp.today().strftime('%Y%m%d')

## Paths

results_dir = 'results'
inputs_dir = 'inputs'

param['results_path'] = os.path.join(param['project_path'], results_dir)
param['inputs_path'] = os.path.join(param['project_path'], inputs_dir)

if not os.path.exists(param['results_path']):
    os.makedirs(param['results_path'])

if not os.path.exists(param['inputs_path']):
    os.makedirs(param['inputs_path'])

## Other
param['crc_status'] = ['Issued - Active', 'Terminated - Surrendered', 'Terminated - Expired', 'Terminated - Replaced', 'Terminated - Cancelled']

