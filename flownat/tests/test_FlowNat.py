# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:25:41 2019

@author: michaelek
"""
import pytest
from flownat import FlowNat
import pandas as pd

pd.options.display.max_columns = 10

from_date='2010-07-01'
to_date='2018-06-30'
min_gaugings=8
rec_data_code='Primary'
#output_path=r'E:\ecan\git\Cwms\Ecan.Cwms.Ashburton\results'
input_sites1 = ['69618', '69635', '168833']
input_sites2 = ['69618', '69635']
input_sites3 = ['168833']

########################################
### Tests


def test_init_FlowNat():
    f1 = FlowNat(from_date, to_date, input_sites=input_sites1, catch_del='rec')

    assert len(f1.summ) > 400


@pytest.mark.parametrize('input_sites', [input_sites1, input_sites2, input_sites3])
def test_nat(input_sites):
    f1 = FlowNat(from_date, to_date, input_sites=input_sites)

    nat_flow = f1.naturalisation()

    assert (len(f1.summ) >= 1) & (len(nat_flow) > 2900)




