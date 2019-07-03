# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:25:41 2019

@author: michaelek
"""
from flownat import FlowNat
import pandas as pd

pd.options.display.max_columns = 10



from_date='2010-07-01'
to_date='2018-06-30'
min_gaugings=8
rec_data_code='Primary'
output_path=r'E:\ecan\git\Cwms\Ecan.Cwms.Ashburton\results'
input_sites = ['69618', '69635', '69616', '69615', '168833']
#input_sites = ['69618', '69635', '69616', '69615']
#input_sites = ['168833']


self = FlowNat(from_date, to_date, input_sites=input_sites, load_rec=True)
catch_del1 = self.catch_del()
allo_wap = self.upstream_takes()
usage_rate = self.usage_est()

flow = self.flow_est()

nat_flow = self.naturalisation()























