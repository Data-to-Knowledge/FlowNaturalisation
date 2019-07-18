# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:25:41 2019

@author: michaelek
"""
from flownat import FlowNat
import pandas as pd

pd.options.display.max_columns = 10

from_date = '2009-07-01'
to_date = '2018-06-30'
min_gaugings = 8
rec_data_code = 'Primary'
output_path = r'S:\Surface Water\shared\projects\ashburton\naturalisation'
input_sites1 = ['68801', '168834']

# Up to 1996-06-30, there was a max of ~9 m3/s taken from the river.
# After that it increases to over 20 m3/s until 2013, then it goes quite high.

########################################
### flow nat


f1 = FlowNat(from_date, to_date, input_sites=input_sites1, output_path=output_path)

#f1 = FlowNat(from_date, to_date, input_sites=input_sites1)

#up1 = f1.upstream_takes()

nat_flow = f1.naturalisation()

for s in input_sites1:
    nat_flow1 = f1.plot(s)

#nat_flow = self.naturalisation()




