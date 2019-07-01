# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:15:35 2019

@author: michaelek
"""
import os
import pandas as pd
from hydrointerp import Interp
from ecandbparams import sql_arg
from pdsql import mssql
import parameters as param
import eto_estimates as eto

pd.options.display.max_columns = 10

###################################
### Parameters

sql1 = sql_arg()

swaz_gis_dict = sql1.get_dict('swaz_gis')

## Selection of specific SWAZ groups - remove for final method
where_in = {'where_in': {'ZONE_GROUP_NAME': param.swaz_grps}}

swaz_gis_dict.update(where_in)

eto_swaz_csv = 'eto_swaz_{}.csv'.format(param.run_time)


#################################
### Estimate ETo at swaz locations
print('ETo at swaz locations')

try:
    swaz_eto = pd.read_csv(os.path.join(param.inputs_path, eto_swaz_csv), parse_dates=['time'], infer_datetime_format=True)
    print('-> loaded from local file')

except:
    print('-> Estimate ETo at SWAZs via spatial interp')
    ## Read in data

    print('Read in SWAZ data')

    swaz1 = mssql.rd_sql(**swaz_gis_dict)

    swaz2 = swaz1.drop('geometry', axis=1).copy()

    swaz2['x'] = swaz1.centroid.x
    swaz2['y'] = swaz1.centroid.y

    ## Estimate ETo

    eto1 = eto.eto0[(eto.eto0.time >= param.from_date) & (eto.eto0.time <= param.to_date)].copy()

    interp1 = Interp(eto1, 'time', 'x', 'y', 'ETo', 4326)
    swaz_eto1 = interp1.points_to_points(swaz2, 2193, method='cubic', min_val=0)

    swaz_eto = pd.merge(swaz2, swaz_eto1.reset_index(), on=['x', 'y'])

    swaz_eto.to_csv(os.path.join(param.inputs_path, eto_swaz_csv), index=False)


#################################
### Testing

#df = eto.eto0.copy()
#time_name = 'time'
#x_name = 'x'
#y_name = 'y'
#data_name = 'ETo'
#point_data = swaz2.copy()
#from_crs = 4326
#to_crs = 2193
#method = 'linear'
#digits = 2
#min_val = None

#swaz_etpo1 = points_to_points(eto.eto0, 'time', 'x', 'y', 'ETo', swaz2, 4326, 2193)


