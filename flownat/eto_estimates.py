# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 08:16:57 2019

@author: michaelek
"""
import os
import pandas as pd
from thinsos import SOS
from hydrointerp import interp2d
from eto import ETo
import parameters as param

pd.options.display.max_columns = 10

###############################
### Parameters

url = 'https://climate-sos.niwa.co.nz'

bbox = [[169, -45.5], [174.2, -41.7]]

observableProperty = {'MTHLY_STATS: MEAN DAILY RADIATION (Global) (MTHLY: MEAN RAD (G))': ['R_s', 1],
                      'MTHLY_STATS: MEAN MAXIMUM TEMPERATURE from daily Maxs (MTHLY: MEAN MAX TEMP)': ['T_max', 1],
                      'MTHLY_STATS: MEAN MINIMUM TEMPERATURE from daily Mins (MTHLY: MEAN MIN TEMP)': ['T_min', 1],
                      'MTHLY_STATS: MEAN VAPOUR PRESSURE (MTHLY: MEAN VP)': ['P', 0.1],
                      'MTHLY_STATS: MEAN WIND SPEED (NO direction) (MTHLY: MEAN WIND SPEED)': ['U_z', 1],
                      'MTHLY_STATS: TOTAL SUNSHINE (MTHLY: TOTAL SUN)': 'n_sun',
                      'MTHLY_STATS: MEAN 9AM RELATIVE HUMIDITY (MTHLY: MEAN 9AM RH)': ['RH_mean', 1]}

niwa_data_csv = 'niwa_met_data_{}.csv'.format(param.run_time)
eto_data_csv = 'eto_data_{}.csv'.format(param.run_time)

##############################
### Read met data

print('Read in met data')

try:
    data2 = pd.read_csv(os.path.join(param.inputs_path, niwa_data_csv), parse_dates=['time'], infer_datetime_format=True)
    print('-> loaded from local file')

except:
    print('-> pulling from NIWA SOS')
    sos1 = SOS(url)

    foi1 = sos1.get_foi(bbox=bbox)

    summ1 = sos1.data_availability.copy()

    summ2 = summ1[summ1.featureOfInterest.isin(foi1.identifier)]

    summ3 = summ2[summ2.observedProperty.isin(observableProperty.keys())].copy()

    data_lst1 = []

    for index, row in summ3.iterrows():
        df1 = sos1.get_observation(row.featureOfInterest, row.observedProperty)
        data_lst1.append(df1)

    data1 = pd.concat(data_lst1).drop(['type', 'procedure', 'uom'], axis=1)

    for key, val in observableProperty.items():
        data1.replace({'observableProperty': {key: val[0]}}, inplace=True)
        data1.loc[data1.observableProperty == key, 'result'] = data1.loc[data1.observableProperty == key, 'result'] * val[1]

    data1.rename(columns={'identifier': 'site', 'observableProperty': 'mtype', 'result': 'value', 'resultTime': 'time'}, inplace=True)

    data2 = data1.groupby(['site', 'mtype', pd.Grouper(key='time', freq='M')]).first().reset_index()

    data2.to_csv(os.path.join(param.inputs_path, niwa_data_csv), index=False)


###############################
### Calc ETo

print('Read in ETo data')

try:
    eto0 = pd.read_csv(os.path.join(param.inputs_path, eto_data_csv), parse_dates=['time'], infer_datetime_format=True)
    print('-> loaded from local file')

except:
    print('-> Calc from Met data')
    eto_lst = []

    data3 = data2.set_index('time')

    grp1 = data2.groupby('site')

    z = 500

    for index, grp in grp1:
        first1 = grp.iloc[0]
        lat = first1['lat']
        lon = first1['lon']

        set1 = grp.pivot_table('value', 'time', 'mtype')
        set2 = set1.dropna(subset=['T_min', 'T_max'])

        eto1 = ETo(set2, 'M', z, lat).eto_fao(interp='time')['ETo_FAO_interp_mm']
        eto1.name = 'ETo'

        eto2 = eto1.reset_index()

        eto2['y'] = lat
        eto2['x'] = lon
        eto2['site'] = index

        eto_lst.append(eto2)

    eto0 = pd.concat(eto_lst)

    eto0.to_csv(os.path.join(param.inputs_path, eto_data_csv), index=False)



