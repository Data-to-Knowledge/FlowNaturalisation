# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 12:20:30 2019

@author: michaelek
"""
import os
import pandas as pd
import upstream_takes as takes
import usage_estimates as ue
import flow_estimates as flow
from parameters import param

pd.options.display.max_columns = 10

#######################################
### Parameters

flow_site_sw_usage_csv = 'flow_site_sw_usage_rate_{}.csv'.format(param['run_time'])
nat_flow_csv = 'nat_flow_{}.csv'.format(param['run_time'])

#####################################
### Naturalisation

print('Naturalise the flow data')

### Resample to daily rate
waps1 = takes.waps_gdf.drop(['geometry', 'SwazGroupName', 'SwazName'], axis=1).copy()

usage_rate = ue.usage_rate[ue.usage_rate.Wap.isin(waps1.Wap.unique())].copy()

days1 = usage_rate.Date.dt.daysinmonth
days2 = pd.to_timedelta((days1/2).round().astype('int32'), unit='D')

usage_rate0 = usage_rate.copy()

usage_rate0['Date'] = usage_rate0['Date'] - days2

grp1 = usage_rate.groupby('Wap')
first1 = grp1.first()
last1 = grp1.last()

first1.loc[:, 'Date'] = pd.to_datetime(first1.loc[:, 'Date'].dt.strftime('%Y-%m') + '-01')

usage_rate1 = pd.concat([first1, usage_rate0.set_index('Wap'), last1], sort=True).reset_index()

usage_rate1.set_index('Date', inplace=True)

usage_daily_rate = usage_rate1.groupby('Wap').apply(lambda x: x.resample('D').interpolate(method='pchip')['SwUsageRate']).reset_index()

## Combine usage with site data

print('-> Combine usage with site data')

usage_rate3 = pd.merge(waps1, usage_daily_rate.reset_index(), on='Wap')

site_rate = usage_rate3.groupby(['FlowSite', 'Date'])[['SwUsageRate']].sum().reset_index()

site_rate.to_csv(os.path.join(param['results_path'], flow_site_sw_usage_csv), index=False)

## Add usage to flow
print('-> Add usage to flow')

flow1 = flow.flow.stack().reset_index()
flow1.columns = ['Date', 'FlowSite', 'Flow']

flow2 = pd.merge(flow1, site_rate, on=['FlowSite', 'Date'], how='left').set_index(['FlowSite', 'Date']).sort_index()
flow2.loc[flow2.SwUsageRate.isnull(), 'SwUsageRate'] = 0

flow2['NatFlow'] = flow2['Flow'] + flow2['SwUsageRate']

nat_flow = flow2.unstack(0).round(3)

nat_flow.to_csv(os.path.join(param['results_path'], nat_flow_csv))



############################################
### Testing


