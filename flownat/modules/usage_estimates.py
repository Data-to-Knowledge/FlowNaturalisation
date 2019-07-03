# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:47:14 2019

@author: michaelek
"""
import os
import pandas as pd
from allotools import AlloUsage
from pdsql import mssql
import upstream_takes as takes
from parameters import param

pd.options.display.max_columns = 10

###################################
### Parameters

grp = ['Wap', 'WaterUse']
datasets = ['Allo', 'RestrAllo', 'Usage']

use_type_dict = {'industrial': 'other', 'municipal': 'other'}

server = param['ts_server']
database = 'hydro'
sites_table = 'ExternalSite'
results_path = param['results_path']

swaz_mon_ratio_csv = 'swaz_mon_ratio_{}.csv'.format(param['run_time'])
allo_usage_wap_swaz_csv = 'allo_usage_wap_swaz_{}.csv'.format(param['run_time'])
wap_sw_mon_usage_csv = 'wap_sw_monthly_usage_rate_{}.csv'.format(param['run_time'])
wap_sw_daily_usage_hdf = 'wap_sw_daily_usage_rate_{}.hd5'.format(param['run_time'])

###################################
### Read existing usage data
print('Read in usage estimates')

try:
    res_swaz5 = pd.read_csv(os.path.join(results_path, swaz_mon_ratio_csv))
    usage4 = pd.read_csv(os.path.join(results_path, allo_usage_wap_swaz_csv), parse_dates=['Date'], infer_datetime_format=True)
    usage_rate = pd.read_csv(os.path.join(results_path, wap_sw_mon_usage_csv), parse_dates=['Date'], infer_datetime_format=True)

    print('-> loaded from local files')

except:

    print('-> Processing usage data from the databases')

    allo1 = AlloUsage(param['from_date'], param['to_date'], site_filter={'SwazGroupName': takes.waps_gdf.SwazGroupName.unique().tolist()})

    usage1 = allo1.get_ts(datasets, 'M', grp)

    usage2 = usage1.loc[usage1.SwRestrAllo > 0, ['SwRestrAllo', 'SwUsage']].reset_index().copy()

    usage2.replace({'WaterUse': use_type_dict}, inplace=True)

    usage2[['SwRestrAlloYr', 'SwUsageYr']] = usage2.groupby(['Wap', 'WaterUse', pd.Grouper(key='Date', freq='A-JUN')]).transform('sum')

    sites1 = mssql.rd_sql(server, database, sites_table, ['ExtSiteID', 'SwazGroupName', 'SwazName'], where_in={'ExtSiteID': usage2.Wap.unique().tolist()})
    sites1.rename(columns={'ExtSiteID': 'Wap'}, inplace=True)

    usage0 = pd.merge(sites1, usage2, on='Wap')
    usage0['Mon'] = usage0.Date.dt.month

    usage0['MonRatio'] = usage0.SwUsage/usage0.SwRestrAllo
    usage0['YrRatio'] = usage0.SwUsageYr/usage0.SwRestrAlloYr

    usage0.set_index(['Wap', 'Date', 'WaterUse'], inplace=True)

    filter1 = (usage0['YrRatio'] >= 0.04) & (usage0['YrRatio'] <= 2) & (usage0['MonRatio'] >= 0.001)

    usage3 = usage0[filter1].reset_index().copy()

    res_swaz1 = usage3.groupby(['SwazGroupName', 'SwazName', 'WaterUse', 'Mon']).MonRatio.mean()
    res_grp1 = usage3.groupby(['SwazGroupName', 'WaterUse', 'Mon']).MonRatio.mean()
    res_grp1.name = 'GrpRatio'

    res_grp2 = usage3.groupby(['WaterUse', 'Mon']).MonRatio.mean()
    res_grp2.name = 'GrossRatio'

    all1 = usage0.groupby(['SwazGroupName', 'SwazName', 'WaterUse', 'Mon']).Mon.first()

    res_swaz2 = pd.concat([res_swaz1, all1], axis=1).drop('Mon', axis=1)
    res_swaz3 = pd.merge(res_swaz2.reset_index(), res_grp1.reset_index(), on=['SwazGroupName', 'WaterUse', 'Mon'], how='left')
    res_swaz4 = pd.merge(res_swaz3, res_grp2.reset_index(), on=['WaterUse', 'Mon'], how='left')

    res_swaz4.loc[res_swaz4.MonRatio.isnull(), 'MonRatio'] = res_swaz4.loc[res_swaz4.MonRatio.isnull(), 'GrpRatio']

    res_swaz4.loc[res_swaz4.MonRatio.isnull(), 'MonRatio'] = res_swaz4.loc[res_swaz4.MonRatio.isnull(), 'GrossRatio']

    res_swaz5 = res_swaz4.drop(['GrpRatio', 'GrossRatio'], axis=1).copy()

    ### Estimate monthly usage by WAP

    usage4 = pd.merge(usage0.drop(['MonRatio', 'YrRatio', 'SwRestrAlloYr', 'SwUsageYr'], axis=1).reset_index(), res_swaz5, on=['SwazGroupName', 'SwazName', 'WaterUse', 'Mon'], how='left').set_index(['Wap', 'Date', 'WaterUse'])

    usage4.loc[~filter1, 'SwUsage'] = usage4.loc[~filter1, 'SwRestrAllo'] * usage4.loc[~filter1, 'MonRatio']

    usage_rate = usage4.groupby(level=['Wap', 'Date'])[['SwUsage']].sum().reset_index().copy()
    usage_rate.rename(columns={'SwUsage': 'SwUsageRate'}, inplace=True)

    days1 = usage_rate.Date.dt.daysinmonth
    usage_rate['SwUsageRate'] = usage_rate['SwUsageRate'] / days1 /24/60/60

    usage4.reset_index(inplace=True)

    ### Save results

    res_swaz5.to_csv(os.path.join(results_path, swaz_mon_ratio_csv), index=False)
    usage4.to_csv(os.path.join(results_path, allo_usage_wap_swaz_csv), index=False)
    usage_rate.to_csv(os.path.join(results_path, wap_sw_mon_usage_csv), index=False)


