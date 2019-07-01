# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:44:32 2019

@author: michaelek
"""
import os
import numpy as np
import pandas as pd
from pdsql import mssql
import geopandas as gpd
import upstream_takes as takes
from hydrolm import LM
from gistools import vector
from parameters import param

pd.options.display.max_columns = 10

#######################################
### Parameters

server = param['ts_server']
database = 'hydro'
site_table = 'ExternalSite'
ts_table = 'TSDataNumericDaily'
ts_summ_table = 'TSDataNumericDailySumm'
dataset_type_table = 'DatasetType'
results_path = param['results_path']

buffer_dis = param['buffer_dis']

flow_csv = 'flow_data_{}.csv'.format(param['run_time'])
reg_flow_csv = 'reg_flow_{}.csv'.format(param['run_time'])


#####################################
### Estimate flow data

print('Estimate flow data')

try:

    flow = pd.read_csv(os.path.join(param.results_path, flow_csv), parse_dates=['DateTime'], infer_datetime_format=True, index_col='DateTime')
    reg_df = pd.read_csv(os.path.join(param.results_path, reg_flow_csv))

    print('-> loaded from local files')

except:

    print('-> processing with regressions')

    ## Read in data
    datasets = mssql.rd_sql(server, database, dataset_type_table, ['DatasetTypeID', 'CTypeID'], where_in={'FeatureID': [1], 'MTypeID': [2], 'CTypeID': [1, 2], 'DataCodeID': [1]})

    site_summ1 = mssql.rd_sql(server, database, ts_summ_table, where_in={'DatasetTypeID': datasets.DatasetTypeID.tolist()})
    site_summ1.FromDate = pd.to_datetime(site_summ1.FromDate)
    site_summ1.ToDate = pd.to_datetime(site_summ1.ToDate)

    rec_datasets = datasets[datasets.CTypeID == 1].DatasetTypeID.tolist()
    man_datasets = datasets[datasets.CTypeID == 2].DatasetTypeID.tolist()

    rec_summ1 = site_summ1[site_summ1.DatasetTypeID.isin(rec_datasets) & (site_summ1.FromDate <= param['from_date']) & (site_summ1.ToDate >= param['to_date'])].sort_values('ToDate', ascending=False).drop_duplicates('ExtSiteID').copy()

    flow_sites_gdf = takes.flow_sites_gdf.copy()

    sites_rec_bool = flow_sites_gdf.FlowSite.isin(rec_summ1.ExtSiteID.unique())

    sites_rec1 = flow_sites_gdf[sites_rec_bool].copy()
    sites_man1 = flow_sites_gdf[~sites_rec_bool].copy()

    flow_rec_sites1 = mssql.rd_sql(server, database, site_table, ['ExtSiteID', 'NZTMX', 'NZTMY'], where_in={'ExtSiteID': rec_summ1.ExtSiteID.unique().tolist()})

    flow_rec_sites2 = vector.xy_to_gpd('ExtSiteID', 'NZTMX', 'NZTMY', flow_rec_sites1)

    ## Estimate flow where recorder doesn't exist

    sites_man2 = sites_man1.copy()
    sites_man2['geometry'] = sites_man1.buffer(buffer_dis)

    rec_sites2 = vector.sel_sites_poly(flow_rec_sites2, sites_man2)

    rec_ts_data1 = mssql.rd_sql_ts(server, database, ts_table, 'ExtSiteID', 'DateTime', 'Value', from_date=param['from_date'], to_date=param['to_date'], where_in={'ExtSiteID': rec_sites2.ExtSiteID.tolist(), 'DatasetTypeID': rec_summ1.DatasetTypeID.unique().tolist()})

    rec_ts_data2 = rec_ts_data1.Value.unstack(0).interpolate('time', limit=10).dropna(axis=1)

    rec_flow1 = rec_ts_data2.loc[:, rec_ts_data2.columns.isin(sites_rec1.FlowSite)].copy()

    man_ts_data1 = mssql.rd_sql_ts(server, database, ts_table, 'ExtSiteID', 'DateTime', 'Value', from_date=param['from_date'], to_date=param['to_date'], where_in={'ExtSiteID': sites_man1.FlowSite.tolist(), 'DatasetTypeID': man_datasets})

    man_ts_data2 = man_ts_data1.Value.unstack(0)

    reg_lst = []
    new_lst = []

    for col in man_ts_data2:
        site0 = sites_man1[sites_man1.FlowSite == col]

        site1 = gpd.GeoDataFrame(geometry=site0.buffer(buffer_dis))

        rec_sites3 = vector.sel_sites_poly(flow_rec_sites2, site1)
        rec_ts_data3 = rec_ts_data2.loc[:, rec_ts_data2.columns.isin(rec_sites3.ExtSiteID)].copy()

        rec_ts_data4 = rec_ts_data3.copy()
        rec_ts_data4[rec_ts_data4 <= 0] = np.nan

        man_ts_data3 = man_ts_data2.loc[:, [site0.FlowSite.iloc[0]]].copy()
        man_ts_data3[man_ts_data3 <= 0] = np.nan

        lm1 = LM(rec_ts_data4, man_ts_data3)
        res1 = lm1.predict(n_ind=1, x_transform='log', y_transform='log', min_obs=param['min_gaugings'])
        res2 = lm1.predict(n_ind=2, x_transform='log', y_transform='log', min_obs=param['min_gaugings'])

        f = [res1.summary_df['f value'].iloc[0], res2.summary_df['f value'].iloc[0]]

        val = f.index(max(f))

        if val == 0:
            reg_lst.append(res1.summary_df)

            s1 = res1.summary_df.iloc[0]

            d1 = rec_ts_data3[s1['x sites']].copy()
            d1[d1 <= 0] = 0.001

            new_data1 = np.exp(np.log(d1) * float(s1['x slopes']) + float(s1['y intercept']))
            new_data1.name = col
            new_data1[new_data1 <= 0] = 0
        else:
            reg_lst.append(res2.summary_df)

            s1 = res2.summary_df.iloc[0]
            x_sites = s1['x sites'].split(', ')
            x_slopes = [float(s) for s in s1['x slopes'].split(', ')]
            intercept = float(s1['y intercept'])

            d1 = rec_ts_data3[x_sites[0]].copy()
            d1[d1 <= 0] = 0.001
            d2 = rec_ts_data3[x_sites[1]].copy()
            d2[d2 <= 0] = 0.001

            new_data1 = np.exp((np.log(d1) * float(x_slopes[0])) + (np.log(d2) * float(x_slopes[1])) + intercept)
            new_data1.name = col
            new_data1[new_data1 <= 0] = 0

        new_lst.append(new_data1)

    new_data2 = pd.concat(new_lst, axis=1)
    reg_df = pd.concat(reg_lst).reset_index()

    flow = pd.concat([rec_flow1, new_data2], axis=1)

    flow.round(3).to_csv(os.path.join(results_path, flow_csv))
    reg_df.to_csv(os.path.join(results_path, reg_flow_csv), index=False)


