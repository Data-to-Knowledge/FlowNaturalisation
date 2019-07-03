# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 09:32:35 2019

@author: michaelek
"""
import os
import pandas as pd
import geopandas as gpd
from gistools import rec, vector
from ecandbparams import sql_arg
from allotools import AlloUsage
from pdsql import mssql
from parameters import param

pd.options.display.max_columns = 10

#######################################
### Parameters

site_csv = 'sites.csv'

ts_server = param['ts_server']
permit_server = param['permit_server']
ts_database = 'hydro'
permit_database = 'ConsentsReporting'
sites_table = 'ExternalSite'
crc_wap_table = 'reporting.CrcAlloSiteSumm'
results_path = param['results_path']

rec_rivers_sql = 'rec_rivers_gis'
rec_catch_sql = 'rec_catch_gis'

catch_del_shp = 'catch_del_{}.shp'.format(param['run_time'])
catch_del_base = 'catch_del_'
allo_csv = 'allo_{}.csv'.format(param['run_time'])
waps_shp = 'waps_{}.shp'.format(param['run_time'])
flow_sites_shp = 'flow_sites_{}.shp'.format(param['run_time'])

######################################
### Read in data
print('Read in allocation and sites data')

try:
    flow_sites_gdf = gpd.read_file(os.path.join(results_path, flow_sites_shp))
    waps_gdf = gpd.read_file(os.path.join(results_path, waps_shp))
    waps_gdf.rename(columns={'SwazGroupN': 'SwazGroupName', 'min_flow_s': 'min_flow_site'}, inplace=True)
    allo = pd.read_csv(os.path.join(results_path, allo_csv))

    print('-> loaded from local files')

except:
    print('-> Processing data from the databases')

    sites1 = mssql.rd_sql(ts_server, ts_database, sites_table, ['ExtSiteID', 'NZTMX', 'NZTMY', 'SwazGroupName', 'SwazName'])

    input_sites1 = pd.read_csv(os.path.join(param['inputs_path'], site_csv)).site.astype(str)

    sites0 = sites1[sites1.ExtSiteID.isin(input_sites1)].copy()
    sites0.rename(columns={'ExtSiteID': 'FlowSite'}, inplace=True)

    flow_sites_gdf = vector.xy_to_gpd('FlowSite', 'NZTMX', 'NZTMY', sites0)
    flow_sites_gdf.to_file(os.path.join(results_path, flow_sites_shp))

    ###################################
    ### Catchment delineation

    try:
        catch_del_shp = [p for p in os.listdir(results_path) if (catch_del_base in p) and ('.shp' in p)][-1]
        catch_gdf = gpd.read_file(os.path.join(results_path, catch_del_shp))
    except:
        sql1 = sql_arg()

        rec_rivers_dict = sql1.get_dict(rec_rivers_sql)
        rec_catch_dict = sql1.get_dict(rec_catch_sql)

        rec_rivers = mssql.rd_sql(**rec_rivers_dict)
        rec_catch = mssql.rd_sql(**rec_catch_dict)

        catch_gdf = rec.catch_delineate(flow_sites_gdf, rec_rivers, rec_catch)
        catch_gdf.to_file(os.path.join(results_path, catch_del_shp))

    ###################################
    ### WAP selection

    wap1 = mssql.rd_sql(permit_server, permit_database, crc_wap_table, ['ExtSiteID'], where_in={'ConsentStatus': param['crc_status']}).ExtSiteID.unique()

    sites3 = sites1[sites1.ExtSiteID.isin(wap1)].copy()
    sites3.rename(columns={'ExtSiteID': 'Wap'}, inplace=True)

    sites4 = vector.xy_to_gpd('Wap', 'NZTMX', 'NZTMY', sites3)
    sites4 = sites4.merge(sites3.drop(['NZTMX', 'NZTMY'], axis=1), on='Wap')

    waps_gdf, poly1 = vector.pts_poly_join(sites4, catch_gdf, 'FlowSite')
    waps_gdf.to_file(os.path.join(results_path, waps_shp))

    ##################################
    ### Get crc data

    allo1 = AlloUsage(crc_filter={'ExtSiteID': waps_gdf.Wap.unique().tolist(), 'ConsentStatus': param['crc_status']}, from_date=param['from_date'], to_date=param['to_date'])

    allo_wap1 = allo1.allo.copy()
    allo_wap = pd.merge(allo_wap1.reset_index(), waps_gdf[['Wap', 'FlowSite']], on='Wap')

    allo_wap.to_csv(os.path.join(results_path, allo_csv), index=False)

