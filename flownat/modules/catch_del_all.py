# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:08:09 2019

@author: michaelek
"""
import os
import numpy as np
from pdsql import mssql
from gistools import rec, vector
import pandas as pd
from flownat import FlowNat
import geopandas as gpd

pd.options.display.max_columns = 10

####################################
### Parameters

min_gaugings = 8
rec_data_code = 'Primary'
output_path = r'C:\ecan\shared\projects\catchment_delineation'
shp1 = 'catchment_delineation_rec_2019-07-17b.shp'
gpkg1 = 'catchment_delineation_rec_2019-07-17b.gpkg'
geojson1 = 'catchment_delineation_rec_2019-07-17b.geojson'
pkl1 = 'catchment_delineation_rec_2019-07-17b.pkl'
pkl2 = 'catchment_delineation_rec_2019-07-17b.pkl.xz'
pkl3 = 'catchment_delineation_rec_2019-07-17b.pkl.bz2'
pkl4 = 'catchment_delineation_rec_2019-07-17b.pkl.gzip'
pkl5 = 'catchment_delineation_rec_2019-07-17b.pkl.zip'

pkl2c = 'catchment_delineation_rec_2019-07-17c.pkl.xz'
pkl2f = 'rec_2-4b.pkl.xz'
pkl2g = 'rec_catch_2-4b.pkl.xz'

rec_rivers_path = r'P:\WaterDataProgramme\Source Data\GDBs\river-environment-classification-canterbury-2010\river-environment-classification-canterbury-2010.gpkg'
rec_catch_path = r'P:\WaterDataProgramme\Source Data\GDBs\river-environment-classification-watershed-canterbury-2010\river-environment-classification-watershed-canterbury-2010.gpkg'

###################################
###

f1 = FlowNat(output_path=output_path, load_rec=True)

sites1 = f1.summ[['ExtSiteID', 'NZTMX', 'NZTMY']].copy()

sites_gdf = vector.xy_to_gpd('ExtSiteID', 'NZTMX', 'NZTMY', sites1)

## Catch del
catch_gdf = catch_delineate(sites_gdf, f1.rec_rivers, f1.rec_catch)

catch_gdf.to_file(os.path.join(output_path, shp1))

c2 = catch_gdf.drop(['NZREACH', 'area'], axis=1).copy()
c2['geometry'] = catch_gdf.simplify(30)

c2.plot()

catch_gdf.plot()
c2.to_file(os.path.join(output_path, gpkg1), driver='GPKG')
#
### Save if required
#if hasattr(self, 'output_path'):
#    run_time = pd.Timestamp.today().strftime('%Y-%m-%dT%H%M')
#    catch_del_shp = param['output']['catch_del_shp'].format(run_date=run_time)
#    catch_gdf.to_file(os.path.join(self.output_path, catch_del_shp))
#
### Return
#setattr(self, 'catch_gdf', catch_gdf)
#return catch_gdf


c3 = vector.multipoly_to_poly(c2)
c3.to_file(os.path.join(output_path, gpkg1), driver='GPKG')

c3.to_file(os.path.join(output_path, geojson1), driver='GeoJSON')

c3.to_pickle(os.path.join(output_path, pkl1))

df1 = pd.read_pickle(os.path.join(output_path, pkl1))

c3.to_pickle(os.path.join(output_path, pkl2))


for p in [pkl1, pkl2, pkl3, pkl4, pkl5]:
#    %timeit c3.to_pickle(os.path.join(output_path, p))
    df1 = pd.read_pickle(os.path.join(output_path, p))

df1 = pd.read_pickle(os.path.join(output_path, pkl2))


c2.to_pickle(os.path.join(output_path, pkl2))


rec_rivers.to_pickle(os.path.join(output_path, pkl2f))
rec_catch.to_pickle(os.path.join(output_path, pkl2g))

rec_catch1 = rec_catch[rec_catch.NZREACH.isin(rec_rivers.NZREACH)]


rec_r1 = gpd.read_file(rec_rivers_path)
rec_r2 = rec_r1[['NZREACH', 'NZFNODE', 'NZTNODE', 'ORDER', 'geometry']]

rec_c1 = gpd.read_file(rec_catch_path)
rec_c2 = rec_c1[['NZREACH', 'geometry']]

rec_r2.to_pickle(os.path.join(output_path, pkl2f))
rec_c2.to_pickle(os.path.join(output_path, pkl2g))




