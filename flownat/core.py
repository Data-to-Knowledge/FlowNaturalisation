# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:04:41 2019

@author: michaelek
"""
import numpy as np
from pdsql import mssql
from gistools import rec, vector
from allotools import AlloUsage
from hydrolm import LM
from gistools import vector
import geopandas as gpd
import os
import yaml
import pandas as pd
from ecandbparams import sql_arg

#####################################
### Parameters

base_dir = os.path.realpath(os.path.dirname(__file__))

with open(os.path.join(base_dir, 'parameters.yml')) as param:
    param = yaml.safe_load(param)


#######################################
### Class


class FlowNat(object):
    """

    """

    def __init__(self, from_date=None, to_date=None, min_gaugings=8, rec_data_code='Primary', input_sites=None, output_path=None):
        """

        """
        setattr(self, 'from_date', from_date)
        setattr(self, 'to_date', to_date)
        setattr(self, 'min_gaugings', min_gaugings)
        setattr(self, 'rec_data_code', rec_data_code)
        self.save_path(output_path)
        summ1 = self.flow_datasets(from_date=from_date, to_date=to_date, min_gaugings=8, rec_data_code=rec_data_code)
        if input_sites is not None:
            input_summ1 = self.process_sites(input_sites)
        pass

#    def data_code(self, rec_data_code='Primary'):
#        """
#        Options are RAW and Primary.
#        """
#
#
#    def date_range(self, from_date=None, to_date=None):
#        """
#
#        """
#
#
#    def min_gaugings(self, min_gaugings=8):
#        """
#
#        """
#
#
#    def buffer_dis(self, buffer_dis=50000):
#        """
#
#        """


    def flow_datasets_all(self, rec_data_code='Primary'):
        """

        """
        ## Get dataset types
        datasets1 = mssql.rd_sql(param['input']['ts_server'], param['input']['ts_database'], param['input']['ts_dataset_table'], where_in={'Feature': ['River'], 'MeasurementType': ['Flow'], 'DataCode': ['Primary', 'RAW']})
        man_datasets1 = datasets1[(datasets1['CollectionType'] == 'Manual Field') & (datasets1['DataCode'] == 'Primary')].copy()
        rec_datasets1 = datasets1[(datasets1['CollectionType'] == 'Recorder') & (datasets1['DataCode'] == rec_data_code)].copy()

        ## Get ts summaries
        man_summ1 = mssql.rd_sql(param['input']['ts_server'], param['input']['ts_database'], param['input']['ts_summ_table'], ['ExtSiteID', 'DatasetTypeID', 'Min', 'Median', 'Mean', 'Max', 'Count', 'FromDate', 'ToDate'], where_in={'DatasetTypeID': man_datasets1['DatasetTypeID'].tolist()}).sort_values('ToDate')
        man_summ2 = man_summ1.drop_duplicates(['ExtSiteID'], keep='last').copy()
        man_summ2['CollectionType'] = 'Manual Field'

        rec_summ1 = mssql.rd_sql(param['input']['ts_server'], param['input']['ts_database'], param['input']['ts_summ_table'], ['ExtSiteID', 'DatasetTypeID', 'Min', 'Median', 'Mean', 'Max', 'Count', 'FromDate', 'ToDate'], where_in={'DatasetTypeID': rec_datasets1['DatasetTypeID'].tolist()}).sort_values('ToDate')
        rec_summ2 = rec_summ1.drop_duplicates(['ExtSiteID'], keep='last').copy()
        rec_summ2['CollectionType'] = 'Recorder'

        ## Combine
        summ2 = pd.concat([man_summ2, rec_summ2], sort=False)

        summ2['FromDate'] = pd.to_datetime(summ2['FromDate'])
        summ2['ToDate'] = pd.to_datetime(summ2['ToDate'])

        ## Add in site info
        sites1 = mssql.rd_sql(param['input']['ts_server'], param['input']['ts_database'], param['input']['sites_table'], ['ExtSiteID', 'NZTMX', 'NZTMY', 'SwazGroupName', 'SwazName'])

        summ3 = pd.merge(summ2, sites1, on='ExtSiteID')

        ## Assign objects
        setattr(self, 'sites', sites1)
        setattr(self, 'rec_data_code', rec_data_code)
        setattr(self, 'summ_all', summ3)


    def flow_datasets(self, from_date=None, to_date=None, min_gaugings=8, rec_data_code='Primary'):
        """

        """
        if not hasattr(self, 'summ_all') | (rec_data_code != self.rec_data_code):
            self.flow_datasets_all(rec_data_code=rec_data_code)

        summ1 = self.summ_all.copy()
        if isinstance(from_date, str):
            summ1 = summ1[summ1.FromDate <= from_date]
        if isinstance(to_date, str):
            summ1 = summ1[summ1.ToDate >= to_date]
        summ2 = summ1[summ1.Count >= min_gaugings].sort_values('CollectionType').drop_duplicates('ExtSiteID', keep='last').copy()

        setattr(self, 'summ', summ2)
        return summ2


    def save_path(self, output_path=None):
        """

        """
        if output_path is None:
            pass
        elif isinstance(output_path, str):
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            setattr(self, 'output_path', output_path)

#        output_dict1 = {k: v.split('_{run_date}')[0] for k, v in param['output'].items()}

#        file_list = [f for f in os.listdir(output_path) if ('catch_del' in f) and ('.shp' in f)]

    def process_sites(self, input_sites):
        """

        """
        ## Checks
        if isinstance(input_sites, (str, int)):
            input_sites = [input_sites]
        elif not isinstance(input_sites, list):
            raise ValueError('input_sites must be a str, int, or list')

        ## Convert sites to gdf
        sites_gdf = vector.xy_to_gpd(['ExtSiteID', 'CollectionType'], 'NZTMX', 'NZTMY', self.summ.drop_duplicates('ExtSiteID'))
        input_summ1 = self.summ[self.summ.ExtSiteID.isin(input_sites)].copy()

        bad_sites = [s for s in input_sites if s not in input_summ1.ExtSiteID.unique()]

        if bad_sites:
            print(', '.join(bad_sites) + ' sites are not available for naturalisation')

        flow_sites_gdf = sites_gdf[sites_gdf.ExtSiteID.isin(input_sites)].copy()
        ## Save if required
        if hasattr(self, 'output_path'):
            run_time = pd.Timestamp.today().strftime('%Y-%m-%dT%H%M')
            flow_sites_shp = param['output']['flow_sites_shp'].format(run_date=run_time)
            flow_sites_gdf.to_file(os.path.join(self.output_path, flow_sites_shp))

        setattr(self, 'sites_gdf', sites_gdf)
        setattr(self, 'flow_sites_gdf', flow_sites_gdf)
        setattr(self, 'input_summ', input_summ1)
        return input_summ1


    def catch_del(self):
        """

        """
        ## Read in GIS data
        sql1 = sql_arg()

        rec_rivers_dict = sql1.get_dict(param['input']['rec_rivers_sql'])
        rec_catch_dict = sql1.get_dict(param['input']['rec_catch_sql'])

        rec_rivers = mssql.rd_sql(**rec_rivers_dict)
        rec_catch = mssql.rd_sql(**rec_catch_dict)

        ## Catch del
        catch_gdf = rec.catch_delineate(self.flow_sites_gdf, rec_rivers, rec_catch)

        ## Save if required
        if hasattr(self, 'output_path'):
            run_time = pd.Timestamp.today().strftime('%Y-%m-%dT%H%M')
            catch_del_shp = param['output']['catch_del_shp'].format(run_date=run_time)
            catch_gdf.to_file(os.path.join(self.output_path, catch_del_shp))

        ## Return
        setattr(self, 'catch_gdf', catch_gdf)
        return catch_gdf


    def upstream_takes(self):
        """

        """
        if not hasattr(self, 'catch_gdf'):
            catch_gdf = self.catch_del()
        else:
            catch_gdf = self.catch_gdf.copy()

        ### WAP selection
        wap1 = mssql.rd_sql(param['input']['permit_server'], param['input']['permit_database'], param['input']['crc_wap_table'], ['ExtSiteID'], where_in={'ConsentStatus': param['input']['crc_status']}).ExtSiteID.unique()

        sites3 = self.sites[self.sites.ExtSiteID.isin(wap1)].copy()
        sites3.rename(columns={'ExtSiteID': 'Wap'}, inplace=True)

        sites4 = vector.xy_to_gpd('Wap', 'NZTMX', 'NZTMY', sites3)
        sites4 = sites4.merge(sites3.drop(['NZTMX', 'NZTMY'], axis=1), on='Wap')

        waps_gdf, poly1 = vector.pts_poly_join(sites4, catch_gdf, 'ExtSiteID')

        ### Get crc data
        allo1 = AlloUsage(crc_filter={'ExtSiteID': waps_gdf.Wap.unique().tolist(), 'ConsentStatus': param['input']['crc_status']}, from_date=self.from_date, to_date=self.to_date)

        allo_wap1 = allo1.allo.copy()
        allo_wap = pd.merge(allo_wap1.reset_index(), waps_gdf[['Wap', 'ExtSiteID']], on='Wap')

        ## Save if required
        if hasattr(self, 'output_path'):
            run_time = pd.Timestamp.today().strftime('%Y-%m-%dT%H%M')

            waps_shp = param['output']['waps_shp'].format(run_date=run_time)
            waps_gdf.to_file(os.path.join(self.output_path, waps_shp))

            allo_data_csv = param['output']['allo_data_csv'].format(run_date=run_time)
            allo_wap.to_csv(os.path.join(self.output_path, allo_data_csv), index=False)

        ## Return
        setattr(self, 'waps_gdf', waps_gdf)
        setattr(self, 'allo_wap', allo_wap)
        return allo_wap


    def flow_est(self, buffer_dis=50000):
        """

        """

        if self.input_summ.CollectionType.isin(['Recorder']).any():
            rec_summ1 = self.input_summ[self.input_summ.CollectionType.isin(['Recorder'])].copy()
            rec_ts_data1 = mssql.rd_sql_ts(param['input']['ts_server'], param['input']['ts_database'], param['input']['ts_table'], ['ExtSiteID', 'DatasetTypeID'], 'DateTime', 'Value', from_date=self.from_date, to_date=self.to_date, where_in={'ExtSiteID': rec_summ1.ExtSiteID.tolist(), 'DatasetTypeID': rec_summ1.DatasetTypeID.unique().tolist()}).reset_index()
            rec_ts_data1 = pd.merge(rec_summ1[['ExtSiteID', 'DatasetTypeID']], rec_ts_data1, on=['ExtSiteID', 'DatasetTypeID']).drop('DatasetTypeID', axis=1).set_index(['ExtSiteID', 'DateTime'])
            rec_ts_data2 = rec_ts_data1.Value.unstack(0)

        else:
            rec_ts_data2 = pd.DataFrame()

        if self.input_summ.CollectionType.isin(['Manual Field']).any():
            man_summ1 = self.input_summ[self.input_summ.CollectionType.isin(['Manual Field'])].copy()
            man_sites1 = self.sites_gdf[self.sites_gdf.ExtSiteID.isin(man_summ1.ExtSiteID)].copy()

            ## Determine which sites are within the buffer of the manual sites

            buff_sites_dict = {}
            man_buff1 = man_sites1.set_index(['ExtSiteID']).copy()
            man_buff1['geometry'] = man_buff1.buffer(buffer_dis)

            rec_sites_gdf = self.sites_gdf[self.sites_gdf.CollectionType == 'Recorder'].copy()

            for index in man_buff1.index:
                buff_sites1 = vector.sel_sites_poly(rec_sites_gdf, man_buff1.loc[[index]])
                buff_sites_dict[index] = buff_sites1.ExtSiteID.tolist()

            buff_sites_list = [item for sublist in buff_sites_dict.values() for item in sublist]
            buff_sites = set(buff_sites_list)

            ## Pull out recorder data needed for all manual sites
            man_ts_data1 = mssql.rd_sql_ts(param['input']['ts_server'], param['input']['ts_database'], param['input']['ts_table'], ['ExtSiteID', 'DatasetTypeID'], 'DateTime', 'Value', from_date=self.from_date, to_date=self.to_date, where_in={'ExtSiteID': man_summ1.ExtSiteID.tolist(), 'DatasetTypeID': man_summ1.DatasetTypeID.unique().tolist()}).reset_index()
            man_ts_data1 = pd.merge(man_summ1[['ExtSiteID', 'DatasetTypeID']], man_ts_data1, on=['ExtSiteID', 'DatasetTypeID']).drop('DatasetTypeID', axis=1).set_index(['ExtSiteID', 'DateTime'])
            man_ts_data2 = man_ts_data1.Value.unstack(0)

            man_rec_summ1 = self.summ[self.summ.ExtSiteID.isin(buff_sites)].copy()
            man_rec_ts_data1 = mssql.rd_sql_ts(param['input']['ts_server'], param['input']['ts_database'], param['input']['ts_table'], ['ExtSiteID', 'DatasetTypeID'], 'DateTime', 'Value', from_date=self.from_date, to_date=self.to_date, where_in={'ExtSiteID': man_rec_summ1.ExtSiteID.tolist(), 'DatasetTypeID': man_rec_summ1.DatasetTypeID.unique().tolist()}).reset_index()
            man_rec_ts_data1 = pd.merge(man_rec_summ1[['ExtSiteID', 'DatasetTypeID']], man_rec_ts_data1, on=['ExtSiteID', 'DatasetTypeID']).drop('DatasetTypeID', axis=1).set_index(['ExtSiteID', 'DateTime'])
            man_rec_ts_data2 = man_rec_ts_data1.Value.unstack(0).interpolate('time', limit=10)

            ## Run through regressions
            reg_lst = []
            new_lst = []

            for key, lst in buff_sites_dict.items():
                man_rec_ts_data3 = man_rec_ts_data2.loc[:, lst].copy()
                man_rec_ts_data3[man_rec_ts_data3 <= 0] = np.nan

                man_ts_data3 = man_ts_data2.loc[:, [key]].copy()
                man_ts_data3[man_ts_data3 <= 0] = np.nan

                lm1 = LM(man_rec_ts_data3, man_ts_data3)
                res1 = lm1.predict(n_ind=1, x_transform='log', y_transform='log', min_obs=self.min_gaugings)
                res2 = lm1.predict(n_ind=2, x_transform='log', y_transform='log', min_obs=self.min_gaugings)

                f = [res1.summary_df['f value'].iloc[0], res2.summary_df['f value'].iloc[0]]

                val = f.index(max(f))

                if val == 0:
                    reg_lst.append(res1.summary_df)

                    s1 = res1.summary_df.iloc[0]

                    d1 = man_rec_ts_data3[s1['x sites']].copy()
                    d1[d1 <= 0] = 0.001

                    new_data1 = np.exp(np.log(d1) * float(s1['x slopes']) + float(s1['y intercept']))
                    new_data1.name = key
                    new_data1[new_data1 <= 0] = 0
                else:
                    reg_lst.append(res2.summary_df)

                    s1 = res2.summary_df.iloc[0]
                    x_sites = s1['x sites'].split(', ')
                    x_slopes = [float(s) for s in s1['x slopes'].split(', ')]
                    intercept = float(s1['y intercept'])

                    d1 = man_rec_ts_data3[x_sites[0]].copy()
                    d1[d1 <= 0] = 0.001
                    d2 = man_rec_ts_data3[x_sites[1]].copy()
                    d2[d2 <= 0] = 0.001

                    new_data1 = np.exp((np.log(d1) * float(x_slopes[0])) + (np.log(d2) * float(x_slopes[1])) + intercept)
                    new_data1.name = key
                    new_data1[new_data1 <= 0] = 0

                new_lst.append(new_data1)

            new_data2 = pd.concat(new_lst, axis=1)
            reg_df = pd.concat(reg_lst).reset_index()
        else:
            new_data2 = pd.DataFrame()
            reg_df = pd.DataFrame()

        flow = pd.concat([rec_ts_data2, new_data2], axis=1).round(3)

        ## Save if required
        if hasattr(self, 'output_path'):
            run_time = pd.Timestamp.today().strftime('%Y-%m-%dT%H%M')

            if not reg_df.empty:
                reg_flow_csv = param['output']['reg_flow_csv'].format(run_date=run_time)
                reg_df.to_csv(os.path.join(self.output_path, reg_flow_csv), index=False)

            flow_csv = param['output']['flow_csv'].format(run_date=run_time)
            flow.to_csv(os.path.join(self.output_path, flow_csv))

        setattr(self, 'flow', flow)
        setattr(self, 'reg_flow', reg_df)
        return flow















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























































