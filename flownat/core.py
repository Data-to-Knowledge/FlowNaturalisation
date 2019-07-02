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

    def __init__(self, from_date=None, to_date=None, min_gaugings=8, rec_data_code='Primary', input_sites=None, output_path=None, load_rec=False):
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

        if load_rec:
            self.load_rec()

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


    def load_rec(self):
        """

        """
        if not hasattr(self, 'rec_rivers'):
            sql1 = sql_arg()

            rec_rivers_dict = sql1.get_dict(param['input']['rec_rivers_sql'])
            rec_catch_dict = sql1.get_dict(param['input']['rec_catch_sql'])

            rec_rivers = mssql.rd_sql(**rec_rivers_dict)
            rec_catch = mssql.rd_sql(**rec_catch_dict)

            setattr(self, 'rec_rivers', rec_rivers)
            setattr(self, 'rec_catch', rec_catch)

        pass


    def catch_del(self):
        """

        """
        ## Read in GIS data
        if not hasattr(self, 'rec_rivers'):
            self.load_rec()

        ## Catch del
        catch_gdf = rec.catch_delineate(self.flow_sites_gdf, self.rec_rivers, self.rec_catch)

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


    def usage_est(self):
        """

        """
        if not hasattr(self, 'waps_gdf'):
            allo_wap = self.upstream_takes()

        waps_gdf = self.waps_gdf.copy()

        ## Get allo and usage data
        allo1 = AlloUsage(self.from_date, self.to_date, site_filter={'SwazGroupName': waps_gdf.SwazGroupName.unique().tolist()})

        usage1 = allo1.get_ts(['Allo', 'RestrAllo', 'Usage'], 'M', ['Wap', 'WaterUse'])

        usage2 = usage1.loc[usage1.SwRestrAllo > 0, ['SwRestrAllo', 'SwUsage']].reset_index().copy()

        usage2.replace({'WaterUse': {'industrial': 'other', 'municipal': 'other'}}, inplace=True)

        usage2[['SwRestrAlloYr', 'SwUsageYr']] = usage2.groupby(['Wap', 'WaterUse', pd.Grouper(key='Date', freq='A-JUN')]).transform('sum')

        sites1 = mssql.rd_sql(param['input']['ts_server'], param['input']['ts_database'], param['input']['sites_table'], ['ExtSiteID', 'SwazGroupName', 'SwazName'], where_in={'ExtSiteID': usage2.Wap.unique().tolist()})
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

        ## Save results
        if hasattr(self, 'output_path'):
            run_time = pd.Timestamp.today().strftime('%Y-%m-%dT%H%M')

            swaz_mon_ratio_csv = param['output']['swaz_mon_ratio_csv'].format(run_date=run_time)
            res_swaz5.to_csv(os.path.join(self.output_path, swaz_mon_ratio_csv), index=False)
            allo_usage_wap_swaz_csv = param['output']['allo_usage_wap_swaz_csv'].format(run_date=run_time)
            usage4.to_csv(os.path.join(self.output_path, allo_usage_wap_swaz_csv), index=False)
            wap_sw_mon_usage_csv = param['output']['wap_sw_mon_usage_csv'].format(run_date=run_time)
            usage_rate.to_csv(os.path.join(self.output_path, wap_sw_mon_usage_csv), index=False)

        setattr(self, 'mon_swaz_usage_ratio', res_swaz5)
        setattr(self, 'usage_rate', usage_rate)
        return usage_rate


    def naturalisation(self):
        """

        """
        if not hasattr(self, 'usage_rate'):
            usage_rate = self.usage_est()
        else:
            usage_rate = self.usage_rate.copy()
        if not hasattr(self, 'flow'):
            flow = self.flow_est()
        else:
            flow = self.flow.copy()

        waps1 = self.waps_gdf.drop(['geometry', 'SwazGroupName', 'SwazName'], axis=1).copy()

        usage_rate = usage_rate[usage_rate.Wap.isin(waps1.Wap.unique())].copy()

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

#        print('-> Combine usage with site data')

        usage_rate3 = pd.merge(waps1, usage_daily_rate.reset_index(), on='Wap')

        site_rate = usage_rate3.groupby(['ExtSiteID', 'Date'])[['SwUsageRate']].sum().reset_index()

        ## Add usage to flow
#        print('-> Add usage to flow')

        flow1 = flow.stack().reset_index()
        flow1.columns = ['Date', 'ExtSiteID', 'Flow']

        flow2 = pd.merge(flow1, site_rate, on=['ExtSiteID', 'Date'], how='left').set_index(['ExtSiteID', 'Date']).sort_index()
        flow2.loc[flow2.SwUsageRate.isnull(), 'SwUsageRate'] = 0

        flow2['NatFlow'] = flow2['Flow'] + flow2['SwUsageRate']

        nat_flow = flow2.unstack(0).round(3)

        ## Save results
        if hasattr(self, 'output_path'):
            run_time = pd.Timestamp.today().strftime('%Y-%m-%dT%H%M')

            nat_flow_csv = param['output']['nat_flow_csv'].format(run_date=run_time)
            nat_flow.to_csv(os.path.join(self.output_path, nat_flow_csv))

        setattr(self, 'nat_flow', nat_flow)
        return nat_flow



