FlowNaturalisation
==================================

This git repository contains project code for the flow naturalisation procedure. The procedure has several modules for performing different tasks that ultimately combine for the naturalisation.

The base class (FlowNat) initialises the tool with a from_date, to_date, min_gaugings, input_sites, and output_path. This sets up and prepares a lot of datasets for the successive modules.

Modules:
  - Querying and/or estimating flow at the input_sites
  - Catchment delineation above the input_sites
  - Selecting the upstream water abstraction sites from the catchment delineation
  - Querying and Estimating water usage when the usage doesn't exist
  - Flow naturalisation

Input parameters
----------------
The base class (FlowNat) initialises the tool with a from_date, to_date, min_gaugings, input_sites, rec_data_code, and output_path. This sets up and prepares a lot of datasets for the successive modules. If all of those input parameters are defined at initialisation, then all of the successive modules/methods will not require any other input.

Example scripts
---------------
An example script can be found here: https://github.com/Data-to-Knowledge/FlowNaturalisation/blob/master/flownat/tests/utest_ash_2019-07-19.py.
The output_path will need to be modified for the specific user.

Background and package dependencies
-----------------------------------
The modules use several python packages for their procedures.

The catchment delineation module uses the python package gistools which has a catchment delineation function. This functions uses the REC stream network version 2 and the associated catchments for determining the catchments above specific points. The flow locations are used to delineate the upstream catchments. The upstream catchments are then used to select the WAPs that are within each catchment. The WAPs were taken from a summary of Accela.

Not all flow locations have a continuous record from a recorder. Consequently, the flow sites with only gaugings need to be correlated to flow sites with (nearly) continuous recorders. This is done via the hydrolm package that uses ordinary least squares regressions of one or two recorders. The F statistic is used to determine the best regression.

Water usage data also needs to be estimated when it doesn't already exist. This was done by grouping the consents by SWAZ/catchment and use type and estimating the ratio of usage to allocation by month. These ratios were then applied at all consents without existing water usage data. This analysis was performed on a monthly scale.

General methodology
-------------------

Introduction
~~~~~~~~~~~~~~~
River flows are affected by many factors. These include climate, geology, vegetation, water abstractions, and others. To be able to appropriately allocate water for abstraction, river flows need to be estimated without the influence of existing water abstractions for the consideration of other important values associated with surface waters (e.g. environmental, cultural, recreational, etc.). The process of estimating a new flow series by removing the effects of upstream water abstractions is called naturalisation.

The naturalisation of river flows is the process of ‘adding back’ the surface water abstractions and hydraulically connected groundwater abstractions to a flow record. Naturalisation is a process to recreate the flow record if the upstream water abstractions did not take place.
The goal of this memo is to describe a new tool to accurately and efficiently naturalise surface water flows in Canterbury. This tool has been developed in Python as an installable package from pypi. The github repository can be found here: https://github.com/Data-to-Knowledge/FlowNaturalisation

Requirements
~~~~~~~~~~~~~
The requirements for the tool includes:

- Datasets available to ECan and maintained
- Consistent with existing ECan methods
- Relatively simple
- Work for the entire record of any recorder or gauging site in Canterbury with sufficient data
- Reproducible
- Produce results quickly (a few minutes)

Methods and assumptions
~~~~~~~~~~~~~~~~~~~~~~~
The methods can be broken down into five modules:

- Querying and/or estimating flow at the input sites
- Catchment delineation above the input sites
- Selecting the upstream water abstraction sites from the catchment delineation
- Querying and Estimating water usage when the usage doesn't exist
- Flow naturalisation calculation

This method is a daily lumped model, which means all inputs and outputs are daily time series and assume that there is no lag time between flow and abstractions. The model also assumes that all abstractions that affect the downstream flow only come from the upstream delineated surface water catchment. Both of these assumptions are oversimplifications. The consequence of the first assumption is that stream depletion effects will occur sooner and end sooner than in real life. The second assumption will affect the seasonality lags due to groundwater storage and depletion from outside the catchment area.

Querying and/or estimating flow at the input sites
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
One of the input data required for the naturalisation is continuous flow records at particular locations prior to naturalisation. This usually comes in two forms: recorded continuous flow at a site with a recorder, or a site where manual flow measurements are taken but do not have a recorder.

If the site has a recorder, then the tool simply pulls the data out from ECan’s systems and uses it directly as long as the flow record covers the requested date period for the naturalisation. If it doesn’t cover the requested period, then manual flows for that site is used.
If manual flows for sites are used then a log-log regression to flow records with sufficient length and a maximum distance (default is 50 km) is used. The best correlation is selected based on the highest F value and a new continuous flow record is estimated from the correlation.
All flows (and other time series data used in this naturalisation) are daily means. Currently, the regressions use the full range of flows.

Catchment delineation above the input sites
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The catchments upstream of each flow site needs to be delineated to determine the water abstraction points (WAPs) that affect the flow record. Currently, the stream and associated catchments layer used for this delineation is the River Environmental Classification (REC) river network and watersheds.

The nearest river segments to flow sites are identified, then upstream river network delineation is performed and the catchments are extracted and merged to have a single catchment above each flow site. The delineation is at the network resolution of the REC dataset, which means that the delineation will likely not be directly at the site. Rather, it will be slightly downstream of the site and consequently the catchment will be of a slightly larger area than it would be realistically. The benefit of using a pregenerated river network and catchments is the significant reduction in the processing time and complexity of the tool.

Selecting the upstream water abstraction sites from the catchment delineation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once the catchments have been delineated above all flow sites, then the WAPs that are considered stream depleting are selected and assigned to each catchment/flow site. The WAPs and the associated consents have already been preassigned whether they are stream depleting and by how much using an ECan implementation of the Theis method for in Smith, 2015. Both the WAPs and the associated stream depletion rates are extracted from ECan’s database during this step.

Querying and Estimating water usage when the usage doesn't exist
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This is probably the most complicated module of the entire naturalisation process and this section will not go into exhaustive detail about it’s implementation. But generally, this involves querying the existing usage data associated with all WAPs/consents found in the prior module, then estimating the usage where it doesn’t exist.

This module primarily uses the EcanAlloUsageTools python package for extracting water usage data from ECan’s databases. This tool pulls from a summary of the Accela data specifically for stream depleting consents that are considered "consumptive". Given the limitations of the data in the Accela database, any consents that have "temporary wavers" or have conditions that are shared between multiple consents (e.g. non-concurrence) are ignored (because the info is not available). The minimum flow restrictions were accounted for in the consented rates and volumes. If a consent was restricted from taking water for some period of the year, that volume was deducted from the values used in the processing. These minimum flow restricted rates and volumes for the consents are split proportionally across their WAPs (if there are more than one WAP on a consent).

First, all of the existing water usage data for the upstream WAPs are extracted. Given that the water usage data does not have much quality controls, three filters are used to ensure that the usage values are “realistic”. These are usage/allocation ratios at the daily, monthly, and yearly scales. The defaults are 2, 3, and 2 respectively. These were found to be generous enough to retain real-looking usage values and exclude erroneous ones. Though it is possible to have gotten false positives and negatives using these filters.
Once the data has been filtered, abstraction/allocation ratios were calculated and lumped by month of the year, catchment, and use type. These ratios were then applied to the WAPs that did not have usage data to estimate the usage data by month. The results of the querying and estimating of the usage data is that all consented WAPs that are considered both stream depleting and consumptive have usage data.
If usage data already existed, then the daily values are used, if not then the monthly estimates are resampled to daily using the pchip interpolation method in Python. This method is to make the result more smooth and consequently more natural looking.

Flow naturalisation calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once the usage has been estimated for all upstream catchments and the flow record has also been estimated, then its a simple matter of adding the two daily time series together to get the final naturalised flow record estimate.

Potential improvements
----------------------

Regression options
~~~~~~~~~~~~~~~~~~~
Currently, the flow regressions use the entire flow record for the correlation to create a continuous flow record. This may be useful for some purposes, but other purposes (e.g. estimating low flow stats) using the entire record may not be appropriate. Additional options to constrain the flow range of the regression (e.g. 1.5 * median) could be useful.

Extend flow recorder data
~~~~~~~~~~~~~~~~~~~~~~~~~
At the moment, if the requested from and to dates extend beyond the record of a flow recorder site, then it uses that gaugings to create a new longer record. It would probably be better to extend the existing flow recorder record rather than using gaugings.

This option is already possible in hydrolm (which is used in the package). It just needs to be integrated into this package.

Finer resolution catchment delineation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Currently, the catchment delineation uses the REC v2.4 and consequently it's resolution. If finer resolution delineation is required, then a different dataset is needed. At the same time, I don't want to increase the run time significantly higher than it already is.

There are a couple options:

We could pre-generate all catchments above all flow measurement sites. This would still require a rivers layer and associated DEM.
This already used as a shortcut in the existing implementation, but it uses the REC. Combining the OSM waterways and the LINZ 8m DEM could do this.

The other alternative if any arbitrary point along a river needs to be delineated on-the-fly would be to do something similar to the first option, but only create an OSM-like network that extends all the way up to all 2nd order streams. Using only the streams layer, all WAPs upstream can be selected based on a nearest neighbor query to the stream network rather than having catchments. The only downside of not using polygon catchments is that there is a possibility (though VERY slim) that WAPs might be in a different catchment even if a different river is technically closer. I do think in practice this won't be an issue if we use all 2nd order and higher streams.

Transient abstractions
~~~~~~~~~~~~~~~~~~~~~~~
Currently abstractions regardless of the distance to the streams are instantaneous abstractions to the stream on that daily. This is of course not true. Lag times would need to be assigned based on the distance to the stream to be more appropriate.

Graphical user interface
~~~~~~~~~~~~~~~~~~~~~~~~~
A GUI on top of the tool would make using it easier for non-programmers.
