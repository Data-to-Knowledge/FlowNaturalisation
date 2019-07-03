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

Input Parameters
----------------
The base class (FlowNat) initialises the tool with a from_date, to_date, min_gaugings, input_sites, rec_data_code, and output_path. This sets up and prepares a lot of datasets for the successive modules. If all of those input parameters are defined at initialisation, then all of the successive modules/methods will not require any other input.

Methods
-------
The modules use several python packages for their procedures.

The catchment delineation module uses the python package gistools which has a catchment delineation function. This functions uses the REC stream network version 2 and the associated catchments for determining the catchments above specific points. The flow locations are used to delineate the upstream catchments. The upstream catchments are then used to select the WAPs that are within each catchment. The WAPs were taken from a summary of Accela.

Not all flow locations have a continuous record from a recorder. Consequently, the flow sites with only gaugings need to be correlated to flow sites with (nearly) continuous recorders. This is done via the hydrolm package that uses ordinary least squares regressions of one or two recorders. The F statistic is used to determine the best regression.

Water usage data also needs to be estimated when it doesn't already exist. This was done by grouping the consents by SWAZ and use type and estimating the ratio of usage to allocation. These ratios were then applied at all consents without existing water usage data. This analysis was performed on a monthly scale.
