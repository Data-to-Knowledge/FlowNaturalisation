Ashburton CWMS repo
==================================

This git repository contains project code for Ashburton CWMS projects.

The main process that this repo contains at the moment is the naturalisation procedure for flow sites. The procedure has several modules for performing different tasks that ultimately combine for the naturalisation.

Modules:
  - Catchment delineation and selection of upstream takes
  - Estimating flow when the flow doesn't exist during the required period
  - Estimate water usage when the usage doesn't exist during the required period
  - Naturalise the flows

Inputs
------
The first step is to create a csv file of site numbers that should be naturalised. This csv should be placed into the inputs folder structured according to the example csv file currently there.

Parameters
----------
In the python folder there is a parameters.ini that sets several global parameters for the modules. The key ones are the project_path, from_date, and to_date. The project_path is the root path for the naturalisation project. Below this paath should be the inputs and results folders. The from_date and to_date parameters is the time period that the naturalisation will be performed over.

Run the procedure
-----------------
To run the code, go to the python subfolder and run the install_env.bat (you might need admin permissions). This will install the python environment needed to run the code. Once that completes, run the main.bat and the procedures to naturalise the flows will run.

The results are placed in the results folder with a date stamp as part of the name. If those results have not been run for the current day, the procedure re-runs all procedures and produces new results. If results already exist for the current day, then the procedure will simply read in those existing results for the further processes. To force a procedure to re-run on the current day, simply delete (or rename) the particular result file and re-run the main.bat.

Methods
-------
The modules use several python packages for their procedures.

The catchment delineation module uses the python package gistools which has a catchment delineation function. This functions uses the REC stream network version 2 and the associated catchments for determining the catchments above specific points. The flow locations are used to delineate the upstream catchments. The upstream catchments are then used to select the WAPs that are within each catchment. The WAPs where taken from a summary of Accela.

Not all flow locations have a continuous record from a recorder. Consequently, the flow sites with only gaugings need to be correlated to flow sites with (nearly) continuous recorders. This is done via the hydrolm package that uses ordinary least squares regressions of one or two recorders. The F statistic is used to determine the best regression.

Water usage data also needs to be estimated when it doesn't already exist. This was done by grouping the consents by SWAZ and use type and estimating the ratio of usage to allocation. These ratios were then applied at all consents without existing water usage data. This analysis was performed on a monthly scale.
