
# ______________________________________________________________________________________
## Input values
# Edit as required - controls run_budget_calculator.
# ______________________________________________________________________________________
import numpy as np
import os

# The target temperature changes to achieve. (Units: C)
dT_targets = np.arange(1.1, 2.6, 0.1)
# The number of loops performed for each temperature. Runs in seconds for ~ 10^6, higher
# reduces statistical error but takes longer
n_loops = 1000000  # Default: 1000000
# ZEC: the change in temperature that will occur after zero emissions has been reached.
# (Units: C)
zec_mean = 0.0  # Default: 0.0
zec_sd = 0.19   # Default 0.19
# Should we interpret the distribution of ZEC in an asymmetric way, i.e. ignore when its
# negative but include when positive?
zec_asym = False  # Default: False
# The temperature difference already seen. (Units: C)
historical_dT = 1.07  # Default: 1.07
# The distribution of the TCRE function - either "normal", "lognormal mean match" or
# "lognormal". The latter two cases are lognormal distributions, in the first
# case matching the mean and sd of the normal distribution which fits the likelihood,
# in the second case matching the likelihood limits itself but exhibiting some skew.
tcre_dist = "normal"  # Default: "normal"
# Constant to convert between Pg C and Gt CO2.
convert_PgC_to_GtCO2 = 3.664
# The upper and lower bounds of the distribution of TCRE. We use units of degC per GtCO2
# (TCRE = Transient climate response to cumulative carbon emissions)
tcre_low = 1.0 / 1000 / convert_PgC_to_GtCO2  # default: 1.0 / 1000 / convert_PgC_to_GtCO2
tcre_high = 2.3 / 1000 / convert_PgC_to_GtCO2 # 2.3 / 1000 / convert_PgC_to_GtCO2
# likelihood is the probability that results fit between the low and high value
likelihood = 0.6827
# Average CO2 emissions per degree C from temperature-dependent Earth feedback loops.
# (Units: GtCO2/C)
earth_feedback_co2_per_C_av = 7.1 * convert_PgC_to_GtCO2  # 7.1 * convert_PgC_to_GtCO2
# Standard dev of CO2 emissions per degree C from temperature-dependent Earth feedback
# St dev CO2 emissions per degree C from temperature-dependent Earth feedback loops.
# loops (Units: Gt CO2/C).
earth_feedback_co2_per_C_stdv = 26.7 * convert_PgC_to_GtCO2 # 26.7 * convert_PgC_to_GtCO2
# Any emissions that have taken place too recently to have factored into the measured
# temperature change, and therefore must be subtracted from the budget (Units: GtCO2)
recent_emissions = 277  # Default: 277
# We will present the budgets at these probability quantiles. (Switch is provided to
# go between easy reading and data for plots)
allquant = False  # False
quantiles_to_report = np.arange(0.01, 0.991, 0.0025) if allquant else np.array(
    [0.1, 0.17, 0.33, 0.5, 0.66, 0.83, 0.9])

# Run version should be ar6wg3, sr15ccbox71, or sr15wg1, corresponding to running the
# code using the AR6 database as used for WG3, or the SR1.5 database with either the
# cross-chapter box 7.1/ Nicholls 2021 configuration of MAGICC or the older Meinshausen
# 2020 configuration, as was used in the AR6 WG1 report.
runver = "ar6wg3"  # Default: "ar6wg3"

# Name of the output folder
if runver == "ar6wg3":
    output_folder = "../Output/ar6wg3/"
    arsr = "ar6"
elif runver == "sr15ccbox71":
    output_folder = "../Output/sr15ccbox71/"
    arsr = "sr15"
elif runver == "sr15wg1":
    output_folder = "../Output/sr15wg1/"
    arsr = "sr15"
else:
    raise ValueError(f"runver {runver} not available, choose ar6wg3, sr15ccbox71 or sr15wg1")

# Output file location for budget data. Includes {} sections detailing inclusion of
# TCRE, inclusion of magic/fair, earth system feedback and likelihood. More added later
output_file = ("allquant" if allquant else "") + \
    "budget_{}_magicc_{}_fair_{}_esf_{}pm{}_likeli_{}_nonCO2pc{}_GtCO2_permaf_{}_zecsd_{}_asym_{}_hdT_{}"

# Output location for figure of peak warming vs non-CO2 warming. More appended later
output_figure_file = "non_co2_cont_to_peak_warming_magicc_{}_fair_{}_permaf_{} zecsd_{}_asym_{}_nonCO2pc_{}_nonlin_{}"
# Quantile fit lines to plot on the temperatures graph.
# If use_as_median_non_co2 evaluates as True, this must include that quantile (by
# default 0.5)
quantiles_to_plot = [0.05, 0.5, 0.95]
# How should we dot these lines? This list must be as long as quantiles_to_plot.
line_dotting = ["--", "-", "--"]
# Should we use the median value from quantile regression (True) or the least-squares
# best fit (False) for the non-CO2 relationship? If instead a number is used, we use
# that quantile, which must be in quantiles_to_plot.
use_as_median_non_co2 = True  # default: True
# Where should we save the results of the figure with trend lines? Not plotted if
# use_median_non_co2 evaluates as True.
if use_as_median_non_co2 != True:
    output_file = output_file + "_nonco2scenfrac" + str(use_as_median_non_co2)

# should we assume the relationship between total warming and non-CO2 warming is linear?
# Default (None) yes; alternatively: "rollingQuantiles" means that rolling quantiles are
# taken point-by-point for scenarios across non-CO2 and total warming space; "QRW"
# resulting in the Quantile Rolling Windows approach, which also takes
# rolling quantiles but does so with weights according to the proximity to the
# evaluation point (see Silicone documentation for more details). "all" uses the linear
# trend in the calculation but plots QRW too. "interp" draws lines between points and
# interpolates between them - it is best used in conjunction with specific for_each_model
# scenarios.
nonlinear_nonco2 = "all"  # default: "all"
if nonlinear_nonco2:
    output_file = output_file + f"NonlinNonCO2_{nonlinear_nonco2}"
output_all_trends = "TrendLinesWithMagicc_permaf_{}_LinearCO2_{}.pdf"
# Do we want to calculate this non-CO2 component separately for each model in the
# database? Options: False (evaluates the normal carbon budget), True (evaluates for
# each model), or a specific string (e.g. "SSP1"),
# in which case we also filter the scenario for that string being its first part.
for_each_model = False  # default: False
if for_each_model:
    # How many scenarios are required for analysis to be done?
    min_scenarios = 3
    output_folder = output_folder + "each/"
    if type(for_each_model) == str:
        output_folder = output_folder + for_each_model + "/"
    # We have a real problem with filenames being too long
    output_file = output_file.replace("budget_", "")
os.makedirs(output_folder, exist_ok=True)

###       Information for reading in files used to calculate non-CO2 component:

# Should we use a variant means of measuring the non-CO2 warming?
# Default = None; ignore scenarios with non-peaking cumulative CO2 emissions, use
# non-CO2 warming in the year of peak cumulative CO2.
# If "peakNonCO2Warming", we use the highest non-CO2 temperature,
# irrespective of emissions peak.
# If "nonCO2AtPeakTot", computes the non-CO2 component at the time of peak total
# temperature.
# If "officialNZ" uses the date of net zero in the metadata used to validate the
# scenarios - validation file must also be used.
peak_version = None  # default: None
output_file += "_" + str(peak_version) + "_recEm" + str(round(recent_emissions)) + ".csv"
output_figure_file += "_" + str(peak_version) + ".pdf"
# We want a list of bools indicating whether to run the code with values from MAGICC,
# FaIR or both.
include_list = [(True, False), (True, True), (False, True)]

##  Load data for MAGICC and FaIR
# We wish to recall one of several MAGICC runs, depending on the situation.
if runver == "ar6wg3":
    # AR6 WG3 version.
    # Strings to describe the runs, filed with a job number
    jobno = "20211019-ar6"
    magiccver = "7.5.3"
    # The folder and files in which we find the MAGICC model estimate for the non-carbon
    # and carbon contributions to temperature change.
    input_folder = "../InputData/MAGICCAR6emWG3scen/"
    # We also check that the scenarios used in the MAGICC are those that pass the vetting process
    vetted_scen_list_file = input_folder + "ar6_full_metadata_indicators2021_10_14_v3.xlsx"
    vetted_scen_list_file_sheet = "meta_Ch3vetted_withclimate"
    # We may search for processed FaIR data here if required
    fair_processed_file = "../InputData/fair163_ar6/fair_processed_data.csv"
    # The folders for the unscaled anthropological temperature changes files (many nc
    # files), skipped if processed data is available
    fair_anthro_folder = "../InputData/fair163_ar6/all_temps/"
    fair_co2_only_folder = "../InputData/fair163_ar6/co2_temps/"
    fair_filestr = "FaIRv1.6.2__"
elif runver == "sr15ccbox71":
    # SR1.5 with later MAGICC version as in Cross Chapter Box 7.1 of AR6WG1
    jobno = "20211014-sr15"
    magiccver = "7.5.3"
    input_folder = "../InputData/MAGICCCCB71_sr15scen/"
    # We use a compound vetting file
    vetted_scen_list_file = input_folder + "sr15_scenario_runs_mocked_vetting.xlsx"
    vetted_scen_list_file_sheet = "Sheet1"
    fair_processed_file = "../InputData/fair163_sr15/fair_processed_data.csv"
    fair_anthro_folder = "../InputData/fair163_sr15/SR15_all_temps/"
    fair_co2_only_folder = "../InputData/fair163_sr15/SR15_co2_temps/"
    fair_filestr = "FaIRv1.6.2__"
elif runver == "sr15wg1":
    # SR1.5 with MAGICC using Meinshausen et al. (2020) input files (as was used for
    # main WG1 RCB calculations)
    jobno = "20210224-sr15"
    magiccver = "7.5.1"
    input_folder = "../InputData/MAGICCMeinshausenInputs_sr15scen/"
    vetted_scen_list_file = input_folder + "sr15_scenario_runs_mocked_vetting.xlsx"
    vetted_scen_list_file_sheet = "meta_Ch3vetted_withclimate"
    fair_processed_file = "../InputData/fair141_sr15/fair_processed_data.csv"
    fair_anthro_folder = "../InputData/fair141_sr15/FAIR141anthro_unscaled/"
    fair_co2_only_folder = "../InputData/fair141_sr15/FAIR141CO2_unscaled/"
    fair_filestr = "IPCCSR15_"
else:
    raise ValueError(f"runver {runver} not available, choose ar6wg3, sr15ccbox71 or sr15wg1")

non_co2_magicc_file_permafrost = (
    input_folder + f"job-{jobno}-nonco2_Raw-GSAT-Non-CO2.csv"
)
non_co2_magicc_file_no_permafrost = non_co2_magicc_file_permafrost
tot_magicc_file_permafrost = input_folder + f"job-{jobno}-nonco2_Raw-GSAT.csv"
tot_magicc_file_nopermafrost = tot_magicc_file_permafrost
# The file in which we find the emissions data
emissions_file = input_folder + f"job-{jobno}-nonco2_Emissions-CO2.csv"
# The name of the non-CO2 warming column output from in the MAGICC model file analysis
magicc_non_co2_col = (
    "non-co2 warming (rel. to 2010-2019) at peak cumulative emissions co2"
)
# The name of the peak temperature column output
magicc_temp_col = "peak surface temperature (rel. to 2010-2019)"
# The percentile to use for non-CO2 temperature change (for each scenario separately)
nonco2_percentile = 50
# The names of the temperature variables in MAGICC files (also specifies the quantile)
magicc_nonco2_temp_variable = "{} climate diagnostics|Raw Surface Temperature (GSAT)|Non-CO2|MAGICCv{}|{}.0th Percentile".format(
    arsr.upper(), magiccver, nonco2_percentile
)
magicc_tot_temp_variable = "{} climate diagnostics|Raw Surface Temperature (GSAT)|MAGICCv{}|50.0th Percentile".format(
    arsr.upper(), magiccver
)
# Do we want to save the output of the MAGICC analysis? If so, give a file name with a
# variable in it. Otherwise leave as None
magicc_savename = "magicc_nonCO2_temp_{}Percentile".format(
    nonco2_percentile
) + str(peak_version) + "permaf_{}.csv"
# If FaIR data is wanted, we will attempt to load it from here, and if not available we
# will generate it and save that for reuse later
fair_savename = magicc_savename.replace("magicc", "fair")
# Years over which we set the average temperature to 0.
# Note that the upper limit of the range is not included in python.
temp_offset_years = np.arange(2010, 2020, 1)
# Use permafrost may be True, False or both (iterates over the list)
List_use_permafrost = [False]
