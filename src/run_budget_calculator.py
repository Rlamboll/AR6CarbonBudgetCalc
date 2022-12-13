# Main script to run to get values of the carbon budget.
# Also contains the controls of how the carbon budget is run.
import os

import numpy as np
import pandas as pd
import src.distributions_of_inputs as distributions
import src.budget_calculator_functions as budget_func
import matplotlib.pyplot as plt
import time
import waterfall_chart as waterfall
# ______________________________________________________________________________________
## Input values
# Edit as required - controls the main calculation.
# ______________________________________________________________________________________
# The target temperature changes to achieve. (Units: C)
dT_targets = np.arange(1.1, 2.6, 0.1)
# The number of loops performed for each temperature. Runs in tens of seconds for
# ~ 10^6, higher reduces statistical error but takes longer
n_loops = 10000000  # Default: 1000000
# ZEC: the change in temperature that will occur after zero emissions has been reached.
# (Units: C)
zec_mean = 0.0  # Default: 0.0
zec_sd = 0.19   # Default 0.19
# Should we interpret the distribution of ZEC in an asymmetric way, i.e. ignore when its
# negative but include when positive?
zec_asym = False  # Default: False
# The temperature difference already seen. (Units: C)
historical_dT = 1.07  # Default: 1.07
# Uncertainty in this value (Units: C). Only used if doing waterfall plot.
historical_uncertainty = 0.2
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
recent_emissions = 326  # Default: 326 (until end of 2022)
# The uncertainty in recent emissions is about 1.8 GtCO2/year. Here we assume errors are
# correlated. This is not used for most calculations
recent_emissions_uncertainty = 1.8 * 8
# We will present the budgets at these probability quantiles. (Switch is provided to
# go between easy reading and data for plots)
allquant = False  # False
quantiles_to_report = np.arange(0.01, 0.991, 0.0025) if allquant else np.array(
    [0.1, 0.17, 0.33, 0.5, 0.66, 0.83, 0.9])

# Run version should be ar6wg3, sr15ccbox71, or sr15prewg1, corresponding to running the
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
elif runver == "sr15prewg1":
    output_folder = "../Output/sr15prewg1/"
    arsr = "sr15"
else:
    raise ValueError(f"runver {runver} not available, choose ar6wg3, sr15ccbox71 or sr15prewg1")

# Output file location for budget data. Includes {} sections detailing inclusion of
# TCRE, inclusion of magic/fair, earth system feedback and likelihood. More added later
output_file = ("allquant" if allquant else "") + \
    "budget_{}_magicc_{}_fair_{}_esf_{}pm{}_likeli_{}_nonCO2pc{}_GtCO2_permaf_{}_zecsd_{}_asym_{}_hdT_{}"

# Output location for figure of peak warming vs non-CO2 warming. More appended later
output_figure_file = "non_co2_cont_to_peak_warming_magicc_{}_fair_{}_permaf_{} zecsd_{}_asym_{}_nonCO2pc_{}_nonlin_{}"
# Quantile fit lines to plot on the temperatures graph.
# If use_as_median_non_co2 evaluates as True, this must include that quantile (by
# default 0.5)
quantiles_to_plot = [0.17, 0.5, 0.83]
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
# if not None, do a waterfall plot of the contributions of each component to the budget
# and save it with this string.
# If not none, should have {} for magicc, Fair, peak, temp, recem, ZEC, permafrost. E.g.
# "waterfall_contributions_MAGICC_{}_FaIR_{}_peak{}_temp{}_recem{}_ZEC{}_pf{}.png"
waterfall_plot = None  # default: None
if waterfall_plot:
    # Error bars on the non-CO2 component in the waterfall plot are calculated by
    # separate runs with different quantiles of non-CO2 warming.
    nonco2_waterfall_uncertainty = {
        1.5: [82, 124.3], 2.0: [95, 178]
    }

###       Information for reading in files used to calculate non-CO2 component:

# Should we use a variant means of measuring the non-CO2 warming?
# Default = None; ignore scenarios with non-peaking cumulative CO2 emissions, use
# non-CO2 warming in the year of peak cumulative CO2.
# If "peakNonCO2Warming", we use the highest non-CO2 temperature,
# irrespective of emissions peak.
# If "nonCO2AtPeakTot", computes the non-CO2 component at the time of peak total
# temperature.
# If "nonCO2AtPeakTotIfNZ", computes the non-CO2 component at the time of peak total
# temperature but ignores scenarios that do not reach net zero.
# If "nonCO2AtPeakTotIfOfNZ, computes the non-CO2 component at the time of peak total
# temperature but ignores scenarios that did not reach net zero prior to
#  harmonisation.
# If "nonCO2AtPeakTotMagicc", computes the non-CO2 component at the time of peak total
# temperature in MAGICC, provided that scenarios reach net zero, and the same warming is
# applied to FaIR.
# If "officialNZ" uses the date of net zero in the metadata used to validate the
# scenarios - validation file must also be used.
# If "nonCO2AtPeakAverage" uses the time of peak temperature of net zero scenarios for
# the separate MAGICC and FaIR calculations but averages the temperature trends before
# calculating the trends in the combined assessment.
peak_version = None  # default: None
output_file += "_" + str(peak_version) + "_recEm" + str(round(recent_emissions)) + ".csv"
output_figure_file += "_" + str(peak_version) + ".pdf"
# We want a list of bools indicating whether to run the code with values from MAGICC,
# FaIR or both.
include_list = [(True, True), (True, False), (False, True)]

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
    fair_folder = "../InputData/fair163_ar6/"
    fair_filestr = "FaIRv1.6.2__"
    fair_filter = None
elif runver == "sr15ccbox71":
    # SR1.5 with later MAGICC version as in Cross Chapter Box 7.1 of AR6WG1
    jobno = "20211014-sr15"
    magiccver = "7.5.3"
    input_folder = "../InputData/MAGICCCCB71_sr15scen/"
    # We use a compound vetting file
    vetted_scen_list_file = input_folder + "sr15_scenario_runs_mocked_vetting.xlsx"
    vetted_scen_list_file_sheet = "Sheet1"
    fair_folder = "../InputData/fair163_sr15/"
    fair_filestr = "FaIRv1.6.2__"
    fair_filter = None
elif runver == "sr15prewg1":
    # SR1.5 with MAGICC using Meinshausen et al. (2020) input files (as was used for
    # main WG1 RCB calculations)
    jobno = "20210224-sr15"
    magiccver = "7.5.1"
    input_folder = "../InputData/MAGICCMeinshausenInputs_sr15scen/"
    vetted_scen_list_file = input_folder + "sr15_scenario_runs_mocked_vetting.xlsx"
    vetted_scen_list_file_sheet = "meta_Ch3vetted_withclimate"
    fair_folder = "../InputData/FAIR13_sr15/"
    fair_filestr = "IPCCSR15_"
    fair_filter = "../InputData/FAIR13_sr15/constrained/IPCCSR15_REMIND-MAgPIE 1.5_SSP2-45_GAS.SCEN.csv"
else:
    raise ValueError(f"runver {runver} not available, choose ar6wg3, sr15ccbox71 or sr15prewg1")
# FaIR filenames
fair_processed_file = "fair_processed_data_{}.csv"
# The folders for the unscaled anthropological temperature changes files (many nc
# files), skipped if processed data is available
fair_anthro_folder = "anthro_temps/"
fair_co2_only_folder = "co2_temps/"
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
# The percentile to use for non-CO2 temperature change (for each scenario separately).
# This is a string, and must be 50.0 the first time FaIR data is run.
nonco2_percentile = "50.0"
# The names of the temperature variables in MAGICC files (also specifies the quantile)
magicc_nonco2_temp_variable = "{} climate diagnostics|Raw Surface Temperature (GSAT)|Non-CO2|MAGICCv{}|{}th Percentile".format(
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
# Should we normalise FaIR and MAGICC temperature trends to 2010-2019 before calculating
# quantiles?
norm_nonco2_years = False
if norm_nonco2_years:
    output_folder = output_folder + "ny/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
# ______________________________________________________________________________________

t0 = time.time()
if peak_version and (peak_version == "officialNZ"):
    assert vetted_scen_list_file is not None
for use_permafrost in List_use_permafrost:
    permafrost_str = ("permafrost_" if use_permafrost else "no_perm_")
    sr15_rename = (arsr == "sr15")
    if use_permafrost:
        non_co2_magicc_file = non_co2_magicc_file_permafrost
        tot_magicc_file = tot_magicc_file_permafrost
    else:
        non_co2_magicc_file = non_co2_magicc_file_no_permafrost
        tot_magicc_file = tot_magicc_file_nopermafrost
    # If we want to calculate using quantiles normalised to 2010-2019 values we need to
    # load different files, which may not exist
    newallfile = tot_magicc_file[:-4] + permafrost_str + "yearnorm.csv"
    newnonco2file = non_co2_magicc_file[:-4] + permafrost_str + "yearnorm.csv"
    yearnorm_files = [newallfile, newnonco2file]
    # Proceed with initial calculation
    magicc_peak_version = peak_version if peak_version not in [
        "nonCO2AtPeakTotMagicc", "nonCO2AtPeakAverage"] else "nonCO2AtPeakTotIfNZ"
    if norm_nonco2_years and not any(
            [not os.path.exists(X) for X in yearnorm_files]
    ):
        non_co2_magicc_file1 = newnonco2file
        tot_magicc_file1 = newallfile
    else:
        non_co2_magicc_file1 = non_co2_magicc_file
        tot_magicc_file1 = tot_magicc_file
    magicc_db_full = distributions.load_data_from_summary(
        non_co2_magicc_file1,
        tot_magicc_file1,
        emissions_file,
        magicc_non_co2_col,
        magicc_temp_col,
        magicc_nonco2_temp_variable,
        magicc_tot_temp_variable,
        temp_offset_years,
        magicc_peak_version,
        permafrost=use_permafrost,
        vetted_scen_list_file=vetted_scen_list_file,
        vetted_scen_list_file_sheet=vetted_scen_list_file_sheet,
        sr15_rename=sr15_rename,
    )
    # If we want to normalise by 2010-2019 values first, we must load the full set of
    # values from file. This must be done after a run from the traditional file, since
    # the list of scenarios is longer and must be restricted to match. After the first
    # run, this isn't a problem.
    if norm_nonco2_years and any([not os.path.exists(X) for X in yearnorm_files]):
        allsummary, nonco2summary = distributions.preprocess_MAGICC_data(
            input_folder,
            tot_magicc_file,
            permafrost_str,
            magicc_db_full.reset_index(),
            temp_offset_years,
            magicc_nonco2_temp_variable,
            magicc_tot_temp_variable,
        )
        allsummary.to_csv(newallfile, index=False)
        nonco2summary.to_csv(newnonco2file, index=False)
        magicc_db_full = distributions.load_data_from_summary(
            newnonco2file,
            newallfile,
            emissions_file,
            magicc_non_co2_col,
            magicc_temp_col,
            magicc_nonco2_temp_variable,
            magicc_tot_temp_variable,
            temp_offset_years,
            magicc_peak_version,
            permafrost=use_permafrost,
            vetted_scen_list_file=vetted_scen_list_file,
            vetted_scen_list_file_sheet=vetted_scen_list_file_sheet,
            sr15_rename=sr15_rename,
        )

    magicc_db = magicc_db_full[np.isfinite(magicc_db_full["hits_net_zero"])]

    if magicc_savename:
        magicc_db.to_csv(output_folder + magicc_savename.format(use_permafrost))
        magicc_db_full.to_csv(output_folder + magicc_savename.format(
            use_permafrost).replace('.csv', '-all-scenarios.csv'))

    # We interpret the higher quantiles as meaning a smaller budget
    inverse_quantiles_to_report = 1 - quantiles_to_report
    # Construct the container for saved results
    all_fit_lines = []
    # Modify the following loop to use subsets of data for robustness checks
    for case_ind in range(len(include_list)):
        include_magicc = include_list[case_ind][0]
        include_fair = include_list[case_ind][1]
        if include_fair:
            if os.path.exists(fair_folder + fair_processed_file.format("alltemp" + f"_normyears_{norm_nonco2_years}")):
                nonco2_fair_db = pd.read_csv(fair_folder + fair_processed_file.format("nonco2temp" + f"_normyears_{norm_nonco2_years}"))
                all_fair_db = pd.read_csv(fair_folder + fair_processed_file.format("alltemp" + f"_normyears_{norm_nonco2_years}"))
            else:
                if not os.path.exists(fair_folder):
                    os.makedirs(fair_folder)
                if norm_nonco2_years:
                    all_fair_db, nonco2_fair_db = distributions.preprocess_yearnorm_quantiles_of_fair_data(
                        fair_folder + fair_anthro_folder,
                        fair_folder + fair_co2_only_folder,
                        magicc_db_full.reset_index(),
                        magicc_nonco2_temp_variable,
                        magicc_tot_temp_variable,
                        fair_filestr,
                        fair_filter,
                        temp_offset_years,
                    )
                else:
                    all_fair_db, nonco2_fair_db = distributions.preprocess_FaIR_data(
                        fair_folder + fair_anthro_folder,
                        fair_folder + fair_co2_only_folder,
                        magicc_db_full.reset_index(),
                        magicc_nonco2_temp_variable,
                        magicc_tot_temp_variable,
                        fair_filestr,
                        fair_filter,
                    )
                all_fair_db.to_csv(fair_folder + fair_processed_file.format("alltemp" + f"_normyears_{norm_nonco2_years}"))
                nonco2_fair_db.to_csv(fair_folder + fair_processed_file.format("nonco2temp" + f"_normyears_{norm_nonco2_years}"))
            if peak_version != "nonCO2AtPeakTotMagicc":
                fair_peak_version = peak_version if peak_version != "nonCO2AtPeakAverage" else "nonCO2AtPeakTotIfNZ"
                fair_vetting_file = vetted_scen_list_file
            else:
                fair_peak_version = "officialNZ"
                fair_vetting_file = magicc_db_full
            non_co2_dT_fair = distributions.load_data_from_summary(
                fair_folder + fair_processed_file.format("nonco2temp" + f"_normyears_{norm_nonco2_years}"),
                fair_folder + fair_processed_file.format("alltemp" + f"_normyears_{norm_nonco2_years}"),
                emissions_file,
                magicc_non_co2_col,
                magicc_temp_col,
                magicc_nonco2_temp_variable,
                magicc_tot_temp_variable,
                temp_offset_years,
                fair_peak_version,
                permafrost=None,
                vetted_scen_list_file=fair_vetting_file,
                vetted_scen_list_file_sheet=vetted_scen_list_file_sheet,
                sr15_rename=sr15_rename,
            )
            non_co2_dT_fair = non_co2_dT_fair[np.isfinite(non_co2_dT_fair["hits_net_zero"])]
            if fair_savename:
                non_co2_dT_fair.to_csv(output_folder + fair_savename.format("None"), index=False)
            if include_magicc:
                assert len(non_co2_dT_fair) == len(magicc_db), "FaIR and MAGICC mismatch"
                if peak_version != "nonCO2AtPeakAverage":
                    master_all_non_co2 = (non_co2_dT_fair + magicc_db) / 2
                else:
                    master_all_non_co2 = distributions.load_data_from_summary(
                        non_co2_magicc_file,
                        tot_magicc_file,
                        emissions_file,
                        magicc_non_co2_col,
                        magicc_temp_col,
                        magicc_nonco2_temp_variable,
                        magicc_tot_temp_variable,
                        temp_offset_years,
                        magicc_peak_version,
                        permafrost=use_permafrost,
                        vetted_scen_list_file=vetted_scen_list_file,
                        vetted_scen_list_file_sheet=vetted_scen_list_file_sheet,
                        sr15_rename=sr15_rename,
                        second_non_co2_file=fair_folder + fair_processed_file.format("nonco2temp" + f"_normyears_{norm_nonco2_years}"),
                        second_tot_file=fair_folder + fair_processed_file.format("alltemp" + f"_normyears_{norm_nonco2_years}"),
                    )
                master_all_non_co2 = master_all_non_co2.loc[
                    magicc_db.index, [magicc_non_co2_col, magicc_temp_col]
                ]
            else:
                master_all_non_co2 = non_co2_dT_fair[[magicc_non_co2_col, magicc_temp_col]]
        else:
            master_all_non_co2 = magicc_db[[magicc_non_co2_col, magicc_temp_col]]
        if for_each_model:
            models = list(dict.fromkeys(master_all_non_co2.reset_index()["model"]))
            model_size = pd.Series(index=models, dtype=int)
        else:
            models = [""]
        for model in models:
            budget_quantiles = pd.DataFrame(index=dT_targets,
                                            columns=quantiles_to_report)
            budget_quantiles.index.name = "dT_targets"
            if for_each_model:
                all_non_co2_db = master_all_non_co2.loc[
                    [x == model for x in master_all_non_co2.index.get_level_values("model")],
                    :
                ]
                if type(for_each_model) == str:
                    all_non_co2_db = all_non_co2_db.loc[
                        [x[:len(for_each_model)] == for_each_model for x in all_non_co2_db.index.get_level_values("scenario")],
                        :
                    ]
                print(f"length of model {model} is {len(all_non_co2_db)}")
                model = model.replace("/", "_").replace(" ", "_")
                # We can't reasonably compute values with very few scenarios
                if all_non_co2_db.shape[0] < min_scenarios:
                    continue
                model_size[model] = all_non_co2_db.shape[0]
            else:
                all_non_co2_db = master_all_non_co2
            if use_as_median_non_co2:
                if type(use_as_median_non_co2) == bool:
                    use_as_median_non_co2 = 0.5
                assert use_as_median_non_co2 in quantiles_to_plot, (
                    "The median value, use_as_median_non_co2, normally=0.5, "
                    "must be included if use_as_median_non_co2 is not false"
                )
                # The quantile regression program is temperamental, so we ensure the data has
                # the correct numeric format before passing it
                x = all_non_co2_db[magicc_temp_col].astype(np.float64)
                y = all_non_co2_db[magicc_non_co2_col].astype(np.float64)
                xy_df = pd.DataFrame({"x": x, "y": y})
                xy_df = xy_df.reset_index(drop=True)
                if (not nonlinear_nonco2) | (nonlinear_nonco2 == "all"):
                    quantile_reg_trends = budget_func.quantile_regression_find_relationships(
                        xy_df, quantiles_to_plot
                    )
                    non_co2_dTs = distributions.establish_median_temp_dep_linear(
                        quantile_reg_trends, dT_targets - historical_dT,
                        use_as_median_non_co2)
                elif nonlinear_nonco2 == "rollingQuantiles":
                    quantile_reg_trends_nonlin = budget_func.quantile_regression_find_relationships_nonlin(
                        xy_df, quantiles_to_plot, smoothing=20
                    )
                    non_co2_dTs = distributions.establish_median_temp_dep_nonlinear(
                        quantile_reg_trends_nonlin, dT_targets - historical_dT, use_as_median_non_co2
                    )
                elif nonlinear_nonco2 == "QRW":
                    quantile_reg_trends_nonlin = budget_func.quantile_regression_quantile_rolling_windows(
                        xy_df, quantiles_to_plot,
                    )
                    non_co2_dTs = distributions.establish_median_temp_dep_nonlinear(
                        quantile_reg_trends_nonlin, dT_targets - historical_dT,
                        use_as_median_non_co2
                    )
                elif nonlinear_nonco2 == "interp":
                    # Other quantiles don't make sense with this
                    quantiles_to_plot = [0.5]
                    quantile_reg_trends_nonlin = budget_func.quantile_regression_find_relationships_interpolate(xy_df)
                    non_co2_dTs = distributions.establish_median_temp_dep_nonlinear(
                        quantile_reg_trends_nonlin, dT_targets - historical_dT,
                        use_as_median_non_co2
                    )
                else:
                    raise ValueError(f"Bad input for nonlinear_nonco2, {nonlinear_nonco2}")

                if nonlinear_nonco2 == "all":
                    quantile_reg_trends_nonlin = budget_func.quantile_regression_find_relationships_nonlin(
                        xy_df, quantiles_to_plot, smoothing=20
                    )
                    quantile_reg_trends_nonlin_qrw = budget_func.quantile_regression_quantile_rolling_windows(
                        xy_df, quantiles_to_plot
                    )

            else:
                # If not quantile regression, we use the least squares fit to the non-CO2 data
                non_co2_dTs = distributions.establish_least_sq_temp_dependence(
                    all_non_co2_db,
                    dT_targets - historical_dT,
                    magicc_non_co2_col,
                    magicc_temp_col,
                )

            for dT_target in dT_targets:
                earth_feedback_co2 = budget_func.calculate_earth_system_feedback_co2(
                    dT_target - historical_dT,
                    earth_feedback_co2_per_C_av,
                    earth_feedback_co2_per_C_stdv,
                    n_loops,
                )
                non_co2_dT = non_co2_dTs.loc[dT_target - historical_dT]
                tcres = distributions.tcre_distribution(
                    tcre_low, tcre_high, likelihood, n_loops, tcre_dist
                )
                zec = distributions.zec_dist(zec_mean, zec_sd, zec_asym, n_loops)
                budgets = budget_func.calculate_budget(
                    dT_target, zec, historical_dT, non_co2_dT, tcres, earth_feedback_co2
                )
                budgets = budgets - recent_emissions
                budget_quantiles.loc[dT_target] = np.quantile(
                    budgets, inverse_quantiles_to_report
                )

            # Save output in the correct format
            budget_quantiles = budget_quantiles.reset_index()
            budget_quantiles["Future_warming"] = budget_quantiles["dT_targets"] - historical_dT
            budget_quantiles = budget_quantiles.set_index("Future_warming")
            budget_quantiles.to_csv(
                output_folder + model + output_file.format(
                    tcre_dist,
                    include_magicc,
                    include_fair,
                    round(earth_feedback_co2_per_C_av/convert_PgC_to_GtCO2, 2),
                    round(earth_feedback_co2_per_C_stdv/convert_PgC_to_GtCO2, 2),
                    likelihood,
                    nonco2_percentile,
                    use_permafrost,
                    zec_sd,
                    zec_asym,
                    historical_dT,
                )
            )
            """
            # Convert the data to PgC and save again
            PgC_budget_quantiles = budget_quantiles
            PgC_budget_quantiles[quantiles_to_report] *= 1/3.664
            PgC_budget_quantiles.to_csv(
                output_folder + model + output_file.format(
                    tcre_dist,
                    include_magicc,
                    include_fair,
                    round(earth_feedback_co2_per_C_av/convert_PgC_to_GtCO2, 2),
                    round(earth_feedback_co2_per_C_stdv/convert_PgC_to_GtCO2, 2),
                    likelihood,
                    nonco2_percentile,
                    use_permafrost,
                    zec_sd,
                    zec_asym,
                    historical_dT,
                ).replace("GtCO2", "PgC")
            )
            """
            # Make plots of the data
            temp_plot_limits = [
                min(all_non_co2_db[magicc_temp_col]) + historical_dT,
                max(all_non_co2_db[magicc_temp_col]) + historical_dT,
            ]
            non_co2_plot_limits = [
                min(all_non_co2_db[magicc_non_co2_col]),
                max(all_non_co2_db[magicc_non_co2_col]),
            ]

            def add_fringe(limits, fringe):
                # Helper function for adding a small amount either side of the limits
                assert len(limits) == 2
                offset = fringe * (limits[1] - limits[0])
                limits[0] = limits[0] - offset
                limits[1] = limits[1] + offset
                return limits

            # 0.10 is chosen for the fringes for aesthetic reasons
            temp_plot_limits = add_fringe(temp_plot_limits, 0.11)
            non_co2_plot_limits = add_fringe(non_co2_plot_limits, 0.11)
            plt.close()
            fig = plt.figure(figsize=(6.4*1.2, 4.8*1.2))
            ax = fig.add_subplot(111)
            legend_text = []
            if for_each_model:
                plt.scatter(all_non_co2_db[magicc_temp_col] + historical_dT, all_non_co2_db[magicc_non_co2_col], color="blue")
            else:
                if include_magicc:
                    plt.scatter(
                        magicc_db[magicc_temp_col] + historical_dT, magicc_db[magicc_non_co2_col], color="navy", s=6
                    )
                    legend_text.append("MAGICC")
                if include_fair:
                    plt.scatter(
                        non_co2_dT_fair[magicc_temp_col] + historical_dT, non_co2_dT_fair[magicc_non_co2_col], color="cornflowerblue", s=6
                    )
                    legend_text.append("FaIR")
                if include_fair and include_magicc:
                    plt.scatter(
                        all_non_co2_db[magicc_temp_col] + historical_dT, all_non_co2_db[magicc_non_co2_col], color="b", s=6
                    )
                    legend_text.append("Averaged")
            #plt.xlim(temp_plot_limits)
            #plt.ylim(non_co2_plot_limits)

            plt.ylabel("Non-CO$_2$ warming relative to 2010-2019 (C)")
            plt.xlabel("Peak total warming (C)")
            if not use_as_median_non_co2:
                x = all_non_co2_db[magicc_temp_col] + historical_dT
                y = all_non_co2_db[magicc_non_co2_col]
                equation_of_fit = np.polyfit(x, y, 1)
                all_fit_lines.append(equation_of_fit)
                plt.plot(
                    np.unique(x), np.poly1d(equation_of_fit)(np.unique(x)), color="dimgrey"
                )
                legend_text.append("line of best fit")
                # Write the equation
                equation_text = "y = " + str(
                    round(equation_of_fit[0], 4)
                ) + "x" " + " + str(
                    round(equation_of_fit[1], 4)
                )
                plt.text(
                    0.9,
                    0.1,
                    equation_text,
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
                quantiles_of_plot = budget_func.rolling_window_find_quantiles(
                    xs=all_non_co2_db[magicc_temp_col],
                    ys=all_non_co2_db[magicc_non_co2_col],
                    quantiles=quantiles_to_plot,
                    nwindows=10,
                )
                x = quantiles_of_plot.index.values
                for col in quantiles_of_plot.columns:
                    y = quantiles_of_plot[col].values
                    if col == 0.5:
                        dashes = [1, 0]
                        color = "black"
                    else:
                        dashes = [6, 2]
                        color = "grey"
                    plt.plot(
                        np.unique(x),
                        np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),
                        dashes=dashes,
                        color=color,
                    )
                    legend_text.append(f"QRW {col} quantile")
            else:
                minT = temp_plot_limits[0]
                maxT = temp_plot_limits[1]
                if (not nonlinear_nonco2) | (nonlinear_nonco2 == "all"):
                    for i in range(len(quantile_reg_trends)):
                        plt.plot(
                            (minT, maxT),
                            (
                                quantile_reg_trends["b"][i] * (minT - historical_dT) + quantile_reg_trends["a"][i],
                                quantile_reg_trends["b"][i] * (maxT - historical_dT) + quantile_reg_trends["a"][i],
                            ),
                            ls=line_dotting[i],
                            color="dimgrey", alpha=0.8
                        )
                        legend_text.append(
                            "Linear quantile {}".format(quantile_reg_trends.loc[i, "quantile"])
                        )

                if nonlinear_nonco2 in ["rollingQuantiles", "QRW", "interp"]:
                    for i, q in enumerate(quantile_reg_trends_nonlin.columns[1:]):
                        plt.plot(
                            quantile_reg_trends_nonlin["x"] + historical_dT,
                            quantile_reg_trends_nonlin[q],
                            ls=line_dotting[i],
                            color="cyan",
                        )
                        if nonlinear_nonco2 == "QRW":
                            legend_text.append(f"QRW {q} quantile")
                        elif nonlinear_nonco2 == "interp":
                            legend_text.append("Interpolated fit")
                        else:
                            legend_text.append(f"Rolling {q} quantile")
                if nonlinear_nonco2 == "all":
                    for i, q in enumerate(quantile_reg_trends_nonlin_qrw.columns[1:]):
                        plt.plot(
                            quantile_reg_trends_nonlin_qrw["x"] + historical_dT,
                            quantile_reg_trends_nonlin_qrw[q],
                            ls=line_dotting[i],
                            color="darkorange",
                        )
                        legend_text.append(f"QRW {q} quantile")

            plt.legend(legend_text)
            if not for_each_model:
                plt.plot(
                    [temp_plot_limits[0]+0.05, temp_plot_limits[0]+0.05],
                    [non_co2_plot_limits[1] - 0.03 - (tcre_high+tcre_low)/2*100,
                     non_co2_plot_limits[1] - 0.03],
                    lw=10
                )
                plt.text(
                    temp_plot_limits[0]+0.07, non_co2_plot_limits[1] - 0.0575,
                    "Warming from 100 GtCO$_2$"
                )
            fig.savefig(
                output_folder + model + output_figure_file.format(
                    include_magicc, include_fair, use_permafrost, zec_sd, zec_asym,
                    nonco2_percentile, nonlinear_nonco2
                ),
                bbox_inches="tight"
            )
        plt.close()
        # Plot all trendlines together, if data was processed without median values
        if not use_as_median_non_co2:
            x = temp_plot_limits
            y = non_co2_plot_limits
            fig = plt.figure(figsize=(12, 7))
            ax = fig.add_subplot(111)
            colours = ["black", "blue", "orange"]
            for eqn in all_fit_lines:
                plt.plot(np.unique(x), np.poly1d(eqn)(np.unique(x)), color=colours[0])
                colours = colours[1:]
            plt.xlim(temp_plot_limits)
            plt.ylim(non_co2_plot_limits)
            plt.legend(["MAGICC and FaIR", "MAGICC only", "FaIR only"])
            plt.ylabel(magicc_non_co2_col)
            plt.xlabel(magicc_temp_col)
            fig.savefig(
                output_folder + model + output_all_trends.format(
                    use_permafrost, nonlinear_nonco2
                )
            )
        if waterfall_plot:
            for temp in [1.5, 2.0]:
                plt.close()
                TCRE = (tcre_high + tcre_low) / 2
                xvals = [
                    "Historical Warming", "Recent emissions warming",
                    "Other Earth Systems Feedback",
                    "Future non-CO$_2$ warming", "Zero Emissions Commitment",
                    "Remaining CO$_2$ warming",
                ]
                yvals = [
                    historical_dT, recent_emissions * TCRE,
                                   (
                                               temp - historical_dT) * earth_feedback_co2_per_C_av * TCRE,
                    non_co2_dTs[
                        np.isclose(non_co2_dTs.index, temp - historical_dT)].iloc[0],
                    zec_mean
                ]
                yvals = yvals + [temp - np.cumsum(yvals)[-1]]

                if (not nonlinear_nonco2) or (nonlinear_nonco2 == "all"):
                    lowind = np.isclose(quantile_reg_trends["quantile"], 0.17)
                    highind = np.isclose(quantile_reg_trends["quantile"], 0.83)
                    nonco2errorlow = \
                    non_co2_dTs[np.isclose(non_co2_dTs.index, temp - historical_dT)
                    ].iloc[0] - (quantile_reg_trends.loc[lowind, "a"].iloc[0] +
                                 quantile_reg_trends.loc[
                                     lowind, "b"].iloc[0] * (temp - historical_dT))
                    nonco2errorhigh = -non_co2_dTs[
                        np.isclose(non_co2_dTs.index, temp - historical_dT)
                    ].iloc[0] + (quantile_reg_trends.loc[highind, "a"].iloc[0] + \
                                 quantile_reg_trends.loc[
                                     highind, "b"].iloc[0] * (temp - historical_dT))
                else:
                    nonco2errorlow = distributions.establish_median_temp_dep_nonlinear(
                        quantile_reg_trends_nonlin, [temp - historical_dT], 0.17
                    ).iloc[0]
                    nonco2errorhigh = distributions.establish_median_temp_dep_nonlinear(
                        quantile_reg_trends_nonlin, [temp - historical_dT], 0.83
                    ).iloc[0]
                # The uncertainty in historical warming is added elsewhere
                errorlow = [
                    0, recent_emissions_uncertainty * TCRE,
                       TCRE * (temp - historical_dT) * earth_feedback_co2_per_C_stdv,
                       nonco2_waterfall_uncertainty[temp][0] * TCRE,
                    zec_sd,
                ]
                totallowerror = (np.sum([x ** 2 for x in errorlow]) +
                            historical_uncertainty ** 2 + nonco2errorlow ** 2) ** 0.5
                totallowerror_unsyst = np.sum([x ** 2 for x in errorlow]) ** 0.5
                errorlow = errorlow + [totallowerror_unsyst]
                errorhigh = [
                    0, recent_emissions_uncertainty * TCRE,
                       TCRE * (temp - historical_dT) * earth_feedback_co2_per_C_stdv,
                       nonco2_waterfall_uncertainty[temp][1] * TCRE,
                    zec_sd
                ]
                totalhigherror = (np.sum(
                    [x ** 2 for x in errorhigh]) + historical_uncertainty ** 2 +
                                  nonco2errorhigh ** 2) ** 0.5
                totalhigherror_unsyst = np.sum([x ** 2 for x in errorhigh]) ** 0.5
                errorhigh = errorhigh + [totalhigherror_unsyst]
                waterfall.plot(
                    xvals,
                    yvals,
                    red_color="orangered",
                    green_color="cornflowerblue",
                    blue_color="mediumblue",
                    net_label="Temperature target"
                )
                plt.xticks(rotation=45, horizontalalignment="right")
                plt.ylabel("Temperature rise from 1850-1900 ($^o$C)")
                cumsumy = np.cumsum(yvals)
                plt.errorbar([0.15, len(yvals) - 1 + 0.15],
                             [cumsumy[0], cumsumy[len(yvals) - 1]],
                             yerr=np.array([
                                 [historical_uncertainty, totallowerror],
                                 [historical_uncertainty, totalhigherror]
                             ]), capsize=2.5, fmt='.', c="deeppink")
                plt.errorbar(np.arange(len(xvals))-0.15, cumsumy,
                             yerr=np.array([errorlow, errorhigh]), fmt='.',
                             c="royalblue", capsize=3)
                plt.errorbar(
                    3, cumsumy[3],
                    np.array([[nonco2errorlow], [nonco2errorhigh]]),
                    fmt='.', alpha=0.7,
                    c="cyan", capsize=3)
                plt.tight_layout()
                plt.savefig(
                    output_folder + waterfall_plot.format(
                        include_magicc, include_fair, peak_version, temp, recent_emissions,
                        zec_mean, use_permafrost
                    )
                )
                print(f"Total temperature error for MAGICC {include_magicc}, FaIR {include_fair} temp {temp}: {totallowerror}, {totalhigherror}")
if for_each_model:
    model_size.to_csv(
        output_folder + f"num_scenarios_for_model_{for_each_model}{'_' + peak_version if peak_version else ''}.csv"
    )
print("Time taken: ", time.time() - t0)
print("The analysis has completed.")
