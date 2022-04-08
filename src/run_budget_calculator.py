import os

import numpy as np
import pandas as pd
import src.distributions_of_inputs as distributions
import src.budget_calculator_functions as budget_func
import matplotlib.pyplot as plt
import time

## Input values
# Edit as required
# ______________________________________________________________________________________

# The target temperature changes to achieve. (Units: C)
dT_targets = np.arange(1.1, 2.6, 0.1)
# The number of loops performed for each temperature. Runs in seconds for ~ 10^6, higher
# reduces statistical error but takes longer
n_loops = 10000000
# The change in temperature that will occur after zero emissions has been reached.
# (Units: C)
zec = 0.0
# The temperature difference already seen. (Units: C)
historical_dT = 1.07
# The distribution of the TCRE function - either "normal", "lognormal mean match" or
# "lognormal". The latter two cases are lognormal distributions, in the first
# case matching the mean and sd of the normal distribution which fits the likelihood,
# in the second case matching the likelihood limits itself but exhibiting some skew.
tcre_dist = "normal"
# Constant to convert between Pg C and Gt CO2.
convert_PgC_to_GtCO2 = 3.664
# The upper and lower bounds of the distribution of TCRE. We use units of degC per GtCO2
# (TCRE = Transient climate response to cumulative carbon emissions)
tcre_low = 1.0 / 1000 / convert_PgC_to_GtCO2
tcre_high = 2.3 / 1000 / convert_PgC_to_GtCO2
# likelihood is the probability that results fit between the low and high value
likelihood = 0.6827
# Average CO2 emissions per degree C from temperature-dependent Earth feedback loops.
# (Units: GtCO2/C)
earth_feedback_co2_per_C_av = 7.1 * convert_PgC_to_GtCO2
# Standard dev of CO2 emissions per degree C from temperature-dependent Earth feedback
# St dev CO2 emissions per degree C from temperature-dependent Earth feedback loops.
# loops (Units: Gt CO2/C).
earth_feedback_co2_per_C_stdv = 26.7 * convert_PgC_to_GtCO2
# Any emissions that have taken place too recently to have factored into the measured
# temperature change, and therefore must be subtracted from the budget (Units: GtCO2)
recent_emissions = 208.81  # 0
# We will present the budgets at these probability quantiles.
quantiles_to_report = np.array([0.17, 0.33, 0.5, 0.66, 0.83])

# Run version should be ar6wg3, sr15ccbox71, or sr15wg1, corresponding to running the
# code using the AR6 database as used for WG3, or the SR1.5 database with either the
# cross-chapter box 7.1/ Nicholls 2021 configuration of MAGICC or the older Meinshausen
# 2020 configuration, as was used in the AR6 WG1 report.
runver = "sr15wg1"

# Name of the output folder
if runver == "ar6wg3":
    output_folder = "../Output/ar6wg3draft2/"
    arsr = "ar6"
elif runver == "sr15ccbox71":
    output_folder = "../Output/sr15-scenarios-ccbox71-magicc/"
    arsr = "sr15"
elif runver == "sr15wg1":
    output_folder = "../Output/sr15-scenarios-meinsahusen-magicc/"
    arsr = "sr15"
else:
    raise ValueError(f"runver {runver} not available, choose ar6wg3, sr15ccbox71 or sr15wg1")
os.makedirs(output_folder, exist_ok=True)

# Output file location for budget data. Includes {} sections detailing inclusion of
# TCRE, inclusion of magic/fair, earth system feedback and likelihood. More added later
output_file = (
    output_folder + "budget_{}_magicc_{}_fair_{}_esf_{}pm{}_likeli_{}_nonCO2pc{}_GtCO2_permaf_{}_hdT_{}"
)
# Output location for figure of peak warming vs non-CO2 warming. More appended later
output_figure_file = output_folder + "non_co2_cont_to_peak_warming_magicc_{}_fair_{}_permaf_{}_nonCO2pc{}"
# Quantile fit lines to plot on the temperatures graph.
# If use_as_median_non_co2 evaluates as True, this must include that quantile (by
# default 0.5)
quantiles_to_plot = [0.05, 0.5, 0.95]
# How should we dot these lines? This list must be as long as quantiles_to_plot.
line_dotting = ["--", "-", "--"]
# Should we use the median value from quantile regression (True) or the least-squares
# best fit (False) for the non-CO2 relationship? If instead a number is used, we use
# that quantile, which must be in quantiles_to_plot.
use_as_median_non_co2 = True
# Where should we save the results of the figure with trend lines? Not plotted if
# use_median_non_co2 evaluates as True.
if use_as_median_non_co2 != True:
    output_file = output_file + "_nonco2scenfrac" + str(use_as_median_non_co2)
output_all_trends = output_folder + "TrendLinesWithMagicc_permaf_{}.pdf"

#       Information for reading in files used to calculate non-CO2 component:

#       MAGICC files
# Should we use a variant means of measuring the non-CO2 warming?
# Default = None; ignore scenarios with non-peaking cumulative CO2 emissions, use
# non-CO2 warming in the year of peak cumulative CO2.
# If "peakNonCO2Warming", we use the highest non-CO2 temperature,
# irrespective of emissions peak.
# If "nonCO2AtPeakTot", computes the non-CO2 component at the time of peak total
# temperature.
# If "officialNZ" uses the date of net zero in the metadata used to validate the
# scenarios - validation file must also be used.
peak_version = None

output_file += "_" + str(peak_version) + "_recEm" + str(round(recent_emissions)) + ".csv"
output_figure_file += "_" + str(peak_version) + ".pdf"

# We wish to recall one of several MAGICC runs, depending on the situation
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
elif runver == "sr15ccbox71":
    # SR1.5 with later MAGICC version as in CCBox 7.1
    jobno = "20211014-sr15"
    magiccver = "7.5.3"
    input_folder = "../InputData/MAGICCCCB71_sr15scen/"
    # We use a compound vetting file
    vetted_scen_list_file = input_folder + "sr15_scenario_runs_mocked_vetting.xlsx"
    vetted_scen_list_file_sheet = "Sheet1"
elif runver == "sr15wg1":
    # # SR1.5 with MAGICC using Meinshausen et al. (2020) input files (as was used for WG1 RCB calculations)
    jobno = "20210224-sr15"
    magiccver = "7.5.1"
    input_folder = "../InputData/MAGICCMeinshausenInputs_sr15scen/"
    vetted_scen_list_file = input_folder + "sr15_scenario_runs_mocked_vetting.xlsx"
    vetted_scen_list_file_sheet = "meta_Ch3vetted_withclimate"
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
magicc_savename = output_folder + "magicc_nonCO2_temp_{}Percentile".format(
    nonco2_percentile
) + str(peak_version) + "permaf_{}.csv"
# Years over which we set the average temperature to 0.
# Note that the upper limit of the range is not included in python.
temp_offset_years = np.arange(2010, 2020, 1)
# Use permafrost may be True, False or both (iterates over the list)
List_use_permafrost = [False]

# ______________________________________________________________________________________
# The parts below should not need editing
t0 = time.time()
if peak_version and (peak_version == "officialNZ"):
    assert vetted_scen_list_file is not None
for use_permafrost in List_use_permafrost:
    sr15_rename = (arsr == "sr15")
    if use_permafrost:
        non_co2_magicc_file = non_co2_magicc_file_permafrost
        tot_magicc_file = tot_magicc_file_permafrost
    else:
        non_co2_magicc_file = non_co2_magicc_file_no_permafrost
        tot_magicc_file = tot_magicc_file_nopermafrost

    magicc_db_full = distributions.load_data_from_MAGICC(
        non_co2_magicc_file,
        tot_magicc_file,
        emissions_file,
        magicc_non_co2_col,
        magicc_temp_col,
        magicc_nonco2_temp_variable,
        magicc_tot_temp_variable,
        temp_offset_years,
        peak_version,
        permafrost=use_permafrost,
        vetted_scen_list_file=vetted_scen_list_file,
        vetted_scen_list_file_sheet=vetted_scen_list_file_sheet,
        sr15_rename=sr15_rename,
    )
    try:
        magicc_db = magicc_db_full[magicc_db_full["hits_net_zero"]].drop("hits_net_zero", axis="columns")
    except KeyError:
        magicc_db = magicc_db_full.copy()

    if magicc_savename:
        magicc_db.to_csv(magicc_savename.format(use_permafrost))
        magicc_db_full.to_csv(magicc_savename.format(use_permafrost).replace('.csv', '-all-scenarios.csv'))

    # We interpret the higher quantiles as meaning a smaller budget
    inverse_quantiles_to_report = 1 - quantiles_to_report
    # Construct the container for saved results
    all_fit_lines = []
    # Modify the following loop to use subsets of data for robustness checks
    for case_ind in range(1):
        # At some point we may want to check robustnesss to including alternative simplified
        # models to evaluate non-CO2 impacts. Currently only MAGICC data is used.
        include_magicc = True
        include_fair = False

        budget_quantiles = pd.DataFrame(index=dT_targets, columns=quantiles_to_report)
        budget_quantiles.index.name = "dT_targets"

        all_non_co2_db = magicc_db[[magicc_non_co2_col, magicc_temp_col]]

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
            quantile_reg_trends = budget_func.quantile_regression_find_relationships(
                xy_df, quantiles_to_plot
            )
            non_co2_dTs = distributions.establish_median_temp_dep(
                quantile_reg_trends, dT_targets - historical_dT, use_as_median_non_co2
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
            output_file.format(
                tcre_dist,
                include_magicc,
                include_fair,
                round(earth_feedback_co2_per_C_av/convert_PgC_to_GtCO2, 2),
                round(earth_feedback_co2_per_C_stdv/convert_PgC_to_GtCO2, 2),
                likelihood,
                nonco2_percentile,
                use_permafrost,
                historical_dT,
            )
        )
        # Convert the data to PgC and save again
        PgC_budget_quantiles = budget_quantiles
        PgC_budget_quantiles[quantiles_to_report] *= 1/3.664
        PgC_budget_quantiles.to_csv(
            output_file.format(
                tcre_dist,
                include_magicc,
                include_fair,
                round(earth_feedback_co2_per_C_av/convert_PgC_to_GtCO2, 2),
                round(earth_feedback_co2_per_C_stdv/convert_PgC_to_GtCO2, 2),
                likelihood,
                nonco2_percentile,
                use_permafrost,
                historical_dT,
            ).replace("GtCO2", "PgC")
        )

        # Make plots of the data
        temp_plot_limits = [
            min(magicc_db[magicc_temp_col]),
            max(magicc_db[magicc_temp_col]),
        ]
        non_co2_plot_limits = [
            min(magicc_db[magicc_non_co2_col]),
            max(magicc_db[magicc_non_co2_col]),
        ]

        def add_fringe(limits, fringe):
            # Helper function for adding a small amount either side of the limits
            assert len(limits) == 2
            offset = fringe * (limits[1] - limits[0])
            limits[0] = limits[0] - offset
            limits[1] = limits[1] + offset
            return limits

        # 0.04 is chosen for the fringes for aesthetic reasons
        temp_plot_limits = add_fringe(temp_plot_limits, 0.04)
        non_co2_plot_limits = add_fringe(non_co2_plot_limits, 0.04)
        plt.close()
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111)
        legend_text = []
        if include_magicc:
            plt.scatter(
                magicc_db[magicc_temp_col], magicc_db[magicc_non_co2_col], color="blue"
            )
            legend_text.append("MAGICC")

        plt.xlim(temp_plot_limits)
        plt.ylim(non_co2_plot_limits)
        plt.legend(legend_text)
        plt.ylabel(magicc_non_co2_col)
        plt.xlabel(magicc_temp_col)
        if not use_as_median_non_co2:
            x = all_non_co2_db[magicc_temp_col]
            y = all_non_co2_db[magicc_non_co2_col]
            equation_of_fit = np.polyfit(x, y, 1)
            all_fit_lines.append(equation_of_fit)
            plt.plot(
                np.unique(x), np.poly1d(equation_of_fit)(np.unique(x)), color="black"
            )
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
        else:
            minT = temp_plot_limits[0]
            maxT = temp_plot_limits[1]
            for i in range(len(quantile_reg_trends)):
                plt.plot(
                    (minT, maxT),
                    (
                        quantile_reg_trends["b"][i] * minT + quantile_reg_trends["a"][i],
                        quantile_reg_trends["b"][i] * maxT + quantile_reg_trends["a"][i],
                    ),
                    ls=line_dotting[i],
                    color="black",
                )
        fig.savefig(
            output_figure_file.format(
                include_magicc, include_fair, use_permafrost, nonco2_percentile
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
        fig.savefig(output_all_trends.format(use_permafrost))
print("Time taken: ", time.time() - t0)
print("The analysis has completed.")
