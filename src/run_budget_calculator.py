import os

import numpy as np
import pandas as pd
import src.distributions_of_inputs as distributions
import src.budget_calculator_functions as budget_func
import matplotlib.pyplot as plt
import time

from config import *

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
    magicc_db = magicc_db_full[np.isfinite(magicc_db_full["hits_net_zero"])]

    if magicc_savename:
        magicc_db.to_csv(output_folder + magicc_savename.format(use_permafrost))
        magicc_db_full.to_csv(output_folder + magicc_savename.format(use_permafrost).replace('.csv', '-all-scenarios.csv'))

    # We interpret the higher quantiles as meaning a smaller budget
    inverse_quantiles_to_report = 1 - quantiles_to_report
    # Construct the container for saved results
    all_fit_lines = []
    # Modify the following loop to use subsets of data for robustness checks
    for case_ind in range(len(include_list)):
        include_magicc = include_list[case_ind][0]
        include_fair = include_list[case_ind][1]
        if include_fair:
            if magicc_savename:
                fair_file_name = output_folder + fair_savename
                if os.path.exists(fair_file_name):
                    non_co2_dT_fair = pd.read_csv(fair_file_name)
            if (magicc_savename == None) or not os.path.exists(fair_file_name):
                model_col = "model"
                scenario_col = "scenario"
                fair_offset_years = np.arange(2010, 2020, 1)
                non_co2_dT_fair = distributions.load_data_from_FaIR(
                    fair_anthro_folder,
                    fair_co2_only_folder,
                    magicc_db.reset_index(),
                    model_col,
                    scenario_col,
                    magicc_non_co2_col,
                    magicc_temp_col,
                    fair_offset_years,
                    fair_filestr,
                    peak_version,
                )
                non_co2_dT_fair.to_csv(fair_file_name, index=False)
            if include_magicc:
                non_co2_dT_fair = non_co2_dT_fair.set_index("magicc_ind")
                master_all_non_co2 = pd.DataFrame(
                    index=non_co2_dT_fair.index,
                    columns=[magicc_non_co2_col, magicc_temp_col]
                )
                master_all_non_co2[magicc_non_co2_col] = (non_co2_dT_fair[
                    magicc_non_co2_col] + magicc_db.reset_index()[magicc_non_co2_col]) /2
                master_all_non_co2[magicc_temp_col] = (non_co2_dT_fair[
                     magicc_temp_col] + magicc_db.reset_index()[magicc_temp_col]) / 2
                master_all_non_co2.index = magicc_db.index
            else:
                master_all_non_co2 = non_co2_dT_fair[[magicc_non_co2_col, magicc_temp_col]]
                master_all_non_co2.index = magicc_db.index
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
            temp_plot_limits = add_fringe(temp_plot_limits, 0.10)
            non_co2_plot_limits = add_fringe(non_co2_plot_limits, 0.10)
            plt.close()
            fig = plt.figure(figsize=(12, 7))
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
            plt.xlim(temp_plot_limits)
            plt.ylim(non_co2_plot_limits)

            plt.ylabel("Non-CO$_2$ warming relative to 2010-2019 (C)")
            plt.xlabel("Peak total warming (C)")
            if not use_as_median_non_co2:
                x = all_non_co2_db[magicc_temp_col] + historical_dT
                y = all_non_co2_db[magicc_non_co2_col]
                equation_of_fit = np.polyfit(x, y, 1)
                all_fit_lines.append(equation_of_fit)
                plt.plot(
                    np.unique(x), np.poly1d(equation_of_fit)(np.unique(x)), color="black"
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
                            color="black",
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
                            color="red",
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
if for_each_model:
    model_size.to_csv(
        output_folder + f"num_scenarios_for_model_{for_each_model}{'_' + peak_version if peak_version else ''}.csv"
    )
print("Time taken: ", time.time() - t0)
print("The analysis has completed.")
