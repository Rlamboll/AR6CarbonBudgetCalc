import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import waterfall_chart as waterfall
import seaborn as sns

results_folder = "../Output/"
subfolders = ["sr15prewg1/", "ar6wg3/", "sr15ccbox71/"]
nonco2_model_uncertainty = 100

# The results will go into the output folder, in this subfolder:
plot_folder = "Plots/"
if not os.path.exists(results_folder + plot_folder):
    os.makedirs(results_folder + plot_folder)

# If plot_distn is "allquant" we plot the effect of the TCRE distribution on results.
# If the empty string, we calculate the impact of many different changes
plot_distn = ""
# Files to read will have this basic structure:
file_format = plot_distn + "budget_{}_magicc_{}_fair_{}_esf_{}pm26.7_likeli_0.6827_" \
              "nonCO2pc50.0_GtCO2_permaf_{}_zecsd_{}_asym_{}_hdT_1.07NonlinNonCO2_{}_{}_recEm{}.csv"
cols = ["Database", "TCRE distribution", "MAGICC", "FaIR", "ESF", "Permafrost", "ZECsd", "ZEC asymmetry","NonCO2", ]
results_table = pd.DataFrame(
    columns=cols
)
for subfolder in subfolders:
    for distribution in ["normal", "lognormal"]:
        for MAGICC in [True, False]:
            for FaIR in [True, False]:
                for ESF in [7.1]:
                    for Permafrost in [True, False]:
                        for NonCO2 in ["all", "QRW"]:
                            for zecsd in ["0.0", "0.19", "0.3"]:
                                for zecasym in [True, False]:
                                    for peak in [
                                        "None", "peakNonCO2Warming", "nonCO2AtPeakTot",
                                        "officialNZ", "nonCO2AtPeakTotIfNZ", "nonCO2AtPeakAverage"
                                    ]:
                                        for recem in ["326", "209"]:
                                            for normyear in [True, False]:
                                                try:

                                                    filename = file_format.format(
                                                        distribution, MAGICC, FaIR,
                                                        ESF,
                                                        Permafrost, zecsd, zecasym,
                                                        NonCO2, peak, recem
                                                    )
                                                    if normyear:
                                                        results = pd.read_csv(
                                                            results_folder + subfolder + "ny/" + filename
                                                        )
                                                    else:
                                                        results = pd.read_csv(
                                                            results_folder + subfolder + filename
                                                        )
                                                except FileNotFoundError:
                                                    continue

                                                for col, res in [
                                                    ("Database", subfolder[:-1].upper()),
                                                    ("TCRE distribution", distribution),
                                                    ("MAGICC", MAGICC),
                                                    ("FaIR", FaIR),
                                                    ("ESF", ESF),
                                                    ("Permafrost", Permafrost),
                                                    ("ZECsd", zecsd),
                                                    ("ZEC asymmetry", zecasym),
                                                    ("NonCO2", NonCO2),
                                                    ("peak", peak),
                                                    ("recem", recem),
                                                    ("normyear", normyear),
                                                ]:
                                                    results[col] = res
                                                results_table = results_table.append(results)
results_table = results_table.reset_index(drop=True)

# Put in some defaults
d0 = "AR6WG3"
ESF0 = 7.1
zec_sd0 = "0.19"
p0 = False
as0 = False
NonCO20 = "all"
peak0 = "None"
ZECas0 = False
recem0 = "326"
recemorig = "209"
normyear0 = False

if not plot_distn:
    for futwarm in [0.43, 0.93]:
        plt.close()
        use_results = results_table.loc[
            ([r in ["SR15CCBOX71", "SR15PREWG1"] for r in results_table["Database"]]) &
            (np.isclose(results_table["Future_warming"], futwarm)) &
            (results_table["TCRE distribution"] == "normal") &
            (results_table["Permafrost"] == False) &
            (results_table["ESF"] == 7.1) &
            (results_table["peak"] == peak0) &
            (results_table["ZEC asymmetry"] == ZECas0) &
            (results_table["recem"] == recem0) &
            (results_table["normyear"] == normyear0),
            :
        ].copy()
        use_results.loc[:, "Updated"] = ["yes" if y == "SR15CCBOX71" else "no" for y in use_results["Database"]]
        use_results.loc[:, "Model"] = "MAGICC"
        use_results.loc[~use_results["MAGICC"].astype(bool), "Model"] = "FaIR"
        use_results.loc[use_results["MAGICC"] & use_results["FaIR"], "Model"] = "MAGICC and FaIR"
        use_results = use_results.melt(
            var_name="Quantile",
            value_name="Budget",
            value_vars=["0.1", "0.17", "0.33", "0.5", "0.66", "0.83", "0.9"],
            id_vars=cols + ["Updated", "Model"]
        )
        use_results["NonCO2"] = [x if x!="all" else "linear" for x in use_results["NonCO2"]]

        def add_median_labels(ax, fmt='.0f'):
            lines = ax.get_lines()
            boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
            lines_per_box = int(len(lines) / len(boxes))
            for median in lines[4:len(lines):lines_per_box]:
                x, y = (data.mean() for data in median.get_data())
                # choose value depending on horizontal or vertical plot orientation
                value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
                text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                               fontweight='bold', color='white')
                # create median-colored border around white text for contrast
                text.set_path_effects([
                    path_effects.Stroke(linewidth=3, foreground=median.get_color()),
                    path_effects.Normal(),
                ])
        cplot = sns.catplot(
            data=use_results, x="Model", hue="Updated", y="Budget", kind="box", hue_order=["no", "yes"], col="NonCO2"
        )
        cplot.set(ylabel="Remaining Budget (GtCO$_2$)")
        for ax in cplot.axes.ravel():
            add_median_labels(ax)
        plt.savefig(
            results_folder + plot_folder + f"updates_Distn_ftwarm{futwarm}.png")
        for quant in ["0.5", "0.66"]:
            estPostupdate = use_results["Budget"][
                (use_results["Updated"] == "yes") &
                (use_results["Model"] == "MAGICC and FaIR") &
                (use_results["NonCO2"] == "linear") &
                (use_results["Quantile"] == quant)
            ]
            assert len(estPostupdate) == 1
            estPostupdate = estPostupdate.iloc[0]
            estPreupdate = use_results["Budget"][
                (use_results["Updated"] == "no") &
                (use_results["Model"] == "MAGICC and FaIR") &
                (use_results["NonCO2"] == "linear") &
                (use_results["Quantile"] == quant)
            ]
            assert len(estPreupdate) == 1
            estPreupdate = estPreupdate.iloc[0]
            print(f"Effect of updates on {futwarm} budget at p {quant} combined estimate: {(estPostupdate - estPreupdate)/estPreupdate}")

    # Calculate fractional change caused by some modifications
    compare = []
    abs_comp = []
    quant_list = ["0.5", "0.66", "0.83", "0.9"]
    hist_warm = 1.07
    for futwarm in [0.43, 0.93]:
        basebool = pd.DataFrame({
            "Database": (results_table["Database"] == d0),
            "TCRE distribution": (results_table["TCRE distribution"]=="normal"),
            "Future_warming": np.isclose(results_table["Future_warming"], futwarm),
            "SCM": results_table["MAGICC"] & results_table["FaIR"],
            "ESF": (results_table["ESF"] == ESF0),
            "Permafrost": (results_table["Permafrost"] == p0),
            "ZECsd": (results_table["ZECsd"] == zec_sd0),
            "ZEC asymmetry": (results_table["ZEC asymmetry"] == as0),
            "NonCO2": (results_table["NonCO2"] == NonCO20),
            "Peak": (results_table["peak"] == peak0),
            "recem": (results_table["recem"] == recem0),
            "normyear": (results_table["normyear"] == normyear0),
        })

        baseline = results_table.iloc[
            [bool(x) for x in np.product(basebool, axis=1)]
        ]
        assert len(baseline) == 1
        try:
            for ind, column, trueval, description in [
                (0, "Database",       "SR15CCBOX71", "Use SR1.5 database"),
                (1, "TCRE distribution", "lognormal", "Lognormal TCRE distribution"),
                (5, "Permafrost",       True, "Include permafrost in MAGICC results"),
                (6, "ZECsd",            "0.0", "ZEC standard deviation 0"),
                (6, "ZECsd",            "0.3", "ZEC standard deviation 0.3"),
                (7, "ZEC asymmetry",    True, "ZEC only impacts if positive"),
                (8, "NonCO2",           "QRW", "Use QRW for non-CO$_2$ fit"),
                (9, "peak",             "peakNonCO2Warming", "Maximum Non-CO$_2$ warming"),
                (9, "peak",             "nonCO2AtPeakTot", "Non-CO$_2$ warming at peak total temp"),
                (9, "peak",             "officialNZ", "Non-CO$_2$ warming at preharmonised NZ"),
                (9, "peak",             "nonCO2AtPeakTotIfNZ", "Non-CO$_2$ warming at peak total, only NZ scenarios"),
                (9, "peak",             "nonCO2AtPeakAverage", "Non-CO$_2$ warming at peak average total, only NZ scenarios"),
                (10, "recem",           "209", "Recent emissions"),
                (11, "normyear",        True, "Non-CO2 normalised 2010-2019")
            ]:
                db = results_table.iloc[
                    [bool(x) for x in (np.product(basebool.iloc[:, 0:ind], axis=1).astype(int) &
                        (np.product(basebool.iloc[:, ind + 1:], axis=1)).astype(int) &
                        (results_table[column] == trueval))]
                ]
                assert len(db) == 1, f"Missing scenario with {column} value {trueval}"
                compare.append(
                    [futwarm + hist_warm, quant_list, description, (
                            (db[quant_list].values - baseline[quant_list].values) /
                            baseline[quant_list].values
                    )[0]]
                )
                abs_comp.append(
                    [futwarm + hist_warm, quant_list, description,
                    (db[quant_list].values - baseline[quant_list].values)[0]]
                )

            comparedf = pd.DataFrame(
                compare, columns=["Warming", "Quantile", "Change", "Percentage change"]
            )
            comparedf = comparedf.explode(["Quantile", "Percentage change"])
            comparedf["Percentage change"] = [round(100 * x, 1) for x in comparedf["Percentage change"]]
            comparedf = comparedf.pivot(
                columns="Quantile",
                index=[c for c in comparedf.columns if c not in ["Quantile", "Percentage change"]],
                values="Percentage change"
            ).reset_index()
            comparedf.to_csv(results_folder + "relative_impact_of_changes.csv")
            abschangecol = "Impact (GtCO2)"
            abs_comp_df = pd.DataFrame(
                abs_comp, columns=["Warming", "Quantile", "Change", abschangecol]
            )
            abs_comp_df = abs_comp_df.explode(["Quantile", abschangecol])
            abs_comp_df["Impact (GtCO2)"] = [round(x, 0) for x in abs_comp_df["Impact (GtCO2)"]]
            abs_comp_df = abs_comp_df.pivot(
                columns="Quantile",
                index=[c for c in abs_comp_df.columns if
                       c not in ["Quantile", abschangecol]],
                values=abschangecol
            ).reset_index()
            abs_comp_df.to_csv(results_folder + "absolute_impact_of_changes.csv")
            # Plot contribution of different factors
        except AssertionError:
            print(f"Not enough values to calculate {column} value {trueval}")

    for (futwarm, quant_want) in [(0.43, "0.5"), (0.93, "0.5")]:
        basebool = pd.DataFrame({
            "Future_warming": np.isclose(results_table["Future_warming"], futwarm),
            "TCRE distribution": (results_table["TCRE distribution"]=="normal"),
            "MAGICC": results_table["MAGICC"],
            "ESF": (results_table["ESF"] == ESF0),
            "ZECsd": (results_table["ZECsd"] == zec_sd0),
            "ZEC asymmetry": (results_table["ZEC asymmetry"] == as0),
            "Permafrost": (results_table["Permafrost"] == p0),
            "Peak": (results_table["peak"] == peak0),
            "NonCO2": (results_table["NonCO2"] == NonCO20),
            "FaIR": (results_table["FaIR"] == False),
            "recem": (results_table["recem"] == recemorig),
            "Database": (results_table["Database"] == "SR15PREWG1"),
            "normyear": (results_table["normyear"] == normyear0)
        })

        origline = results_table.iloc[
            [bool(x) for x in np.product(basebool, axis=1)]
        ]
        yvals = [origline[quant_want].iloc[0]]
        xvals = ["AR6 WGI"]
        quanthigh = "0.33"
        quantlow = "0.66"
        errorlow = [yvals[0] - origline[quantlow].iloc[0]]
        errorhigh = [origline[quanthigh].iloc[0]-yvals[0]]
        ycent = yvals.copy()
        plt.close()
        for (ind, condition, xlabel) in [
            (11, (results_table["Database"] == "SR15CCBOX71") & (results_table["normyear"] == normyear0), "Update MAGICC"),
            (11, (results_table["Database"] == "AR6WG3"), "Use AR6 DB"),
            (10, ((results_table["Database"] == "AR6WG3") & (results_table["recem"] == recem0)) & (results_table["normyear"] == normyear0), "Recent emissions"),
            (9, (results_table["Database"] == "SR15CCBOX71") & (
                    results_table["FaIR"] == True) & (results_table["recem"] == recem0) & (results_table["normyear"] == normyear0), "Include FaIR"),
            (8, (results_table["Database"] == "AR6WG3") & (results_table["FaIR"] == True) &
                (results_table["NonCO2"] == "QRW") & (results_table["recem"] == recem0) & (results_table["normyear"] == normyear0), "Use QRW"),
            (7, (results_table["Database"] == "AR6WG3") & (results_table["FaIR"] == True) &
                (results_table["NonCO2"] == "QRW") & (results_table["peak"]=="nonCO2AtPeakAverage")
                & (results_table["recem"] == recem0) & (results_table["normyear"] == normyear0), "Change non-CO$_2$ time"),
            (6, (results_table["Database"] == "AR6WG3") & (results_table["FaIR"] == True) &
                (results_table["NonCO2"] == "QRW") & (results_table["peak"]=="nonCO2AtPeakAverage") &
                (results_table["Permafrost"] == True) & (results_table["recem"] == recem0) & (results_table["normyear"] == normyear0), "Permafrost in MAGICC"),
            (6, (results_table["Database"] == "AR6WG3") & (results_table["FaIR"] == True) &
                (results_table["NonCO2"] == "QRW") & (results_table["peak"]=="nonCO2AtPeakAverage") &
                (results_table["Permafrost"] == True) & (results_table["recem"] == recem0) &
                (results_table["normyear"] == True), "Normalise non-CO$_2$ years")
        ]:
            db = results_table.iloc[
                [bool(x) for x in (np.product(basebool.iloc[:, 0:ind], axis=1).astype(int) &
               condition)]
            ]
            assert len(db) == 1
            ycent.append(db[quant_want].iloc[0])
            errorhigh.append(db[quanthigh].iloc[0] - db[quant_want].iloc[0])
            errorlow.append(db[quant_want].iloc[0] - db[quantlow].iloc[0])
            try:
                yvals.append(db[quant_want].iloc[0] - dbold[quant_want].iloc[0])
            except NameError:
                yvals.append(db[quant_want].iloc[0] - yvals[-1])
            xvals.append(xlabel)
            dbold = db.copy()

        waterfall.plot(
            xvals,
            yvals,
            red_color="orangered",
            green_color="cornflowerblue",
            blue_color="mediumblue",
            net_label="Recommended update",
            formatting="{:,.0f}"
        )
        plt.xticks(rotation=45, horizontalalignment="right")
        plt.ylabel("Remaining budget (GtCO$_2$)")
        del dbold
        #plt.errorbar(xvals, ycent, yerr=np.array([errorlow, errorhigh]), fmt='.', alpha=0.5, c="lightblue",)
        plt.errorbar(xvals[0], ycent[0], yerr=np.array([[errorlow[0]], [errorhigh[0]]]), fmt='.', c="lightblue", alpha=0.5)
        plt.errorbar(xvals, ycent, yerr=np.array([errorlow, errorhigh]), fmt='.',
                     alpha=0.0)
        plt.errorbar(["Updated budget"], ycent[-1],
                     yerr=np.array([[errorlow[-1]], [errorhigh[-1]]]), fmt='.',
                     c="lightblue", alpha=0.5)

        plt.xticks()
        plt.ylim([0, ycent[0] + errorhigh[0] + 20])
        plt.tight_layout()
        plt.title(f"RCB for 50% chance of {round(futwarm + 1.07, 1)}$^o$C", y=1.0, pad=-14)
        plt.savefig(f"../Output/Plots/waterfall_changes_in_budget_{futwarm}C_p{quant_want}_errorbars_outer.png")
        plt.close()

    initialbug = results_table.iloc[
        [bool(x) for x in np.product(basebool.iloc[:, 1:], axis=1).astype(int)]
    ]
    finalbug = results_table.iloc[
        [bool(x) for x in (np.product(basebool.iloc[:, 1:ind], axis=1).astype(int) &
        (results_table["recem"] == recem0) & condition)]
    ]
    assert len(finalbug) == len(initialbug)
    plt.plot(hist_warm + initialbug["Future_warming"], initialbug["0.5"], color="mediumblue")
    plt.plot(hist_warm + finalbug["Future_warming"], finalbug["0.5"], color="orangered")
    plt.axhline(0, c="darkgrey", linewidth=1, zorder=0)
    plt.fill_between(hist_warm + initialbug["Future_warming"], initialbug["0.33"], initialbug["0.66"], alpha=0.25, color="mediumblue")
    plt.fill_between(hist_warm + finalbug["Future_warming"], finalbug["0.33"],
                     finalbug["0.66"], alpha=0.25, color="orangered")
    plt.xlabel("Peak warming (C)")
    plt.ylabel("Remaining budget (GtCO$_2$)")
    plt.legend(["AR6 WGI report", "Recommended estimate"], loc="upper left")
    plt.savefig(f"../Output/Plots/compare_updated_and_orig_budget_over_temps.png")


if plot_distn:
    for futwarm in [0.43, 0.93]:
        plt.close()
        use_results = results_table.loc[
            (results_table["Database"] == d0) &
            (results_table["ESF"] == ESF0) &
            (results_table["Permafrost"] == p0) &
            np.isclose(results_table["Future_warming"], futwarm) &
            (results_table["MAGICC"]) & (results_table["FaIR"]) &
            (results_table["ZECsd"]==zec_sd0) &
            (results_table["NonCO2"] == NonCO20) &
            (results_table["recem"] == recem0),
            :
        ]
        use_results = use_results.set_index(cols + ["Future_warming", "dT_targets", "peak", "recem", "normyear"])
        use_results.columns = [str(round(float(c), 4)) for c in use_results.columns]
        use_results = use_results.sort_index(axis=1)

        use_results = use_results.reset_index().melt(
            var_name="Quantile",
            value_name="Budget",
            value_vars=[str(round(x, 4)) for x in np.arange(0.01, 0.991, 0.0025)],
            id_vars=cols
        )
        sns.set_theme(style="whitegrid")
        violinplot = sns.violinplot(
            data=use_results, x="TCRE distribution", hue="ZEC asymmetry",
            y="Budget", cut=0, split=True, bw=0.18, inner="quartile",
        )
        violinplot.set(ylabel="Remaining Budget (GtCO$_2$)")

        medians = use_results.groupby(['TCRE distribution', "ZEC asymmetry"], sort=False)['Budget'].median().round(0)
        vertical_offset = use_results['Budget'].median() * 0.05  # offset from median for display
        for xtick in violinplot.get_xticks():
            violinplot.text(xtick+0.1, medians[xtick * 2], str(int(medians[xtick * 2])),
                          horizontalalignment='center', size='small', color='w',
                          weight='semibold')
            violinplot.text(xtick-0.1, medians[xtick * 2 + 1], str(int(medians[xtick * 2 + 1])),
                          horizontalalignment='center', size='small', color='w',
                          weight='semibold')
        plt.savefig(results_folder + plot_folder + f"violinplot_TCREZECDistn_ftwarm{futwarm}.png")

# Calculate the difference made by the change in non-CO2 quantiles:
if plot_distn == "":
    inds = ["0.17", "0.5", "0.83"]
    for nonlin, peak, perm in [("all", "None", False), ("QRW", "nonCO2AtPeakAverage", True)]:
        for temp in [1.5, 2]:
            resqant = {}
            for quantile_str in ["83.3", "50.0", "16.7"]:
                filename = plot_distn + f'budget_normal_magicc_True_fair_True_esf_{ESF0}pm26.7_likeli_0.6827_nonCO2pc{quantile_str}_' \
                    f'GtCO2_permaf_{perm}_zecsd_{zec_sd0}_asym_{ZECas0}_hdT_1.07NonlinNonCO2_{nonlin}_{peak}_recEm{recem0}.csv'
                results = pd.read_csv(
                    results_folder + "ar6wg3/" + "ny/" + filename
                )
                resqant[quantile_str] = results.loc[[round(r, 2) == temp for r in results.dT_targets], inds].values.squeeze()
            resqant = pd.DataFrame(resqant, index=inds)
            print(f"\n nonlin {nonlin} peak {peak}, perm {perm}, temp {temp}: ")
            print(resqant)
            negnonco2 = resqant['16.7'] - resqant['50.0']
            posnonco2 = resqant['50.0'] - resqant['83.3']
            bestest = resqant.loc["0.5", "50.0"]
            print(f"negative non-CO2 error: {negnonco2['0.5']}")
            print(f"positive non-CO2 error: {posnonco2['0.5']}")
            negcombined = (negnonco2["0.5"]**2 + (resqant.loc["0.17", "50.0"] - bestest)**2 + nonco2_model_uncertainty**2)**0.5
            poscombined = (posnonco2["0.5"]**2 + (bestest - resqant.loc["0.83", "50.0"])**2 + nonco2_model_uncertainty**2)**0.5
            print(f"total negative error {negcombined}")
            print(f"total positive error {poscombined}")
            print(f"Range: {bestest} ({bestest-negcombined}-{bestest+poscombined})")
