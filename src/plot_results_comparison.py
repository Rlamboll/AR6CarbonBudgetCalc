import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

results_folder = "../Output/"
subfolders = ["sr15wg1/", "ar6wg3/", "sr15ccbox71/"]

# The results will go into the output folder, in this subfolder:
plot_folder = "Plots/"
if not os.path.exists(results_folder + plot_folder):
    os.makedirs(results_folder + plot_folder)

# If plot_distn is "allquant" we plot the effect of the TCRE distribution on results.
# If the empty string, we plot the impact
plot_distn = ""
# Files to read will have this basic structure:
file_format = plot_distn + "budget_{}_magicc_{}_fair_{}_esf_{}pm26.7_likeli_0.6827_" \
              "nonCO2pc50_GtCO2_permaf_{}_hdT_1.07NonlinNonCO2_{}_None_recEm209.csv"
cols = ["Database", "TCRE Distribution", "MAGICC", "FaIR", "ESF", "Permafrost", "NonCO2"]
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
                            try:
                                filename = file_format.format(
                                        distribution, MAGICC, FaIR, ESF, Permafrost, NonCO2
                                    )
                                results = pd.read_csv(
                                    results_folder + subfolder + filename
                                )
                            except FileNotFoundError:
                                print("Didn't find " + filename)
                                continue

                            for col, res in [
                                ("Database", subfolder[:-1].upper()),
                                ("TCRE Distribution", distribution),
                                ("MAGICC", MAGICC),
                                ("FaIR", FaIR),
                                ("ESF", ESF),
                                ("Permafrost", Permafrost),
                                ("NonCO2", NonCO2)
                            ]:
                                results[col] = res
                            results_table = results_table.append(results)

if not plot_distn:
    for nonco2 in ["all", "QRW"]:
        for futwarm in [0.43, 0.93]:
            plt.close()
            use_results = results_table.loc[
                ([r in ["SR15CCBOX71", "SR15WG1"] for r in results_table["Database"]]) & (
                np.isclose(results_table["Future_warming"], futwarm)) & (
                results_table["TCRE Distribution"]=="normal") & (results_table["Permafrost"] == False) & (
                results_table["ESF"] == 7.1) & (results_table["NonCO2"] == nonco2),
                :
            ]
            use_results["Updated"] = ["yes" if y == "SR15CCBOX71" else "no" for y in use_results["Database"]]
            use_results["Model"] = "MAGICC"
            use_results.loc[~use_results["MAGICC"].astype(bool), "Model"] = "FaIR"
            use_results.loc[use_results["MAGICC"] & use_results["FaIR"], "Model"] = "MAGICC and FaIR"
            use_results = use_results.melt(
                var_name="Quantile",
                value_name="Budget",
                value_vars=["0.17", "0.33", "0.5", "0.66", "0.83"],
                id_vars=cols + ["Updated", "Model"]
            )

            sns.catplot(
                data=use_results, x="Model", hue="Updated", y="Budget", kind="box", hue_order=["no", "yes"]
            )
            plt.savefig(
                results_folder + plot_folder + f"updates_Distn_ftwarm{futwarm}_nonco2_{nonco2}.png")


if plot_distn:
    for futwarm in [0.43, 0.93]:
        plt.close()
        use_results = results_table.loc[
            (results_table["Database"] == "AR6WG3") &
            np.isclose(results_table["Future_warming"], futwarm) & (
                results_table["MAGICC"]) & (results_table["FaIR"]==False),
            :
        ]
        for pos in np.arange(0.015, 0.9851, 0.01):
            use_results[str(round(pos, 4))] = np.nan
        use_results = use_results.set_index(cols + ["Future_warming", "dT_targets"])
        use_results.columns = [str(round(float(c), 4)) for c in use_results.columns]
        use_results = use_results.sort_index(axis=1)
        use_results = use_results.interpolate(axis=1, method="linear")

        use_results = use_results.reset_index().melt(
            var_name="Quantile",
            value_name="Budget",
            value_vars=[str(round(x, 4)) for x in np.arange(0.01, 0.991, 0.005)],
            id_vars=cols
        )
        sns.set_theme(style="whitegrid")
        sns.violinplot(
            data=use_results, x="TCRE Distribution", hue="Permafrost",
            y="Budget", cut=0, split=True, bw=0.4, inner="quartile",
        )
        plt.savefig(results_folder + plot_folder + f"violinplot_PermaforstDistn_ftwarm{futwarm}.png")
