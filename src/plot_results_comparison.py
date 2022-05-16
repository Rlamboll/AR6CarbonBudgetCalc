import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import scipy.interpolate

results_folder = "../Output/"
subfolder = "ar6wg3/"

# The results will go into the output folder, in this subfolder:
plot_folder = "Plots/"
if not os.path.exists(results_folder + plot_folder):
    os.makedirs(results_folder + plot_folder)
# Files to read will have this basic structure:
file_format = "allquantbudget_{}_magicc_{}_fair_{}_esf_{}pm26.7_likeli_0.6827_" \
              "nonCO2pc50_GtCO2_permaf_{}_hdT_1.07NonlinNonCO2_{}_None_recEm209.csv"
cols = ["TCRE Distribution", "MAGICC", "FaIR", "ESF", "Permafrost", "NonCO2"]
results_table = pd.DataFrame(
    columns=cols
)
for distribution in ["normal", "lognormal"]:
    for MAGICC in [True]:
        for FaIR in [False]:
            for ESF in [0.0, 7.1]:
                for Permafrost in [True, False]:
                    for NonCO2 in ["all"]:
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
                            ("TCRE Distribution", distribution),
                            ("MAGICC", MAGICC),
                            ("FaIR", FaIR),
                            ("ESF", ESF),
                            ("Permafrost", Permafrost),
                            ("NonCO2", NonCO2)
                        ]:
                            results[col] = res
                        results_table = results_table.append(results)
for futwarm in [0.43, 0.93]:
    plt.close()
    use_results = results_table.loc[
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
        y="Budget", inner="quartile", cut=0, split=True, bw=0.4
    )
    plt.savefig(results_folder + plot_folder + f"violinplot_PermaforstDistn_ftwarm{futwarm}.png")
