import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

# Plots the results of running the model-and-SSP specific budgets.

# Database should be the foldername where files are stored, e.g. ar6wg3 or sr15ccbox71
database = "ar6wg3"
results_folder = f"../Output/{database}/"
subfolders0 = ["each/SSP1/", "each/SSP2/", "each/SSP3/", "each/SSP4/", "each/SSP5/"]
startstrings = []
for fold in subfolders0:
    all_files = os.listdir(results_folder + fold)
    valid_files = [
        i.split("normal")[0] for i in all_files if (i[:4] != "fair") & (i[:6] != "magicc") &
                                (i[-4:] == ".csv") & (i[:4] != "num_")
    ]
    valid_files = list(set([i for i in valid_files if i[-3:] != "log"]))
    startstrings.append([fold + i for i in valid_files])
startstrings = [i for j in startstrings for i in j]
# The results will go into the output folder, in this subfolder:
plot_folder = f"../Output/Plots/each/{database}/"
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

# Files to read will have this basic structure:
file_format ="{}normal_magicc_{}_fair_{}_esf_7.1pm26.7_likeli_0.6827_nonCO2pc50.0_GtCO2_permaf_False_zecsd_0.19_asym_False_hdT_1.07NonlinNonCO2_{}_{}_recEm277.csv"
cols = ["Database", "NonCO2"]
historic_warming = 1.07
results_table = pd.DataFrame(
    columns=cols
)
magicc0 = True
fair0 = True
peak0 = "nonCO2AtPeakTot"
nonco20 = "interp"
baseline_file = file_format.format("budget_", magicc0, fair0, nonco20, peak0)
results = pd.read_csv(
    results_folder + baseline_file
)
for col, res in [
    ("Database", "AR6"),
    ("MAGICC", magicc0),
    ("FaIR", fair0),
    ("NonCO2", nonco20),
]:
    results[col] = res
results_table = results
for startstr in startstrings:
    for MAGICC in [magicc0]:
        for FaIR in [fair0]:
            for NonCO2 in ["interp"]:
                try:
                    filename = file_format.format(
                        startstr, MAGICC, FaIR, NonCO2, peak0
                    )
                    results = pd.read_csv(
                        results_folder + filename
                    )
                except FileNotFoundError:
                    print("Didn't find " + filename)
                    continue

                for col, res in [
                    ("Database", startstr),
                    ("MAGICC", MAGICC),
                    ("FaIR", FaIR),
                    ("NonCO2", NonCO2)
                ]:
                    results[col] = res
                results_table = results_table.append(results)
results_table = results_table.reset_index(drop=True)
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)
results_table.to_csv(plot_folder + "model_specific_summary.csv")
databasestr = "AR6" if (database == "ar6wg3") else "SR1.5"
results_table["Scenario"] = [x[5: 9] if len(x) > 3 else f"All {databasestr}" for x in results_table["Database"]]
results_table["Model"] = [x[10:] if len(x) > 3 else f"All {databasestr}" for x in results_table["Database"]]

temp = 1.5
quantiles = ["0.1", "0.17", "0.33","0.5", "0.66", "0.83", "0.9"]
to_plot = results_table.melt(
    var_name="Quantile", value_name="Budget", value_vars=quantiles, id_vars=["dT_targets", "Scenario", "Model"]
)
to_plot = to_plot.loc[
    np.isclose(to_plot["dT_targets"], temp),
    :
]
plt.close()
sns.catplot(
    data=to_plot, x="Scenario", hue="Model", y="Budget", kind="box", aspect=1.2
)
plt.savefig(plot_folder + "scenario_budgets_catplot.png")
# Also plot this the other way around
plt.close()
sns.catplot(
    data=to_plot, x="Model", hue="Scenario", y="Budget", kind="box", aspect=1.2,
    legend=False
)
plt.xticks(rotation=45, horizontalalignment="right")
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.tight_layout()
plt.savefig(plot_folder + "model_then_scenario_budgets_catplot.png")

# Plot different peak/non-CO2 waring of scenario families on the same plot
magicc_db = pd.read_csv(
    results_folder + "magicc_nonCO2_temp_50.0PercentilenonCO2AtPeakTotpermaf_False.csv"
)
fair_db = pd.read_csv(
    results_folder + "fair_nonCO2_temp_50.0PercentilenonCO2AtPeakTotpermaf_None.csv"
)
magicc_non_co2_col = (
    "non-co2 warming (rel. to 2010-2019) at peak cumulative emissions co2"
)
magicc_temp_col = "peak surface temperature (rel. to 2010-2019)"
master_all_non_co2 = pd.DataFrame(
    index=magicc_db.index,
    columns=[magicc_non_co2_col, magicc_temp_col]
)
master_all_non_co2[magicc_non_co2_col] = (fair_db[
    magicc_non_co2_col] + magicc_db.reset_index()[magicc_non_co2_col]) / 2
master_all_non_co2[magicc_temp_col] = (fair_db[
     magicc_temp_col] + magicc_db.reset_index()[magicc_temp_col]) / 2
master_all_non_co2.index = magicc_db.index
master_all_non_co2["Scenario"] = magicc_db["scenario"]
master_all_non_co2["Model"] = magicc_db["model"]

image_data = master_all_non_co2.loc[master_all_non_co2["Model"] == "IMAGE 3.0.1"]
legends = []
plt.close()
for scenario in ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]:
    ind = [x[:len(scenario)] == scenario for x in image_data["Scenario"]]
    plt.plot(historic_warming + image_data[magicc_temp_col].iloc[ind],
        image_data[magicc_non_co2_col].iloc[ind]
    )
    legends.append(scenario)
image_lincoefs = np.polyfit(image_data[magicc_temp_col], image_data[magicc_non_co2_col], 1)
image_linfun = np.poly1d(image_lincoefs)
plt.plot(
    [historic_warming + min(image_data[magicc_temp_col]), historic_warming + max(image_data[magicc_temp_col])],
    image_linfun([min(image_data[magicc_temp_col]), max(image_data[magicc_temp_col])]),
    linewidth=1, c="grey", linestyle="--"
)
legends.append("Linear fit")
plt.legend(legends)
plt.xlabel("Peak total warming (C)")
plt.ylabel("Non-CO$_2$ warming relative to 2010-2019 (C)")
plt.savefig(plot_folder + "IMAGE_3.0.1_ssps.png")

# Calculate the standard deviation and range of the distribution of values
for futwarm, wantquant in [(0.43, "0.5"), (0.93, "0.66")]:
    valid_inds = np.isclose(results_table["Future_warming"], futwarm) & (results_table["Database"] != "AR6")
    rangeofvals = max(results_table.loc[valid_inds, wantquant]) - min(results_table.loc[valid_inds, wantquant])
    sdofvals = np.std(results_table.loc[valid_inds, wantquant], ddof=1)
    minval = min(results_table.loc[valid_inds, wantquant])
    print(f"For future warming {futwarm} quantile {wantquant} range is {rangeofvals} and stdev {sdofvals}, min is {minval}")