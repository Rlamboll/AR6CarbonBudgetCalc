import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.interpolate

from src.distributions_of_inputs import _read_and_clean_summary_csv, _clean_columns_magicc

# Plots the distribution of total temperatures and non-CO2 temperatures

# load temp trajectories
fair_df_all = pd.read_csv("../InputData/fair163_ar6/fair_processed_data_alltemp.csv")
fair_df_non = pd.read_csv("../InputData/fair163_ar6/fair_processed_data_nonco2temp.csv")
magicc_file_all = "../InputData/MAGICCAR6emWG3scen/job-20211019-ar6-nonco2_Raw-GSAT.csv"
magicc_file_non = "../InputData/MAGICCAR6emWG3scen/job-20211019-ar6-nonco2_Raw-GSAT-Non-CO2.csv"


# Load data and work out net zero date
nzyearcol = "year of netzero CO2 emissions"
emissions = pd.read_csv(
    "../InputData/MAGICCAR6emWG3scen/job-20211019-ar6-nonco2_Emissions-CO2.csv"
)
magicc_non_co2_col = (
    "non-co2 warming (rel. to 2010-2019) at peak cumulative emissions co2"
)
magicc_temp_col = "peak surface temperature (rel. to 2010-2019)"
yeardf = _clean_columns_magicc(emissions, None, sr15_rename=False)
empty_cols = [
    col for col in yeardf.columns
    if ((yeardf[col].isnull().all()) or (col == "permafrost"))
]
yeardf.drop(empty_cols, axis=1, inplace=True)
scenario_cols = ["model", "region", "scenario"]
del yeardf["unit"]
total_co2 = yeardf.groupby(scenario_cols).sum()
total_co2.columns = [int(col[:4]) for col in total_co2.columns]
zero_years = pd.Series(index=total_co2.index)
for index, row in total_co2.iterrows():
    try:
        change_sign = np.where(row < 0)[0][0]
        zero_year = scipy.interpolate.interp1d(
            [row.iloc[change_sign - 1], row.iloc[change_sign]],
            [row.index[change_sign - 1], row.index[change_sign]],
        )(0)
        zero_years[index] = np.round(zero_year)
    except IndexError:
        zero_years[index] = np.nan

cols = ["model", "region", "scenario"]
quant = 0.5
magiccquantname = "AR6 climate diagnostics|Raw Surface Temperature (GSAT)|{}MAGICCv7.5.3|50.0th Percentile"
fair_df_non = fair_df_non[fair_df_non["Unnamed: 0"]==quant]
fair_df_all = fair_df_all[fair_df_all["Unnamed: 0"]==quant]
fair_df_non = fair_df_non.set_index(cols)
fair_df_all = fair_df_all.set_index(cols)
temp_df = pd.DataFrame(
            index=total_co2.index, columns=[magicc_temp_col, magicc_non_co2_col], dtype=np.float64
        )
magicc_df_non = _read_and_clean_summary_csv(
        scenario_cols, temp_df, magiccquantname.format("Non-CO2|"), magicc_file_non,
    None, use_permafrost=False, sr15_rename=False,
)
magicc_df_all = _read_and_clean_summary_csv(
        scenario_cols, temp_df, magiccquantname.format(""), magicc_file_all,
    None, use_permafrost=False, sr15_rename=False,
)
future_dates = range(2020, 2101)

# Plot data
hsv = plt.get_cmap('nipy_spectral')
imageinds = magicc_df_all.index[np.where(magicc_df_all.index.get_level_values("model") == "IMAGE 3.0.1")[0]]
plotnum = len(imageinds)
colors = hsv(np.linspace(0, 1.0, plotnum))
for i in range(plotnum):
    if np.isfinite(zero_years.loc[imageinds[i]]):
        plt.plot(
            magicc_df_all.loc[imageinds[i], future_dates],
            magicc_df_non.loc[imageinds[i], future_dates],
            alpha=0.5, color=colors[i]
        )
        plt.scatter(
            magicc_df_all.loc[imageinds[i], int(zero_years.loc[imageinds[i]])],
            magicc_df_non.loc[imageinds[i], int(zero_years.loc[imageinds[i]])],
            marker="o", alpha=0.65, color=colors[i], s=60
        )
    else:
        plt.plot(
            magicc_df_all.loc[imageinds[i], future_dates],
            magicc_df_non.loc[imageinds[i], future_dates],
            linewidth=0.3, alpha=0.45, color=colors[i]
        )
plt.xlim([1.25, 2.5])
plt.ylim([0.2, 0.6])
plt.xlabel("Total warming (C)")
plt.ylabel("Non-CO$_2$ warming (C)")
plt.savefig("../Output/Plots/nonco2_co2_relation_over_time_fair.png")
plt.close()
future_dates = [str(c) for c in future_dates]
for i in range(plotnum):
    if np.isfinite(zero_years.loc[imageinds[i]]):
        plt.plot(
            fair_df_all.loc[imageinds[i], future_dates],
            fair_df_non.loc[imageinds[i], future_dates],
            alpha=0.5, color=colors[i]
        )
        plt.scatter(
            fair_df_all.loc[imageinds[i], int(zero_years.loc[imageinds[i]])],
            fair_df_non.loc[imageinds[i], int(zero_years.loc[imageinds[i]])],
            marker="o", alpha=0.65, color=colors[i], s=60
        )
    else:
        plt.plot(
            fair_df_all.loc[imageinds[i], future_dates],
            fair_df_non.loc[imageinds[i], future_dates],
            linewidth=0.3, alpha=0.45, color=colors[i]
        )
plt.xlim([1.25, 2.5])
plt.ylim([0.1, 0.6])
plt.xlabel("Total warming (C)")
plt.ylabel("Non-CO$_2$ warming (C)")
plt.savefig("../Output/Plots/nonco2_co2_relation_over_time_fair.png")

for i in range(30):
    if np.isfinite(zero_years.iloc[i]):
        plt.plot(
            magicc_df_all.loc[:, future_dates].iloc[i],
            magicc_df_non.loc[:, future_dates].iloc[i]
        )
        plt.scatter(
            magicc_df_all.loc[:, str(int(zero_years.iloc[i]))].iloc[i],
            magicc_df_non.loc[:, str(int(zero_years.iloc[i]))].iloc[i],
            marker="o", alpha=0.5
        )
    else:
        # To keep colors aligned.
        plt.plot(
            fair_df_all.loc[:, future_dates].iloc[i],
            fair_df_non.loc[:, future_dates].iloc[i],
            linewidth=0.4
        )
        plt.scatter([], [])
plt.xlim([1.25, 2.5])
plt.ylim([0, 0.6])
plt.savefig("../Output/Plots/nonco2_co2_relation_over_time_magicc.png")
