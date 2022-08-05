import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.interpolate

from src.distributions_of_inputs import _read_and_clean_summary_csv, _clean_columns_magicc

# Plots the distribution of total temperatures and non-CO2 temperatures over time

# load temp trajectories
fair_file_all = "../InputData/fair163_ar6/fair_processed_data_alltemp.csv"
fair_file_non = "../InputData/fair163_ar6/fair_processed_data_nonco2temp.csv"
magicc_file_all = "../InputData/MAGICCAR6emWG3scen/job-20211019-ar6-nonco2_Raw-GSAT.csv"
magicc_file_non = "../InputData/MAGICCAR6emWG3scen/job-20211019-ar6-nonco2_Raw-GSAT-Non-CO2.csv"


# Load data and work out net zero date
nzyearcol = "year of netzero CO2 emissions"
emissions = pd.read_csv(
    "../InputData/MAGICCAR6emWG3scen/job-20211019-ar6-nonco2_Emissions-CO2.csv"
)
vetted_scen_list_file = "../InputData/MAGICCAR6emWG3scen/" + "ar6_full_metadata_indicators2021_10_14_v3.xlsx"
vetted_scen_list_file_sheet = "meta_Ch3vetted_withclimate"
nzyearcol = "year of netzero CO2 emissions"
vetted_scens = pd.read_excel(
    vetted_scen_list_file, sheet_name=vetted_scen_list_file_sheet
).loc[:, ["model", "scenario", "exclude", nzyearcol]]
vetted_scens["region"] = "World"
scenario_cols = ["model", "region", "scenario"]
vetted_scens = vetted_scens.set_index(scenario_cols)
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
del yeardf["unit"]
total_co2 = yeardf.groupby(scenario_cols).sum()
total_co2.columns = [int(col[:4]) for col in total_co2.columns]
zero_years = pd.Series(index=total_co2.index, dtype=np.float64)
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

quant = 0.5
magiccquantname = "AR6 climate diagnostics|Raw Surface Temperature (GSAT)|{}MAGICCv7.5.3|50.0th Percentile"
temp_df = pd.DataFrame(
    index=total_co2.index, columns=[magicc_temp_col, magicc_non_co2_col], dtype=np.float64
)
fair_df_non = _read_and_clean_summary_csv(
    scenario_cols, temp_df, magiccquantname.format("Non-CO2|"), fair_file_non,
    None, use_permafrost=None, sr15_rename=False,
)
fair_df_all = _read_and_clean_summary_csv(
    scenario_cols, temp_df, magiccquantname.format(""), fair_file_all,
    None, use_permafrost=None, sr15_rename=False,
)
magicc_df_non = _read_and_clean_summary_csv(
        scenario_cols, temp_df, magiccquantname.format("Non-CO2|"), magicc_file_non,
    None, use_permafrost=False, sr15_rename=False,
)
magicc_df_all = _read_and_clean_summary_csv(
        scenario_cols, temp_df, magiccquantname.format(""), magicc_file_all,
    None, use_permafrost=False, sr15_rename=False,
)
future_dates = range(2015, 2101)

# Plot data
hsv = plt.get_cmap('Blues_r')
imageinds = magicc_df_all.index
plotnum = len(imageinds)
chosen_inds = [
    ("REMIND-MAgPIE 1.5", "World", "SSP2-19"),
    ("IMAGE 3.2", "World", "SSP1_SPA1_19I_D"),
    ("AIM/CGE 2.0", "World", "SSP1-26"),
    ("MESSAGE-GLOBIOM 1.0", "World", "SSP2-26"),
    ("GCAM 5.3", "World", "R_MAC_95_n8"),
]
assert len([i for i in chosen_inds if i not in imageinds]) == 0
colors = hsv(np.linspace(0, 1.0, len(chosen_inds) + 2))


def plot_temps(magicc_df_all, magicc_df_non, future_dates, title):
    global i
    for i in range(plotnum):
        if imageinds[i] not in chosen_inds:
            plt.plot(
                magicc_df_all.loc[imageinds[i], future_dates],
                magicc_df_non.loc[imageinds[i], future_dates],
                linewidth=0.3, alpha=0.15, color="darkorange"
            )
    first = True
    for (i, ci) in enumerate(chosen_inds):
        if np.isfinite(zero_years.loc[ci]):
            plt.plot(
                magicc_df_all.loc[ci, future_dates],
                magicc_df_non.loc[ci, future_dates],
                alpha=0.95, color=colors[i], zorder=2,
            )
            plt.scatter(
                magicc_df_all.loc[ci, int(zero_years.loc[ci])],
                magicc_df_non.loc[ci, int(zero_years.loc[ci])],
                marker="*", color=colors[i], s=60,
                zorder=2.5, alpha=0.65,
                label="Actual net zero" if first else None
            )
            plt.scatter(
                magicc_df_all.loc[ci, int(vetted_scens.loc[ci, nzyearcol])],
                magicc_df_non.loc[ci, int(vetted_scens.loc[ci, nzyearcol])],
                marker="x", color=colors[i], s=60,
                zorder=2.5, alpha=0.65,
                label="Original net zero" if first else None
            )
            maxtind = np.where(
                magicc_df_all.loc[ci, :] == max(magicc_df_all.loc[ci, :]))[0][0]
            plt.scatter(
                magicc_df_all.loc[ci].iloc[maxtind],
                magicc_df_non.loc[ci].iloc[maxtind],
                marker=">", color=colors[i], s=60,
                zorder=2.5, alpha=0.65,
                label="Max total temp" if first else None
            )
            maxtind = np.where(
                magicc_df_non.loc[ci, :] == max(magicc_df_non.loc[ci, :]))[0][0]
            plt.scatter(
                magicc_df_all.loc[ci].iloc[maxtind],
                magicc_df_non.loc[ci].iloc[maxtind],
                marker="^", color=colors[i], s=60,
                zorder=2.5, alpha=0.65,
                label="Max non-CO$_2$ temp" if first else None
            )
            first = False
        else:
            print(f"No value for scenario {ci}")
    plt.xlabel("Total warming (C)")
    plt.ylabel("Non-CO$_2$ warming (C)")
    plt.legend()
    plt.title(title)

plot_temps(magicc_df_all, magicc_df_non, future_dates, "MAGICC")
plt.xlim([1.15, 2.5])
plt.ylim([0.2, 0.6])
plot_folder = "../Output/Plots/"
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)
plt.savefig(plot_folder + "nonco2_co2_relation_over_time_magicc.png")
plt.close()

# Repeat for fair
plot_temps(fair_df_all, fair_df_non, future_dates, "FaIR")
plt.xlim([1.11, 2.5])
plt.ylim([0.15, 0.6])
plt.savefig(plot_folder + "nonco2_co2_relation_over_time_fair.png")
