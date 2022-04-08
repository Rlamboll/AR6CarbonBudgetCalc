import os.path

import pandas as pd


AR6_WG3_META_FILE = os.path.join("..", "InputData", "MAGICCCCB71_sr15scen", "ar6_full_metadata_indicators2021_10_14_v3.xlsx")
SR15_META_FILE = os.path.join("..", "InputData", "sr15_metadata_indicators_r2.0.xlsx")
SR15_RUNS_OUTPUT_FILE = os.path.join("..", "InputData", "MAGICCCCB71_sr15scen", "job-20211014-sr15-nonco2_Emissions-CO2.csv")

OUT_FILE = os.path.join("..", "InputData", "sr15_scenario_runs_mocked_vetting.xlsx")


ar6_wg3_meta = pd.read_excel(AR6_WG3_META_FILE, sheet_name="meta")
sr15_meta = pd.read_excel(SR15_META_FILE, sheet_name="meta")
sr15_runs_output = pd.read_csv(SR15_RUNS_OUTPUT_FILE)


def rename_scenario_to_sr15_names(s):
    replacements = (
        ("SocioeconomicFactorCM", "SFCM"),
        ("TransportERL", "TERL"),
    )
    out = s

    for old, new in replacements:
        out = out.replace(old, new)

    return out


def rename_model_to_sr15_names(m):
    replacements = (
        ("AIM_", "AIM/CGE "),
        ("2_0", "2.0"),
        ("2_1", "2.1"),
        ("5_005", "5.005"),
        ("GCAM_4_2", "GCAM 4.2"),
        ("WEM", "IEA World Energy Model 2017"),
        ("IMAGE_", "IMAGE "),
        ("3_0_1", "3.0.1"),
        ("3_0_2", "3.0.2"),
        ("MERGE-ETL_6_0", "MERGE-ETL 6.0"),
        ("MESSAGE-GLOBIOM_1_0", "MESSAGE-GLOBIOM 1.0"),
        ("MESSAGE_V_3", "MESSAGE V.3"),
        ("MESSAGEix-GLOBIOM_1_0", "MESSAGEix-GLOBIOM 1.0"),
        ("POLES_", "POLES "),
        ("CDL", "CD-LINKS"),
        ("REMIND_", "REMIND "),
        ("REMIND-MAgPIE_", "REMIND-MAgPIE "),
        ("1_5", "1.5"),
        ("1_7", "1.7"),
        ("3_0", "3.0"),
        ("WITCH-GLOBIOM_", "WITCH-GLOBIOM "),
        ("3_1", "3.1"),
        ("4_2", "4.2"),
        ("4_4", "4.4"),
    )
    out = m

    for old, new in replacements:
        out = out.replace(old, new)

    return out


sr15_runs_output["model"] = sr15_runs_output["model"].apply(rename_model_to_sr15_names)
sr15_runs_output["scenario"] = sr15_runs_output["scenario"].apply(rename_scenario_to_sr15_names)

sr15_runs_used_scenarios = sr15_runs_output[["model", "scenario"]].drop_duplicates()


no_non_wg3_scens = 0
out = []
for _, row in sr15_runs_used_scenarios.iterrows():
    tmp = {"model": row.model, "scenario": row.scenario}
    nz_year = ar6_wg3_meta.loc[
        (ar6_wg3_meta["model"] == row.model)
        & (ar6_wg3_meta["scenario"] == row.scenario),
        "year of netzero CO2 emissions"
    ]

    if nz_year.empty:
        no_non_wg3_scens += 1
        nz_year = sr15_meta.loc[
            (sr15_meta["model"] == row.model)
            & (sr15_meta["scenario"] == row.scenario),
            "year of netzero CO2 emissions"
        ]

    tmp["year of netzero CO2 emissions"] = float(nz_year)
    out.append(tmp)


print(f"{no_non_wg3_scens} non-WG3 scenarios")

out = pd.DataFrame(out)
out["exclude"] = False
out.to_excel(OUT_FILE)
