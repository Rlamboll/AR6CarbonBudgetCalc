import src.distributions_of_inputs as distributions
import src.budget_calculator_functions as calc
import numpy as np
import pandas as pd
import pytest


def test_peaking_strats():
    magicc_file_for_tests_to_use = "magicc_file_for_tests_to_use.csv"
    magicc_nonco2_temp_variable = "SR15 climate diagnostics|Raw Surface Temperature (GSAT)|Non-CO2|MAGICCv7.5.1|10.0th Percentile"
    offset_years = np.arange(2010, 2020, 1)
    magicc_file_for_tests_to_use = "magicc_file_for_tests_to_use_many_scenarios.csv"
    total_magicc_file = "magicc_file_for_tests_to_use_totaltemp_many_scenarios.csv"
    vetting_file = "metadata_indicators_test.xlsx"
    magicc_nonco2_temp_variable = "SR15 climate diagnostics|Raw Surface Temperature (GSAT)|Non-CO2|MAGICCv7.5.1|10.0th Percentile"
    magicc_tot_temp_variable = "SR15 climate diagnostics|Raw Surface Temperature (GSAT)|MAGICCv7.5.1|10.0th Percentile"
    emissions_file = "emissions_test_file_many_scen.csv"
    offset_years = np.arange(2010, 2020, 1)
    for peak, num_scen in [(None, 2), ("peakNonCO2Warming", 3), ("nonCO2AtPeakTot", 3), ("officialNZ", 2)]:
        scenarios = distributions.load_data_from_summary(
            magicc_file_for_tests_to_use,
            total_magicc_file,
            emissions_file,
            magicc_nonco2_temp_variable,
            magicc_tot_temp_variable,
            magicc_nonco2_temp_variable,
            magicc_tot_temp_variable,
            offset_years,
            vetted_scen_list_file=vetting_file,
            vetted_scen_list_file_sheet="meta_Ch3vetted_withclimate",
            peak_version=peak,
            permafrost=True,
        )
        assert scenarios.shape[0] == 3
        assert len(scenarios[np.isfinite(scenarios["hits_net_zero"])]) == num_scen
        if peak != "officialNZ":
            scenarios = distributions.load_data_from_summary(
                magicc_file_for_tests_to_use,
                total_magicc_file,
                emissions_file,
                magicc_nonco2_temp_variable,
                magicc_tot_temp_variable,
                magicc_nonco2_temp_variable,
                magicc_tot_temp_variable,
                offset_years,
                peak_version=peak,
                permafrost=True,
            )
            assert scenarios.shape[0] == 4
            assert len(scenarios[np.isfinite(scenarios["hits_net_zero"])]) == num_scen + 1

def test_tcre_bad_limits():
    low = 5.8
    high = 2.5
    likelihood = 0.6827  # we use the 1-sigma range
    distn = "normal"
    with pytest.raises(
        AssertionError, match="High and low limits are the wrong way around"
    ):
        returned = distributions.tcre_distribution(
            low, high, likelihood, n_return=1, tcre_dist=distn
        )


def test_tcre_normal_distribution():
    low = 0.8
    high = 2.5
    likelihood = 0.6827  # we use the 1-sigma range
    normal_mean = (low + high) / 2
    normal_sd = (high - low) / 2
    n_return = 1000000
    distn = "normal"
    returned = distributions.tcre_distribution(
        low, high, likelihood, n_return=n_return, tcre_dist=distn
    )
    assert abs(np.mean(returned) - normal_mean) < 0.005
    assert abs(np.std(returned) - normal_sd) < 0.005
    assert abs(sum(returned > normal_mean) / n_return - 0.5) < 0.01
    likely_fraction = 1 - (sum(low > returned) + sum(high < returned)) / len(returned)
    assert abs(likely_fraction - likelihood) < 0.01


def test_tcre_lognormal_mean_match_distribution():
    low = 0.8
    high = 2.5
    likelihood = 0.6827  # we use the 1-sigma range
    normal_mean = (0.8 + 2.5) / 2
    normal_sd = (2.5 - 0.8) / 2
    n_return = 1000000
    distn = "lognormal mean match"
    returned = distributions.tcre_distribution(
        low, high, likelihood, n_return=n_return, tcre_dist=distn
    )
    assert abs(np.mean(returned) - normal_mean) < 0.005
    assert abs(np.std(returned) - normal_sd) < 0.005
    assert sum(returned < 0) == 0
    likely_fraction = 1 - (sum(0.8 > returned) + sum(2.5 < returned)) / len(returned)
    assert likely_fraction > likelihood


def test_tcre_lognormal_pde_distribution():
    low = 0.8
    high = 2.5
    likelihood = 0.6827  # we use the 1-sigma range
    # The median value is the geometric mean
    expected_median = (0.8 * 2.5) ** 0.5
    n_return = 1000000
    distn = "lognormal"
    returned = distributions.tcre_distribution(
        low, high, likelihood, n_return=n_return, tcre_dist=distn
    )
    assert abs(np.median(returned) - expected_median) < 0.005
    assert sum(returned < 0) == 0
    likely_fraction = 1 - (sum(0.8 > returned) + sum(2.5 < returned)) / len(returned)
    assert abs(likely_fraction - likelihood) < 0.01


def test_establish_median_temp_dep():
    xy_df = pd.DataFrame(
        {"x": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3], "y": [0.0, 0.1, 0.2, 0.2, 0.3, 0.4]}
    )
    quantiles_to_plot = [0.5]
    temp = np.array([2, 3])
    relations = calc.quantile_regression_find_relationships(xy_df, quantiles_to_plot)
    returned = distributions.establish_median_temp_dep_linear(relations, temp,
                                                              quantiles_to_plot)
    # The above data has a 1:1 relationship, so we expect to receive the temp back again
    assert all(abs(x - y) < 1e-14 for x, y in zip(returned, temp))


def test_establish_median_temp_dep_not_skewed():
    # This test is largely equivalent to the above, but includes some very large values
    # that should not impact the overall results.
    xy_df = pd.DataFrame(
        {
            "x": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
            "y": [0.0, 0.1, 0.2, 1000, 1000, 1000, 0.1, 0.2, 0.3],
        }
    )
    quantiles_to_plot = [0.5]
    temp = np.array([0.12, 0.26])
    relations = calc.quantile_regression_find_relationships(xy_df, quantiles_to_plot)
    returned = distributions.establish_median_temp_dep_linear(relations, temp, 0.5)
    # The above data has a 1:1 relationship, so we expect to receive the temp back again
    assert np.allclose(returned, temp)
    # Also should work nonlinearly, although there is a slight error on the uppermost point
    nonlin_rel = calc.quantile_regression_find_relationships_nonlin(xy_df, quantiles_to_plot, smoothing=3)
    nonlin_returned = distributions.establish_median_temp_dep_nonlinear(nonlin_rel, temp, quantiles_to_plot)
    assert np.allclose(nonlin_returned[0.12], temp[0])
    assert np.allclose(nonlin_returned[0.26], temp[1], atol=0.03)



def test_load_data_from_summary():
    magicc_file_for_tests_to_use = "magicc_file_for_tests_to_use.csv"
    magicc_nonco2_temp_variable = "SR15 climate diagnostics|Raw Surface Temperature (GSAT)|Non-CO2|MAGICCv7.5.1|10.0th Percentile"
    offset_years = np.arange(2010, 2020, 1)
    # The name of the peak temperature column output
    with pytest.raises(ValueError,
                   match="The input data contains permafrost information but we have not "
                         "given instructions with what to do with it."
                   ):
        distributions.load_data_from_summary(
            magicc_file_for_tests_to_use,
            magicc_file_for_tests_to_use,
            magicc_file_for_tests_to_use,
            magicc_nonco2_temp_variable,
            magicc_nonco2_temp_variable,
            magicc_nonco2_temp_variable,
            magicc_nonco2_temp_variable,
            offset_years,
            vetted_scen_list_file=None,
            peak_version=None,
            permafrost=None,
        )


def test_summary_averages_two():
    magicc_file_for_tests_to_use = "magicc_file_for_tests_to_use.csv"
    total_magicc_file = "magicc_file_for_tests_to_use_totaltemp.csv"
    total_fair_file = "fair_file_for_tests_to_use_totaltemp.csv"
    nonco2_fair_file = "fair_file_for_tests_to_use.csv"

    magicc_nonco2_temp_variable = "SR15 climate diagnostics|Raw Surface Temperature (GSAT)|Non-CO2|MAGICCv7.5.1|10.0th Percentile"
    magicc_tot_temp_variable = "SR15 climate diagnostics|Raw Surface Temperature (GSAT)|MAGICCv7.5.1|10.0th Percentile"
    emissions_file = "emissions_test_file.csv"
    offset_years = np.arange(2010, 2020, 1)
    # The name of the peak temperature column output
    with pytest.raises(AssertionError):
        distributions.load_data_from_summary(
            magicc_file_for_tests_to_use,
            total_magicc_file,
            emissions_file,
            magicc_nonco2_temp_variable,
            magicc_tot_temp_variable,
            magicc_nonco2_temp_variable,
            magicc_tot_temp_variable,
            offset_years,
            peak_version=None,
            permafrost=True,
            second_tot_file=total_fair_file
        )
    added_fair = distributions.load_data_from_summary(
        magicc_file_for_tests_to_use,
        total_magicc_file,
        emissions_file,
        magicc_nonco2_temp_variable,
        magicc_tot_temp_variable,
        magicc_nonco2_temp_variable,
        magicc_tot_temp_variable,
        offset_years,
        peak_version=None,
        permafrost=True,
        second_tot_file=total_fair_file,
        second_non_co2_file=nonco2_fair_file,
    )
    no_fair = distributions.load_data_from_summary(
        magicc_file_for_tests_to_use,
        total_magicc_file,
        emissions_file,
        magicc_nonco2_temp_variable,
        magicc_tot_temp_variable,
        magicc_nonco2_temp_variable,
        magicc_tot_temp_variable,
        offset_years,
        peak_version=None,
        permafrost=True,
    )
    double_magicc = distributions.load_data_from_summary(
        magicc_file_for_tests_to_use,
        total_magicc_file,
        emissions_file,
        magicc_nonco2_temp_variable,
        magicc_tot_temp_variable,
        magicc_nonco2_temp_variable,
        magicc_tot_temp_variable,
        offset_years,
        peak_version=None,
        permafrost=True,
        second_tot_file=total_magicc_file,
        second_non_co2_file=magicc_file_for_tests_to_use
    )
    assert no_fair.equals(double_magicc)
    # The first element of fair has a non-CO2 temp equal to the total

    assert np.isclose(no_fair.iloc[0, 0:1] / 2, added_fair.iloc[0, 0:1])
    assert np.allclose(no_fair.iloc[1, 1], added_fair.iloc[1, 1])


def test_magicc_loader_works_with_permafrost():
    magicc_file_for_tests_to_use = "magicc_file_for_tests_to_use.csv"
    total_magicc_file = "magicc_file_for_tests_to_use_totaltemp.csv"
    magicc_nonco2_temp_variable = "SR15 climate diagnostics|Raw Surface Temperature (GSAT)|Non-CO2|MAGICCv7.5.1|10.0th Percentile"
    magicc_tot_temp_variable = "SR15 climate diagnostics|Raw Surface Temperature (GSAT)|MAGICCv7.5.1|10.0th Percentile"
    emissions_file = "emissions_test_file.csv"
    offset_years = np.arange(2010, 2020, 1)
    # The name of the peak temperature column output
    permafrost_on = distributions.load_data_from_summary(
        magicc_file_for_tests_to_use,
        total_magicc_file,
        emissions_file,
        magicc_nonco2_temp_variable,
        magicc_tot_temp_variable,
        magicc_nonco2_temp_variable,
        magicc_tot_temp_variable,
        offset_years,
        peak_version=None,
        permafrost=True,
    )
    permafrost_off = distributions.load_data_from_summary(
        magicc_file_for_tests_to_use,
        total_magicc_file,
        emissions_file,
        magicc_nonco2_temp_variable,
        magicc_tot_temp_variable,
        magicc_nonco2_temp_variable,
        magicc_tot_temp_variable,
        offset_years,
        peak_version=None,
        permafrost=False,
    )
    assert len(permafrost_on) == 2
    assert len(permafrost_off) == 2
    compare = permafrost_on - permafrost_off
    assert not (compare.isna().all().all())
    assert not any([a == 0 for a in compare])


def test_scenario_filter():
    # We provide a set of 4 scenarios, one of which is not found in the vetting database
    magicc_file_for_tests_to_use = "magicc_file_for_tests_to_use_many_scenarios.csv"
    total_magicc_file = "magicc_file_for_tests_to_use_totaltemp_many_scenarios.csv"
    vetting_file = "metadata_indicators_test.xlsx"
    magicc_nonco2_temp_variable = "SR15 climate diagnostics|Raw Surface Temperature (GSAT)|Non-CO2|MAGICCv7.5.1|10.0th Percentile"
    magicc_tot_temp_variable = "SR15 climate diagnostics|Raw Surface Temperature (GSAT)|MAGICCv7.5.1|10.0th Percentile"
    emissions_file = "emissions_test_file_many_scen.csv"
    offset_years = np.arange(2010, 2020, 1)
    # The name of the peak temperature column output
    fewer_scenarios = distributions.load_data_from_summary(
        magicc_file_for_tests_to_use,
        total_magicc_file,
        emissions_file,
        magicc_nonco2_temp_variable,
        magicc_tot_temp_variable,
        magicc_nonco2_temp_variable,
        magicc_tot_temp_variable,
        offset_years,
        vetted_scen_list_file=vetting_file,
        vetted_scen_list_file_sheet="meta_Ch3vetted_withclimate",
        peak_version=None,
        permafrost=True,
    )

    all_scenarios = distributions.load_data_from_summary(
        magicc_file_for_tests_to_use,
        total_magicc_file,
        emissions_file,
        magicc_nonco2_temp_variable,
        magicc_tot_temp_variable,
        magicc_nonco2_temp_variable,
        magicc_tot_temp_variable,
        offset_years,
        peak_version=None,
        permafrost=True,
    )
    # One of the passing years doesn't peak, so one of the 4 scenarios always fails
    assert len(fewer_scenarios[np.isfinite(fewer_scenarios["hits_net_zero"])]) == 2
    assert len(fewer_scenarios) == 3
    assert len(all_scenarios) == 4
    assert len(all_scenarios[np.isfinite(all_scenarios["hits_net_zero"])]) == 3
    assert [
               i for i in all_scenarios[np.isfinite(all_scenarios["hits_net_zero"])].index if
               i not in fewer_scenarios[np.isfinite(fewer_scenarios["hits_net_zero"])].index
           ] == [("REMIND_1_5", "World", "not good ")]
    assert all(
        fewer_scenarios[np.isfinite(fewer_scenarios["hits_net_zero"])].index.get_level_values("scenario") == ["Fine", "Acceptable"]
    )

def test_scenario_filter_official_NZ():
    magicc_file_for_tests_to_use = "magicc_file_for_tests_to_use_many_scenarios.csv"
    total_magicc_file = "magicc_file_for_tests_to_use_totaltemp_many_scenarios.csv"
    vetting_file = "metadata_indicators_test.xlsx"
    magicc_nonco2_temp_variable = "SR15 climate diagnostics|Raw Surface Temperature (GSAT)|Non-CO2|MAGICCv7.5.1|10.0th Percentile"
    magicc_tot_temp_variable = "SR15 climate diagnostics|Raw Surface Temperature (GSAT)|MAGICCv7.5.1|10.0th Percentile"
    emissions_file = "emissions_test_file_many_scen.csv"
    offset_years = np.arange(2010, 2020, 1)
    # The name of the peak temperature column output
    more_scenarios = distributions.load_data_from_summary(
        magicc_file_for_tests_to_use,
        total_magicc_file,
        emissions_file,
        magicc_nonco2_temp_variable,
        magicc_tot_temp_variable,
        magicc_nonco2_temp_variable,
        magicc_tot_temp_variable,
        offset_years,
        vetted_scen_list_file=vetting_file,
        peak_version="officialNZ",
        permafrost=True,
        vetted_scen_list_file_sheet="meta_Ch3vetted_withclimate",
        sr15_rename=False,
    )
    with pytest.raises(UnboundLocalError):
        distributions.load_data_from_summary(
            magicc_file_for_tests_to_use,
            total_magicc_file,
            emissions_file,
            magicc_nonco2_temp_variable,
            magicc_tot_temp_variable,
            magicc_nonco2_temp_variable,
            magicc_tot_temp_variable,
            offset_years,
            peak_version="officialNZ",
            permafrost=True,
            sr15_rename=False,
        )
    fewer_scenarios = more_scenarios[np.isfinite(more_scenarios["hits_net_zero"])]
    # One of the passing years doesn't peak, so one of the 4 scenarios always fails
    assert len(more_scenarios) == 3
    assert len(fewer_scenarios) == 2
    assert all(
        fewer_scenarios.index.get_level_values("scenario") == ["Acceptable", "Acceptable"]
    )
    # Calculate the expected temp from the excel file directly
    expected_remind_accept = 0.324777364 - np.mean([
        -0.01125151, -0.020053354, -0.020859599, -0.017984689, -0.012561602,
        -0.007388461, -0.000822226, 0.003277905, 0.006592823, 0.008149571
    ])

    assert np.isclose(
        fewer_scenarios.loc[("REMIND_1_5", "World", "Acceptable"),
        magicc_tot_temp_variable], expected_remind_accept
    )


def test_dist_of_zec():
    zec_mean = 0.5
    zec_sd = 1
    n_loops = 100000
    zec_s = distributions.zec_dist(zec_mean, zec_sd, False, n_loops)
    assert len(zec_s) == n_loops
    assert type(zec_s) == type(np.array(0))
    assert np.isclose(zec_s.mean(), zec_mean, atol=1e-2)
    assert np.isclose(zec_s.std(), zec_sd, atol=1e-2)


    zec_a = distributions.zec_dist(zec_mean, zec_sd, True, n_loops)
    assert len(zec_a) == n_loops
    assert type(zec_a) == type(np.array(0))
    assert not any([x < 0 for x in zec_a])
    assert zec_a.mean() > zec_mean + 0.1
    assert zec_a.std() < zec_sd