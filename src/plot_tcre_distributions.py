from src.distributions_of_inputs import tcre_distribution
import matplotlib.pyplot as plt
import numpy as np

"""
This script exists to create plots of TCRE distribution functions. 
It does this by directly generating large numbers of results and plotting the 
probability distribution function, hence will work for any distribution. 
"""
# Input instructions:
# A list of the tcre distributions to plot.
tcre_dists = ["normal", "lognormal"]
# The mean of the distribution of TCRE. We use units of C per GtCO2.
# (TCRE = Transient climate response to cumulative carbon emissions)
tcre_low = 1.1 / 3664 * 1000
# The standard deviation of the distribution of TCRE.
tcre_high = 2.2 / 3664 * 1000
# Likelihood is the probability that results fit between the low and high value
likelihood = 0.66
# Number to return such that the granularity is negligible.
n_runs = 100000000
# Where the graphs will be saved. Should contain a {} sign for each tcre_dist, tcre_low,
# tcre_high,and likelihood.
save_location = "../Output/tcre_distribution_{}_low_{}_high_{}_likelihood_{}.pdf"

# Functional code (the graph limits may need adjusting though):
for tcre_dist in tcre_dists:
    xs = tcre_distribution(tcre_low, tcre_high, likelihood, n_runs, tcre_dist)
    normed = np.histogram(xs, bins=800, density=True)
    plt.close()
    fig = plt.figure(figsize=(12, 7))
    plt.plot(normed[1][1:], normed[0])
    plt.xlabel(u"\N{DEGREE SIGN}" + "C/Tt CO2")
    plt.ylabel("Probability density")
    plt.xlim([-0.25, 1.25])
    plt.ylim([0, 3])
    fig.savefig(save_location.format(tcre_dist, tcre_low, tcre_high, likelihood))
