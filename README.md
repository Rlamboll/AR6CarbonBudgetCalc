# Carbon Budget Calculator
Calculates remaining carbon budget given atmospheric simulation inputs. 
The workhorse script is `src/run_budget_calculator.py`, with options to change it found 
at the top of the file.  

The calculation is based on the framework in 
*Estimating and tracking the remaining carbon budget for stringent climate targets*, 
Rogelj et al. 2019. This revolves around five terms: the warming to date (input 
directly), the non-CO2 warming (calculated from scenarios analysed previously by MAGICC 
or FaIR), the zero-emissions commitment (ZEC, input directly), the  transient climate 
response to cumulative emissions of CO2 (TCRE, parameters are input for a distribution
which may be either lognormal or normal)
and unrepresented Earth feedbacks (a linear function of temperature change).

The default values in version v1.0.3 replicate the data in the IGCC report for 2024.    

By changing the default values this code can replicate the values in the 
IPCC reports for AR6 WGI and WGIII.  
With `runver = sr15prewg1, zec_sd = 0, recent_emissions=209` this replicates Table 5.8 
in Chapter 5 (Canadell et al, 2021), Table TS.3 in the Technical Summary (Arias et al, 
2021) and Table SPM.2 of the Summary for Policymakers (IPCC, 2021) of the IPCC AR6 WGI 
report. Note that values reported in the report are rounded to the nearest 10 PgC or 50 
GtCO2 and the impact of ZEC standard deviation are tabulated separately. All the 
reports use the MAGICC True, FaIR False scenario.  

With `zec_sd = 0, recent_emissions=209` this replicates the carbon 
budget results in chapter 3 (Riahi et al, 2022) of the IPCC AR6 WG3 report.

With `peak_version = "nonCO2AtPeakAverage", List_use_permafrost=[True], norm_nonco2_years=True, nonlinear_nonco2 = "QRW"`
this replicates our recommended update.

With `recent_emissions=204, historical_dT = 1.15 ` this replicates the results in 
"Indicators of Global Climate Change 2022: annual update".

The code expects to find data from MAGICC and/or FaIR simulations in the `InputData` 
folder in the format output for a pyam dataframe. Alternatively unprocessed FaIR data 
may be provided as folders of .nc or .hfd files, which can be processed into pyam format.

# Bibliography:
Arias, P.A., Bellouin, N., Coppola, E., Jones, R.G., Krinner, G., Marotzke, J., Naik, V., Palmer, M.D., Plattner, G.-K., Rogelj, J., Rojas, M., Sillmann, J., Storelvmo, T., Thorne, P.W., Trewin, B., Achuta Rao, K., Adhikary, B., Allan, R.P., Armour, K., Bala, G., Barimalala, R., Berger, S., Canadell, J.G., Cassou, C., Cherchi, A., Collins, W., Collins, W.D., Connors, S.L., Corti, S., Cruz, F., Dentener, F.J., Dereczynski, C., Di Luca, A., Diongue Niang, A., Doblas-Reyes, F.J., Dosio, A., Douville, H., Engelbrecht, F., Eyring, V., Fischer, E., Forster, P., Fox-Kemper, B., Fuglestvedt, J.S., Fyfe, J.C., Gillett, N.P., Goldfarb, L., Gorodetskaya, I., Gutierrez, J.M., Hamdi, R., Hawkins, E., Hewitt, H.T., Hope, P., Islam, A.S., Jones, C., Kaufman, D.S., Kopp, R.E., Kosaka, Y., Kossin, J., Krakovska, S., Lee, J.-Y., Li, J., Mauritsen, T., Maycock, T.K., Meinshausen, M., Min, S.-K., Monteiro, P.M.S., Ngo-Duc, T., Otto, F., Pinto, I., Pirani, A., Raghavan, K., Ranasinghe, R., Ruane, A.C., Ruiz, L., Sallée, J.-B., Samset, B.H., Sathyendranath, S., Seneviratne, S.I., Sörensson, A.A., Szopa, S., Takayabu, I., Treguier, A.-M., van den Hurk, B., Vautard, R., von Schuckmann, K., Zaehle, S., Zhang, X., Zickfeld, K., 2021. Technical Summary, in: Masson-Delmotte, V., Zhai, P., Pirani, A., Connors, S.L., Péan, C., Berger, S., Caud, N., Chen, Y., Goldfarb, L., Gomis, M.I., Huang, M., Leitzell, K., Lonnoy, E., Matthews, J.B.R., Maycock, T.K., Waterfield, T., Yelekçi, O., Yu, R., Zhou, B. (Eds.), Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change. Cambridge University Press.
 
Shukla, P.R. and Skea, J. and Slade, R. and Khourdajie, A. Al and van Diemen, R. and McCollum, D. and Pathak, M. and Some, S. and Vyas, P. and Fradera, R. and Belkacemi, M. and Hasija, A. and Lisboa, G. and Luz, S. and Malley, J., 2022. Climate Change 2022: Mitigation of Climate Change. Contribution of Working Group III to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change 

Canadell, J.G., Monteiro, P.M.S., Costa, M.H., Cotrim da Cunha, L., Cox, P.M., Eliseev, A.V., Henson, S., Ishii, M., Jaccard, S., Koven, C., Lohila, A., Patra, P.K., Piao, S., Rogelj, J., Syampungani, S., Zaehle, S., Zickfeld, K., 2021. Global Carbon and other Biogeochemical Cycles and Feedbacks, in: Masson-Delmotte, V., Zhai, P., Pirani, A., Connors, S.L., Péan, C., Berger, S., Caud, N., Chen, Y., Goldfarb, L., Gomis, M.I., Huang, M., Leitzell, K., Lonnoy, E., Matthews, J.B.R., Maycock, T.K., Waterfield, T., Yelekçi, O., Yu, R., Zhou, B. (Eds.), Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change. Cambridge University Press. 

IPCC, 2021. Summary for Policymakers. In: Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change. 

Forster, P. M., Smith, C. J., Walsh, T., Lamb, W. F., Lamboll, R., Hauser, M., Ribes, A., Rosen, D., Gillett, N., Palmer, M. D., Rogelj, J., von Schuckmann, K., Seneviratne, S. I., Trewin, B., Zhang, X., Allen, M., Andrew, R., Birt, A., Borger, A., Boyer, T., Broersma, J. A., Cheng, L., Dentener, F., Friedlingstein, P., Gutiérrez, J. M., Gütschow, J., Hall, B., Ishii, M., Jenkins, S., Lan, X., Lee, J.-Y., Morice, C., Kadow, C., Kennedy, J., Killick, R., Minx, J. C., Naik, V., Peters, G. P., Pirani, A., Pongratz, J., Schleussner, C.-F., Szopa, S., Thorne, P., Rohde, R., Corradi, M. R., Schumacher, D., Vose, R., Zickfeld, K., Masson-Delmotte, V. and Zhai, P.: Indicators of Global Climate Change 2022: annual update of large-scale indicators of the state of the climate system and human influence, Earth Syst. Sci. Data, 15(6), 2295–2327, doi:10.5194/ESSD-15-2295-2023, 2023.

When re-using the data of global warming induced by non-CO2 species, please reference Cross-Chapter Box 7.1 (Nicholls et al, 2021).  

Nicholls, Z., Meinshausen, M., Forster, P., Armour, K., Berntsen, T., Collins, W., Jones, C., Lewis, J., Marotzke, J., Milinski, S., Rogelj, J., Smith, C., 2021. Cross-Chapter Box 7.1: Physical emulation of Earth System Models for scenario classification and knowledge integration in AR6, in: Masson-Delmotte, V., Zhai, P., Pirani, A., Connors, S.L., Péan, C., Berger, S., Caud, N., Chen, Y., Goldfarb, L., Gomis, M.I., Huang, M., Leitzell, K., Lonnoy, E., Matthews, J.B.R., Maycock, T.K., Waterfield, T., Yelekçi, O., Yu, R., Zhou, B. (Eds.), Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change. Cambridge University Press.

# Acknowledgements:
 This project has received funding from the European Union's Horizon 2020 research and 
 innovation programme under grant agreement no. 820829 (CONSTRAIN).