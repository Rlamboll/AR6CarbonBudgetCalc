# Carbon Budget Calculator
Calculates remaining carbon budget given atmospheric simulation inputs. 
The workhorse script is `src/run_budget_calculator.py`
Expects to find data from MAGICC or FaIR simulations in the `InputData` folder in the 
standard format output for a pyam dataframe. Specification of how to read this data 
and options for running the code are all found at the beginning of 
`src/run_budget_calculator.py`. 

The calculation is based on the framework in 
*Estimating and tracking the remaining carbon budget for stringent climate targets*, 
Rogelj et al. 2019. This revolves around five terms: the warming to date (input 
directly), the non-CO2 warming (calculated from scenarios analysed previously by MAGICC 
or FaIR), the zero-emissions commitment (ZEC, input directly), the  transient climate 
response to cumulative emissions of CO2 (TCRE, parameters are input for a distribution
which may be either lognormal or normal)
and unrepresented Earth feedbacks (a linear function of temperature change).   

By changing the value `runver` this code can replicate the values in the IPCC reports 
for AR6 WGI and WGIII.  
With `runver = sr15wg1` this replicates Table 5.8 in Chapter 5 (Canadell et al, 2021), 
Table TS.3 in the Technical Summary (Arias et al, 2021) and Table SPM.2 of the Summary 
for Policymakers (IPCC, 2021) of the IPCC AR6 WGI report. Note that values reported in 
the report are rounded to the nearest 10 PgC or 50 GtCO2.

With `runver = ar6wg3` this replicates the carbon budget results in chapter 3 (Riahi et 
al, 2022) of the IPCC AR6 WG3 report. 


Arias, P.A., Bellouin, N., Coppola, E., Jones, R.G., Krinner, G., Marotzke, J., Naik, V., Palmer, M.D., Plattner, G.-K., Rogelj, J., Rojas, M., Sillmann, J., Storelvmo, T., Thorne, P.W., Trewin, B., Achuta Rao, K., Adhikary, B., Allan, R.P., Armour, K., Bala, G., Barimalala, R., Berger, S., Canadell, J.G., Cassou, C., Cherchi, A., Collins, W., Collins, W.D., Connors, S.L., Corti, S., Cruz, F., Dentener, F.J., Dereczynski, C., Di Luca, A., Diongue Niang, A., Doblas-Reyes, F.J., Dosio, A., Douville, H., Engelbrecht, F., Eyring, V., Fischer, E., Forster, P., Fox-Kemper, B., Fuglestvedt, J.S., Fyfe, J.C., Gillett, N.P., Goldfarb, L., Gorodetskaya, I., Gutierrez, J.M., Hamdi, R., Hawkins, E., Hewitt, H.T., Hope, P., Islam, A.S., Jones, C., Kaufman, D.S., Kopp, R.E., Kosaka, Y., Kossin, J., Krakovska, S., Lee, J.-Y., Li, J., Mauritsen, T., Maycock, T.K., Meinshausen, M., Min, S.-K., Monteiro, P.M.S., Ngo-Duc, T., Otto, F., Pinto, I., Pirani, A., Raghavan, K., Ranasinghe, R., Ruane, A.C., Ruiz, L., Sallée, J.-B., Samset, B.H., Sathyendranath, S., Seneviratne, S.I., Sörensson, A.A., Szopa, S., Takayabu, I., Treguier, A.-M., van den Hurk, B., Vautard, R., von Schuckmann, K., Zaehle, S., Zhang, X., Zickfeld, K., 2021. Technical Summary, in: Masson-Delmotte, V., Zhai, P., Pirani, A., Connors, S.L., Péan, C., Berger, S., Caud, N., Chen, Y., Goldfarb, L., Gomis, M.I., Huang, M., Leitzell, K., Lonnoy, E., Matthews, J.B.R., Maycock, T.K., Waterfield, T., Yelekçi, O., Yu, R., Zhou, B. (Eds.), Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change. Cambridge University Press.
 
Shukla, P.R. and Skea, J. and Slade, R. and Khourdajie, A. Al and van Diemen, R. and McCollum, D. and Pathak, M. and Some, S. and Vyas, P. and Fradera, R. and Belkacemi, M. and Hasija, A. and Lisboa, G. and Luz, S. and Malley, J., 2022. Climate Change 2022: Mitigation of Climate Change. Contribution of Working Group III to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change 

Canadell, J.G., Monteiro, P.M.S., Costa, M.H., Cotrim da Cunha, L., Cox, P.M., Eliseev, A.V., Henson, S., Ishii, M., Jaccard, S., Koven, C., Lohila, A., Patra, P.K., Piao, S., Rogelj, J., Syampungani, S., Zaehle, S., Zickfeld, K., 2021. Global Carbon and other Biogeochemical Cycles and Feedbacks, in: Masson-Delmotte, V., Zhai, P., Pirani, A., Connors, S.L., Péan, C., Berger, S., Caud, N., Chen, Y., Goldfarb, L., Gomis, M.I., Huang, M., Leitzell, K., Lonnoy, E., Matthews, J.B.R., Maycock, T.K., Waterfield, T., Yelekçi, O., Yu, R., Zhou, B. (Eds.), Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change. Cambridge University Press. 

IPCC, 2021. Summary for Policymakers. In: Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change. 

When re-using the data of global warming induced by non-CO2 species, please reference Cross-Chapter Box 7.1 (Nicholls et al, 2021).  

Nicholls, Z., Meinshausen, M., Forster, P., Armour, K., Berntsen, T., Collins, W., Jones, C., Lewis, J., Marotzke, J., Milinski, S., Rogelj, J., Smith, C., 2021. Cross-Chapter Box 7.1: Physical emulation of Earth System Models for scenario classification and knowledge integration in AR6, in: Masson-Delmotte, V., Zhai, P., Pirani, A., Connors, S.L., Péan, C., Berger, S., Caud, N., Chen, Y., Goldfarb, L., Gomis, M.I., Huang, M., Leitzell, K., Lonnoy, E., Matthews, J.B.R., Maycock, T.K., Waterfield, T., Yelekçi, O., Yu, R., Zhou, B. (Eds.), Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change. Cambridge University Press. 