# MicrobialGrowth

This repository is part of my master thesis with the main goal of comparing three different individual-based growth-division models encompassing two different time- and two different trait scales.

For each of the models there is a file (BasicModel.jl,VariGrowthModel.jl,ProteinModel.jl) defining the basic functions and performing a Bayesian inference over the involved parameters.
The data which is used is stored in the data folder. The modified_Susman18_physical_units.csv contains 249 cell cycles of an individual E.coli cell.

In the ModelComparison.jl file we compare each of the model by estimating the predictive density and compare the forward simulated division data containing growth rate, division factor and division time visually.

The Python file CorrPlots.py was used to construct the correlation plots for each of the models.

Finally, the .jld files contain all variable and parameters used in the performed inferences.
