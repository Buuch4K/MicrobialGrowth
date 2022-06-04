using Roots, Distributions, Statistics, StatsPlots, Plots, AffineInvariantMCMC, CSV, JLD2

ex_gen = Array{Float64}(undef,N,3);
ex_gen[:,1] = gendata.divratio;
ex_gen[:,2] = gendata.growth;
ex_gen[:,3] = gendata.time;

corrplot(ex_gen,label=["f","alpha","tau"],tickfontsize=4,guidefontsize=6)

readdata = PMreal_data;
ex_read = Array{Float64}(undef,length(readdata.time),3);
ex_read[:,1] = vcat(readdata.divratio,0.49);
ex_read[:,2] = readdata.growth;
ex_read[:,3] = readdata.time;

corrplot(ex_read,label=["f","alpha","tau"],tickfontsize=4,guidefontsize=6)

CSV.write("pmsyn.csv",DataFrame(f=ex_gen[:,1],alpha=ex_gen[:,2],tau=ex_gen[:,3]));
CSV.write("pmreal.csv",DataFrame(f=ex_read[:,1],alpha=ex_read[:,2],tau=ex_read[:,3]));
