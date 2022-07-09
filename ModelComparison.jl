using JLD2, Plots, Roots, StatsPlots, Distributions, Statistics, AffineInvariantMCMC, CSV, JLD2, DataFrames

struct Data
    time
    growth
    mass
    divratio
end


function basic_generate_data(si::Float64,N::Int64)
    #= this function computes a synthetic data set of size N and initial cell size si.
    Input:  si - initial cell size
            N - number of division times
    Output: object DATA containing the data set
    =#
    X = Float64[si]; #sizes of the cell at division
    Y = Array{Float64}(undef,N); #division times
    k = rand(Uniform(0,1),N);
    for i = 1:N
        if X[i] < u
            t0 = 1/o1*log(u/X[i])
            f = t -> log(k[i]) + o2/(v+u)*(u/o1*exp(o1*t) + v*t - u/o1)
        else
            t0 = 0.
            f = t -> log(k[i]) + o2/(v+u)*(X[i]/o1*exp(o1*t) + v*t - X[i]/o1)
        end
        fx = ZeroProblem(f, 1);
        Y[i] = solve(fx) + t0;
        next_size = (X[i] * exp(o1*Y[i]))/2;
        push!(X, next_size)
    end
    return Data(Y,X[1:N],[1/2 for i=1:N],[o1 for i=1:N])
end


function varying_generate_data(si::Float64,num::Int64)
    #= this function computes a synthetic data set of size num and initial cell size si.
    Input:  si - initial cell size
            num - number of division times
    Output: object DATA containing the data set
    =#
    X = Float64[si];
    Y = Array{Float64}(undef,num);
    k = rand(Uniform(0,1),num);
    alpha = rand(Gamma(o1^2/sig,sig/o1),num);
    f = rand(Beta((b1^2*(1-b1)-b2*b1)/b2,(b1*(1-b1)^2-b2*(1-b1))/b2),num);
    for n = 1:num
        if X[n] < u
            t0 = 1/alpha[n]*log(u/X[n]);
            h = t -> log(k[n]) + o2/(v+u)*(u/alpha[n]*exp(alpha[n]*t) + v*t - u/alpha[n])
        else
            t0 = 0;
            h = t -> log(k[n]) + o2/(v+u)*(X[n]/alpha[n]*exp(alpha[n]*t) + v*t - X[n]/alpha[n])
        end
        hx = ZeroProblem(h, 1)
        Y[n] = solve(hx)+t0;
        next_size = X[n] * exp(alpha[n]*Y[n]) * f[n]
        push!(X, next_size)
    end
    return Data(Y,alpha,X[1:num],f[1:num]) 
end


function protein_generate_data(si::Float64,num::Int64)
    #= this function computes a synthetic data set of size num and initial cell size si.
    Input:  si - initial cell size
            num - number of division times
    Output: object DATA containing the data set
    =#
    X = Float64[si];
    Y = Array{Float64}(undef,num);
    k = rand(Uniform(0,1),num);
    alpha = rand(Gamma(o1^2/sig,sig/o1),num);
    f = rand(Beta((b1^2*(1-b1)-b2*b1)/b2,(b1*(1-b1)^2-b2*(1-b1))/b2),num);
    for n = 1:num
        t0 = 1/alpha[n]*log((u+c*X[n])/(c*X[n]));
        h = t -> log(k[n]) + o2/(u+v)*((c*X[n])/alpha[n]*exp(alpha[n]*(t+t0)) - c*X[n]*t + v*t - (c*X[n])/alpha[n]*exp(alpha[n]*t0))
        hx = ZeroProblem(h, 1)
        Y[n] = solve(hx)+t0;
        next_size = X[n] * exp(alpha[n]*Y[n]) * f[n]
        push!(X, next_size)
    end
    return Data(Y,alpha,X[1:num],f[1:num])
end


function read_data(filename::String)
    # reads the data from a csv file and returns an object Data
    data = CSV.File(filename,select=["growth_rate","generationtime","length_birth","division_ratio"]);
    div_ratio = convert(Array{Float64},vcat(data.division_ratio[2:end],mean(data.division_ratio[2:end])));
    return Data(data.generationtime,data.growth_rate,data.length_birth,div_ratio);
end


function split_data(all::Data,test_size = 100)
    # splits all data in training and testing data sets where test data has fixed size 100
    test_idx = sort(rand(1:length(all.time),test_size));
    train_idx = setdiff(1:length(all.time),test_idx);
    train = Data(all.time[train_idx],all.growth[train_idx],all.mass[train_idx],all.divratio[train_idx]);
    test = Data(all.time[test_idx],all.growth[test_idx],all.mass[test_idx],all.divratio[test_idx]);
    return train,test
end


function remove_stuck_chain(chain,llhood,nwalk::Int64)
    # takes the inference chain and removes chains where all entries are equal to the first one.
    bad_idx = []; 
    for k=1:nwalk
        if all(y -> y == first(chain[2,k,:]), chain[2,k,:])
            push!(bad_idx, k);
        end
    end
    idx = setdiff(1:nwalk,bad_idx)
    println(length(idx))
    return chain[:,idx,:],llhood[:,idx,:]
end


function bm_loglikeli(p::Vector,D::Data)
    # log likelihood function of basic model
    # para = [o1,o2,u,v]
    if any(x -> x.<0,p)
        return -Inf
    else
        like = 0.;
        for k in 1:length(D.time)
            t0 = max(0,1/p[1]*log(p[3]/D.mass[k]))
            if D.time[k] < t0
                return -Inf
            else
                temp = log(p[2]/(p[3]+p[4])*(D.mass[k]*exp(p[1]*D.time[k]) + p[4])) + (-p[2]/(p[3]+p[4])*(D.mass[k]/p[1]*(exp(p[1]*D.time[k])-exp(p[1]*t0)) + p[4]*(D.time[k]-t0)))
            end
            like += temp
        end
        return like
    end
end


function bm_logprior(p::Vector)
    # log prior of basic model
    if p[3] > p[4]
        return -Inf
    else
        return sum([logpdf(pri_bm,p[k]) for k = 1:length(p)])
    end
end


function vm_loglikeli(p::Vector,D::Data)
    # log likelihood function of varying model
    # p = [o1,sig,b1,b2,o2,u,v]
    if any(x->x.<0,p)
        return -Inf
    elseif p[4] >= p[3]*(1-p[3])
        return -Inf
    else
        like = 0.;
        for k = 1:length(D.time)
            t0 = max(0,1/D.growth[k]*log(p[6]/D.mass[k]));
            if D.time[k] < t0
                return -Inf
            else
                temp = log(p[5]/(p[6]+p[7])*(p[7] + D.mass[k]*exp(D.growth[k]*D.time[k]))) + (-p[5]/(p[6]+p[7])*(D.mass[k]/D.growth[k]*(exp(D.growth[k])*D.time[k] - exp(D.growth[k]*t0)) + p[7]*(D.time[k] - t0)))
            end
            like += temp
        end
        return like + sum([logpdf(Gamma(p[1]^2/p[2],p[2]/p[1]),D.growth[k]) for k=1:length(D.growth)]) + sum([logpdf(Beta(p[3]/p[4]*(p[3]*(1-p[3])-p[4]),(1-p[3])/p[4]*(p[3]*(1-p[3])-p[4])),D.divratio[k]) for k=1:length(D.divratio)])
    end
end


function vm_logprior(p::Vector)
    # log prior of varying model
    # p = [o1,sig,b1,b2,o2,u,v]
    if p[6] > p[7]
        return -Inf
    else
        return logpdf(pri_gamma,p[1]) + logpdf(pri_sigma,p[2]) + logpdf(pri_beta,p[3]) + logpdf(pri_sigma,p[4]) + sum([logpdf(pri_vm,p[k]) for k=5:length(p)])
    end
end


function pm_loglikeli(p::Vector,D::Data)
    # og likelihood function of protein model
    # p = [o1,sig,b1,b2,o2,u,v]
    if any(x->x.<0,p)
        return -Inf
    elseif p[4] >= p[3]*(1-p[3])
        return -Inf
    else
        like = 0.;
        for k = 1:length(D.time)
            t0 = max(0,1/D.growth[k]*log((p[6]+c*D.mass[k])/(c*D.mass[k])))
            if D.time[k] < t0
                return -Inf
            else
                temp = log(p[5]/(p[6]+p[7])*(p[7]+c*D.mass[k]*(exp(D.growth[k]*D.time[k])-1))) + (-p[5]/(p[6]+p[7])*((c*D.mass[k])/D.growth[k]*(exp(D.growth[k]*D.time[k]) - exp(D.growth[k]*t0)) + (p[7]-c*D.mass[k])*D.time[k] + (c*D.mass[k]-p[7])*t0))
            end
            like += temp
        end
        return like + sum([logpdf(Gamma(p[1]^2/p[2],p[2]/p[1]),D.growth[k]) for k = 1:length(D.growth)]) + sum([logpdf(Beta(p[3]/p[4]*(p[3]*(1-p[3])-p[4]),(1-p[3])/p[4]*(p[3]*(1-p[3])-p[4])),D.divratio[k]) for k = 1:length(D.divratio)])
    end
end

function pm_logprior(p::Vector)
    # log prior of protein model
    # p = [o1,sig,b1,b2,o2,u,v]
    if p[6] > p[7]
        return -Inf
    else
        return logpdf(pri_gamma,p[1]) + logpdf(pri_sigma,p[2]) + logpdf(pri_beta,p[3]) + logpdf(pri_sigma,p[4]) + sum([logpdf(pri_pm,p[k]) for k=5:length(p)])
    end
end



#################### global setup
all_data = read_data("data/modified_Susman18_physical_units.csv");
const max_iter = 30;

#################### Compute posterior and predictive density of basic model
pri_bm = Uniform(2,10); 
numdims = 4; numwalkers = 20; thinning = 10; numsamples_perwalker = 20000; burnin = 2000;
bm_pd = Array{Float64}(undef,max_iter);

for k=1:max_iter
    println("STATUS: $k. iteration of $max_iter");

    train_data,test_data = split_data(all_data);
    bm_logpost = x -> bm_loglikeli(x, train_data) + bm_logprior(x);
    x = rand(pri_bm,numdims,numwalkers);

    chain, llhoodvals = AffineInvariantMCMC.sample(bm_logpost,numwalkers,x,burnin,1);
    chain, llhoodvals = AffineInvariantMCMC.sample(bm_logpost,numwalkers,chain[:, :, end],numsamples_perwalker,thinning);
    chain, llhoodvals = remove_stuck_chain(chain,llhoodvals,numwalkers);
    bm_flatchain,bm_flatllhood = AffineInvariantMCMC.flattenmcmcarray(chain,llhoodvals);

    bm_mle = bm_flatchain[:,argmax(bm_flatllhood)]; # maximum likelihood estimates
    bm_pd[k] = bm_loglikeli(bm_mle,test_data);
end
bm_pd_mean = mean(deleteat!(bm_pd,findall(x->x<0,bm_pd))) # 59.606 / 61.570 / 54.322


#################### Compute posterior and predictive density of varying growth and division model
pri_gamma = Uniform(0,3);pri_beta = Uniform(0.4,0.6);pri_sigma = Uniform(0,0.2);pri_vm = Uniform(0,5);
numdims = 7; numwalkers = 20; thinning = 10; numsamples_perwalker = 40000; burnin = 4000;
vm_pd = Array{Float64}(undef,max_iter);

for k=1:max_iter
    println("STATUS: $k. iteration of $max_iter");
    
    train_data,test_data = split_data(all_data);
    vm_logpost = x -> vm_loglikeli(x, train_data) + vm_logprior(x);
    x = vcat(rand(pri_gamma,1,numwalkers),rand(pri_sigma,1,numwalkers),rand(pri_beta,1,numwalkers),rand(pri_sigma,1,numwalkers),rand(pri_vm,numdims-4,numwalkers));
    
    chain, llhoodvals = AffineInvariantMCMC.sample(vm_logpost,numwalkers,x,burnin,1);
    chain, llhoodvals = AffineInvariantMCMC.sample(vm_logpost,numwalkers,chain[:, :, end],numsamples_perwalker,thinning);
    chain, llhoodvals = remove_stuck_chain(chain,llhoodvals,numwalkers);
    vm_flatchain, vm_flatllhood = AffineInvariantMCMC.flattenmcmcarray(chain,llhoodvals);

    vm_mle = vm_flatchain[:,argmax(vm_flatllhood)]; # maximum likelihood estimates
    vm_pd[k] = vm_loglikeli(vm_mle,test_data);
end
vm_pd_mean = mean(deleteat!(vm_pd,findall(x->x<0,vm_pd))) # 201.438 / 197.248 / 200.100


#################### Compute posterior and predictive density of protein model
pri_gamma = Uniform(0.8,1.8);pri_pm = Uniform(0.2,1.8);
const c = 1.; numdims = 7; numwalkers = 20; thinning = 10; numsamples_perwalker = 60000; burnin = 6000;
pm_pd = Array{Float64}(undef,max_iter);

for k=1:max_iter
    println("STATUS: $k. iteration of $max_iter");

    train_data,test_data = split_data(all_data);
    pm_logpost = x -> pm_loglikeli(x,train_data) + pm_logprior(x);
    x = vcat(rand(pri_gamma,1,numwalkers),rand(pri_sigma,1,numwalkers),rand(pri_beta,1,numwalkers),rand(pri_sigma,1,numwalkers),rand(pri_pm,numdims-4,numwalkers));

    chain, llhoodvals = AffineInvariantMCMC.sample(pm_logpost,numwalkers,x,burnin,1);
    chain, llhoodvals = AffineInvariantMCMC.sample(pm_logpost,numwalkers,chain[:, :, end],numsamples_perwalker,thinning);
    chain, llhoodvals = remove_stuck_chain(chain,llhoodvals,numwalkers);
    pm_flatchain, pm_flatllhood = AffineInvariantMCMC.flattenmcmcarray(chain,llhoodvals);

    pm_mle = pm_flatchain[:,argmax(pm_flatllhood)]; # maximum likelihood estimates
    pm_pd[k] = pm_loglikeli(pm_mle,test_data);
end
pm_pd_mean = mean(deleteat!(pm_pd,findall(x->x<0,pm_pd))) # 240.854 / 232.833 / 232.408


########## basic model forward simulation to compare data distributions
@load "bmreal.jld"
const o1,o2,u,v = BMreal_flatchain[:,argmax(BMreal_flatllhood)];
bm_gendata = basic_generate_data(2.6,2500);

histogram([BMreal_data.time,bm_gendata.time],label=["real world" "simulated"],normalize=true,fillalpha=0.4,fillcolor=[:blue :green],bins=0:0.1:1.3)


########## varying growth and division model forward simulation to compare data distributions
@load "vmreal.jld"
const o1,sig,b1,b2,o2,u,v = VMreal_flatchain[:,argmax(VMreal_flatllhood)];
vm_gendata = varying_generate_data(2.6,2500);

histogram([VMreal_data.growth,vm_gendata.growth],label=["real world" "simulated"],normalize=true,fillalpha=0.4,fillcolor=[:blue :green],bins=0.5:0.05:2.75)
histogram([VMreal_data.divratio,vm_gendata.divratio],label=["real world" "simulated"],normalize=true,fillalpha=0.4,fillcolor=[:blue :green],bins=0.3:0.01:0.7)
histogram([VMreal_data.time,vm_gendata.time],label=["real world" "simulated"],normalize=true,fillalpha=0.4,fillcolor=[:blue :green],bins=0:0.1:1.6)


########## protein model forward simulation to compare data distributions
@load "pmreal.jld"
const o1,sig,b1,b2,o2,u,v = PMreal_flatchain[:,argmax(PMreal_flatllhood)]; const c=1;
pm_gendata = protein_generate_data(2.6,2500);

histogram([PMreal_data.growth,pm_gendata.growth],label=["real world" "simulated"],normalize=true,fillalpha=0.4,fillcolor=[:blue :green],bins=0.5:0.05:2.75)
histogram([PMreal_data.divratio,pm_gendata.divratio],label=["real world" "simulated"],normalize=true,fillalpha=0.4,fillcolor=[:blue :green],bins=0.3:0.01:0.7)
histogram([PMreal_data.time,pm_gendata.time],label=["real world" "simulated"],normalize=true,fillalpha=0.4,fillcolor=[:blue :green],bins=0:0.1:1.6)