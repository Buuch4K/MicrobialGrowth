using Roots, Distributions, Statistics, StatsPlots, Plots, AffineInvariantMCMC, CSV

struct Data
    time
    mass
    divratio
    growth
end


function generate_data(si::Float64,N::Int64)
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


function read_data(filename::String)
    # this function takes the memory location of the csv file containing the real world data and
    # returns the values in an object DATA
    data = CSV.File(filename,select=["division_ratio","generationtime","length_birth","growth_rate"]);
    N = length(data.generationtime);
    return Data(data.generationtime,data.length_birth,[1/2 for i=1:N],[1. for i=1:N])
end


function plot_survival(s,t)
    #= This function scatters the survival function S(t).
    Input:  s - initial size of the cell
            t - time points the function is evaluated at
    =#
    result = Array{Float64}(undef,length(t));
    if s < u
        for k = 1:length(t)
            temp = exp(-o2/(v+u)*(u/o1*exp(o1*t[k]) + v*t[k] - u/o1 - v*1/o1*log(u/s)))
            result[k] = min(1,temp)
        end
    else
        for k = 1:length(t)
            result[k] = exp(-o2/(v+u)*(s/o1*exp(o1*t[k]) + v*t[k] - s/o1))
        end
    end
    scatter(t,result)
end


function plot_data(D::Data,growth = 1)
    # This function takes a dataset as input and visualizes the timeseries of the growth and division process.
    t = Array{Float64}(undef,length(D.time)*10);
    result = Array{Float64}(undef,length(D.time)*10);
    for k = 1:length(D.time)
        start = sum(D.time[1:(k-1)])
        t[(k-1)*10+1:k*10] = range(start,start+D.time[k],10)
        result[(k-1)*10+1:k*10] = D.mass[k] .* exp.(growth*range(0,D.time[k],10)) 
    end
    plot(t,result, label = false)
end


function log_likeli(p::Vector,D::Data)
    # This function takes all parameters and a dataset as input and returns the likelihood.
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


function log_prior(p::Vector)
    # this function takes all parameters as input and returns the log_prior value.
    # para = [o1,o2,u,v]
    if p[3] > p[4]
        return -Inf
    else
        return sum([logpdf(pri,p[k]) for k = 1:length(p)])
    end
end


function remove_stuck_chain(chain,llhood,nwalk::Int64)
    #= This function removes chains from the result which always contain the same value, i.e. are stuck.
    Input:  chain - realization of the sampler,
            llhood - corresponding likelihood values,
            nwalk - number of parallel chains
    Output: chain, llhood =#
    bad_idx = []; 
    for k=1:nwalk
        if all(y -> y == first(chain[3,k,:]), chain[3,k,:])
            push!(bad_idx, k);
        end
    end
    idx = setdiff(1:20,bad_idx)
    println(length(idx))
    return chain[:,idx,:],llhood[:,idx,:]
end



# initial values for generating data
const o1 = 1.; #exponential growth rate
const o2 = 0.8; #hazard rate functions constant
const u = 0.2; #lower treshhold for division
const v = 5.; #upper treshhold for division

# define prior distribution
pri = Uniform(2,10);

# initial parameters for the data generation
N = 250; #number of observations
m0 = 2.4; #initial size
gendata = generate_data(m0,N);

# read data from csv file
readdata = read_data("data/modified_Susman18_physical_units.csv");

plot_data(gendata)

# sampling all parameters using SYNTHETIC data
numdims = 4; numwalkers = 20; thinning = 10; numsamples_perwalker = 20000; burnin = 2000;
logpost = x -> log_likeli(x,gendata) + log_prior(x);
x = rand(pri,numdims,numwalkers); # define initial points

chain, llhood = AffineInvariantMCMC.sample(logpost,numwalkers,x,burnin,1);
chain, llhood = AffineInvariantMCMC.sample(logpost,numwalkers,chain[:, :, end],numsamples_perwalker,thinning);
chain, llhood = remove_stuck_chain(chain,llhood,numwalkers);
BMsyn_flatchain, BMsyn_flatllhood = AffineInvariantMCMC.flattenmcmcarray(chain,llhoodvals);

#sampling all parameters using REAL data
logpost = x -> log_likeli(x,readdata) + log_prior(x);
x = rand(pri,numdims,numwalkers); # define initial points

chain, llhood = AffineInvariantMCMC.sample(logpost,numwalkers,x,burnin,1);
chain, llhood = AffineInvariantMCMC.sample(logpost,numwalkers,chain[:, :, end],numsamples_perwalker,thinning);
chain, llhood = remove_stuck_chain(chain,llhood,numwalkers);
BMreal_flatchain, BMreal_flatllhood = AffineInvariantMCMC.flattenmcmcarray(chain,llhood);

# plotting the correlation plots for both simulations
corrplot(transpose(BMsyn_flatchain),title="synthetic data",label=["o1","o2","u","v"],tickfontsize=4,guidefontsize=6)

corrplot(transpose(BMreal_flatchain),title="real world data",label=["o1","o2","u","v"],tickfontsize=4,guidefontsize=6)