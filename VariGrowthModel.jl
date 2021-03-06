using Roots, Distributions, Statistics, StatsPlots, Plots, AffineInvariantMCMC, CSV

struct Data
    time
    growth
    mass
    divratio
end


function generate_data(si::Float64,num::Int64)
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


function read_data(filename::String)
    # this function takes the memory location of the csv file containing the real world data and
    # returns the values in an object DATA
    data = CSV.File(filename,select=["generationtime","length_birth","growth_rate","division_ratio"]);
    div_ratio = convert(Array{Float64},vcat(data.division_ratio[2:end],mean(data.division_ratio[2:end])));
    return Data(data.generationtime,data.growth_rate,data.length_birth,div_ratio) 
end


function plot_data(D::Data)
    # This function takes a dataset as input and visualizes the timeseries of the growth and division process.
    n = length(D.time)
    t = Array{Float64}(undef,n*10);
    result = Array{Float64}(undef,n*10);
    for k = 1:n
        start = sum(D.time[1:(k-1)]);
        t[(k-1)*10+1:k*10] = range(start,start+D.time[k],10)
        result[(k-1)*10+1:k*10] = D.mass[k] .* exp.(D.growth[k]*range(0,D.time[k],10))
    end
    plot(t,result, label=false)
end


function log_likeli(p::Vector,D::Data)
    # This function takes all parameters and a dataset as input and returns the likelihood.
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


function log_prior(p::Vector)
    # this function takes all parameters as input and returns the log_prior value.
    # p = [o1,sig,b1,b2,o2,u,v]
    if p[6] >= p[7]
        return -Inf
    else
        return logpdf(pri_gamma,p[1]) + logpdf(pri_sigma,p[2]) + logpdf(pri_beta,p[3]) + logpdf(pri_sigma,p[4]) + sum([logpdf(pri,p[k]) for k=5:length(p)])
    end
end


function remove_stuck_chain(chain,llhood,nwalk::Int64)
    #= This function removes chains from the result which always contain the same value, i.e. are stuck.
    Input:  chain - realization of the sampler,
            llhood - corresponding likelihood values,
            nwalk - number of parallel chains
    Output: chain, llhood
    =#
    bad_idx = []; 
    for k=1:nwalk
        if all(y -> y == first(chain[2,k,:]), chain[2,k,:])
            push!(bad_idx, k);
        end
    end
    idx = setdiff(1:20,bad_idx)
    println(length(idx))
    return chain[:,idx,:],llhood[:,idx,:]
end


# initial parameters
const o1 = 1.4; # growth distribution
const sig = 0.04;

const b1 = 0.5; # division distribution
const b2 = 0.002;

const o2 = 2.6; #hazard rate functions constant
const u = 2.8; #lower treshhold for division
const v = 3.; #upper treshhold for division

# define prior distributions
pri_gamma = Uniform(0,3);
pri_beta = Uniform(0.4,0.6); 
pri_sigma = Uniform(0,0.2); 
pri = Uniform(0,5);

# generate data using defined model
N = 250; #number of observations
m0 = 2.6; #initial mass of cell
gendata = generate_data(m0,N);

# read data from dataset
readdata = read_data("data/modified_Susman18_physical_units.csv");

plot_data(readdata)

# sampling all parameters using SYNTHETIC data
numdims = 7; numwalkers = 20; thinning = 10; numsamples_perwalker = 40000; burnin = 2000;
logpost = x -> log_likeli(x,gendata) + log_prior(x);
x = vcat(rand(pri_gamma,1,numwalkers),rand(pri_sigma,1,numwalkers),rand(pri_beta,1,numwalkers),rand(pri_sigma,1,numwalkers),rand(pri,numdims-4,numwalkers));

chain, llhood = AffineInvariantMCMC.sample(logpost,numwalkers,x,burnin,1);
chain, llhood = AffineInvariantMCMC.sample(logpost,numwalkers,chain[:, :, end],numsamples_perwalker,thinning);
chain, llhood = remove_stuck_chain(chain,llhood,numwalkers);
VMsyn_flatchain, VMsyn_flatllhoodvals = AffineInvariantMCMC.flattenmcmcarray(chain,llhood);

# sampling all parameters using REAL data
logpost = x -> log_likeli(x,readdata) + log_prior(x);
x = vcat(rand(pri_gamma,1,numwalkers),rand(pri_sigma,1,numwalkers),rand(pri_beta,1,numwalkers),rand(pri_sigma,1,numwalkers),rand(pri,numdims-4,numwalkers));

chain, llhood = AffineInvariantMCMC.sample(logpost,numwalkers,x,burnin,1);
chain, llhood = AffineInvariantMCMC.sample(logpost,numwalkers,chain[:, :, end],numsamples_perwalker,thinning);
chain, llhood = remove_stuck_chain(chain,llhood,numwalkers);
VMsyn_flatchain, VMsyn_flatllhoodvals = AffineInvariantMCMC.flattenmcmcarray(chain,llhood);


# plotting the correlation plots for both simulations
corrplot(transpose(BMsyn_flatchain),title="synthetic data",label=["mu_a","sigma_a","mu_f","sigma_f","o2","u","v"],tickfontsize=4,guidefontsize=6)

corrplot(transpose(BMreal_flatchain),title="real world data",label=["mu_a","sigma_a","mu_f","sigma_f","o2","u","v"],tickfontsize=4,guidefontsize=6)

