using Roots, Distributions, Statistics, StatsPlots, Plots, AffineInvariantMCMC, CSV

struct Data
    time
    growth
    mass
    divratio
end


function generate_data(si::Float64,num::Int64)
    X = Float64[si];
    Y = Array{Float64}(undef,num);
    k = rand(Uniform(0,1),num);
    alpha = rand(Gamma(o1,sig),num);
    f = rand(Beta(b1,b2),num);
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
    return Data(Y,alpha,X[1:num],f[1:num-1]) 
end


function read_data(filename::String)
    data = CSV.File(filename,select=["generationtime","length_birth","growth_rate","division_ratio"]);
    return Data(data.generationtime,data.growth_rate,data.length_birth,data.division_ratio[2:end]) 
end


function plot_data(D::Data)
    n = length(D.time)
    t = Array{Float64}(undef,n*10);
    result = Array{Float64}(undef,n*10);
    for k = 1:n
        start = sum(D.time[1:(k-1)]);
        t[(k-1)*10+1:k*10] = range(start,start+D.time[k],10)
        result[(k-1)*10+1:k*10] = D.mass[k] .* exp.(D.growth[k]*range(0,D.time[k],10))
    end
    plot(t,result)
end

function log_likeli_gd(p::Vector,D::Data)
    # p = [o1,sig,b1,b2]
    if any(x->x.<0,p)
        return -Inf
    else
        return sum([logpdf(Gamma(p[1],p[2]),D.growth[k]) for k=1:length(D.growth)]) + sum([logpdf(Beta(p[3],p[4]),D.divratio[k]) for k=1:length(D.divratio)])
    end
end


function log_prior_gd(p::Vector)
    # p = [01,sig,b1,b2]
    return logpdf(pri_Gamma,p[1]) + logpdf(pri_1,p[2]) + logpdf(pri_Beta,p[3]) + logpdf(pri_Beta,p[4])
end


function log_likeli(p::Vector,D::Data,f::Vector)
    # p = [o2,u,v]
    if any(x->x.<0,p)
        return -Inf
    else
        like = 0.;
        for k = 1:length(D.time)
            t0 = max(0,1/D.growth[k]*log(p[2]/D.mass[k]));
            if D.time[k] < t0
                return -Inf
            else
                temp = log((p[1]*p[3])/(p[2]+p[3]) + (p[1]*D.mass[k])/(p[2]+p[3])*exp(D.growth[k]*D.time[k])) + (-p[1]/(p[2]+p[3])*(D.mass[k]/D.growth[k]*(exp(D.growth[k])*D.time[k] - exp(D.growth[k]*t0)) + p[3]*(D.time[k] - t0)))
            end
            like += temp
        end
        return like + sum([logpdf(Gamma(f[1],f[2]),D.growth[k]) for k=1:length(D.growth)]) + sum([logpdf(Beta(f[3],f[4]),D.divratio[k]) for k=1:length(D.divratio)])
    end
end


function log_prior(p::Vector)
    #p = [o2,u,v]
    if p[2] >= p[3]
        return -Inf
    else
        return sum([logpdf(pri_2,p[k]) for k=1:length(p)])
    end
end


function remove_stuck_chain(chain,llhood,nwalk)
    bad_idx = []; 
    for k=1:nwalk
        if all(y -> y == first(chain[2,k,:]), chain[2,k,:])
            push!(bad_idx, k);
        end
    end
    idx = setdiff(1:20,bad_idx)
    return chain[:,idx,:],llhood[:,idx,:]
end


# initial parameters
const o1 = 27.; # growth distribution
const sig = 0.05;

const b1 = 15.; # division distribution
const b2 = 15.;

const o2 = 1.33; #hazard rate functions constant
const u = 0.2; #lower treshhold for division
const v = 5.5; #upper treshhold for division

#prior distributions
pri_Gamma = Uniform(20,36);
pri_Beta = Uniform(12,24);
pri_1 = Uniform(0,10);
pri_2 = Uniform(1,9);


# generate data using defined model
N = 200; #number of observations
m0 = 2.6; #initial mass of cell
gendata = generate_data(m0,N);

# read data from dataset
readdata = read_data("data/modified_Susman18_physical_units.csv");

plot_data(readdata)

scatter(gendata.divratio .* exp.(gendata.growth[2:end].*gendata.time[2:end]))

# applying the MH algo for the posterior Distribution in two steps
numdims = 4; numwalkers = 20; thinning = 10; numsamples_perwalker = 20000; burnin = 1000;
logpost_gd = x -> log_likeli_gd(x,readdata) + log_prior_gd(x);

# step one: infer the parameters for the growth and division distribution
x = vcat(rand(pri_Gamma,1,numwalkers),rand(pri_1,1,numwalkers),rand(pri_Beta,2,numwalkers));
chain1, llhoodvals1 = AffineInvariantMCMC.sample(logpost_gd,numwalkers,x,burnin,1);
chain1, llhoodvals1 = AffineInvariantMCMC.sample(logpost_gd,numwalkers,chain1[:, :, end],numsamples_perwalker,thinning);
flatchain1, flatllhoodvals1 = AffineInvariantMCMC.flattenmcmcarray(chain1,llhoodvals1);
flatchain1 = permutedims(flatchain1,[2,1]);
fixed = mean(flatchain1,dims=1)[1,:]

# step two: infer the parameters o2,u,v
numdims = 3; logpost = x -> log_likeli(x,readdata,fixed) + log_prior(x);
x = rand(pri_2,numdims,numwalkers);
chain2, llhoodvals2 = AffineInvariantMCMC.sample(logpost,numwalkers,x,burnin,1);
chain2, llhoodvals2 = AffineInvariantMCMC.sample(logpost,numwalkers,chain2[:, :, end],numsamples_perwalker,thinning);
flatchain2, flatllhoodvals2 = AffineInvariantMCMC.flattenmcmcarray(chain2,llhoodvals2);

mod_chain2, mod_llhoodvals2 = remove_stuck_chain(chain2,llhoodvals2,numwalkers);
mod_flatchain2, mod_flatllhoodvals2 = AffineInvariantMCMC.flattenmcmcarray(mod_chain2,mod_llhoodvals2);
mod_flatchain2 = permutedims(mod_flatchain2,[2,1]);