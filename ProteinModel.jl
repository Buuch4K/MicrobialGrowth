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
    return Data(Y,alpha,X[1:num],f[1:num-1])
end


function read_data(filename::String)
    data = CSV.File(filename, select = ["growth_rate","generationtime","length_birth","division_ratio"]);
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
    plot(t,result,label=false)
end


function log_likeli(p::Vector,D::Data)
    # p = [o1,sig,b1,b2,o2,u,v,c]
    if any(x->x.<0,p)
        return -Inf
    else
        like = 0.;
        for k = 1:length(D.time)
            t0 = max(0,1/D.growth[k]*log((p[6]+p[8]*D.mass[k])/(p[8]*D.mass[k])))
            if D.time[k] < t0
                return -Inf
            else
                temp = log(p[5]/(p[6]+p[7])*(p[7]+p[8]*D.mass[k]*(exp(D.growth[k]*D.time[k])-1))) + (-p[5]/(p[6]+p[7])*((p[8]*D.mass[k])/D.growth[k]*(exp(D.growth[k]*D.time[k]) - exp(D.growth[k]*t0)) + (p[7]-p[8]*D.mass[k])*D.time[k] + (p[8]*D.mass[k]-p[7])*t0))
            end
            like += temp
        end
        return like + sum([logpdf(Gamma(p[1]^2/p[2],p[2]/p[1]),D.growth[k]) for k = 1:length(D.growth)]) + sum([logpdf(Beta(p[3],p[4]),D.divratio[k]) for k = 1:length(D.divratio)])
    end
end


function log_prior(p::Vector)
    # p = [o1,sig,b1,b2,o2,u,v,c]
    if p[6] >= p[7]
        return -Inf
    else
        return sum([logpdf(pri_gamma,p[k]) for k=1:2]) + sum([logpdf(pri_beta,p[k]) for k=3:4]) + sum([logpdf(pri,p[k]) for k=5:7])
    end
end


function remove_stuck_chain(chain,llhood,nwalk::Int64)
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


function extract_beta!(flat::Matrix)
    for k = 1:size(flat)[2]
        b1 = flat[3,k]; b2 = flat[4,k];
        flat[3,k] = b1/(b1+b2)
        flat[4,k] = (b1*b2)/((b1+b2)^2*(b1+b2+1))
    end
    return flat
end


# initial parameters
const o1 = 1.405; # growth distribution
const sig = 0.05;

const b1 = 0.5; # division distribution
const b2 = 0.008;

const o2 = 1.33; #hazard ratio
const u = 0.4; #lower treshhold for division
const v = 0.8; #upper treshhold for division
const c = 1.; #protein constant

#prior distributions
pri_gamma = Uniform(0,3);
pri_beta = Uniform(8,24);
pri = Uniform(0.2,5); # gendata (0,5), readdata (0.2,5)

# generate data using defined model
N = 252; #number of observations
m0 = 2.6; #initial mass of cell
gendata = generate_data(m0,N);

# read data from data set
readdata = read_data("data/modified_Susman18_physical_units.csv");

plot_data(gendata)

# applying the MH algo for the posterior Distribution
numdims = 7; numwalkers = 20; thinning = 10; numsamples_perwalker = 50000; burnin = 5000;
logpost = x -> log_likeli([x[1],x[2],x[3],x[4],x[5],x[6],x[7],c],readdata) + log_prior([x[1],x[2],x[3],x[4],x[5],x[6],x[7],c]);


x = vcat(rand(pri_gamma,2,numwalkers),rand(pri_beta,2,numwalkers),rand(pri,numdims-4,numwalkers));
chain, llhoodvals = AffineInvariantMCMC.sample(logpost,numwalkers,x,burnin,1);
chain, llhoodvals = AffineInvariantMCMC.sample(logpost,numwalkers,chain[:, :, end],numsamples_perwalker,thinning);
flatchain, flatllhoodvals = AffineInvariantMCMC.flattenmcmcarray(chain,llhoodvals);

mod_chain, mod_llhoodvals = remove_stuck_chain(chain,llhoodvals,numwalkers);
mod_flatchain, mod_flatllhoodvals = AffineInvariantMCMC.flattenmcmcarray(mod_chain,mod_llhoodvals);
beta_distr = extract_beta(mod_flatchain);