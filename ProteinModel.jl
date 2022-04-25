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
    plot(t,result)
end


function log_likeli(D::Data,p::Vector)
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
                temp = log(p[5]/(p[6]+p[7])*(p[7]+p[8]*D.mass[k]*(exp(D.growth[k]*D.time[k])-1))) - (p[5]/(p[6]+p[7])*((p[8]*D.mass[k])/D.growth[k]*(exp(D.growth[k]*D.time[k]) - exp(D.growth[k]*t0)) + (p[7]+p[8]*D.mass[k])*D.time[k] + (p[8]*D.mass[k]-p[7])*t0))
            end
            like += temp
        end
        return like + sum([logpdf(Gamma(p[1],p[2]),D.growth[k]) for k = 1:length(D.growth)]) + sum([logpdf(Beta(p[3],p[4]),D.divratio[k]) for k = 1:length(D.divratio)])
    end
end


function log_prior(p::Vector)
    # p = [o1,sig,b1,b2,o2,u,v,c]
    if p[6] >= p[7]
        return -Inf
    else
        gam = logpdf(pri_Gamma,para[1]) + logpdf(pri,para[2]);
        be = logpdf(pri_Beta,para[3]) + logpdf(pri_Beta,para[4]);
        re = sum([logpdf(pri,para[k]) for k=5:length(p)]);
        return gam+be+re
    end
end


# initial parameters
const o1 = 27.1232; # growth distribution
const sig = 0.05189;

const b1 = 16.0817; # division distribution
const b2 = 16.0425;

const o2 = 1.7761; #hazard ratio
const u = 2.5; #lower treshhold for division
const v = 5.5; #upper treshhold for division
const c = 1.; #protein constant

# prior distributions
const pri_Gamma = Uniform(20,34);
const pri_Beta = Uniform(10,22);
const pri = Uniform(0,6);

# generate data using defined model
N = 200; #number of observations
m0 = 2.6; #initial mass of cell
gendata = generate_data(m0,N);

# read data from data set
readdata = read_data("data/modified_Susman18_physical_units.csv")

plot_data(gendata)

# applying the MH algo for the posterior Distribution
numdims = 2; numwalkers = 20; thinning = 10; numsamples_perwalker = 20000; burnin = 1000;
logpost = x -> log_likeli(gendata,[x[1],x[2],b1,b2,o2,u,v,c]) + log_prior([x[1],x[2],b1,b2,o2,u,v,c]);

x = rand(pri,numdims,numwalkers); # define initial points with all same prior
chain, llhoodvals = AffineInvariantMCMC.sample(logpost,numwalkers,x,burnin,1);
chain, llhoodvals = AffineInvariantMCMC.sample(logpost,numwalkers,chain[:, :, end],numsamples_perwalker,thinning);
flatchain, flatllhoodvals = AffineInvariantMCMC.flattenmcmcarray(chain,llhoodvals);

# permute dimensions to simplify plotting
chain = permutedims(chain, [1,3,2]);
flatchain = permutedims(flatchain,[2,1]);

corrplot(flatchain)