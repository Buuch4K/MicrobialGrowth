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
    alpha = rand(LogNormal(o1,sig),num);
    f = rand(Beta(2,3),num);
    for n = 1:num
        t0 = max(0,1/alpha[n]*log((u+c*X[n])/(c*X[n])));
        h = t -> log(k[n]) + o2/(u+v)*((c*X[n])/alpha[n]*exp(alpha[n]*t)*exp(alpha[n]*t0) - c*X[n]*t + v*t - (c*X[n])/alpha[n]*exp(alpha[n]*t0))
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
    # p = [o1,sig,o2,u,v,c]
    if any(x->x.<0,p)
        return -Inf
    else
        like = 0.;
        for k = 1:length(D.time)
            t0 = max(0,1/D.growth[k]*log((u+c*D.mass[k])/(c*D.mass[k])))
            if D.time[k] < t0
                return -Inf
            else
                fac = -p[3]/(p[4]+p[5]) * ((p[6]*D.mass[k])/D.growth[k]*exp(D.growth[k]*D.time[k]) - p[6]*D.mass[k]*D.time[k] + p[5]*D.time[k] - (p[6]*D.mass[k])/D.growth[k]*exp(D.growth[k]*t0) + p[6]*D.mass[k]*t0 - p[5]*t0)
                temp = log(p[3]/(p[4]+p[5])*(p[5] - p[6]*D.mass[k] + p[6]*D.mass[k]*exp(D.growth[k]*D.time[k])))*fac
            end
            like += temp
        end
        return like + sum([logpdf(LogNormal(p[1],p[2]),D.growth[k]) for k = 1:length(p)])
    end
end


function log_prior(p::Vector)
    # p = [o1,sig,o2,u,v,c]
    if p[3] >= p[4]
        return -Inf
    else
        return sum([logpdf(pri,p[k]) for k=1:length(p)])
    end
end


# initial parameters
const o1 = 0.8; #mean of growth rate distribution
const sig = 0.8; #sd of growth rate distribution
const o2 = 0.5; #hazard rate functions constant
const u = 2.5; #lower treshhold for division
const v = 5.5; #upper treshhold for division
const c = 1.2; #protein constant

const pri = Uniform(0,6); #prior distribution for all parameters

# generate data using defined model
N = 200; #number of observations
m0 = 2.6; #initial mass of cell
gendata = generate_data(m0,N);

# read data from data set
readdata = read_data("data/modified_Susman18_physical_units.csv")

plot_data(gendata)

# applying the MH algo for the posterior Distribution
numdims = 3; numwalkers = 20; thinning = 10; numsamples_perwalker = 20000; burnin = 1000;
logpost = x -> log_likeli(gendata,[x[1],x[2],x[3],u,v,c]) + log_prior([x[1],x[2],x[3],u,v,c]);

x = rand(pri,numdims,numwalkers); # define initial points with all same prior
chain, llhoodvals = AffineInvariantMCMC.sample(logpost,numwalkers,x,burnin,1);
chain, llhoodvals = AffineInvariantMCMC.sample(logpost,numwalkers,chain[:, :, end],numsamples_perwalker,thinning);
flatchain, flatllhoodvals = AffineInvariantMCMC.flattenmcmcarray(chain,llhoodvals);

# permute dimensions to simplify plotting
chain = permutedims(chain, [1,3,2]);
flatchain = permutedims(flatchain,[2,1]);

