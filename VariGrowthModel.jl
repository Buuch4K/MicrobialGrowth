using Roots, Distributions, Statistics, StatsPlots, Plots, AffineInvariantMCMC, CSV

struct Data
    time
    growth
    mass
end

function generate_data(si::Float64,num::Int64)
    X = Float64[si];
    Y = Array{Float64}(undef,num);
    k = rand(Uniform(0,1),num);
    alpha = rand(LogNormal(o1,sig),num);
    for n = 1:num
        if X[n] < u
            t0 = 1/alpha[n]*log(u/X[n]);
            f = t -> log(k[n]) + o2/(v+u)*(u/alpha[n]*exp(alpha[n]*t) + v*t - u/alpha[n])
        else
            t0 = 0;
            f = t -> log(k[n]) + o2/(v+u)*(X[n]/alpha[n]*exp(alpha[n]*t) + v*t - X[n]/alpha[n])
        end
        fx = ZeroProblem(f, 1)
        Y[n] = solve(fx)+t0;
        next_size = (X[n] * exp(alpha[n]*Y[n]))/2
        push!(X, next_size)
    end
    return Data(Y,alpha,X[1:num])
end


function read_data(lineage::Float64,filename::String)
    data = CSV.File(filename,select=["lineage_ID", "generationtime","length_birth","growth_rate"]);
    X = Float64[]; Y = Float64[]; alpha = Float64[];
    for k = 1:length(data.lineage_ID)
        if data.lineage_ID[k] == lineage
            push!(alpha, data.growth_rate[k])
            push!(Y, data.generationtime[k])
            push!(X, data.length_birth[k])
        end
    end
    return Data(Y,alpha,X)
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


function log_posterior(D::Data,para::Vector)
    if any(x->x.<0,para)
        return -Inf
    else
        like = 0.;
        for k = 1:length(D.time)
            if D.mass[k] < para[3]
                t0 = 1/D.growth[k]*log(para[3]/D.mass[k]);
                if D.time[k] < t0
                    return -Inf
                else
                    temp = log((para[4]*para[2])/(para[4]+para[3]) + (D.mass[k]*para[2])/(para[4]+para[3])*exp(D.growth[k]*D.time[k])) + ((para[2]/(para[4]+para[3]))*(para[3]/D.growth[k] - (D.mass[k]*exp(D.growth[k]*D.time[k]))/D.growth[k] - para[4]*D.time[k] + para[4]*t0))
                end
            else
                temp = log((para[4]*para[2])/(para[4]+para[3]) + (D.mass[k]*para[2])/(para[4]+para[3])*exp(D.growth[k]*D.time[k])) + ((para[2]/(para[4]+para[3]))*(D.mass[k]/D.growth[k] - (D.mass[k]*exp(D.growth[k]*D.time[k]))/D.growth[k] - para[4]*D.time[k]))
            end
            like += temp
        end
        return like + sum([logpdf(LogNormal(para[1],para[5]),D.growth[k]) for k=1:length(D.growth)]) + sum([logpdf(pri,para[k]) for k=1:length(para)])
    end
end


function log_prior(para::Vector)
    return sum([logpdf(pri,para[k]) for k=1:length(para)])
end


# initial parameters
const u = 0.6; #lower treshhold for division
const v = 2.; #upper treshhold for division
const o1 = 1.; #mean of growth rate distribution
const sig = 0.5; #sd of growth rate distribution
const o2 = 0.5; #hazard rate functions constant

const pri = Uniform(0,10); #prior distribution

generate = true;
if generate
    # Initial variables
    N = 10; #number of observations
    m0 = 8.; #initial mass of cell

    gendata = generate_data(m0,N);
else
    readdata = read_data(15.,"data/Susman18_physical_units.csv");
end
plot_data(gendata)

# applying the MH algo for the posterior Distribution
numdims = 4; numwalkers = 20; thinning = 10; numsamples_perwalker = 20000; burnin = 1000;
loglhood = x -> log_posterior(gendata,[x[1],x[2],x[4],v,x[3]]);

x = rand(pri,numdims,numwalkers); # define initial points for parameters
chain, llhoodvals = AffineInvariantMCMC.sample(loglhood,numwalkers,x,burnin,1);
chain, llhoodvals = AffineInvariantMCMC.sample(loglhood,numwalkers,chain[:, :, end],numsamples_perwalker,thinning);
flatchain, flatllhoodvals = AffineInvariantMCMC.flattenmcmcarray(chain,llhoodvals);



