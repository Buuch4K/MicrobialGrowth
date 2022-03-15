using Roots, Distributions, Statistics, StatsPlots, Plots, AffineInvariantMCMC, CSV

struct Data
    time
    growth
    mass
end

function generate_data(s::Float64,num::Int64)
    X = Float64[s];
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
    s = Float64[]; t = Float64[]; g = Float64[];
    for k = 1:length(data.lineage_ID)
        if data.lineage_ID[k] == lineage
            push!(g, data.growth_rate[k])
            push!(t, data.generationtime[k])
            push!(s,data.length_birth[k])
        end
    end
    return Data(t,g,s)
end


function plot_data(S::Data)
    n = length(S.time)
    t = Array{Float64}(undef,n*10);
    result = Array{Float64}(undef,n*10);
    for k = 1:n
        start = sum(S.time[1:(k-1)]);
        t[(k-1)*10+1:k*10] = range(start,start+S.time[k],10)
        result[(k-1)*10+1:k*10] = S.mass[k] .* exp.(S.growth[k]*range(0,S.time[k],10))
    end
    plot(t,result)
end


function log_posterior(S::Data,para::Vector)
    if any(x->x.<0,para)
        return -Inf
    else
        like = 0.;
        for k = 1:length(S.time)
            if S.mass[k] < para[3]
                t0 = 1/S.growth[k]*log(para[3]/S.mass[k]);
                if S.time[k] < t0
                    return -Inf
                else
                    temp = log((para[4]*para[2])/(para[4]+para[3]) + (S.mass[k]*para[2])/(para[4]+para[3])*exp(s.growth[k]*S.time[k])) + ((para[2]/(para[4]+para[3]))*(para[3]/S.growth[k] - (S.mass[k]*exp(S.growth[k]*S.time[k]))/S.growth[k] - para[4]*S.time[k] + para[4]*t0))
                end
            else
                temp = log((para[4]*para[2])/(para[4]+para[3]) + (S.mass[k]*para[2])/(para[4]+para[3])*exp(S.growth[k]*S.time[k])) + ((para[2]/(para[4]+para[3]))*(S.mass[k]/S.growth[k] - (S.mass[k]*exp(S.growth[k]*S.time[k]))/S.growth[k] - para[4]*S.time[k]))
            end
            like += temp
        end
        return like + sum([logpdf(pri,para[k]) for k=1:length(para)])
    end
end


function log_prior(para::Vector)
    return sum([logpdf(pri,para[k]) for k=1:length(para)])
end


# Initial parameters
const N = 10; #number of observations
const m0 = 8.; #initial mass of cell

const u = 0.6; #lower treshhold for division
const v = 2.; #upper treshhold for division
const o1 = 1.; #mean of growth rate distribution
const sig = 0.5; #sd of growth rate distribution
const o2 = 0.5; #hazard rate functions constant

const pri = Uniform(0,10); #prior distribution with mean 2 and sd 1

gendata = generate_data(m0,N);
# realdata = read_data(15.,"data/Susman18_physical_units.csv")
plot_data(gendata)


