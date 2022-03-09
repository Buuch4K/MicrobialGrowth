using Roots, Distributions, Statistics, StatsPlots, Plots, AffineInvariantMCMC


function generate_data(size_of_cell,N)
    #=
        This function generates a data set of length N for initial size "size_of_cell"
        output:
            X: vector of all sizes at division time (x0,x1,...,xN)
            Y: vector of all observed division times (tau1,...,tauN)
            init_para: initial values of all parameters
    =#
    X = Float64[size_of_cell]; #sizes of the cell at division
    Y = Float64[]; #division times
    k = rand(Uniform(0,1),N);
    for n = 1:N
        if X[n] < u
            t0 = 1/o1*log(u/X[n])
            f = t -> log(k[n]) + o2/(v+u)*(u/o1*exp(o1*t) + v*t - u/o1)
        else
            t0 = 0.
            f = t -> log(k[n]) + o2/(v+u)*(X[n]/o1*exp(o1*t) + v*t - X[n]/o1)
        end
        fx = ZeroProblem(f, 1)
        push!(Y, solve(fx) + t0)
        next_size = (X[n] * exp(o1*Y[n]))/2
        push!(X, next_size)
    end
    return (Y,X)
end


function plot_survival(t,s)
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


function plot_data(Y,X)
    t = Array{Float64}(undef,N*10);
    result = Array{Float64}(undef,N*10);
    for k = 1:N
        start = sum(Y[1:(k-1)])
        t[(k-1)*10+1:k*10] = range(start,start+Y[k],10)
        result[(k-1)*10+1:k*10] = X[k] .* exp.(o1*range(0,Y[k],10)) 
    end
    plot(t,result)
end


function log_posterior(time,s,para::Vector)
    #=
        This function computes the likelihood for an observation Y given the initial parameters (theta, xi) 
    =#
    if any(x-> x.<0,para)
        return -Inf
    else
        like = 0.;
        for k in 1:length(time)
            if s[k] < para[3]
                t0 = 1/para[1]*log(para[3]/s[k]);
                temp = log((para[4]*para[2])/(para[4]+para[3]) + (s[k]*para[2])/(para[4]+para[3])*exp(para[1]*time[k])) + ((para[2]/(para[4]+para[3]))*(para[3]/para[1] - (s[k]*exp(para[1]*time[k]))/para[1] - para[4]*time[k] + para[4]*t0))
            else
                temp = log((para[4]*para[2])/(para[4]+para[3]) + (s[k]*para[2])/(para[4]+para[3])*exp(para[1]*time[k])) + ((para[2]/(para[4]+para[3]))*(s[k]/para[1] - (s[k]*exp(para[1]*time[k]))/para[1] - para[4]*time[k]))
            end
            like += temp
        end
        return like + sum([logpdf(pri,para[k]) for k = 1:length(para)])
    end
end


function log_prior(para::Vector)
    return sum([logpdf(pri,para[k]) for k = 1:length(para)])
end


# initial parameters for the data generation
const N = 10; #number of observations
const m0 = 8.; #initial size

const u = 0.6; #lower treshhold for division
const v= 2.; #upper treshhold for division
const o1 = 1.; #exponential growth rate
const o2 = 0.5; #hazard rate functions constant

const pri = Uniform(0,10); #define prior distribution with mean 2 and sd 1

# generating the first dataset
div_time, mass = generate_data(m0,N);
plot_data(div_time,mass)

plot_survival(range(0,1,10), mass[8])

log_likeli(div_time,mass,[o1,o2,u,v])


# applying the MH algo for the posterior Distribution
numdims = 2; numwalkers = 100; thinning = 10; numsamples_perwalker = 1000; burnin = 1000;
loglhood = x -> log_posterior(div_time,mass,[x[1],x[2],0.6,2.]);

x = rand(pri,numdims,numwalkers); # define initial points for parameters
chain, llhoodvals = AffineInvariantMCMC.sample(loglhood,numwalkers,x,burnin,1);
chain, llhoodvals = AffineInvariantMCMC.sample(loglhood,numwalkers,chain[:, :, end],numsamples_perwalker,thinning);
flatchain, flatllhoodvals = AffineInvariantMCMC.flattenmcmcarray(chain,llhoodvals);


plot(flatchain[1,:])
plot(flatllhoodvals)
histogram(flatchain[1,:], xlims = [0,3])
