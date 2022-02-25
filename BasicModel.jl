using Roots, Distributions, Random, Statistics, StatsPlots, Plots
include("./SimpleMCMC.jl")
using .SimpleMCMC
# Random.seed!(4)


function generate_data(size_of_cell,N)
    #=
        This function generates a data set of length N for initial size "size_of_cell"
        output:
            X: vector of all sizes at division time (x0,x1,...,xN)
            Y: vector of all observed division times (tau1,...,tauN)
            init_para: initial values of all parameters
    =#
    u_init = 0.6; #lower treshhold for division
    v_init = 2.0; #upper treshhold for division
    o1_init = 1.0; #exponential growth rate
    o2_init = 0.5; #hazard rate functions constant    
    init_para = [o1_init,o2_init,u_init,v_init];
    
    X = Float64[size_of_cell]; #sizes of the cell at division
    Y = Float64[]; #division times
    k = rand(Uniform(0,1),N)
    for n = 1:N
        if X[n] < u_init
            t0 = 1/o1_init*log(u_init/X[n])
            f = t -> log(k[n]) + o2_init/(v_init+u_init)*((u_init)/o1_init*exp(o1_init*t) + v_init*t - u_init/o1_init)
        else
            t0 = 0
            f = t -> log(k[n]) + o2_init/(v_init+u_init)*(X[n]/o1_init*exp(o1_init*t) + v_init*t - X[n]/o1_init)
        end
        fx = ZeroProblem(f, 1)
        push!(Y, solve(fx) + t0)
        next_size = X[n]/2 * exp(o1_init*Y[n])
        push!(X, next_size)
    end
    return (X, Y, init_para)
end


function plot_survival(t,s,o1,o2,lb,ub)
    result = Array{Float64}(undef,length(t));
    if s < lb
        for k = 1:length(t)
            temp = exp(-o2/(ub+lb)*(lb/o1*exp(o1*t[k]) + ub*t[k] - lb/o1 - ub*1/o1*log(lb/s)))
            result[k] = min(1,temp)
        end
    else
        for k = 1:length(t)
            result[k] = exp(-o2/(ub+lb)*(s/o1*exp(o1*t[k]) + ub*t[k] - s/o1))
        end
    end
    scatter(t,result)
end


function log_likeli(time,s,o1,o2,lb,ub)
    #=
        This function computes the likelihood for an observation Y given the initial parameters (theta, xi) 
    =#
    check = zeros(N)
    like = 0.0;
    for k in 1:length(time)
        if s[k] < lb
            temp = ((ub*o2)/(ub+lb) + (s[k]*o2)/(ub+lb)*exp(o1*time[k]))* exp((o2/(ub+lb))*(lb/o1 - (s[k]*exp(o1*time[k]))/o1 - ub*time[k] + ub*1/o1*log(lb/s[k])))
        else
            temp = ((ub*o2)/(ub+lb) + (s[k]*o2)/(ub+lb)*exp(o1*time[k]))* exp((o2/(ub+lb))*(s[k]/o1 - (s[k]*exp(o1*time[k]))/o1 - ub*time[k]))
        end
        check[k] = temp;
        like += log(temp)
    end
    return like
end

# initial parameters for the data generation
N = 100; #number of observations
x0 = 0.725; #initial size

#generating the first dataset
size, div_time, para_init = generate_data(x0,N);
plot_survival(range(0,1,100), size[8], para_init[1], para_init[2], para_init[3], para_init[4])

log_likeli(div_time,size,para_init[1],para_init[2],para_init[3], para_init[4])

#define prior Distributions
u = LogNormal(log(0.6)-1^2/2,1); # mu = 0.8, sigma = 1
v = LogNormal(log(2.0)-1^2/2,1);
omega1 = LogNormal(log(1.0)-1^2/2,1);
omega2 = LogNormal(log(0.5)-1^2/2,1);
para = Distribution[omega1,omega2,u,v]

# plot(u, xlim=(0,10), ylim=(0, 2), yflip = false)

#applying the MH algo for the posterior Distribution
p = SimpleMCMC.MetropolisHastings(div_time, size, para, log_likeli, samples = 10000, burnedinsamples = 1000);
SimpleMCMC.describe_paramvec("omega1","omega1",vec(p[:,1]))
SimpleMCMC.describe_paramvec("omega2","omega2",vec(p[:,2]))
SimpleMCMC.describe_paramvec("u","u",vec(p[:,3]))
SimpleMCMC.describe_paramvec("v","v",vec(p[:,4]))

corrplot(p[:,3:4])




