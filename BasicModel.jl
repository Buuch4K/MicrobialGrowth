using Roots, Distributions, Random, Statistics, StatsPlots
include("./SimpleMCMC.jl")
using .SimpleMCMC
# Random.seed!(4)


function generate_data(size_of_cell,N)
    #=
        This function generates a data set of length N for initial size "size_of_cell"
        output:
            X: vector of all sizes at division time (x0,x1,...,xN)
            Y_obs: vector of all observed division times (tau1,...,tauN)
            init_para: initial values of all parameters
    =#
    function find_division(k,s,o1,o2,lb,ub)
        #=
            find_division computes the division time for a specific cell cycle
            input:
                k: random number in [0,1]
                s: size of the cell after previous division
                o1: initial value for omega1
                02: initial value for omega2
                lb: initial value for u
                ub: initial value for v
            output:
                division time for this cell cycle
        =#
        function f(t)
            return log(k) + o2*ub*t/(ub+lb) + (o2*s)/(o1*(ub+lb))*exp(o1*t) - (o2*s)/(o1*(ub+lb))
        end
        fx = ZeroProblem(f, 1)
        return solve(fx)
    end

    u_init = 0.6; #lower treshhold for division
    v_init = 1; #upper treshhold for division
    omega1_init = 1; #exponential growth rate
    omega2_init = 0.5; #hazard rate functions constant    
    init_para = [omega1_init,omega2_init,u_init,v_init];
    
    X = Float64[size_of_cell]; #sizes of the cell after division
    Y_obs = Float64[]; #division times
    for n = 1:N
        k = rand(Uniform(0,1),1)[1]
        push!(Y_obs, find_division(k,X[n],omega1_init,omega2_init,u_init,v_init))
        next_size = X[n]/2 * exp(omega1_init*Y_obs[n])
        push!(X, next_size)
    end
    return (X, Y_obs, init_para)
end


function log_likeli(time,size,o1,o2,lb,ub)
    #=
        This function computes the likelihood for an observation Y given the initial parameters (theta, xi) 
    =#
    like = 0.0;
    for k in 1:length(time)
        temp = (o2*ub/(ub+lb) + o2*size[k]/(ub+lb)*exp(o1*time[k])) * exp(- o2*ub*time[k]/(ub+lb) - o2*size[k]/(o1*(ub+lb))*exp(o1*time[k]) + o2*size[k]/(o1*(ub+lb)))
        like += log(temp)
    end
    return like
end

# initial parameters for the data generation
N = 100; #number of observations
x0 = 0.725; #initial size

#generating the first dataset
size, div_time, para_init = generate_data(x0,N)

#define prior Distributions
# u = LogNormal(log(2)-1^2/2,1);
u = LogNormal(log(2)-1^2/2,1);
v = LogNormal(log(2)-1^2/2,1);
omega1 = LogNormal(log(2)-1^2/2,1);
omega2 = LogNormal(log(2)-1^2/2,1);
para = Distribution[omega1,omega2,u,v]


#applying the MH algo for the posterior Distribution
p = SimpleMCMC.MetropolisHastings(div_time, size, para, log_likeli, samples = 50000, burnedinsamples = 1000)
SimpleMCMC.describe_paramvec("omega1","omega1",vec(p[:,1]))
SimpleMCMC.describe_paramvec("omega2","omega2",vec(p[:,2]))
SimpleMCMC.describe_paramvec("u","u",vec(p[:,3]))
SimpleMCMC.describe_paramvec("v","v",vec(p[:,4]))

corrplot(p[:,3:4])

