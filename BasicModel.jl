using Roots, Distributions, Random, Statistics, StatsPlots
include("./SimpleMCMC.jl")
# Random.seed!(4)


function generate_data(size_init,N)
    #=
        This function generates a dataset containing the size of the zell and the division time for a single cell
        
    =#
    function find_division(k,size_of_cell,u, v, omega1,omega2)
        function f(t)
            #= 
                In this function we have a relation of the division time and the survival probability.
                we draw a random k between 0 and 1 and then solve for t
            =#
            return log(k) + omega2*v*t/(v+u) + (omega2*size_of_cell)/(omega1*(v+u))*exp(omega1*t) - (omega2*size_of_cell)/(omega1*(v+u))
        end
        fx = ZeroProblem(f, 1)
        return solve(fx)
    end

    u_init = 0.6; #lower treshhold for division
    v_init = 1; #upper treshhold for division
    omega1_init = 1; #exponential growth rate
    omega2_init = 0.302; #hazard rate functions constant

    X = Float64[size_init]; #sizes of the cell after division
    Y_obs = Float64[]; #division times
    for n = 1:N
        k = rand(Uniform(0,1),1)[1]
        push!(Y_obs, find_division(k,X[n],u_init,v_init,omega1_init,omega2_init))
        next_size = X[n]/2 * exp(omega1_init*Y_obs[n])
        push!(X, next_size)
    end
    return (X, Y_obs)
end


function loglikelyhood(time)
    #=
        This function computes the liklyhood for an observation Y given the initial parameters (theta, x0) 
    =#
    like = 1;
    for t in time
        temp = (omega2*v/(v+u) + omega2*x0/(v+u)*exp(omega1*t)) * exp(- omega2*v*t/(v+u) - (omega2*x0)/(omega1*(v+u))*exp(omega1*t))
        like += temp
    end
    return like
end

# initial parameters for the data generation
N = 10; #number of observations
x0 = 0.725; #initial size

#generating the first dataset
size, div_time = generate_data(x0,N)

#define prior Distributions
d = LogNormal(log(2)-1^2/2,1);
u = d
v = d
omega1 = d
omega2 = d
para = Distribution[u,v,omega_1,omega_2]



#applying the MH algo for the posterior Distribution
p = MetropolisHastings_MCMC(div_time, para, loglikelihood,
                            samples = 1000,
                            burnedinsamples = 500)

