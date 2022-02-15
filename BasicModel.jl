using Roots, RRRMC, Distributions, Random
# Random.seed!(4)

function find_division(k,size_of_cell)
    function f(t)
        #= 
            In this function we have a relation of the division time and the survival probability.
            we draw a random k between 0 and 1 and then solve for t
        =#
        return log(k) + omega_2*v*t/(v+u) + (omega_2*size_of_cell)/(omega_1*(v+u))*exp(omega_1*t) 
    end
    fx = ZeroProblem(f, 1)
    return solve(fx)
end


function compute_likelyhood(time)
    #=
        This function computes the liklyhood for an observation Y given the initial parameters (theta, x0) 
    =#
    like = 1;
    for t in time
        temp = (omega_2*v/(v+u) + omega_2*x0/(v+u)*exp(omega_1*t)) * exp(- omega_2*v*t/(v+u) - (omega_2*x0)/(omega_1*(v+u))*exp(omega_1*t))
        like = like * temp
    end
    return like
end


N = 10; #number of observations
x0 = 5; #initial size

d = LogNormal(log(2)-1^2/2,1);
u = 6.689; #lower treshhold for division
v = 8.977 #upper treshhold for division
omega_1 = 1; #exponential growth rate
omega_2 = 0.302; #hazard rate functions constant

X = Float64[x0]; #sizes of the cells
Y_obs = Float64[]; #division times

for n = 1:N
    k = rand(Uniform(0,1),1)[1]
    push!(Y_obs, find_division(k,X[n]))
    next_size = X[n]/2 * exp(omega_1*Y_obs[n])
    push!(X, next_size)
end

println(X)
println(Y_obs)

likelyhood = compute_likelyhood(Y_obs)







