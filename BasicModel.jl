using Roots, AdvancedMH, Distributions, Random
Random.seed!(7)

function find_division(k,x)
    function f(t)
        #= 
            In this function we have a relation of the division time and the survival probability.
            we draw a random k between 0 and 1 and then solve for t
        =#
    
        log(k) - omega_2*v*t/(v+u) - (omega_2*x)/(omega_1*(v+u))*exp(omega_1*t) 
    end
    
    return find_zero(f,(0,Inf))
end



N = 10; #number of observations
x0 = 0.1; #initial size

d = LogNormal(0,1)
u = rand(d, 1); #lower treshhold for division
v = rand(d,1); #upper treshhold for division
omega_1 = rand(d,1); #exponential growth rate
omega_2 = rand(d,1); #hazard rate functions constant


X = [x0];
Y_obs = [];

for n = 1:N
    k = rand(1);
    push!(Y_obs, find_division(k,X[n]))
    X[n+1] = X[n]/2*exp(omega_1*Y_obs[n])
end
println(X,Y_obs)







