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

plot_data(gendata)