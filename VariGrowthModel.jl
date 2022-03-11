using Roots, Distributions, Statistics, StatsPlots, Plots, AffineInvariantMCMC


function generate_data(s,N)

    X = Float64[s];
    Y = Array{Float64}(undef,N,2);
    k = rand(Uniform(0,1),N);
    alpha = rand(LogNormal(o1,sig),N);
    Y[:,2] = alpha
    for n = 1:N
        if X[n] < u
            t0 = 1/alpha[n]*log(u/X[n]);
            f = t -> log(k[n]) + o2/(v+u)*(u/alpha[n]*exp(alpha[n]*t) + v*t - u/alpha[n])
        else
            t0 = 0;
            f = t -> log(k[n]) + o2/(v+u)*(X[n]/alpha[n]*exp(alpha[n]*t) + v*t - X[n]/alpha[n])
        end
        fx = ZeroProblem(f, 1)
        Y[n,1] = solve(fx)+t0;
        next_size = (X[n] * exp(alpha[n]*Y[n,1]))/2
        push!(X, next_size)
    end
    return (Y,X)
end


function plot_data(Y,X)
    t = Array{Float64}(undef,N*10);
    result = Array{Float64}(undef,N*10);
    for k = 1:N
        start = sum(Y[1:(k-1),1]);
        t[(k-1)*10+1:k*10] = range(start,start+Y[k,1],10)
        result[(k-1)*10+1:k*10] = X[k] .* exp.(Y[k,2]*range(0,Y[k,1],10))
    end
    plot(t,result)
end


function log_posterior(Y,X,para::Vector)
    if any(x->x.<0,para)
        return -Inf
    else
        like = 0.;
        for k = 1:size(Y,1)
            if X[k] < para[3]
                t0 = 1/Y[k,2]*log(para[3]/X[k]);
                if time[k] < t0
                    return -Inf
                else
                    temp = log((para[4]*para[2])/(para[4]+para[3]) + (s[k]*para[2])/(para[4]+para[3])*exp(Y[k,2]*Y[k,1])) + ((para[2]/(para[4]+para[3]))*(para[3]/Y[k,2] - (s[k]*exp(Y[k,2]*Y[k,1]))/Y[k,2] - para[4]*Y[k,1] + para[4]*t0))
                end
            else
                temp = log((para[4]*para[2])/(para[4]+para[3]) + (s[k]*para[2])/(para[4]+para[3])*exp(Y[k,2]*Y[k,1])) + ((para[2]/(para[4]+para[3]))*(s[k]/Y[k,2] - (s[k]*exp(Y[k,2]*Y[k,1]))/Y[k,2] - para[4]*Y[k,1]))
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
const sig = 0.1; #sd of growth rate distribution
const o2 = 0.5; #hazard rate functions constant

const pri = Uniform(2,1); #prior distribution with mean 2 and sd 1

data, mass = generate_data(m0,N);
plot_data(data,mass)


