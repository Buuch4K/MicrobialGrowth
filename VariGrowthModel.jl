using Roots, Distributions, Statistics, StatsPlots,Plots
include("./SimpleMCMC.jl")
using .SimpleMCMC


function generate_data(s,N)

    u_init = 0.6;
    v_init = 2.0;
    o1_init = 1.0;
    o2_init = 0.5;
    sigma_init = 0.1;
    init_para = [o1_init,o2_init,u_init,v_init,sigma_init];

    X = Float64[s];
    Y = Array{Float64}(undef,N,2);
    k = rand(Uniform(0,1),N);
    alpha = rand(LogNormal(log(o1_init)-sigma_init^2/2,sigma_init),N);
    Y[:,2] = alpha
    for n = 1:N
        if X[n] < u_init
            t0 = 1/alpha[n]*log(u_init/X[n]);
            f = t -> log(k[n]) + o2_init/(v_init+u_init)*((u_init)/alpha[n]*exp(alpha[n]*t) + v_init*t - u_init/alpha[n])
        else
            t0 = 0
            f = t -> log(k[n]) + o2_init/(v_init+u_init)*(X[n]/alpha[n]*exp(alpha[n]*t) + v_init*t - X[n]/alpha[n])
        end
        fx = ZeroProblem(f, 1)
        Y[n,1] = solve(fx)+t0;
        next_size = (X[n] * exp(alpha[n]*Y[n,1]))/2
        push!(X, next_size)
    end
    return (X,Y,init_para)
end


function plot_data(Y,X)
    t = Array{Float64}(undef,N*10);
    result = Array{Float64}(undef,N*10);
    for k = 1:N
        temp = Array{Float64}(undef,10);
        for j = 1:10
            temp[j] = X[k]*exp(Y[k,2]*(j*Y[k,1])/10) 
        end
        start = sum(Y[1:(k-1),1]);
        t[(k-1)*10+1:k*10] = range(start,start+Y[k],10)
        result[(k-1)*10+1:k*10] = temp
    end
    plot(t,result)
end






N = 10;
size_init = 0.725;

mass,data,para_init = generate_data(size_init,N)
plot_data(data,mass)