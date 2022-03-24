using Roots, Distributions, Statistics, StatsPlots, Plots, AffineInvariantMCMC, CSV


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
    for i = 1:N
        if X[i] < u
            t0 = 1/o1*log(u/X[i])
            f = t -> log(k[i]) + o2/(v+u)*(u/o1*exp(o1*t) + v*t - u/o1)
        else
            t0 = 0.
            f = t -> log(k[i]) + o2/(v+u)*(X[i]/o1*exp(o1*t) + v*t - X[i]/o1)
        end
        fx = ZeroProblem(f, 1)
        push!(Y, solve(fx) + t0)
        next_size = (X[i] * exp(o1*Y[i]))/2
        push!(X, next_size)
    end
    return (Y,X[1:N])
end


function read_data(lineage::Float64,filename::String)
    data = CSV.File(filename,select=["lineage_ID","generationtime","length_birth","growth_rate"]);
    s = Float64[]
    t = Float64[]
    index = Int64[]
    for k = 1:length(data.lineage_ID)
        if data.lineage_ID[k] == lineage
            push!(index,k)
            push!(t, data.generationtime[k])
            push!(s,data.length_birth[k])
        end
    end
    return t,s,mean(data.growth_rate[index[1]:index[end]])
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


function plot_data(Y,X,growth = 1)
    t = Array{Float64}(undef,length(Y)*10);
    result = Array{Float64}(undef,length(Y)*10);
    for k = 1:length(Y)
        start = sum(Y[1:(k-1)])
        t[(k-1)*10+1:k*10] = range(start,start+Y[k],10)
        result[(k-1)*10+1:k*10] = X[k] .* exp.(growth*range(0,Y[k],10)) 
    end
    plot(t,result)
end


function log_likeli(time,s,para::Vector)
    # para = [o1,o2,v,u]
    if any(x-> x.<0,para)
        return -Inf
    else
        like = 0.;
        for k in 1:length(time)
            if s[k] < para[4]
                t0 = 1/para[1]*log(para[4]/s[k]);
                if time[k] < t0
                    return -Inf # division cannot happen until s >= u
                else
                    temp = log((para[3]*para[2])/(para[4]+para[3]) + (s[k]*para[2])/(para[4]+para[3])*exp(para[1]*time[k])) + ((para[2]/(para[4]+para[3]))*(para[4]/para[1] - (s[k]*exp(para[1]*time[k]))/para[1] - para[3]*time[k] + para[3]*t0))
                end
            else
                temp = log((para[3]*para[2])/(para[4]+para[3]) + (s[k]*para[2])/(para[4]+para[3])*exp(para[1]*time[k])) + ((para[2]/(para[4]+para[3]))*(s[k]/para[1] - (s[k]*exp(para[1]*time[k]))/para[1] - para[3]*time[k]))
            end
            like += temp
        end
        return like
    end
end


function log_prior(para::Vector)
    return sum([logpdf(pri,para[k]) for k = 1:length(para)-1]) + logpdf(pri_u,para[end])
end


function check_sample(chain,size_of_cell,n)
    # generates a data set with parameters sampled from a chain
    omega1, omega2, lb = mean(chain, dims=2);
    X = Float64[size_of_cell]; #sizes of the cell at division
    Y = Float64[]; #division times
    k = rand(Uniform(0,1),n);
    for i = 1:n
        if X[i] < lb
            t0 = 1/omega1*log(lb/X[i])
            f = t -> log(k[i]) + omega2/(v+lb)*(lb/omega1*exp(omega1*t) + v*t - lb/omega1)
        else
            t0 = 0.
            f = t -> log(k[i]) + omega2/(v+lb)*(X[i]/omega1*exp(omega1*t) + v*t - X[i]/omega1)
        end
        fx = ZeroProblem(f, 1)
        push!(Y, solve(fx) + t0)
        next_size = (X[i] * exp(omega1*Y[i]))/2
        push!(X, next_size)
    end
    return Y,X
end


const u = 0.6; #lower treshhold for division
const v = 1.; #upper treshhold for division
const o1 = 1.; #exponential growth rate
const o2 = 0.5; #hazard rate functions constant
const pri = Uniform(0,2); #define prior distribution for o1, o2, and v
const pri_u = Uniform(0,v); #define prior distribution for u depending on v

generate = true;
if generate
    # initial parameters for the data generation
    N = 250; #number of observations
    m0 = 8.; #initial size

    div_time, mass = generate_data(m0,N);
else
    div_time,mass,rate = read_data(15.,"data/Susman18_physical_units.csv"); # read data fram csv file
end

plot_data(div_time,mass,rate)


# applying the MH algo for the posterior Distribution
numdims = 4; numwalkers = 100; thinning = 10; numsamples_perwalker = 2000; burnin = 1000;
logpost = x -> log_likeli(div_time,mass,x)+log_prior(x);

# x = rand(pri,numdims,numwalkers); # define initial points with all same prior
x = vcat(rand(pri,numdims-1,numwalkers),rand(pri_u,1,numwalkers)); #define initial point with regard to prior of u
chain, llhoodvals = AffineInvariantMCMC.sample(logpost,numwalkers,x,burnin,1);
chain, llhoodvals = AffineInvariantMCMC.sample(logpost,numwalkers,chain[:, :, end],numsamples_perwalker,thinning);
flatchain, flatllhoodvals = AffineInvariantMCMC.flattenmcmcarray(chain,llhoodvals);

corrplot(transpose(flatchain))
plot(flatllhoodvals)

sampled_time, sampled_mass = check_sample(flatchain,mass[1],length(div_time))
plot_data(sampled_time,sampled_mass)