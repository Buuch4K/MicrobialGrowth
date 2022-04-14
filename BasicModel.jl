using Roots, Distributions, Statistics, StatsPlots, Plots, AffineInvariantMCMC, CSV

struct Data
    time
    mass
end


function generate_data(si,N)
    X = Float64[si]; #sizes of the cell at division
    Y = Array{Float64}(undef,N); #division times
    k = rand(Uniform(0,1),N);
    for i = 1:N
        if X[i] < u
            t0 = 1/o1*log(u/X[i])
            f = t -> log(k[i]) + o2/(v+u)*(u/o1*exp(o1*t) + v*t - u/o1)
        else
            t0 = 0.
            f = t -> log(k[i]) + o2/(v+u)*(X[i]/o1*exp(o1*t) + v*t - X[i]/o1)
        end
        fx = ZeroProblem(f, 1);
        Y[n] = solve(fx) + t0;
        next_size = (X[i] * exp(o1*Y[i]))/2;
        push!(X, next_size)
    end
    return Data(Y,X[1:N])
end


function read_data(filename::String)
    data = CSV.File(filename,select=["lineage_ID","generationtime","length_birth","growth_rate"]);
    return Data(data.generationtime,data.length_birth)
end


function plot_survival(s,t)
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


function plot_data(D::Data,growth = 1)
    t = Array{Float64}(undef,length(D.time)*10);
    result = Array{Float64}(undef,length(D.time)*10);
    for k = 1:length(D.time)
        start = sum(D.time[1:(k-1)])
        t[(k-1)*10+1:k*10] = range(start,start+D.time[k],10)
        result[(k-1)*10+1:k*10] = D.mass[k] .* exp.(growth*range(0,D.time[k],10)) 
    end
    plot(t,result)
end


function log_likeli(D::Data,para::Vector)
    # para = [o1,o2,v,u]
    if any(x-> x.<0,para)
        return -Inf
    else
        like = 0.;
        for k in 1:length(time)
            if D.mass[k] < para[4]
                t0 = 1/para[1]*log(para[4]/D.mass[k]);
                if D.time[k] < t0
                    return -Inf # division cannot happen until s >= u
                else
                    temp = log((para[3]*para[2])/(para[4]+para[3]) + (D.mass[k]*para[2])/(para[4]+para[3])*exp(para[1]*D.time[k])) + ((para[2]/(para[4]+para[3]))*(para[4]/para[1] - (D.mass[k]*exp(para[1]*D.time[k]))/para[1] - para[3]*D.time[k] + para[3]*t0))
                end
            else
                temp = log((para[3]*para[2])/(para[4]+para[3]) + (D.mass[k]*para[2])/(para[4]+para[3])*exp(para[1]*D.time[k])) + ((para[2]/(para[4]+para[3]))*(D.mass[k]/para[1] - (D.mass[k]*exp(para[1]*D.time[k]))/para[1] - para[3]*D.time[k]))
            end
            like += temp
        end
        return like
    end
end


function log_prior(para::Vector)
    if para[4] >= para[3]
        return -Inf
    else
        return sum([logpdf(pri,para[k]) for k = 1:length(para)])
    end
end


const u = 2.3; #lower treshhold for division
const v = 5.5; #upper treshhold for division
const o1 = 1.; #exponential growth rate
const o2 = 0.5; #hazard rate functions constant
const pri = Uniform(0,4); #define prior distribution

# initial parameters for the data generation
N = 200; #number of observations
m0 = 2.3; #initial size

gendata = generate_data(m0,N);


readdata = read_data("data/modified_Susman18_physical_units.csv"); # read data fram csv file

plot_data(gendata,rate)


# applying the MH algo for the posterior Distribution
numdims = 4; numwalkers = 20; thinning = 10; numsamples_perwalker = 20000; burnin = 1000;
logpost = x -> log_likeli(gendata,x)+log_prior(x);

x = rand(pri,numdims,numwalkers); # define initial points
chain, llhoodvals = AffineInvariantMCMC.sample(logpost,numwalkers,x,burnin,1);
chain, llhoodvals = AffineInvariantMCMC.sample(logpost,numwalkers,chain[:, :, end],numsamples_perwalker,thinning);
flatchain, flatllhoodvals = AffineInvariantMCMC.flattenmcmcarray(chain,llhoodvals);
