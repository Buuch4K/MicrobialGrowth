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
        Y[i] = solve(fx) + t0;
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


function log_likeli(D::Data,p::Vector)
    # para = [o1,o2,u,v]
    if any(x -> x.<0,p)
        return -Inf
    else
        like = 0.;
        for k in 1:length(D.time)
            t0 = max(0,1/p[1]*log(p[3]/D.mass[k]))
            if D.time[k] < t0
                return -Inf
            else
                temp = log(p[2]/(p[3]+p[4])*(D.mass[k]*exp(p[1]*D.time[k]) + p[4])) + (p[2]/(p[3]+p[4])*(D.mass[k]/p[1]*(exp(p[1]*t0)-exp(p[1]*D.time[k])) + p[4]*(t0-D.time[k])))
            end
            like += temp
        end
        return like
    end
end


function log_prior(p::Vector)
    if p[3] > p[4]
        return -Inf
    else
        return sum([logpdf(pri,p[k]) for k = 1:length(p)])
    end
end


function remove_stuck_chain(chain,llhood,nwalk)
    bad_idx = []; 
    for k=1:nwalk
        if all(y -> y == first(chain[3,k,:]), chain[3,k,:])
            push!(bad_idx, k);
        end
    end
    idx = setdiff(1:20,bad_idx)
    return chain[:,idx,:],llhood[:,idx,:]
end



# initial values for generating data
const o1 = 1.; #exponential growth rate
const o2 = 0.5; #hazard rate functions constant
const u = 0.2; #lower treshhold for division
const v = 4.; #upper treshhold for division

# define prior distribution
pri = Uniform(0,6);

# initial parameters for the data generation
N = 200; #number of observations
m0 = 2.4; #initial size
gendata = generate_data(m0,N);

# read data from csv file
readdata = read_data("data/modified_Susman18_physical_units.csv");

plot_data(readdata)


# applying the MH algo for the posterior Distribution
numdims = 4; numwalkers = 20; thinning = 10; numsamples_perwalker = 20000; burnin = 1000;
logpost = x -> log_likeli(readdata,x) + log_prior(x);

x = rand(pri,numdims,numwalkers); # define initial points
chain, llhoodvals = AffineInvariantMCMC.sample(logpost,numwalkers,x,burnin,1);
chain, llhoodvals = AffineInvariantMCMC.sample(logpost,numwalkers,chain[:, :, end],numsamples_perwalker,thinning);
flatchain, flatllhoodvals = AffineInvariantMCMC.flattenmcmcarray(chain,llhoodvals);

# remove stuck chains
mod_chain,mod_llhoodvals = remove_stuck_chain(chain,llhoodvals,numwalkers);
mod_flatchain, mod_flatllhoodvals = AffineInvariantMCMC.flattenmcmcarray(mod_chain,mod_llhoodvals);

# permute dimensions to simplify plotting
chain = permutedims(chain, [1,3,2]);
flatchain = permutedims(flatchain,[2,1]);

