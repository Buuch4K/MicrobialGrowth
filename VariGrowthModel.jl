using Roots, Distributions, Statistics, StatsPlots, Plots, AffineInvariantMCMC, CSV

struct Data
    time
    growth
    mass
end

function generate_data(si::Float64,num::Int64)
    X = Float64[si];
    Y = Array{Float64}(undef,num);
    k = rand(Uniform(0,1),num);
    alpha = rand(LogNormal(o1,sig),num);
    f = rand(Beta(2,3),num);
    for n = 1:num
        if X[n] < u
            t0 = 1/alpha[n]*log(u/X[n]);
            h = t -> log(k[n]) + o2/(v+u)*(u/alpha[n]*exp(alpha[n]*t) + v*t - u/alpha[n])
        else
            t0 = 0;
            h = t -> log(k[n]) + o2/(v+u)*(X[n]/alpha[n]*exp(alpha[n]*t) + v*t - X[n]/alpha[n])
        end
        hx = ZeroProblem(h, 1)
        Y[n] = solve(hx)+t0;
        next_size = X[n] * exp(alpha[n]*Y[n]) * f[n]
        push!(X, next_size)
    end
    return Data(Y,alpha,X[1:num]), f[1:num-1]
end


function read_data(filename::String)
    data = CSV.File(filename,select=["lineage_ID","generationtime","length_birth","growth_rate","division_ratio"]);
    return Data(data.generationtime,data.growth_rate,data.length_birth), data.division_ratio[2:end]
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


function log_likeli(D::Data,para::Vector)
    # para = [o1,sig,o2,v,u]
    if any(x->x.<0,para)
        return -Inf
    else
        like = 0.;
        for k = 1:length(D.time)
            if D.mass[k] < para[5]
                t0 = 1/D.growth[k]*log(para[5]/D.mass[k]);
                if D.time[k] < t0
                    return -Inf
                else
                    temp = log((para[3]*para[4])/(para[4]+para[5]) + (D.mass[k]*para[3])/(para[4]+para[5])*exp(D.growth[k]*D.time[k])) + ((para[3]/(para[4]+para[5]))*(para[5]/D.growth[k] - (D.mass[k]*exp(D.growth[k]*D.time[k]))/D.growth[k] - para[4]*D.time[k] + para[4]*t0))
                end
            else
                temp = log((para[3]*para[4])/(para[4]+para[5]) + (D.mass[k]*para[3])/(para[4]+para[5])*exp(D.growth[k]*D.time[k])) + ((para[3]/(para[4]+para[5]))*(D.mass[k]/D.growth[k] - (D.mass[k]*exp(D.growth[k]*D.time[k]))/D.growth[k] - para[4]*D.time[k]))
            end
            like += temp
        end
        return like + sum([logpdf(LogNormal(para[1],para[2]),D.growth[k]) for k=1:length(D.growth)])
    end
end


function log_prior(para::Vector)
    if para[4] <= para[5]
        return -Inf
    else
        return sum([logpdf(pri,para[k]) for k=1:length(para)])
    end
end


# initial parameters
const o1 = 1.; #mean of growth rate distribution
const sig = 0.5; #sd of growth rate distribution
const o2 = 0.5; #hazard rate functions constant
const u = 2.5; #lower treshhold for division
const v = 5.5; #upper treshhold for division

const pri = Uniform(0,6); #prior distribution for o1,sig,o2 and v
# const pri_u = Uniform(0,v); # prior distribution for u depending on v


# generate data using defined model
N = 200; #number of observations
m0 = 2.; #initial mass of cell
gendata, gen_div_ratio = generate_data(m0,N);

# read data from dataset
readdata, div_ratio = read_data("data/modified_Susman18_physical_units.csv");

plot_data(gendata)

scatter(div_ratio .* exp.(readdata.growth[2:end].*readdata.time[2:end]))

# applying the MH algo for the posterior Distribution
numdims = 4; numwalkers = 20; thinning = 10; numsamples_perwalker = 20000; burnin = 1000;
logpost = x -> log_likeli(readdata,[x[1],x[2],x[3],v,x[4]]) + log_prior([x[1],x[2],x[3],v,x[4]]);

x = rand(pri,numdims,numwalkers); # define initial points with all same prior
# x = vcat(rand(pri,numdims-1,numwalkers),rand(pri_u,1,numwalkers)); #define initial point with regard to prior of u
chain, llhoodvals = AffineInvariantMCMC.sample(logpost,numwalkers,x,burnin,1);
chain, llhoodvals = AffineInvariantMCMC.sample(logpost,numwalkers,chain[:, :, end],numsamples_perwalker,thinning);
flatchain, flatllhoodvals = AffineInvariantMCMC.flattenmcmcarray(chain,llhoodvals);

# permute dimensions to simplify plotting
chain = permutedims(chain, [1,3,2]);
flatchain = permutedims(flatchain,[2,1]);


corrplot(flatchain[:,1:3])
corrplot(flatchain[:,4:5])
corrplot(flatchain)
[x[1],x[2],x[3],v,x[4]]


means = mean(flatchain,dims=1);
gro_distr = LogNormal(means[1],means[2]);
poi = range(0,1,100);
histogram(readdata.growth)
plot(poi, pdf.(gro_distr, poi))