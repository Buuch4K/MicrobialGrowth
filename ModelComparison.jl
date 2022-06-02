using JLD2, Plots, Distributions, Statistics, AffineInvariantMCMC,CSV

struct Data
    time
    growth
    mass
    divratio
end

function read_split_data(filename::String,test_size = 100)
    data = CSV.File(filename,select=["growth_rate","generationtime","length_birth","division_ratio"]);
    data.division_ratio = vcat(data.division_ratio[2:end],0.49);
    test_idx = sort(rand(1:length(data.generationtime),test_size));
    train_idx = setdiff(1:length(data.generationtime),test_idx);
    train = Data(data.generationtime[train_idx],data.growth_rate[train_idx],data.length_birth[train_idx],data.division_ratio[train_idx]);
    test = Data(data.generationtime[test_idx],data.growth_rate[test_idx],data.length_birth[test_idx],data.division_ratio[test_idx])
    return train,test
end


function remove_stuck_chain(chain,llhood,nwalk::Int64)
    bad_idx = []; 
    for k=1:nwalk
        if all(y -> y == first(chain[2,k,:]), chain[2,k,:])
            push!(bad_idx, k);
        end
    end
    idx = setdiff(1:nwalk,bad_idx)
    println(length(idx))
    return chain[:,idx,:],llhood[:,idx,:]
end


function bm_loglikeli(p::Vector,D::Data)
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
                temp = log(p[2]/(p[3]+p[4])*(D.mass[k]*exp(p[1]*D.time[k]) + p[4])) + (-p[2]/(p[3]+p[4])*(D.mass[k]/p[1]*(exp(p[1]*D.time[k])-exp(p[1]*t0)) + p[4]*(D.time[k]-t0)))
            end
            like += temp
        end
        return like
    end
end


function bm_logprior(p::Vector)
    if p[3] > p[4]
        return -Inf
    else
        return sum([logpdf(pri_bm,p[k]) for k = 1:length(p)])
    end
end


function vm_loglikeli(p::Vector,D::Data)
    # p = [o1,sig,b1,b2,o2,u,v]
    if any(x->x.<0,p)
        return -Inf
    elseif p[4] >= p[3]*(1-p[3])
        return -Inf
    else
        like = 0.;
        for k = 1:length(D.time)
            t0 = max(0,1/D.growth[k]*log(p[6]/D.mass[k]));
            if D.time[k] < t0
                return -Inf
            else
                temp = log(p[5]/(p[6]+p[7])*(p[7] + D.mass[k]*exp(D.growth[k]*D.time[k]))) + (-p[5]/(p[6]+p[7])*(D.mass[k]/D.growth[k]*(exp(D.growth[k])*D.time[k] - exp(D.growth[k]*t0)) + p[7]*(D.time[k] - t0)))
            end
            like += temp
        end
        return like + sum([logpdf(Gamma(p[1]^2/p[2],p[2]/p[1]),D.growth[k]) for k=1:length(D.growth)]) + sum([logpdf(Beta(p[3]/p[4]*(p[3]*(1-p[3])-p[4]),(1-p[3])/p[4]*(p[3]*(1-p[3])-p[4])),D.divratio[k]) for k=1:length(D.divratio)])
    end
end


function vm_logprior(p::Vector)
    # p = [o1,sig,b1,b2,o2,u,v]
    if p[6] >= p[7]
        return -Inf
    else
        return logpdf(pri_gamma,p[1]) + logpdf(pri_sigma,p[2]) + logpdf(pri_beta,p[3]) + logpdf(pri_sigma,p[4]) + sum([logpdf(pri_vm,p[k]) for k=5:length(p)])
    end
end


function pm_loglikeli(p::Vector,D::Data)
    # p = [o1,sig,b1,b2,o2,u,v,c]
    if any(x->x.<0,p)
        return -Inf
    elseif p[4] >= p[3]*(1-p[3])
        return -Inf
    else
        like = 0.;
        for k = 1:length(D.time)
            t0 = max(0,1/D.growth[k]*log((p[6]+p[8]*D.mass[k])/(p[8]*D.mass[k])))
            if D.time[k] < t0
                return -Inf
            else
                temp = log(p[5]/(p[6]+p[7])*(p[7]+p[8]*D.mass[k]*(exp(D.growth[k]*D.time[k])-1))) + (-p[5]/(p[6]+p[7])*((p[8]*D.mass[k])/D.growth[k]*(exp(D.growth[k]*D.time[k]) - exp(D.growth[k]*t0)) + (p[7]-p[8]*D.mass[k])*D.time[k] + (p[8]*D.mass[k]-p[7])*t0))
            end
            like += temp
        end
        return like + sum([logpdf(Gamma(p[1]^2/p[2],p[2]/p[1]),D.growth[k]) for k = 1:length(D.growth)]) + sum([logpdf(Beta(p[3]/p[4]*(p[3]*(1-p[3])-p[4]),(1-p[3])/p[4]*(p[3]*(1-p[3])-p[4])),D.divratio[k]) for k = 1:length(D.divratio)])
    end
end

function pm_logprior(p::Vector)
    # p = [o1,sig,b1,b2,o2,u,v,c]
    if p[6] >= p[7]
        return -Inf
    else
        return logpdf(pri_gamma,p[1]) + logpdf(pri_sigma,p[2]) + logpdf(pri_beta,p[3]) + logpdf(pri_sigma,p[4]) + sum([logpdf(pri_pm,p[k]) for k=5:length(p)])
    end
end



#### global setup
train_data,test_data = read_split_data("data/modified_Susman18_physical_units.csv")

pri_bm = Uniform(2,10);
pri_gamma = Uniform(0.8,1.8);
pri_beta = Uniform(0.4,0.6);
pri_sigma = Uniform(0,0.2);
pri_vm = Uniform(0,5);
pri_pm = Uniform(0.2,1.8);


#################### Compute posterior and predictive density of basic model
numdims = 4; numwalkers = 20; thinning = 10; numsamples_perwalker = 20000; burnin = 2000;
bm_logpost = x -> bm_loglikeli(x, train_data) + bm_logprior(x);

x = rand(pri_bm,numdims,numwalkers);
chain, llhoodvals = AffineInvariantMCMC.sample(bm_logpost,numwalkers,x,burnin,1);
chain, llhoodvals = AffineInvariantMCMC.sample(bm_logpost,numwalkers,chain[:, :, end],numsamples_perwalker,thinning);
chain, llhoodvals = remove_stuck_chain(chain,llhoodvals,numwalkers);
bm_flatchain,bm_flatllhood = AffineInvariantMCMC.flattenmcmcarray(chain,llhoodvals);

bm_mle = bm_flatchain[:,argmax(bm_flatllhood)]; # maximum likelihood estimates
bm_aic = 2*numdims - 2*bm_loglikeli(bm_mle,train_data)

bm_preddens = 1/size(bm_flatchain)[2]*sum([bm_loglikeli(bm_flatchain[:,k],test_data) for k=1:size(bm_flatchain)[2]])

#################### Compute posterior and predictive density of varying growth and division model
numdims = 7; numwalkers = 20; thinning = 10; numsamples_perwalker = 40000; burnin = 4000;
vm_logpost = x -> vm_loglikeli(x, train_data) + vm_logprior(x);

x = vcat(rand(pri_gamma,1,numwalkers),rand(pri_sigma,1,numwalkers),rand(pri_beta,1,numwalkers),rand(pri_sigma,1,numwalkers),rand(pri_vm,numdims-4,numwalkers));
chain, llhoodvals = AffineInvariantMCMC.sample(vm_logpost,numwalkers,x,burnin,1);
chain, llhoodvals = AffineInvariantMCMC.sample(vm_logpost,numwalkers,chain[:, :, end],numsamples_perwalker,thinning);
chain, llhoodvals = remove_stuck_chain(chain,llhoodvals,numwalkers);
vm_flatchain, vm_flatllhood = AffineInvariantMCMC.flattenmcmcarray(chain,llhoodvals);

vm_mle = vm_flatchain[:,argmax(vm_flatllhood)]; # maximum likelihood estimates
bm_aic = 2*numdims - 2*vm_loglikeli(vm_mle,train_data)

vm_preddens = 1/size(vm_flatchain)[2]*sum([vm_loglikeli(vm_flatchain[:,k],test_data) for k=1:size(vm_flatchain)[2]])


#################### Compute posterior and predictive density of protein model
const c = 1.; numdims = 7; numwalkers = 20; thinning = 10; numsamples_perwalker = 60000; burnin = 6000;
pm_logpost = x -> pm_loglikeli(vcat(x,c),readdata) + pm_logprior(vcat(x,c));

x = vcat(rand(pri_gamma,1,numwalkers),rand(pri_sigma,1,numwalkers),rand(pri_beta,1,numwalkers),rand(pri_sigma,1,numwalkers),rand(pri_pm,numdims-4,numwalkers));
chain, llhoodvals = AffineInvariantMCMC.sample(pm_logpost,numwalkers,x,burnin,1);
chain, llhoodvals = AffineInvariantMCMC.sample(pm_logpost,numwalkers,chain[:, :, end],numsamples_perwalker,thinning);
chain, llhoodvals = remove_stuck_chain(chain,llhoodvals,numwalkers);
pm_flatchain, pm_flatllhood = AffineInvariantMCMC.flattenmcmcarray(chain,llhoodvals);

pm_mle = pm_flatchain[:,argmax(pm_flatllhood)]; # maximum likelihood estimates
pm_aic = 2*numdims - 2*pm_loglikeli(pm_mle,train_data)

pm_preddens = 1/size(pm_flatchain)[2]*sum([pm_loglikeli(pm_flatchain[:,k],test_data) for k=1:size(pm_flatchain)[2]])

