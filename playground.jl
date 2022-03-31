using Roots

f(x) = exp(x)-x^4
cand = 8
fx = ZeroProblem(f,cand)

x0 = find_zero(f,(8,9))

x1 = solve(fx)
println("One root is $x0")
println("Another method yields to the root $x1")


using AffineInvariantMCMC, Statistics, Plots, Random, Distributions, RobustPmap

numdims = 5;
numwalkers = 100;
thinning = 10;
numsamples_perwalker = 100;
burnin = 100;

const stds = exp.(5 * randn(numdims));
const means = 1 .+ 5 * rand(numdims);

llhood = x->begin
	retval = 0.
	for i in 1:length(x)
		retval -= .5 * ((x[i] - means[i]) / stds[i]) ^ 2
	end
	return retval
end

x0 = rand(numdims, numwalkers) * 10 .- 5
chain, llhoodvals = AffineInvariantMCMC.sample(llhood, numwalkers, x0, burnin, 1);
chain, llhoodvals = AffineInvariantMCMC.sample(llhood, numwalkers, chain[:, :, end], numsamples_perwalker, thinning);
flatchain, flatllhoodvals = AffineInvariantMCMC.flattenmcmcarray(chain, llhoodvals);


histogram(flatchain[1,:])
mean(flatchain[1,:])


using InvertibleNetworks

