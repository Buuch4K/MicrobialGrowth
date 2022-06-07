using Roots, Distributions, Statistics, StatsPlots, Plots, AffineInvariantMCMC, CSV, JLD2

struct Data
    time
    growth
    mass
    divratio
end


function generate_data(si::Float64,num::Int64)
    #= this function computes a synthetic data set of size num and initial cell size si.
    Input:  si - initial cell size
            num - number of division times
    Output: object DATA containing the data set
    =#
    X = Float64[si];
    Y = Array{Float64}(undef,num);
    k = rand(Uniform(0,1),num);
    alpha = rand(Gamma(o1^2/sig,sig/o1),num);
    f = rand(Beta((b1^2*(1-b1)-b2*b1)/b2,(b1*(1-b1)^2-b2*(1-b1))/b2),num);
    for n = 1:num
        t0 = 1/alpha[n]*log((u+c*X[n])/(c*X[n]));
        h = t -> log(k[n]) + o2/(u+v)*((c*X[n])/alpha[n]*exp(alpha[n]*(t+t0)) - c*X[n]*t + v*t - (c*X[n])/alpha[n]*exp(alpha[n]*t0))
        hx = ZeroProblem(h, 1)
        Y[n] = solve(hx)+t0;
        next_size = X[n] * exp(alpha[n]*Y[n]) * f[n]
        push!(X, next_size)
    end
    return Data(Y,alpha,X[1:num],f)
end

readdata = PMreal_data;
ex_read = Array{Float64}(undef,length(readdata.time),3);
ex_read[:,1] = vcat(readdata.divratio,0.49);
ex_read[:,2] = readdata.growth;
ex_read[:,3] = readdata.time;

corrplot(ex_read,label=["f","alpha","tau"],tickfontsize=4,guidefontsize=6)

const o1,sig,b1,b2,o2,u,v = PMreal_flatchain[:,argmax(PMreal_flatllhood)]; const c=1;
gendata = generate_data(2.6,2500);
ex_gen = Array{Float64}(undef,2500,3);
ex_gen[:,1] = gendata.divratio;
ex_gen[:,2] = gendata.growth;
ex_gen[:,3] = gendata.time;

corrplot(ex_gen,label=["f","alpha","tau"],tickfontsize=4,guidefontsize=6)


# scatter the three important data values to see their dependencies
scatter(readdata.divratio,readdata.growth,readdata.time,label="read");
scatter!(gendata.divratio,gendata.growth,gendata.time,label="synthetic");
plot!(xlabel="f",ylabel="alpha",zlabel="tau")

# scatter the claimed formula between f, alpha and tau.
scatter(gendata.divratio .* exp.(gendata.growth[2:end].*gendata.time[2:end]),label=false)



# save the real world and synthetic data with mle estimates to a csv file
CSV.write("pmsyn.csv",DataFrame(f=ex_gen[:,1],alpha=ex_gen[:,2],tau=ex_gen[:,3]));
CSV.write("pmreal.csv",DataFrame(f=ex_read[:,1],alpha=ex_read[:,2],tau=ex_read[:,3]));
