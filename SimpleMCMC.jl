module SimpleMCMC

    using Distributions, StatsPlots, Statistics, Printf

    # Version 6
    #  v6: Add Gibbs_MCMC and change module name from MHMCMC to SimpleMCMC
    #  v5: turn it into a module
    #  v4: remove prior from Parameter struct and the struct itself
    #  v3: create JumpingWidthVec, lbVec, ubVec  to improve efficiency
    #  v2: Remove q0 from the main loop to improve efficiency
    #  v1: Turn it into a version that uses pure function(s)

    function MetropolisHastings(data, lop::Vector{Distribution},loglikelihood;samples,burnedinsamples,JumpingWidth=0.01)
        # Find the number of parameters
        numofparam = length(lop)
        # calc the n_samples
        n_samples = samples + burnedinsamples
        # Create a parameter Matrix p with n_samples rows and numofparam cols
        p = zeros(n_samples,numofparam)
        # JumpingWidthVector, lowerboundVector and upperboundbVector
        JumpingWidthVec = zeros(numofparam)
        lbVec = zeros(numofparam)
        ubVec = zeros(numofparam)
        # Create a Vector of our logprior function
        logpriorVec = Array{Function}(undef,numofparam)

        # Set the starting value to a random number between lb and ub
        for k = 1:numofparam  # for each parameter k
            # Set the starting value to a random number between lb and ub
            p[1,k] = rand(lop[k])

            # get 100 random values and find the practical max and practical min
            HundredRandValues = rand(lop[k],100)
            prac_min = minimum(HundredRandValues)
            prac_max = maximum(HundredRandValues)
            JumpingWidthVec[k] = JumpingWidth * (prac_max - prac_min)

            lbVec[k] = minimum(lop[k])
            ubVec[k] = maximum(lop[k])

            # Array of function prior(x) = pdf(distribution,x)
            logpriorVec[k] = x->logpdf(lop[k],x)
        end

        p_old = vec(p[1,:])
        q0 = loglikelihood(data,p_old...) + sum([logpriorVec[k](p_old[k]) for k = 1:numofparam])

        # prepare the p_new array
        p_new = zeros(numofparam)
        # Main loop for MetropolisHastings MCMC
        for i = 2:n_samples
            for k = 1:numofparam  # for each parameter k
                # p_new has a value around the vicinity of p[i-1]
                p_new[k] = rand(Normal(p[i-1,k] , JumpingWidthVec[k]))
                # make sure p_new is between lb and ub
                if p_new[k] < lbVec[k]
                    p_new[k] = lbVec[k] + abs(p_new[k] - lbVec[k])
                elseif p_new[k] > ubVec[k]
                    p_new[k] = ubVec[k] - abs(p_new[k] - ubVec[k])
                end
            end

            # Calc the two posterior
            " q0 is posterior0 = likelihood0 * prior0        "
            " q1 is posterior1 = likelihood1 * prior1        "
            q1 = loglikelihood(data,p_new...) + sum([logpriorVec[k](p_new[k]) for k = 1:numofparam ])
            # The value of p[i] depends on whether the
            # random number is less than q1/q0
            p[i,:] .= log(rand()) < q1-q0 ? (q0=q1; p_old=p_new[:]) : p_old
        end
        # Finally we must not forget to remove the burned in samples
        return p[(burnedinsamples+1):end,:]
    end

    function Gibbs_MCMC(data,lop::Vector{Distribution},loglikelihood;samples,burnedinsamples)
        # Find the number of parameters
        numofparam = length(lop)
        # calc the n_samples
        n_samples = samples + burnedinsamples
        # Create a parameter Matrix p with n_samples rows and numofparam cols
        p = zeros(n_samples,numofparam)
        # Create a Vector of our logprior function
        logpriorVec = Array{Function}(undef,numofparam)
        # Number of holes to drill
        numofdrillholes = 1001
        drillhole = Array{Float64}(undef,numofdrillholes)
        invcdf = Array{Float64}(undef,numofdrillholes+1)
        # prepare the vectors: startposvec stopposvec chunksizevec
        startposvec = zeros(numofparam)
        stopposvec = zeros(numofparam)
        chunksizevec = zeros(numofparam)

        # Set the starting value to a random number between lb and ub
        for k = 1:numofparam  # for each parameter k
            # Set the starting value to a random number between lb and ub
            p[1,k] = rand(lop[k])

            # get 400 random values and find the practical max and practical min
            FourHundredRandValues = rand(lop[k],400)
            startposvec[k] = minimum(FourHundredRandValues)
            stopposvec[k]  = maximum(FourHundredRandValues)
            chunksizevec[k] = (stopposvec[k]-startposvec[k])/(numofdrillholes-1)

            # Array of function prior(x) = pdf(distribution,x)
            logpriorVec[k] = x->logpdf(lop[k],x)
        end

        # Main loop for Gibbs MCMC
        for i = 2:n_samples
            # prepare the p_new vector
            p_new = vec(p[i-1,:])
            # Select a new value for each parameter of p_new
            for k = 1:numofparam  # for each parameter k
                # Calculate chunk size
                startpos,stoppos,chunksize = startposvec[k],stopposvec[k],chunksizevec[k]
                invcdf[1] = 0.0
                # Start drilling the drill holes for parameter k
                for n = 1:numofdrillholes
                    p_new[k] = startpos + (n-1) * chunksize
                    drillhole[n] = exp(  loglikelihood(data,p_new...)  )
                    invcdf[n+1] = invcdf[n] + drillhole[n]
                end
                invcdf /= invcdf[end]
                # Now we perform an inverse CDF sampling
                x = 0.0
                while x == 0.0
                    x = rand()  # Make sure x is not zero
                end
                counter = 2
                while !(invcdf[counter-1] < x <= invcdf[counter])
                    counter += 1
                end
                counter -= 1
                # now find the kth parameter value
                pos = startpos + (counter-1) * chunksize
                # This is the value we are going to jump to
                p_new[k] = pos
                p[i,k] = pos
            end
        end
        # Finally we must not forget to remove the burned in samples
        return p[(burnedinsamples+1):end,:]
    end

    function describe_paramvec(name,namestr,v::Vector)
        m,s,v_sorted = mean(v),std(v),sort(v)
        pc5 = v_sorted[Int64(round(0.055*length(v_sorted)))]
        pc94 = v_sorted[Int64(round(0.945*length(v_sorted)))]
        println(
            "Parameter $(name)   mean ",@sprintf("%6.3f",round(m,sigdigits=3)),
            "   sd ",@sprintf("%6.3f",round(s,sigdigits=3)),
            "   5.5% ",@sprintf("%6.3f",round(pc5,sigdigits=3)),
            "   94.5% ",@sprintf("%6.3f",round(pc94,sigdigits=3))
        )
        histogram(v,legend=false,title="Posterior of $(namestr)") |> display
        return (m,s,pc5,pc94)
    end

end