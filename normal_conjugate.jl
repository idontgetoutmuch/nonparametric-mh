using NPMH
using Turing, Random, Distributions
using DataFrames, CSV

"""
    NP-MH for conjugate normal
"""

# NP-MH

@model function conjugateNormal(data)
    mu ~ Normal(0.0,1.0)
    n = length(data)
    for i in 1:n
        data[i] ~ Normal(mu,1.0)
    end
    # Do we need to return anything? According to
    # https://turing.ml/v0.22/docs/using-turing/quick-start we don't
    # but the infiniteGMM example returns the number of components
    # return mu
end

cn = conjugateNormal(vcat(4.0))

pointwisek = PointwiseAuxKernel((n, t)-> Normal(t[n], 1.0))
cp = NPMHModel(pointwisek, cn)

rng = MersenneTwister(1729)
chain = sample(rng, cp, NPMHSampler(), 30000)

df = DataFrame(A = [chain[i][1] for i in 5001:30000])
CSV.write("normal-conjugate.csv", df)
