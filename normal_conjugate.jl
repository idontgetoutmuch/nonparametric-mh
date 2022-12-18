using NPMH
using Turing, Random, Distributions
using DataFrames, CSV, PlotlyJS

"""
    NP-MH for conjugate normal
"""

# NP-MH
pointwisek = PointwiseAuxKernel((n, t)-> n == 1 ? DiscreteUniform(max(0,t[n]-1),t[n]+1) : Normal(t[n], 1.0))

@model function conjugateNormal(data)
    # This does nothing but without it NPMH becomes unhappy
    k ~ Poisson(3.0)
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

@model function conjugateNormak(data)
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
cm = conjugateNormak(vcat(4.0))

rng = MersenneTwister(1729)

pointwisel = PointwiseAuxKernel((n, t)-> Normal(t[n], 1.0))

cp = NPMHModel(pointwisek, cn)

cq = NPMHModel(pointwisel, cm)

chain = sample(rng, cp, NPMHSampler(), 15000)

chaim = sample(rng, cq, NPMHSampler(), 15000)

df = DataFrame(A = [chain[i][2] for i in 5001:15000])

dg = DataFrame(A = [chaim[i][1] for i in 5001:15000])

CSV.write("foo-julia.csv", dg)
