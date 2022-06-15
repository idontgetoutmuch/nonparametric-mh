using NPLiftedMH
using Turing, Random, Distributions
using DataFrames, CSV, PlotlyJS

"""
    NP-Lifted MH for infinite GMM
"""

seedrange = 1:10
iterations = 5000
datapath = "data/"
filename = "infinite_gmm"

# Data

rng = MersenneTwister(100)
data = vcat(randn(rng, 10), randn(rng, 10) .- 5, randn(rng, 10) .+ 10)
data .-= mean(data)
data /= std(data)

# Infinite GMM Model

@model function infiniteGMM(data)
    k ~ Poisson(3.0)
    # number of components
    K = Int(k)+1

    # means μs, variances vs and weights ws of each component
    μs = tzeros(Float64, K)
    vs = fill(0.3, K)
    ws = tzeros(Float64, K)
    for i in 1:K
        μs[i] ~ Normal(0.0,1.0)
        ws[i] ~ Normal(0.0,1.0)
    end
    # ws ~ Dirichlet(ones(K)/K)
    ws = abs.(ws .+ 0.5)
    ws = ws/sum(ws)

    # the mixture model
    m = MixtureModel(map((μ, v) -> Normal(μ, sqrt(v)), μs, vs), ws)

    # observe data from m
    n = length(data)
    for i in 1:n
        data[i] ~ m
    end
    return K
end
gmm = infiniteGMM(data)

# Sampling
function run_store(model, inference, infname, resultfn)
    println(infname)
    dictk = Dict()
    for i in 1:length(seedrange)
        seed = seedrange[i]
        rng = MersenneTwister(seed)
        chain = sample(rng, model, inference, iterations)
        dictk[join(["seed",string(seed)])] = resultfn(chain)
    end
    CSV.write(join([datapath, filename, "-", infname, ".csv"]), DataFrame(dictk))
end

# Lifted MH
minuskernel = PointwiseAuxKernel((n, t)-> n == 1 ? DiscreteUniform(max(0,t[n]-1),t[n]+1) : TruncatedNormal(t[n], 1.0, -Inf, t[n]))
pluskernel = PointwiseAuxKernel((n, t)-> n == 1 ? DiscreteUniform(max(0,t[n]-1),t[n]+1) : TruncatedNormal(t[n], 1.0, t[n], Inf))
dirk = DirectionAuxKernel(d -> d ? pluskernel : minuskernel)
npliftedmh = NPLiftedMHModel(dirk, gmm)
run_store(npliftedmh, NPLiftedMHSampler(), "NP-MH-P", chn->map(ts -> Int(ts[1])+1, chn))
